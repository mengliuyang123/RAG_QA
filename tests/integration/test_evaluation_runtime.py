# -*- coding: utf-8 -*-
"""使用真实 `.env` 验证隔离评估索引和三种检索策略。"""

from pathlib import Path

import pytest

from app.config import Settings
from app.domain.retrieval import RetrievalStrategy
from evaluation.datasets import load_corpus, load_qa_dataset
from evaluation.judge import JudgeClient
from evaluation.runtime import build_evaluation_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = PROJECT_ROOT / "evaluation"


@pytest.mark.integration
def test_real_evaluation_runtime_runs_one_qa_with_all_strategies(
    tmp_path: Path,
) -> None:
    """真实服务应在临时隔离索引上完成一道题的三策略查询。"""
    settings = Settings()
    missing = []
    if not settings.resolved_llm_api_key:
        missing.append("LLM_API_KEY")
    if not settings.resolved_embedding_api_key:
        missing.append("EMBEDDING_API_KEY")
    if not settings.resolved_reranker_api_key:
        missing.append("RERANKER_API_KEY")
    if not settings.reranker_base_url:
        missing.append("RERANKER_BASE_URL")
    judge_values = (
        settings.evaluation_judge_api_key,
        settings.evaluation_judge_base_url,
        settings.evaluation_judge_model,
    )
    if not all(judge_values):
        missing.append("EVALUATION_JUDGE_*")
    if missing:
        pytest.skip("缺少真实评估凭据：" + ", ".join(missing))

    isolated = settings.model_copy(
        update={
            "evaluation_chroma_path": tmp_path / "chroma",
            "evaluation_collection_prefix": "integration_eval",
        }
    )
    isolated.validate_evaluation_settings(require_judge=True)
    corpus = load_corpus(
        EVALUATION_ROOT / "corpus",
        EVALUATION_ROOT / "corpus_manifest.json",
    )
    qa_cases = load_qa_dataset(
        EVALUATION_ROOT / "qa_dataset.json",
        corpus,
    )
    case = qa_cases[0]
    runtime = build_evaluation_runtime(isolated, corpus, rebuild=True)
    try:
        for strategy in RetrievalStrategy:
            result = runtime.query_service.evaluate_query(
                case.question,
                strategy,
                top_k=5,
            )
            assert result.retrieval_strategy is strategy
            assert result.answer.strip()
            assert len(result.retrieved) <= 5
    finally:
        runtime.close()


@pytest.mark.integration
def test_real_independent_judge_returns_bounded_structured_score() -> None:
    """独立裁判应为明确正确的合成回答返回严格有界分数。"""
    settings = Settings()
    judge_values = (
        settings.evaluation_judge_api_key,
        settings.evaluation_judge_base_url,
        settings.evaluation_judge_model,
    )
    if not all(judge_values):
        pytest.skip("缺少真实独立裁判配置：EVALUATION_JUDGE_*")
    settings.validate_evaluation_settings(require_judge=True)

    judge = JudgeClient.from_settings(settings)
    result = judge.score_semantic_correctness(
        question="中国的首都是哪里？",
        reference="中国的首都是北京。",
        response="中国的首都是北京。",
    )

    assert 0.0 <= result.score <= 1.0
    assert result.reason.strip()
