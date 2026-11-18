from tests.postgres_support import pg_database, pg_connect
# -*- coding: utf-8 -*-
"""默认入口只计划13轮，dry-run不得构造模型、打开索引或创建运行目录。"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.config import Settings
from app.bootstrap.container import build_application_container
from app.bootstrap.factories import AdapterFactories
from app.application.ports.structured_output import ResponseModel
from app.infrastructure.conversations.langgraph_agent import LangGraphConversationAgent
from evaluation.runtime import build_evaluation_runtime
from tests.evaluation.test_runtime import _FakeEmbedder
from tests.test_fast_lane import _Provider
from tests.test_scoped_agent import _Quality
from tests.test_unified_query_api import _Model
from tests.understanding import OfflineUnderstandingProvider
from tests.corpus_state import CORPUS_MISSING_REASON
from tests.corpus_state import evaluation_corpus_complete


@pytest.mark.skipif(
    not evaluation_corpus_complete(), reason=CORPUS_MISSING_REASON,
)
def test_dry_run_uses_env_settings_without_network_or_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """默认扩大为150条、打印密钥或提前构造资源都应使此测试失败。

    Args:
        tmp_path: 未创建的work路径的父目录。
        monkeypatch: 禁止三个有副作用的资源入口。
        capsys: 捕获安全配置概览。
    """
    from evaluation.lanes import cli

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run不得创建远程或存储资源")

    for name in ("build_evaluation_runtime", "build_application_container",
                 "create_evaluation_judge_provider"):
        monkeypatch.setattr(cli, name, forbidden)
    settings = Settings(
        _env_file=None, llm_api_key="generator-secret", llm_model="generator",
        llm_base_url="https://generator.example/v1",
        embedding_api_key="embedding-secret",
        reranker_base_url=None,
        evaluation_judge_api_key="judge-secret",
        evaluation_judge_model="judge",
        evaluation_judge_base_url="https://judge.example/v1",
    )
    work = tmp_path / "work"
    assert cli.main(["--dry-run", "--work-root", str(work)], settings=settings) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["planned_turns"] == 1
    assert payload["planned_episodes"] == 1
    assert payload["strategy"] == "hybrid"
    assert payload["purpose"] == "connectivity_smoke"
    assert "generator-secret" not in output
    assert "embedding-secret" not in output
    assert "judge-secret" not in output
    assert not work.exists()


@pytest.mark.skipif(
    not evaluation_corpus_complete(), reason=CORPUS_MISSING_REASON,
)
def test_full_offline_entry_isolated_and_rescore_does_not_open_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全部固定片段的CLI接线不能碰生产库，续跑评分不能重新准备索引。

    Args:
        tmp_path: 临时的生产哨兵路径和独立评估数据。
        monkeypatch: 仅替代外部模型与Embedding响应边界。
    """
    from evaluation.lanes import cli

    quality = _Quality()
    embedder = _FakeEmbedder()
    factories = replace(
        AdapterFactories(), embedding_provider=lambda *args: embedder,
        llm_provider=lambda *args: _Provider("资料[1]"),
        quality_evaluator=lambda *args: quality,
        understanding_provider=lambda *args: OfflineUnderstandingProvider(),
        conversation_agent=lambda key, url, model, temperature, effort, retriever, saver, top_k, max_calls: LangGraphConversationAgent(
            _Model(), retriever, saver, top_k, max_retrieval_calls=max_calls,
            quality_evaluator=quality,
        ),
    )

    def runtime(settings, corpus):
        return build_evaluation_runtime(settings, corpus, container_builder=lambda s, options: build_application_container(s, options, factories=factories))

    class JudgeProvider:
        """只替代外部独立裁判，其余CLI、窗口、图、索引与存储保持真实。"""

        def generate(self, messages: list[dict[str, object]],
                     response_model: type[ResponseModel]) -> ResponseModel:
            """按适用范围返回合法评分，分数不冒充真实模型评估。

            Args:
                messages: 实际评分输入。
                response_model: 严格评分schema。

            Returns:
                独立评分边界的离线替身。
            """
            payload = json.loads(str(messages[-1]["content"]))
            return response_model.model_validate({
                "correctness": 0.5,
                "supported": False if payload["knowledge_answer"] else None,
                "complete": False, "reason": "离线接线测试，不是能力评分",
            })

        def close(self) -> None:
            """离线响应没有外部资源。

            Returns:
                None。
            """

    monkeypatch.setattr(cli, "build_evaluation_runtime", runtime)
    monkeypatch.setattr(cli, "create_evaluation_judge_provider", lambda settings: JudgeProvider())
    production = tmp_path / "production"
    settings = Settings(
        _env_file=None, llm_api_key="generator", llm_model="generator",
        llm_base_url="https://generator.example/v1", embedding_api_key="embedding",
        evaluation_judge_api_key="judge",
        evaluation_judge_model="judge",
        evaluation_judge_base_url="https://judge.example/v1", reranker_base_url=None,
        artifacts_path=production / "chroma", docs_path=production / "docs",




        evaluation_artifacts_path=tmp_path / "evaluation/chroma",
    )
    work = tmp_path / "work"
    assert cli.main(["--full", "--work-root", str(work)], settings=settings) == 0
    reports = list(work.rglob("report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["generated_turns"] == report["scored_turns"] == 6
    assert report["status"] == "complete"
    # 现有 4 个片段都没有“检索后作答”的轮次，不会写来源文件，所以不再断言评估产物目录存在。
    assert not production.exists()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("评分或完全复用不能打开或重新向量化索引")

    monkeypatch.setattr(cli, "build_evaluation_runtime", forbidden)
    assert cli.main(["--full", "--work-root", str(work), "--phase", "score"], settings=settings) == 0
    assert not production.exists()
