# -*- coding: utf-8 -*-
"""RAGAS collections 适配器的输入与分数归一化测试。"""

from types import SimpleNamespace

import pytest


class _FakeMetric:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[dict[str, object]] = []

    async def ascore(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value)


class _BudgetSensitiveMetric:
    def __init__(self, llm: dict[str, object], **_kwargs: object) -> None:
        self.llm = llm

    async def ascore(self, **_kwargs: object) -> object:
        max_tokens = int(self.llm["max_tokens"])
        if max_tokens < 4096:
            raise RuntimeError("结构化输出预算不足")
        extra_body = self.llm.get("extra_body")
        if extra_body != {"enable_thinking": False}:
            raise RuntimeError("结构化输出不得启用思考模式")
        return SimpleNamespace(value=0.5)


def test_ragas_evaluator_calls_each_collection_metric_exactly() -> None:
    """三项指标必须接收各自固定字段，不能依赖 RAGAS 默认输入。"""
    from evaluation.ragas_eval import RagasCase, RagasEvaluator

    faithfulness = _FakeMetric(0.8)
    relevancy = _FakeMetric(-0.2)
    correctness = _FakeMetric(0.9)
    evaluator = RagasEvaluator(
        faithfulness=faithfulness,
        answer_relevancy=relevancy,
        answer_correctness=correctness,
    )
    case = RagasCase(
        user_input="问题",
        response="回答",
        retrieved_contexts=["上下文一", "上下文二"],
        reference="参考答案",
    )

    scores = evaluator.score(case)

    assert faithfulness.calls == [
        {
            "user_input": "问题",
            "response": "回答",
            "retrieved_contexts": ["上下文一", "上下文二"],
        }
    ]
    assert relevancy.calls == [
        {"user_input": "问题", "response": "回答"}
    ]
    assert correctness.calls == [
        {
            "user_input": "问题",
            "response": "回答",
            "reference": "参考答案",
        }
    ]
    assert scores.faithfulness == 0.8
    assert scores.answer_relevancy_raw == -0.2
    assert scores.answer_relevancy == 0.0
    assert scores.answer_correctness == 0.9


def test_ragas_evaluator_rejects_non_finite_metric_result() -> None:
    """NaN 不得静默进入公开报告。"""
    from evaluation.ragas_eval import RagasCase, RagasEvaluator, RagasScoreError

    evaluator = RagasEvaluator(
        faithfulness=_FakeMetric(float("nan")),
        answer_relevancy=_FakeMetric(0.5),
        answer_correctness=_FakeMetric(0.5),
    )
    case = RagasCase(
        user_input="问题",
        response="回答",
        retrieved_contexts=["上下文"],
        reference="参考",
    )

    with pytest.raises(RagasScoreError, match="必须是有限数值"):
        evaluator.score(case)


def test_ragas_from_settings_supports_long_structured_output() -> None:
    """独立裁判必须为 RAGAS 结构化输出提供足够的 token 预算。"""
    from app.config import Settings
    from evaluation.ragas_eval import (
        RagasCase,
        RagasComponents,
        RagasEvaluator,
    )

    def llm_factory(**kwargs: object) -> dict[str, object]:
        return {
            "max_tokens": kwargs.get("max_tokens", 1024),
            "extra_body": kwargs.get("extra_body"),
        }

    def embedding_factory(**_kwargs: object) -> object:
        return object()

    class FakeAsyncClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

    settings = Settings(
        llm_api_key="generator-key",
        llm_base_url="https://generator.example/v1",
        llm_model="generator-model",
        embedding_api_key="embedding-key",
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        evaluation_judge_api_key="judge-key",
        evaluation_judge_base_url=(
            "https://dashscope.aliyuncs.com/compatible-api/v1"
        ),
        evaluation_judge_model="deepseek-v4-pro-0813",
        _env_file=None,
    )
    components = RagasComponents(
        llm_factory=llm_factory,
        embedding_factory=embedding_factory,
        faithfulness_type=_BudgetSensitiveMetric,
        answer_relevancy_type=_BudgetSensitiveMetric,
        answer_correctness_type=_BudgetSensitiveMetric,
    )
    evaluator = RagasEvaluator.from_settings(
        settings,
        components=components,
        async_client_type=FakeAsyncClient,
    )
    case = RagasCase(
        user_input="问题",
        response="回答",
        retrieved_contexts=["上下文"],
        reference="参考答案",
    )

    scores = evaluator.score(case)

    assert scores.faithfulness == 0.5
    assert scores.answer_relevancy == 0.5
    assert scores.answer_correctness == 0.5


def test_ragas_forwards_explicit_judge_reasoning_effort() -> None:
    """RAGAS 三项裁判指标必须使用显式配置的推理强度。"""
    from app.config import Settings
    from evaluation.ragas_eval import RagasComponents, RagasEvaluator

    llm_calls: list[dict[str, object]] = []

    def llm_factory(**kwargs: object) -> object:
        llm_calls.append(kwargs)
        return object()

    def embedding_factory(**_kwargs: object) -> object:
        return object()

    def metric_factory(**_kwargs: object) -> object:
        return object()

    class FakeAsyncClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

    settings = Settings(
        llm_api_key="generator-key",
        llm_base_url="https://generator.example/v1",
        llm_model="generator-model",
        embedding_api_key="embedding-key",
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        evaluation_judge_api_key="judge-key",
        evaluation_judge_base_url=(
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        evaluation_judge_model="qwen3.8-max",
        evaluation_judge_reasoning_effort="none",
        _env_file=None,
    )
    components = RagasComponents(
        llm_factory=llm_factory,
        embedding_factory=embedding_factory,
        faithfulness_type=metric_factory,
        answer_relevancy_type=metric_factory,
        answer_correctness_type=metric_factory,
    )

    RagasEvaluator.from_settings(
        settings,
        components=components,
        async_client_type=FakeAsyncClient,
    )

    assert llm_calls[0]["reasoning_effort"] == "none"


def test_ragas_04_import_smoke() -> None:
    """eval 依赖组必须能导入实际 RAGAS 0.4 collections API。"""
    from evaluation.ragas_eval import load_ragas_components

    components = load_ragas_components()

    assert components.faithfulness_type.__name__ == "Faithfulness"
    assert components.answer_relevancy_type.__name__ == "AnswerRelevancy"
    assert components.answer_correctness_type.__name__ == "AnswerCorrectness"
