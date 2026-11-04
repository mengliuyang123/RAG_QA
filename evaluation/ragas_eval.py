# -*- coding: utf-8 -*-
"""RAGAS 0.4 collections API 的唯一适配与兼容边界。"""

import asyncio
import importlib
import math
import sys
import types
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings


RAGAS_MAX_TOKENS = 4096


class RagasScoreError(RuntimeError):
    """RAGAS 指标结果不是可发布的有限数值时抛出。"""


class AsyncMetric(Protocol):
    """RAGAS collections 指标使用的最小异步接口。"""

    async def ascore(self, **kwargs: object) -> object:
        """根据指标专用字段返回带 `value` 的结果。

        Args:
            **kwargs: 指标声明的固定输入字段。

        Returns:
            具有数值 `value` 属性的结果对象。
        """


class RagasCase(BaseModel):
    """传给三项 RAGAS 指标的固定字段。

    Attributes:
        user_input: 原始问题。
        response: RAG 生成回答。
        retrieved_contexts: 本次实际检索的有序上下文正文。
        reference: 人工参考答案。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_input: str = Field(min_length=1)
    response: str = Field(min_length=1)
    retrieved_contexts: list[str] = Field(min_length=1)
    reference: str = Field(min_length=1)


class RagasScores(BaseModel):
    """三项 RAGAS 原始与报告归一化分数。

    Attributes:
        faithfulness: 有界忠实度原始值。
        answer_relevancy_raw: 可能因余弦相似度略超范围的原始值。
        answer_relevancy: 限制到零到一的报告值。
        answer_correctness: 有界回答正确性原始值。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy_raw: float
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    answer_correctness: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class RagasComponents:
    """延迟导入的 RAGAS factories 与 collections 类型。

    Attributes:
        llm_factory: RAGAS 0.4 LLM factory。
        embedding_factory: RAGAS 0.4 embedding factory。
        faithfulness_type: Faithfulness collections 类。
        answer_relevancy_type: AnswerRelevancy collections 类。
        answer_correctness_type: AnswerCorrectness collections 类。
    """

    llm_factory: Any
    embedding_factory: Any
    faithfulness_type: type[Any]
    answer_relevancy_type: type[Any]
    answer_correctness_type: type[Any]


class RagasEvaluator:
    """显式持有三项 RAGAS collections 指标的同步适配器。"""

    def __init__(
        self,
        faithfulness: AsyncMetric,
        answer_relevancy: AsyncMetric,
        answer_correctness: AsyncMetric,
    ) -> None:
        """保存已经显式装配的指标实例。

        Args:
            faithfulness: 忠实度指标。
            answer_relevancy: 回答相关性指标。
            answer_correctness: 回答正确性指标。
        """
        self._faithfulness = faithfulness
        self._answer_relevancy = answer_relevancy
        self._answer_correctness = answer_correctness

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        components: RagasComponents | None = None,
        async_client_type: type[Any] = AsyncOpenAI,
    ) -> "RagasEvaluator":
        """使用独立裁判和本次索引 Embedding 显式装配指标。

        Args:
            settings: 独立裁判与 Embedding 配置。
            components: 测试可注入的 RAGAS factories 和指标类型。
            async_client_type: 测试可替换的异步 OpenAI 客户端类型。

        Returns:
            不依赖 RAGAS 默认模型的三指标适配器。
        """
        settings.validate_evaluation_settings(require_judge=True)
        if components is None:
            components = load_ragas_components()
        judge_key = settings.evaluation_judge_api_key
        judge_url = settings.evaluation_judge_base_url
        judge_model = settings.evaluation_judge_model
        if judge_key is None or judge_url is None or judge_model is None:
            raise ValueError("独立裁判配置不完整")
        judge_client = async_client_type(
            api_key=judge_key,
            base_url=judge_url,
        )
        embedding_client = async_client_type(
            api_key=settings.resolved_embedding_api_key,
            base_url=settings.embedding_base_url,
        )
        llm_options: dict[str, object] = {
            "temperature": settings.evaluation_judge_temperature,
            "max_tokens": RAGAS_MAX_TOKENS,
        }
        if settings.evaluation_judge_reasoning_effort is not None:
            llm_options["reasoning_effort"] = (
                settings.evaluation_judge_reasoning_effort
            )
        if _requires_non_thinking_mode(judge_url, judge_model):
            # DeepSeek V4 仅在非思考模式支持结构化输出；关闭思考也避免
            # 推理 token 占满 RAGAS 的结构化回答预算。
            llm_options["extra_body"] = {"enable_thinking": False}
        llm = components.llm_factory(
            model=judge_model,
            provider="openai",
            client=judge_client,
            **llm_options,
        )
        embeddings = components.embedding_factory(
            provider="openai",
            model=settings.embedding_model,
            client=embedding_client,
            interface="modern",
        )
        return cls(
            faithfulness=components.faithfulness_type(llm=llm),
            answer_relevancy=components.answer_relevancy_type(
                llm=llm,
                embeddings=embeddings,
            ),
            answer_correctness=components.answer_correctness_type(
                llm=llm,
                embeddings=embeddings,
            ),
        )

    def score(self, case: RagasCase) -> RagasScores:
        """同步执行三项指标并保留 Answer Relevancy 原始值。

        Args:
            case: 固定问题、回答、上下文和参考答案。

        Returns:
            经过有限性检查与报告归一化的三项分数。

        Raises:
            RagasScoreError: 任一指标不是有限数值或有界指标越界。
        """
        return asyncio.run(self._score_async(case))

    async def _score_async(self, case: RagasCase) -> RagasScores:
        faithfulness_result = await self._faithfulness.ascore(
            user_input=case.user_input,
            response=case.response,
            retrieved_contexts=case.retrieved_contexts,
        )
        relevancy_result = await self._answer_relevancy.ascore(
            user_input=case.user_input,
            response=case.response,
        )
        correctness_result = await self._answer_correctness.ascore(
            user_input=case.user_input,
            response=case.response,
            reference=case.reference,
        )
        faithfulness = _finite_metric_value(faithfulness_result)
        relevancy_raw = _finite_metric_value(relevancy_result)
        correctness = _finite_metric_value(correctness_result)
        if not 0.0 <= faithfulness <= 1.0:
            raise RagasScoreError("Faithfulness 必须在 0 到 1 之间")
        if not 0.0 <= correctness <= 1.0:
            raise RagasScoreError("Answer Correctness 必须在 0 到 1 之间")
        relevancy = min(1.0, max(0.0, relevancy_raw))
        return RagasScores(
            faithfulness=faithfulness,
            answer_relevancy_raw=relevancy_raw,
            answer_relevancy=relevancy,
            answer_correctness=correctness,
        )


def load_ragas_components() -> RagasComponents:
    """延迟导入真实 RAGAS 0.4 factories 和 collections 类型。

    Returns:
        本项目固定使用的 factories 和三项指标类型。
    """
    _install_ragas_vertexai_compatibility()
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        Faithfulness,
    )

    return RagasComponents(
        llm_factory=llm_factory,
        embedding_factory=embedding_factory,
        faithfulness_type=Faithfulness,
        answer_relevancy_type=AnswerRelevancy,
        answer_correctness_type=AnswerCorrectness,
    )


def _finite_metric_value(result: object) -> float:
    try:
        value = float(getattr(result, "value"))
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise RagasScoreError("RAGAS 指标必须是有限数值") from exc
    if not math.isfinite(value):
        raise RagasScoreError("RAGAS 指标必须是有限数值")
    return value


def _requires_non_thinking_mode(base_url: str, model: str) -> bool:
    hostname = urlsplit(base_url).hostname
    if hostname is None or not hostname.lower().endswith(".aliyuncs.com"):
        return False
    return model.strip().lower().startswith("deepseek-v4")


def _install_ragas_vertexai_compatibility() -> None:
    """兼容 RAGAS 0.4.3 对已移除 VertexAI 模块的无条件导入。"""
    module_name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

        class LegacyChatVertexAI:
            pass

        compatibility_module = types.ModuleType(module_name)
        compatibility_module.ChatVertexAI = LegacyChatVertexAI
        sys.modules[module_name] = compatibility_module

    community_llms = importlib.import_module("langchain_community.llms")
    if "VertexAI" not in community_llms.__dict__:

        class LegacyVertexAI:
            pass

        community_llms.VertexAI = LegacyVertexAI
