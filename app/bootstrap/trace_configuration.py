# -*- coding: utf-8 -*-
"""只从显式非敏感执行参数生成配置指纹，不哈希整份Settings。"""

import hashlib
import json

from app.application.tracing.recorder import normalize_origin
from app.config import Settings


_EXECUTION_OPTIONS = (
    "llm_model", "llm_reasoning_effort", "llm_timeout_seconds",
    "llm_max_output_tokens", "llm_output_token_parameter", "model_max_retries",
    "embedding_model", "embedding_timeout_seconds", "embedding_batch_size",
    "reranker_model", "reranker_endpoint", "quality_output_mode",
    "understanding_temperature", "understanding_confidence_threshold",
    "agent_temperature", "agent_max_retrieval_calls", "agent_turn_timeout_seconds",
    "conversation_context_tokens", "conversation_reserved_tokens",
    "conversation_assistant_token_limit", "conversation_history_batch_size",
    "top_k", "top_k_rerank", "chunk_size", "chunk_overlap",
    "question_rate_limit_per_minute", "question_max_active",
    "question_max_waiting", "question_queue_timeout_seconds",
)


def trace_configuration_version(
    settings: Settings, generation_temperature: float | None,
) -> str:
    """区分启动执行参数与运行时温度，不纳入凭据、查询或本地路径。

    Args:
        settings: 已验证的冻结设置。
        generation_temperature: 实际生成适配器的温度覆盖，None表示未指定。

    Returns:
        流水线版本加非敏感参数的稳定SHA256指纹。
    """
    payload = {}
    for name in _EXECUTION_OPTIONS:
        payload[name] = getattr(settings, name)
    payload.update(
        generation_temperature=generation_temperature,
        generation_origin=normalize_origin(settings.llm_base_url),
        embedding_origin=normalize_origin(settings.embedding_base_url),
        reranker_origin=normalize_origin(settings.reranker_base_url or ""),
        understanding_origin=normalize_origin(settings.resolved_understanding_base_url),
        understanding_model=settings.resolved_understanding_model,
        # 记录实际生效值（含沿用 LLM_* 的结果），而不是是否显式配置。
        understanding_reasoning_effort=settings.resolved_understanding_reasoning_effort,
        understanding_timeout_seconds=settings.resolved_understanding_timeout_seconds,
        understanding_max_output_tokens=settings.resolved_understanding_max_output_tokens,
        understanding_output_token_parameter=(
            settings.resolved_understanding_output_token_parameter
        ),
        understanding_context_tokens=settings.resolved_understanding_context_tokens,
        understanding_output_mode=settings.resolved_understanding_output_mode,
        judge_origin=normalize_origin(settings.resolved_judge_base_url),
        judge_model=settings.resolved_judge_model,
        judge_reasoning_effort=settings.resolved_judge_reasoning_effort,
        judge_timeout_seconds=settings.resolved_judge_timeout_seconds,
        retrieval_profile=settings.knowledge_retrieval_strategy.value,
        price_versions=[item.model_dump(mode="json") for item in settings.trace_model_prices],
    )
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    return f"layered-routing-v1:{digest}"
