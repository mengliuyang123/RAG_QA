# -*- coding: utf-8 -*-
"""只提取供应商报告的计数，兼容原始SDK和LangChain消息。"""

from collections.abc import Mapping

from app.domain.tracing import TokenUsage


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def extract_token_usage(value: object, *, no_generation: bool = False) -> TokenUsage:
    """兼容已知字段，不把total_tokens猜测成输入Token。

    Args:
        value: SDK usage对象或LangChain usage_metadata。
        no_generation: Embedding/重排没有生成Token，输入已报告时输出为零。

    Returns:
        实际报告的有效计数，缺字段保持None。
    """
    input_count = _count(_field(value, "prompt_tokens"))
    if input_count is None:
        input_count = _count(_field(value, "input_tokens"))
    output_count = _count(_field(value, "completion_tokens"))
    if output_count is None:
        output_count = _count(_field(value, "output_tokens"))
    if no_generation and input_count is not None:
        output_count = 0
    cached = _count(_field(_field(value, "prompt_tokens_details"), "cached_tokens"))
    if cached is None:
        cached = _count(_field(_field(value, "input_token_details"), "cache_read"))
    reasoning = _count(_field(_field(value, "completion_tokens_details"), "reasoning_tokens"))
    if reasoning is None:
        reasoning = _count(_field(_field(value, "output_token_details"), "reasoning"))
    if cached is not None and input_count is not None and cached > input_count:
        cached = None
    if reasoning is not None and output_count is not None and reasoning > output_count:
        reasoning = None
    return TokenUsage(
        input=input_count, output=output_count,
        cached_input=cached, reasoning_output=reasoning,
    )
