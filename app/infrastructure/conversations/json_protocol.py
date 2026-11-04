# -*- coding: utf-8 -*-
"""从模型展示包装中提取唯一 JSON 对象，不放宽质量 schema。"""

import json
import re
from typing import Any


def parse_quality_object(
    content: str, *, allow_wrapping: bool = True,
) -> dict[str, Any]:
    """允许展示包装，但拒绝思考未闭合、多个对象与重复键。

    Args:
        content: 模型返回的完整文本。
        allow_wrapping: 仅显式提示词兼容模式允许展示包装。

    Returns:
        唯一且完整的 JSON 对象，供严格领域模型进一步校验。

    Raises:
        ValueError: 包装不完整、对象不唯一或 JSON 不合法。
    """
    text = content.strip()
    if not allow_wrapping:
        payload = json.loads(text, object_pairs_hook=_unique_object)
        if not isinstance(payload, dict):
            raise ValueError("结构化结果必须是对象")
        return payload
    while text.startswith("<think>"):
        closing = text.find("</think>")
        if closing < 0:
            raise ValueError("思考包装未闭合")
        text = text[closing + len("</think>"):].strip()
    if text.startswith("```"):
        match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE,
        )
        if match is None:
            raise ValueError("代码包装未闭合")
        text = match.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("未找到 JSON 对象")
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    payload, end = decoder.raw_decode(text, start)
    surrounding = text[:start] + text[end:]
    # JSON 字符串中的标签是待审内容，只有对象外的标签才是包装。
    if "<think>" in surrounding or "</think>" in surrounding:
        raise ValueError("思考包装位置不明确")
    # 多个对象或残缺结构不能通过“取第一个对象”被误判为成功。
    if any(symbol in surrounding for symbol in "{}[]`"):
        raise ValueError("质量 JSON 对象不唯一或包装有歧义")
    if not isinstance(payload, dict):
        raise ValueError("质量判断必须是对象")
    return payload


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("质量 JSON 存在重复键")
        result[key] = value
    return result
