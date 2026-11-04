# -*- coding: utf-8 -*-
"""模拟旧客户端只读取约定字段，新增字段由新契约测试独立验证。"""

from typing import Any


def legacy_answer_view(
    payload: dict[str, Any], fields: tuple[str, ...],
) -> dict[str, Any]:
    """提取手工列出的旧外壳和旧来源字段，不引用生产DTO计算预期值。

    Args:
        payload: 实际JSON结果。
        fields: 旧客户端读取的固定字段名。

    Returns:
        旧客户端能看到的结果。
    """
    result = {field: payload[field] for field in fields}
    if "citations" in result:
        result["citations"] = []
        for citation in payload["citations"]:
            result["citations"].append({
                key: citation[key] for key in (
                    "doc_id", "filename", "page", "chunk_text", "position",
                )
            })
    return result
