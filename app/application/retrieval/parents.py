# -*- coding: utf-8 -*-
"""把命中子块与同一父块的兄弟子块拼成交给证据判断和生成的上下文。"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.documents import DocumentChunk
from app.domain.documents import INDEX_PATH_SEPARATOR
from app.domain.retrieval import RetrievalResult


def join_children(texts: Sequence[str]) -> str:
    """按序拼接同一父块的子块：两侧都是 ASCII 字母数字时补一个空格，其余直接相连。

    子块在切分处去掉了空白，直接相连会把英文单词粘在一起；中文和标点不需要空格。
    """
    parts: list[str] = []
    for text in texts:
        if parts and parts[-1][-1:].isascii() and parts[-1][-1:].isalnum() \
                and text[:1].isascii() and text[:1].isalnum():
            parts.append(" ")
        parts.append(text)
    return "".join(parts)


def parent_context(
    hit: RetrievalResult, siblings: Sequence[DocumentChunk],
) -> str | None:
    """校验兄弟子块后返回父块上下文；不一致时返回 None，调用方只用命中子块。

    Args:
        hit: 排名最高的命中子块。
        siblings: 存储返回的同一 ``(doc_id, parent_index)`` 子块，任意顺序。

    Returns:
        ``文件名 / 章节 / 页码 / 内容`` 格式的上下文；兄弟子块版本与命中块不同、
        序号不连续、缺少命中块或正文不一致时为 None。
    """
    ordered = sorted(siblings, key=lambda chunk: chunk.chunk_index)
    if not ordered:
        return None
    indexes = [chunk.chunk_index for chunk in ordered]
    if indexes != list(range(indexes[0], indexes[0] + len(indexes))):
        return None
    if any(
        chunk.doc_id != hit.doc_id
        or chunk.parent_index != hit.parent_index
        or chunk.doc_updated_at != hit.doc_updated_at
        or chunk.kb_id != hit.kb_id
        for chunk in ordered
    ):
        return None
    matched = [chunk for chunk in ordered if chunk.id == hit.chunk_id]
    if len(matched) != 1 or matched[0].text != hit.text:
        return None
    page = str(hit.page) if hit.page is not None else "无"
    lines = [f"文件名：{hit.filename}"]
    if hit.heading_path:
        lines.append("章节：" + INDEX_PATH_SEPARATOR.join(hit.heading_path))
    lines += [f"页码：{page}", "内容：", join_children([chunk.text for chunk in ordered])]
    return "\n".join(lines)
