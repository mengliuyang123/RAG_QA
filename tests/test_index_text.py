# -*- coding: utf-8 -*-
"""索引文本前缀与结构元数据默认值。"""

from app.domain.documents import DocumentChunk, DocumentSegment, index_text


def _chunk(**overrides) -> DocumentChunk:
    values = {
        "id": "doc:0", "doc_id": "doc", "text": "标准为每晚 500 元",
        "filename": "制度.md", "page": None, "chunk_index": 0,
    }
    values.update(overrides)
    return DocumentChunk(**values)


def test_legacy_shapes_keep_defaults() -> None:
    """不传新字段时构造出旧格式对象。"""
    chunk = _chunk()
    segment = DocumentSegment(text="正文", page=1, filename="a.pdf")

    assert chunk.parent_index is None
    assert chunk.heading_path == ()
    assert segment.heading_path == ()


def test_index_text_without_headings_uses_filename_only() -> None:
    assert index_text(_chunk()) == "制度.md\n标准为每晚 500 元"


def test_index_text_with_heading_path() -> None:
    chunk = _chunk(heading_path=("报销制度", "差旅费", "住宿"), parent_index=2)

    assert index_text(chunk) == "制度.md › 报销制度 › 差旅费 › 住宿\n标准为每晚 500 元"


def test_index_text_collapses_whitespace_and_skips_blank_headings() -> None:
    """标题中的换行、制表符不会把前缀拆成多行；空白标题被忽略。"""
    chunk = _chunk(heading_path=("第一章\n 总则", "  ", "a\tb › c"))

    prefix, body = index_text(chunk).split("\n", 1)
    assert prefix == "制度.md › 第一章 总则 › a b › c"
    assert body == chunk.text
