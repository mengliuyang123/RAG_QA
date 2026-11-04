# -*- coding: utf-8 -*-
"""引用编号提取与来源映射测试。"""

import logging

import pytest

from app.application.query.citations import extract_citations
from app.domain.retrieval import RetrievalResult


def _make_result(
    index: int,
    filename: str,
    page: int | None,
) -> RetrievalResult:
    """创建引用测试使用的检索结果。

    Args:
        index: 文本块在测试列表中的零基下标。
        filename: 来源文件名。
        page: 来源页码。

    Returns:
        带完整来源元数据的检索结果。
    """
    return RetrievalResult(
        chunk_id=f"doc-{index}:{index}",
        doc_id=f"doc-{index}",
        text=f"文本块 {index}",
        filename=filename,
        page=page,
        chunk_index=index,
        score=1.0 - index * 0.1,
    )


def test_extracts_unique_citations_in_first_seen_order() -> None:
    """有效编号应去重并保持在回答中的首次出现顺序。"""
    results = [
        _make_result(0, "a.pdf", 1),
        _make_result(1, "b.pdf", None),
    ]

    citations = extract_citations(
        "结论[2]，补充[1]，再次[2]。",
        results,
    )

    assert [item.model_dump() for item in citations] == [
        {
            "doc_id": "doc-1",
            "filename": "b.pdf",
            "page": None,
            "chunk_text": "文本块 1",
            "position": 2,
            "chunk_id": "doc-1:1", "kb_id": "default", "doc_updated_at": None,
        },
        {
            "doc_id": "doc-0",
            "filename": "a.pdf",
            "page": 1,
            "chunk_text": "文本块 0",
            "position": 1,
            "chunk_id": "doc-0:0", "kb_id": "default", "doc_updated_at": None,
        },
    ]


def test_ignores_invalid_numbers_and_logs_range_violations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """忽略非法形式，并为越界数字留下警告。

    Args:
        caplog: pytest 提供的日志捕获器。
    """
    results = [
        _make_result(0, "a.pdf", 1),
        _make_result(1, "b.pdf", 2),
    ]

    with caplog.at_level(
        logging.WARNING,
        logger="app.application.query.citations",
    ):
        citations = extract_citations(
            "忽略[0]、[-1]、[99]、【2】，仅保留[1]。",
            results,
        )

    assert [item.filename for item in citations] == ["a.pdf"]
    warning_records = [
        record
        for record in caplog.records
        if record.getMessage() == "忽略超出上下文范围的引用编号"
    ]
    assert len(warning_records) == 1
    assert warning_records[0].invalid_citation_numbers == [0, 99]


def test_returns_empty_when_answer_has_no_standard_citation() -> None:
    """普通数字或非标准括号不得被当成引用。"""
    results = [_make_result(0, "a.pdf", 1)]

    citations = extract_citations(
        "普通数字 1，全角引用【1】。",
        results,
    )

    assert citations == []


def test_ignores_full_width_digits_inside_ascii_brackets() -> None:
    """ASCII 方括号中的全角数字不属于标准引用编号。"""
    results = [
        _make_result(0, "a.pdf", 1),
        _make_result(1, "b.pdf", 2),
    ]

    citations = extract_citations("全角数字[１]和[２]。", results)

    assert citations == []
