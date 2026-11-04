# -*- coding: utf-8 -*-
"""BM25 关键词索引测试。"""

from pathlib import Path

import pytest

from app.bootstrap.factories import AdapterFactories
from app.domain.documents import DocumentChunk
from app.infrastructure.indexes import bm25
from app.infrastructure.indexes.bm25 import BM25Index, tokenize
from app.infrastructure.vectorstores.chroma import ChromaVectorStore


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    filename: str = "guide.md",
    page: int | None = None,
) -> DocumentChunk:
    """创建带完整来源信息的 BM25 测试文本块。

    Args:
        chunk_id: ``doc_id:chunk_index`` 形状的文本块标识。
        text: 文本块正文。
        filename: 来源文件名。
        page: 可选来源页码。

    Returns:
        完整的领域文本块。
    """
    doc_id, chunk_index = chunk_id.split(":")
    return DocumentChunk(
        id=chunk_id,
        doc_id=doc_id,
        text=text,
        filename=filename,
        page=page,
        chunk_index=int(chunk_index),
    )


def test_tokenizer_keeps_python_identifier_and_normalizes_case() -> None:
    """标识符应保持完整，英文应统一为小写。"""
    assert tokenize("FastAPI response_model 参数") == [
        "fastapi",
        "response_model",
        "参",
        "数",
    ]


def test_tokenizer_keeps_numbers_and_splits_cjk_characters() -> None:
    """错误码数字和中文单字都应成为可检索 token。"""
    assert tokenize("HTTP 422：请求失败") == [
        "http",
        "422",
        "请",
        "求",
        "失",
        "败",
    ]


def test_bm25_finds_exact_error_code() -> None:
    """精确错误码查询应优先召回包含该错误码的文本块。"""
    index = BM25Index()
    index.rebuild([
        make_chunk("doc:0", "FastAPI application startup guide"),
        make_chunk("doc:1", "HTTP 422 means request validation failed"),
        make_chunk("doc:2", "HTTP 500 means internal server error"),
    ])

    results = index.search("HTTP 422", 3)

    assert results[0].chunk_id == "doc:1"
    assert results[0].text == "HTTP 422 means request validation failed"
    assert results[0].filename == "guide.md"
    assert results[0].score > results[1].score


def test_bm25_rebuild_replaces_previous_corpus() -> None:
    """全量重建后不得继续返回旧文本块。"""
    index = BM25Index()
    index.rebuild([make_chunk("old:0", "legacy_token")])

    index.rebuild([make_chunk("new:0", "current_token")])

    assert [result.chunk_id for result in index.search("current_token", 5)] == [
        "new:0"
    ]
    assert index.search("legacy_token", 5) == []


def test_bm25_refreshes_from_shared_store_before_each_search() -> None:
    """每次查询都应读取共享存储，避免其他 worker 上传后索引永久陈旧。"""
    persisted_chunks = [make_chunk("old:0", "legacy_token")]
    loader_calls = 0

    def load_chunks() -> list[DocumentChunk]:
        nonlocal loader_calls
        loader_calls += 1
        return list(persisted_chunks)

    index = BM25Index(chunk_loader=load_chunks)

    assert index.search("legacy_token", 5)[0].chunk_id == "old:0"

    persisted_chunks[:] = [make_chunk("new:0", "current_token")]

    assert index.search("legacy_token", 5) == []
    assert index.search("current_token", 5)[0].chunk_id == "new:0"
    assert loader_calls == 3


def test_failed_rebuild_preserves_previous_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新索引构建失败时，查询应继续使用完整的上一代状态。

    Args:
        monkeypatch: pytest 提供的属性替换工具。
    """
    index = BM25Index()
    index.rebuild([make_chunk("old:0", "legacy_token")])

    class _FailingBM25:
        def __init__(self, corpus: object) -> None:
            """模拟底层索引构造失败。

            Args:
                corpus: BM25Index 传入的分词语料。

            Raises:
                RuntimeError: 固定的索引构造错误。
            """
            del corpus
            raise RuntimeError("索引构建失败")

    monkeypatch.setattr(bm25, "BM25Okapi", _FailingBM25)

    with pytest.raises(RuntimeError, match="索引构建失败"):
        index.rebuild([make_chunk("new:0", "current_token")])

    assert index.search("legacy_token", 5)[0].chunk_id == "old:0"
    assert index.search("current_token", 5) == []


def test_bm25_returns_empty_for_empty_index_or_tokenless_query() -> None:
    """空索引及无法分词的查询都不应产生结果。"""
    index = BM25Index()

    assert index.search("HTTP 422", 3) == []

    index.rebuild([make_chunk("doc:0", "HTTP 422")])
    assert index.search("!!!", 3) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_bm25_rejects_non_positive_top_k(top_k: int) -> None:
    """结果数量必须是正整数。

    Args:
        top_k: 当前参数化场景的非正结果数量。
    """
    index = BM25Index()

    with pytest.raises(ValueError, match="top_k"):
        index.search("HTTP 422", top_k)


def test_bm25_dependency_builds_index_from_persisted_chunks(
    tmp_path: Path,
) -> None:
    """默认 factory 初始化索引时应立即载入已有的文本块。

    Args:
        tmp_path: pytest 提供的临时 Chroma 目录。
    """
    store = ChromaVectorStore(tmp_path, "bm25_startup")
    store.add_chunks(
        [make_chunk("doc:0", "FastAPI response_model reference")],
        [[1.0, 0.0]],
    )
    index = AdapterFactories().keyword_index(store)

    assert index.search("response_model", 5)[0].chunk_id == "doc:0"
