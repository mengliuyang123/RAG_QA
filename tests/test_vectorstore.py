# -*- coding: utf-8 -*-

from pathlib import Path
from types import MappingProxyType

import pytest

from app.application.errors import VectorStoreError
from app.domain.documents import DocumentChunk
from app.infrastructure.vectorstores.chroma import ChromaVectorStore


def make_chunks() -> list[DocumentChunk]:
    """创建覆盖分页与非分页来源的测试文本块。"""
    return [
        DocumentChunk(
            id="doc-a:0",
            doc_id="doc-a",
            text="第一段",
            filename="guide.pdf",
            page=1,
            chunk_index=0,
        ),
        DocumentChunk(
            id="doc-b:0",
            doc_id="doc-b",
            text="第二段",
            filename="notes.txt",
            page=None,
            chunk_index=0,
        ),
    ]


def test_store_persists_chunks_across_client_recreation(tmp_path: Path) -> None:
    """重建客户端后仍应读取已写入文本块及完整元数据。"""
    store = ChromaVectorStore(tmp_path, "test_collection")
    chunks = make_chunks()
    store.add_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])

    reopened = ChromaVectorStore(tmp_path, "test_collection")
    stored = reopened.list_chunks()

    assert stored == chunks
    assert reopened.count() == 2


def test_store_persists_caller_collection_metadata(tmp_path: Path) -> None:
    """评估指纹应与固定 cosine 配置一起跨客户端持久化。"""
    ChromaVectorStore(
        tmp_path,
        "evaluation_collection",
        collection_metadata={"evaluation_fingerprint": "abc123"},
    )

    reopened = ChromaVectorStore(tmp_path, "evaluation_collection")

    assert reopened.collection_metadata() == {
        "hnsw:space": "cosine",
        "evaluation_fingerprint": "abc123",
    }


def test_store_accepts_read_only_string_and_integer_metadata(
    tmp_path: Path,
) -> None:
    """构造边界应复制只读 Mapping 并保留字符串与整数值。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    metadata = MappingProxyType(
        {
            "evaluation_fingerprint": "abc123",
            "expected_chunk_count": 12,
        }
    )

    store = ChromaVectorStore(
        tmp_path,
        "evaluation_collection",
        collection_metadata=metadata,
    )

    assert store.collection_metadata() == {
        "hnsw:space": "cosine",
        "evaluation_fingerprint": "abc123",
        "expected_chunk_count": 12,
    }


def test_store_rejects_cosine_metadata_override(tmp_path: Path) -> None:
    """调用方不得改变生产检索已依赖的距离空间。"""
    with pytest.raises(VectorStoreError, match="hnsw:space"):
        ChromaVectorStore(
            tmp_path,
            "evaluation_collection",
            collection_metadata={"hnsw:space": "l2"},
        )


def test_store_updates_metadata_without_losing_cosine(tmp_path: Path) -> None:
    """补写预期块数时必须保留已有指纹与 cosine 配置。"""
    store = ChromaVectorStore(
        tmp_path,
        "evaluation_collection",
        collection_metadata={"evaluation_fingerprint": "abc123"},
    )

    store.update_collection_metadata({"expected_chunk_count": 12})

    assert store.collection_metadata() == {
        "hnsw:space": "cosine",
        "evaluation_fingerprint": "abc123",
        "expected_chunk_count": 12,
    }


def test_store_deletes_only_exact_collection(tmp_path: Path) -> None:
    """显式清理评估 collection 不得影响同路径的其他 collection。"""
    selected = ChromaVectorStore(tmp_path, "selected_collection")
    other = ChromaVectorStore(tmp_path, "other_collection")
    selected.add_chunks([make_chunks()[0]], [[1.0, 0.0]])
    other.add_chunks([make_chunks()[1]], [[0.0, 1.0]])

    selected.delete_collection()

    assert ChromaVectorStore(tmp_path, "selected_collection").count() == 0
    assert ChromaVectorStore(tmp_path, "other_collection").count() == 1


def test_store_deletes_only_selected_document(tmp_path: Path) -> None:
    """按 doc_id 清理时不得影响其他文档。"""
    store = ChromaVectorStore(tmp_path, "test_collection")
    store.add_chunks(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    store.delete_document("doc-a")

    assert store.list_chunks() == [make_chunks()[1]]


def test_store_queries_by_embedding(tmp_path: Path) -> None:
    """向量查询应返回最近文本块及距离。"""
    store = ChromaVectorStore(tmp_path, "test_collection")
    store.add_chunks(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    results = store.query_by_embedding([0.9, 0.1], top_k=1)

    assert len(results) == 1
    assert results[0].chunk.id == "doc-a:0"
    assert results[0].distance >= 0


def test_store_rejects_mismatched_chunk_and_embedding_counts(
    tmp_path: Path,
) -> None:
    """文本块与向量数量不一致时不得产生部分写入。"""
    store = ChromaVectorStore(tmp_path, "test_collection")

    with pytest.raises(VectorStoreError, match="数量"):
        store.add_chunks(make_chunks(), [[1.0, 0.0]])

    assert store.count() == 0


def test_store_accepts_empty_batch(tmp_path: Path) -> None:
    """空批次应作为无操作处理。"""
    store = ChromaVectorStore(tmp_path, "test_collection")

    store.add_chunks([], [])

    assert store.count() == 0


def test_store_reports_corrupted_metadata_with_chunk_id(tmp_path: Path) -> None:
    """损坏元数据应转换为包含 chunk ID 的稳定领域异常。"""

    class CorruptedCollection:
        """返回缺少 filename 字段的损坏 Chroma 响应。"""

        def count(self) -> int:
            return 1

        def query(self, **_kwargs):
            return {
                "ids": [["broken-doc:0"]],
                "documents": [["损坏记录"]],
                "metadatas": [[
                    {"doc_id": "broken-doc", "page": 1, "chunk_index": 0}
                ]],
                "distances": [[0.1]],
            }

    store = ChromaVectorStore(tmp_path, "test_collection")
    store._collection = CorruptedCollection()

    with pytest.raises(VectorStoreError, match="broken-doc:0"):
        store.query_by_embedding([1.0, 0.0], top_k=1)


def test_store_hides_vendor_write_error(tmp_path: Path) -> None:
    """Chroma 写入异常不得携带供应商正文越过适配器边界。"""

    class FailingCollection:
        """仅暴露会失败的 Chroma 写入方法。"""

        def add(self, **_kwargs: object) -> None:
            """模拟包含敏感正文的供应商异常。

            Returns:
                None。
            """
            raise RuntimeError("secret vendor response")

    store = ChromaVectorStore(tmp_path, "test_collection")
    store._collection = FailingCollection()

    with pytest.raises(VectorStoreError, match="写入失败") as exc_info:
        store.add_chunks([make_chunks()[0]], [[1.0, 0.0]])

    assert "secret" not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_store_hides_vendor_count_error(tmp_path: Path) -> None:
    """Chroma 读取异常应转换为稳定的向量存储异常。"""

    class FailingCollection:
        """仅暴露会失败的 Chroma 计数方法。"""

        def count(self) -> int:
            """模拟供应商读取失败。

            Returns:
                永远不会返回。
            """
            raise RuntimeError("secret vendor response")

    store = ChromaVectorStore(tmp_path, "test_collection")
    store._collection = FailingCollection()

    with pytest.raises(VectorStoreError, match="计数失败") as exc_info:
        store.count()

    assert "secret" not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.parametrize("top_k", [0, -1])
def test_store_rejects_invalid_top_k(tmp_path: Path, top_k: int) -> None:
    """查询条数必须为正整数。"""
    store = ChromaVectorStore(tmp_path, "test_collection")

    with pytest.raises(ValueError, match="top_k"):
        store.query_by_embedding([1.0, 0.0], top_k=top_k)
