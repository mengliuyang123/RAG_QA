import psycopg
from tests.postgres_support import pg_database, pg_connect
# -*- coding: utf-8 -*-

from pathlib import Path
from types import MappingProxyType

import pytest

from app.application.errors import VectorStoreError
from app.domain.documents import DocumentChunk
from app.infrastructure.vectorstores.postgres import PostgresVectorStore


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
    store = PostgresVectorStore(pg_database(tmp_path), "test_collection")
    chunks = make_chunks()
    store.add_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])

    reopened = PostgresVectorStore(pg_database(tmp_path), "test_collection")
    stored = reopened.list_chunks()

    assert stored == chunks
    assert reopened.count() == 2


def test_store_persists_caller_collection_metadata(tmp_path: Path) -> None:
    """评估指纹应与固定 cosine 配置一起跨客户端持久化。"""
    PostgresVectorStore(
        pg_database(tmp_path),
        "evaluation_collection",
        collection_metadata={"evaluation_fingerprint": "abc123"},
    )

    reopened = PostgresVectorStore(pg_database(tmp_path), "evaluation_collection")

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

    store = PostgresVectorStore(
        pg_database(tmp_path),
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
        PostgresVectorStore(
            pg_database(tmp_path),
            "evaluation_collection",
            collection_metadata={"hnsw:space": "l2"},
        )


def test_store_updates_metadata_without_losing_cosine(tmp_path: Path) -> None:
    """补写预期块数时必须保留已有指纹与 cosine 配置。"""
    store = PostgresVectorStore(
        pg_database(tmp_path),
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
    selected = PostgresVectorStore(pg_database(tmp_path), "selected_collection")
    other = PostgresVectorStore(pg_database(tmp_path), "other_collection")
    selected.add_chunks([make_chunks()[0]], [[1.0, 0.0]])
    other.add_chunks([make_chunks()[1]], [[0.0, 1.0]])

    selected.delete_collection()

    assert PostgresVectorStore(pg_database(tmp_path), "selected_collection").count() == 0
    assert PostgresVectorStore(pg_database(tmp_path), "other_collection").count() == 1


def test_store_deletes_only_selected_document(tmp_path: Path) -> None:
    """按 doc_id 清理时不得影响其他文档。"""
    store = PostgresVectorStore(pg_database(tmp_path), "test_collection")
    store.add_chunks(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    store.delete_document("doc-a")

    assert store.list_chunks() == [make_chunks()[1]]


def test_store_queries_by_embedding(tmp_path: Path) -> None:
    """向量查询应返回最近文本块及距离。"""
    store = PostgresVectorStore(pg_database(tmp_path), "test_collection")
    store.add_chunks(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    results = store.query_by_embedding([0.9, 0.1], top_k=1)

    assert len(results) == 1
    assert results[0].chunk.id == "doc-a:0"
    assert results[0].distance >= 0


def test_store_rejects_mismatched_chunk_and_embedding_counts(
    tmp_path: Path,
) -> None:
    """文本块与向量数量不一致时不得产生部分写入。"""
    store = PostgresVectorStore(pg_database(tmp_path), "test_collection")

    with pytest.raises(VectorStoreError, match="数量"):
        store.add_chunks(make_chunks(), [[1.0, 0.0]])

    assert store.count() == 0


def test_store_accepts_empty_batch(tmp_path: Path) -> None:
    """空批次应作为无操作处理。"""
    store = PostgresVectorStore(pg_database(tmp_path), "test_collection")

    store.add_chunks([], [])

    assert store.count() == 0


def test_store_reports_corrupted_metadata_with_chunk_id(tmp_path: Path) -> None:
    database = pg_database(tmp_path / 'vectors.db')
    store = PostgresVectorStore(database, 'test_collection')
    chunk = make_chunks()[0]
    store.add_chunks([chunk], [[1.0, 0.0]])
    with database.connection() as connection:
        connection.execute("UPDATE vector_chunks SET metadata=metadata-'filename'")
    with pytest.raises(VectorStoreError, match=chunk.id):
        store.query_by_embedding([1.0, 0.0], 1)


def test_store_hides_vendor_write_error(tmp_path: Path) -> None:
    store = PostgresVectorStore(pg_database(tmp_path / 'vectors.db'), 'test_collection')
    chunk = make_chunks()[0]
    store.add_chunks([chunk], [[1.0, 0.0]])
    with pytest.raises(VectorStoreError) as error:
        store.add_chunks([chunk], [[1.0, 0.0]])
    assert isinstance(error.value.__cause__, psycopg.IntegrityError)
    assert 'INSERT' not in str(error.value)


def test_store_hides_vendor_count_error(tmp_path: Path) -> None:
    database = pg_database(tmp_path / 'vectors.db')
    store = PostgresVectorStore(database, 'test_collection')
    with database.connection() as connection:
        connection.execute('DROP TABLE vector_chunks')
    with pytest.raises(VectorStoreError) as error:
        store.count()
    assert isinstance(error.value.__cause__, psycopg.Error)
    assert 'SELECT' not in str(error.value)


@pytest.mark.parametrize("top_k", [0, -1])
def test_store_rejects_invalid_top_k(tmp_path: Path, top_k: int) -> None:
    """查询条数必须为正整数。"""
    store = PostgresVectorStore(pg_database(tmp_path), "test_collection")

    with pytest.raises(ValueError, match="top_k"):
        store.query_by_embedding([1.0, 0.0], top_k=top_k)
