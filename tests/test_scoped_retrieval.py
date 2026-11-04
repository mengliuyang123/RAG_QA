# -*- coding: utf-8 -*-
"""用真实本地索引验证分库过滤发生在召回和Top-K之前。"""

from pathlib import Path

import pytest

from app.application.retrieval.service import RetrievalService
from app.application.retrieval.service import VectorRetriever
from app.domain.documents import DocumentChunk
from app.domain.retrieval import RetrievalResult
from app.domain.retrieval import RetrievalStrategy
from app.infrastructure.indexes.bm25 import BM25Index
from app.infrastructure.vectorstores.chroma import ChromaVectorStore
from tests.test_retriever import FakeQueryEmbedder


def _chunk(chunk_id: str, kb_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id, doc_id=chunk_id, filename="a.md", text=text,
        page=None, chunk_index=0, kb_id=kb_id,
    )


def _stores(tmp_path: Path):
    chunks = [
        _chunk("ops:0", "ops", "needle 与配置"),
        _chunk("dev:0", "dev", "needle needle needle needle"),
        _chunk("dev:1", "dev", "unrelated document"),
    ]
    store = ChromaVectorStore(tmp_path, "scoped_retrieval")
    store.add_chunks(chunks, [[0.0, 1.0], [1.0, 0.0], [1.0, 0.01]])
    index = BM25Index()
    index.rebuild(chunks)
    return store, index


class _Reranker:
    """只替代远端精排，保留真实融合与范围边界。"""

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int,
    ) -> list[RetrievalResult]:
        """远端也不应收到范围之外的正文。

        Args:
            query: 实际问题。
            candidates: 过滤后的融合候选。
            top_k: 输出数量。

        Returns:
            保留来源的候选结果。
        """
        assert all(item.kb_id == "ops" for item in candidates)
        return candidates[:top_k]


def test_vector_scope_is_applied_before_top_k(tmp_path: Path) -> None:
    """范围外向量更相似时，也能召回较远的范围内结果。

    Args:
        tmp_path: 隔离Chroma目录。
    """
    store, _ = _stores(tmp_path)
    matches = store.query_by_embedding([1.0, 0.0], 1, kb_ids=("ops",))
    assert [item.chunk.id for item in matches] == ["ops:0"]
    assert matches[0].chunk.kb_id == "ops"
    assert store.query_by_embedding([1.0, 0.0], 1, kb_ids=()) == []


def test_keyword_scope_is_applied_before_top_k(tmp_path: Path) -> None:
    """关键词强命中范围外时不能挤掉范围内的候选。

    Args:
        tmp_path: 隔离索引目录。
    """
    _, index = _stores(tmp_path)
    results = index.search("needle", 1, kb_ids=("ops",))
    assert [item.chunk_id for item in results] == ["ops:0"]
    assert results[0].kb_id == "ops"
    assert index.search("needle", 1, kb_ids=()) == []
    assert index.search("needle", 1, kb_ids=("missing",)) == []


@pytest.mark.parametrize("strategy", list(RetrievalStrategy))
def test_all_strategies_preserve_the_same_scope(
    tmp_path: Path, strategy: RetrievalStrategy,
) -> None:
    """向量、混合及精排均不能让另一条召回路径突破范围。

    Args:
        tmp_path: 隔离持久化目录。
        strategy: 三种生产检索策略。
    """
    store, index = _stores(tmp_path)
    retriever = RetrievalService(
        VectorRetriever(store, FakeQueryEmbedder()), index, _Reranker(),
    )
    results = retriever.retrieve("needle", strategy, 1, kb_ids=("ops",))
    assert [item.chunk_id for item in results] == ["ops:0"]
    assert results[0].kb_id == "ops"


def test_empty_scope_never_means_global_or_calls_embedding(tmp_path: Path) -> None:
    """空集合是空权限范围，不得因为falsey而变成全库。

    Args:
        tmp_path: 隔离持久化目录。
    """
    store, index = _stores(tmp_path)
    embedder = FakeQueryEmbedder()
    retriever = RetrievalService(VectorRetriever(store, embedder), index)
    assert retriever.retrieve("needle", "hybrid", 1, kb_ids=()) == []
    assert embedder.queries == []


@pytest.mark.parametrize("scope", ["ops", ["ops"], ("",), ("ops", "ops")])
def test_invalid_scope_is_rejected_before_external_calls(
    tmp_path: Path, scope: object,
) -> None:
    """字符串、非法ID和重复集合不能被隐式解释为其他范围。

    Args:
        tmp_path: 隔离持久化目录。
        scope: 违反内部范围契约的输入。
    """
    store, index = _stores(tmp_path)
    embedder = FakeQueryEmbedder()
    with pytest.raises(ValueError):
        RetrievalService(VectorRetriever(store, embedder), index).retrieve(
            "needle", "hybrid", 1, kb_ids=scope,
        )
    assert embedder.queries == []


def test_legacy_metadata_migrates_without_reembedding_or_reassigning_known_kb(
    tmp_path: Path,
) -> None:
    """重开旧索引只给缺少kb_id的记录补默认库，保留向量和其他元数据。

    Args:
        tmp_path: 隔离旧版Chroma目录。
    """
    store = ChromaVectorStore(tmp_path, "legacy_scopes")
    store._collection.add(
        ids=["legacy", "known"], embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["旧正文", "已归属正文"],
        metadatas=[
            {"doc_id": "old", "filename": "a.md", "page": 0,
             "chunk_index": 0, "custom": "preserved"},
            {"doc_id": "known", "filename": "b.md", "page": 1,
             "chunk_index": 0, "kb_id": "dev"},
        ],
    )
    reopened = ChromaVectorStore(tmp_path, "legacy_scopes")
    matches = reopened.query_by_embedding([1.0, 0.0], 1, kb_ids=("default",))
    assert [item.chunk.id for item in matches] == ["legacy"]
    raw = reopened._collection.get(ids=["legacy"], include=["metadatas", "embeddings"])
    assert raw["metadatas"][0]["custom"] == "preserved"
    assert raw["metadatas"][0]["kb_id"] == "default"
    assert list(raw["embeddings"][0]) == [1.0, 0.0]
    known = reopened.query_by_embedding([0.0, 1.0], 1, kb_ids=("dev",))
    assert [item.chunk.id for item in known] == ["known"]
