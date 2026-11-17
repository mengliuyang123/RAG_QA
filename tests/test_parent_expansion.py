# -*- coding: utf-8 -*-
"""检索结果按父块合并与上下文扩展（retrieval-evidence-context 规格）。"""

from datetime import datetime, timezone

import pytest

from app.application.errors import VectorStoreError
from app.application.retrieval.parents import join_children, parent_context
from app.application.retrieval.service import RetrievalService, VectorRetriever
from app.domain.documents import DocumentChunk
from app.domain.retrieval import RetrievalStrategy, VectorSearchResult


V1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
V2 = datetime(2026, 2, 1, tzinfo=timezone.utc)
PATH = ("报销制度", "差旅费", "住宿")


def child(doc: str, index: int, parent: int | None, text: str, *,
          updated=V1, kb_id: str = "default", heading_path=PATH) -> DocumentChunk:
    return DocumentChunk(
        id=f"{doc}:{index}", doc_id=doc, text=text, filename=f"{doc}.md", page=None,
        chunk_index=index, kb_id=kb_id, doc_updated_at=updated,
        parent_index=parent, heading_path=heading_path if parent is not None else (),
    )


class RankedStore:
    """按给定顺序返回向量命中，记录请求的候选数。"""

    def __init__(self, ranked: list[DocumentChunk]) -> None:
        self.ranked = ranked
        self.calls: list[int] = []

    def query_by_embedding(self, vector, top_k, **_scope) -> list[VectorSearchResult]:
        self.calls.append(top_k)
        return [VectorSearchResult(chunk=c, distance=0.1 * i) for i, c in enumerate(self.ranked[:top_k])]


class Embedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class ParentReader:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple] = []

    def read_parent_chunks(self, parents, *, kb_ids=None, excluded_doc_ids=()):
        self.calls.append((parents, kb_ids, excluded_doc_ids))
        wanted = set(parents)
        return [c for c in self.chunks if (c.doc_id, c.parent_index) in wanted]


def service(ranked, siblings, *, top_k_rerank: int = 20):
    store = RankedStore(ranked)
    reader = ParentReader(siblings)
    return RetrievalService(
        VectorRetriever(store, Embedder()), top_k_rerank=top_k_rerank, parent_reader=reader,
    ), store, reader


def retrieve(svc, top_k: int = 5):
    return svc.retrieve("差旅住宿标准", strategy=RetrievalStrategy.VECTOR, top_k=top_k)


def test_children_of_same_parent_merge_into_one_evidence_with_full_context() -> None:
    a0 = child("a", 0, 0, "一线城市")
    a1 = child("a", 1, 0, "标准为每晚 500 元")
    a2 = child("a", 2, 0, "需要发票")
    b0 = child("b", 0, 0, "招待费另行规定")
    svc, store, reader = service([a1, b0, a2], [a0, a1, a2, b0])

    results = retrieve(svc, top_k=5)

    assert store.calls == [15]
    assert [r.chunk_id for r in results] == ["a:1", "b:0"]
    first = results[0]
    assert first.text == "标准为每晚 500 元"
    assert first.context == (
        "文件名：a.md\n章节：报销制度 › 差旅费 › 住宿\n页码：无\n内容：\n"
        "一线城市标准为每晚 500 元需要发票"
    )
    assert reader.calls[0][0] == (("a", 0), ("b", 0))


def test_top_k_counts_parents_and_does_not_pad() -> None:
    ranked = [child("a", i, i // 2, f"A{i}") for i in range(6)]
    svc, _store, _reader = service(ranked, ranked)

    assert [r.chunk_id for r in retrieve(svc, top_k=2)] == ["a:0", "a:2"]
    assert [r.parent_index for r in retrieve(svc, top_k=5)] == [0, 1, 2]


def test_candidate_amplification_is_bounded_by_top_k_rerank() -> None:
    svc, store, _reader = service([child("a", 0, 0, "x")], [], top_k_rerank=8)

    retrieve(svc, top_k=5)

    assert store.calls == [8]


def test_replaced_document_falls_back_to_hit_only() -> None:
    """兄弟子块已是新版本时，不拼入另一版本的内容。"""
    hit = child("a", 1, 0, "旧版本正文")
    newer = [child("a", 0, 0, "新版本开头", updated=V2), child("a", 1, 0, "新版本正文", updated=V2)]
    svc, _store, _reader = service([hit], newer)

    [result] = retrieve(svc)

    assert result.context is None
    assert result.text == "旧版本正文"


def test_non_contiguous_or_missing_hit_falls_back() -> None:
    hit = child("a", 1, 0, "命中")
    gap = [child("a", 0, 0, "x"), hit, child("a", 3, 0, "y")]
    missing = [child("a", 0, 0, "x"), child("a", 2, 0, "y")]

    assert parent_context(retrieve_result(hit), gap) is None
    assert parent_context(retrieve_result(hit), missing) is None
    assert parent_context(retrieve_result(hit), []) is None


def retrieve_result(chunk: DocumentChunk):
    svc, _s, _r = service([chunk], [])
    return retrieve(svc)[0]


def test_legacy_chunks_are_their_own_parent_and_are_not_read() -> None:
    old_a = child("old", 0, None, "旧格式一")
    old_b = child("old", 1, None, "旧格式二")
    svc, _store, reader = service([old_a, old_b], [])

    results = retrieve(svc)

    assert [r.chunk_id for r in results] == ["old:0", "old:1"]
    assert all(r.context is None for r in results)
    assert reader.calls == []


def test_out_of_scope_sibling_is_rejected() -> None:
    hit = child("a", 0, 0, "x", kb_id="ops")

    class LeakyReader(ParentReader):
        def read_parent_chunks(self, parents, *, kb_ids=None, excluded_doc_ids=()):
            return [child("a", 1, 0, "y", kb_id="private")]

    store = RankedStore([hit])
    svc = RetrievalService(VectorRetriever(store, Embedder()), parent_reader=LeakyReader([]))
    with pytest.raises(VectorStoreError, match="父块读取违反知识库范围"):
        svc.retrieve("q", strategy=RetrievalStrategy.VECTOR, top_k=1, kb_ids=("ops",))


def test_without_parent_reader_only_dedupes() -> None:
    a0, a1, b0 = child("a", 0, 0, "x"), child("a", 1, 0, "y"), child("b", 0, 0, "z")
    store = RankedStore([a0, a1, b0])
    svc = RetrievalService(VectorRetriever(store, Embedder()))

    results = svc.retrieve("q", strategy=RetrievalStrategy.VECTOR, top_k=3)

    assert store.calls == [3]
    assert [r.chunk_id for r in results] == ["a:0", "b:0"]
    assert all(r.context is None for r in results)


def test_join_children_spaces_only_between_ascii_words() -> None:
    assert join_children(["hello", "world", "。中文", "继续", "v2", "：说明"]) == "hello world。中文继续v2：说明"
