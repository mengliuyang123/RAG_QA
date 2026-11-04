# -*- coding: utf-8 -*-
"""未发布块必须在Top-K之前排除，不能交给裁判后再隐藏。"""

from pathlib import Path
from uuid import UUID
from uuid import uuid4

import pytest

from app.application.errors import IngestionJobStorageError
from app.application.errors import VectorStoreError
from app.application.retrieval.service import RetrievalService
from app.application.retrieval.service import VectorRetriever
from app.domain.documents import DocumentChunk
from app.domain.ingestion_jobs import IngestionJobSubmission
from app.domain.ingestion_jobs import IngestionJob
from app.domain.knowledge_bases import KnowledgeBase
from app.domain.retrieval import RetrievalResult
from app.infrastructure.indexes.bm25 import BM25Index
from app.infrastructure.persistence.ingestion_jobs import SQLiteIngestionJobRepository
from app.infrastructure.persistence.knowledge_bases import SQLiteKnowledgeBaseRepository
from app.infrastructure.vectorstores.chroma import ChromaVectorStore
from tests.test_retriever import FakeQueryEmbedder


PENDING = UUID("11111111-1111-4111-8111-111111111111")
PUBLISHED = UUID("22222222-2222-4222-8222-222222222222")
FILLER = UUID("33333333-3333-4333-8333-333333333333")


def _setup(
    tmp_path: Path,
) -> tuple[
    ChromaVectorStore, BM25Index, SQLiteIngestionJobRepository, IngestionJob,
]:
    path = tmp_path / "catalog.sqlite3"
    catalog = SQLiteKnowledgeBaseRepository(path)
    catalog.create(KnowledgeBase(id="ops", name="运维"))
    jobs = SQLiteIngestionJobRepository(path)
    queued = jobs.enqueue(IngestionJobSubmission(
        job_id=f"job_{uuid4().hex}", doc_id=PENDING, kb_id="ops",
        filename="private.md", size_bytes=10, uploaded_by=uuid4(),
        content_sha256="a" * 64,
    ), capacity=1)
    chunks = []
    vectors = []
    for index in range(25):
        chunks.append(DocumentChunk(
            id=f"pending:{index}", doc_id=str(PENDING), filename="private.md",
            text="needle needle needle private unfinished", chunk_index=index,
            page=None, kb_id="ops",
        ))
        vectors.append([1.0, 0.0])
    chunks.append(DocumentChunk(
        id="public:0", doc_id=str(PUBLISHED), filename="public.md",
        text="needle published evidence", chunk_index=0, page=None, kb_id="ops",
    ))
    vectors.append([0.0, 1.0])
    for index in range(40):
        chunks.append(DocumentChunk(
            id=f"filler:{index}", doc_id=str(FILLER), filename="filler.md",
            text="unrelated", chunk_index=index, page=None, kb_id="ops",
        ))
        vectors.append([-1.0, 0.0])
    chunks.append(DocumentChunk(
        id="elsewhere:0", doc_id=str(uuid4()), filename="elsewhere.md",
        text="needle needle needle", chunk_index=0, page=None, kb_id="dev",
    ))
    vectors.append([1.0, 0.0])
    store = ChromaVectorStore(tmp_path / "chroma", "publication_test")
    store.add_chunks(chunks, vectors)
    index = BM25Index()
    index.rebuild(chunks)
    return store, index, jobs, queued


class Reranker:
    """只替代收费上游，任何未发布正文都不能跨越这个边界。"""

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int,
    ) -> list[RetrievalResult]:
        """保留真实融合候选并检查安全边界。

        Args:
            query: 实际检索问题。
            candidates: 已过滤的真实索引命中。
            top_k: 结果上限。

        Returns:
            已发布结果。
        """
        for candidate in candidates:
            assert candidate.doc_id != str(PENDING)
            assert candidate.kb_id == "ops"
        return candidates[:top_k]


def test_vector_and_keyword_exclude_unpublished_before_top_k(tmp_path: Path) -> None:
    """25个更强的未完成候选不应挤掉较弱但已发布的第一名。

    Args:
        tmp_path: 真实临时SQL/Chroma索引。
    """
    store, index, _, _ = _setup(tmp_path)
    matches = store.query_by_embedding(
        [1.0, 0.0], 1, kb_ids=("ops",), excluded_doc_ids=(str(PENDING),),
    )
    assert [match.chunk.id for match in matches] == ["public:0"]
    results = index.search(
        "needle", 1, kb_ids=("ops",), excluded_doc_ids=(str(PENDING),),
    )
    assert [result.chunk_id for result in results] == ["public:0"]
    assert store.count_documents_by_kb(excluded_doc_ids=(str(PENDING),)) == {"ops": 2, "dev": 1}


@pytest.mark.parametrize("strategy", ["vector", "hybrid", "hybrid_rerank"])
def test_all_strategies_obey_live_persisted_publication(
    tmp_path: Path, strategy: str,
) -> None:
    """SQL发布前不召回，失败后仍隐藏，重开登记不丢失事实。

    Args:
        tmp_path: 隔离真实索引。
        strategy: 生产检索策略。
    """
    store, index, jobs, queued = _setup(tmp_path)
    service = RetrievalService(
        VectorRetriever(store, FakeQueryEmbedder()), index, Reranker(),
        publication=jobs,
    )
    result = service.retrieve("needle", strategy, 1, kb_ids=("ops",))
    assert [item.chunk_id for item in result] == ["public:0"]
    claim = jobs.claim_next(lease_seconds=30)
    assert claim is not None
    jobs.fail(claim, code="indexing_failed")
    reopened = SQLiteIngestionJobRepository(tmp_path / "catalog.sqlite3")
    service = RetrievalService(
        VectorRetriever(store, FakeQueryEmbedder()), index, Reranker(),
        publication=reopened,
    )
    result = service.retrieve("needle", strategy, 1, kb_ids=("ops",))
    assert [item.chunk_id for item in result] == ["public:0"]
    assert reopened.get(queued.job_id).status == "failed"


def test_publication_becomes_visible_without_query_time_index_rebuild(tmp_path: Path) -> None:
    """成功标记改变可见性，不要求每次查询重新加载正文或重建BM25。

    Args:
        tmp_path: 隔离真实索引。
    """
    store, index, jobs, _ = _setup(tmp_path)
    service = RetrievalService(VectorRetriever(store, FakeQueryEmbedder()), index, publication=jobs)
    before = service.retrieve("needle", "hybrid", 1, kb_ids=("ops",))
    assert [item.doc_id for item in before] == [str(PUBLISHED)]
    claim = jobs.claim_next(lease_seconds=30)
    assert claim is not None
    jobs.advance(claim, stage="embedding", completed_batches=1, total_batches=1)
    jobs.advance(claim, stage="indexing")
    jobs.succeed(claim, chunks_count=25)
    after = service.retrieve("needle", "hybrid", 1, kb_ids=("ops",))
    assert [item.doc_id for item in after] == [str(PENDING)]


def test_publication_read_failure_stops_before_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """登记不可用不能先调用收费模型，再把错误吞成可见全库。

    Args:
        tmp_path: 隔离真实索引。
        monkeypatch: 只替换SQL故障边界。
    """
    store, index, jobs, _ = _setup(tmp_path)

    def unavailable() -> tuple[UUID, ...]:
        raise IngestionJobStorageError("private database detail")

    monkeypatch.setattr(jobs, "unpublished_document_ids", unavailable)
    embedder = FakeQueryEmbedder()
    service = RetrievalService(VectorRetriever(store, embedder), index, publication=jobs)
    with pytest.raises(VectorStoreError):
        service.retrieve("needle", "hybrid", 1)
    assert embedder.queries == []


@pytest.mark.parametrize("strategy", ["vector", "hybrid", "hybrid_rerank"])
def test_job_admitted_during_embedding_cannot_bypass_snapshot(
    tmp_path: Path, strategy: str,
) -> None:
    """模型耗时期间的新任务不在旧快照里，不能因此把部分正文发给重排。

    Args:
        tmp_path: 隔离真实索引与任务表。
        strategy: 所有生产检索策略。
    """
    store, index, jobs, _ = _setup(tmp_path)
    identifier = uuid4()

    class ConcurrentEmbedder(FakeQueryEmbedder):
        """在外部查询向量返回时模拟另一个请求已提交并写入部分块。"""

        def embed_query(self, text: str) -> list[float]:
            """模拟真实请求交错，不替换检索和发布判断。

            Args:
                text: 实际问题。

            Returns:
                固定离线向量。
            """
            jobs.enqueue(IngestionJobSubmission(
                job_id=f"job_{uuid4().hex}", doc_id=identifier, kb_id="ops",
                filename="new.md", size_bytes=10, uploaded_by=uuid4(),
                content_sha256="b" * 64,
            ), capacity=2)
            new = DocumentChunk(
                id="new:0", doc_id=str(identifier), filename="new.md",
                text="NEW PRIVATE UNFINISHED", chunk_index=0, page=None,
                kb_id="ops",
            )
            store.add_chunks([new], [[1.0, 0.0]])
            return [1.0, 0.0]

    service = RetrievalService(
        VectorRetriever(store, ConcurrentEmbedder()), index, Reranker(),
        publication=jobs,
    )
    with pytest.raises(VectorStoreError):
        service.retrieve("needle", strategy, 1, kb_ids=("ops",))
