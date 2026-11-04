# -*- coding: utf-8 -*-
"""真实解析、分块、SQLite进度和本地索引的任务流水线测试。"""

from pathlib import Path
from threading import RLock
from uuid import uuid4

import pytest

from app.application.errors import EmbeddingError
from app.application.errors import EmptyDocumentError
from app.application.ingestion.job_pipeline import JobIngestionPipeline
from app.application.ingestion.pipeline import IngestionPipeline
from app.domain.documents import DocumentChunk
from app.domain.ingestion_jobs import IngestionJobSubmission
from app.domain.ingestion_jobs import IngestionProgress
from app.infrastructure.chunking.recursive import RecursiveDocumentChunker
from app.infrastructure.indexes.bm25 import BM25Index
from app.infrastructure.parsing.factory import ParserFactory
from app.infrastructure.persistence.ingestion_jobs import SQLiteIngestionJobRepository
from app.infrastructure.persistence.knowledge_bases import SQLiteKnowledgeBaseRepository
from app.infrastructure.vectorstores.chroma import ChromaVectorStore


class BatchEmbedder:
    """只替代真实远程向量服务，批次以外组件全部真实运行。"""

    def __init__(self, fail_on: int | None = None) -> None:
        """保存可选故障批次。

        Args:
            fail_on: 从1开始的故障批次，None表示全部成功。

        Returns:
            None。
        """
        self.batches: list[list[str]] = []
        self.fail_on = fail_on

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """返回每条文本的固定二维向量。

        Args:
            texts: 真实分块结果。

        Returns:
            一一对应的离线向量。
        """
        self.batches.append(texts)
        if len(self.batches) == self.fail_on:
            raise EmbeddingError("private upstream credential detail")
        return [[1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """提供兼容的离线查询向量。

        Args:
            text: 用户问题。

        Returns:
            固定向量。
        """
        return [1.0, 0.0]


def _setup(tmp_path: Path, embedder: BatchEmbedder):
    catalog_path = tmp_path / "catalog.sqlite3"
    SQLiteKnowledgeBaseRepository(catalog_path)
    jobs = SQLiteIngestionJobRepository(catalog_path)
    submission = IngestionJobSubmission(
        job_id=f"job_{uuid4().hex}", doc_id=uuid4(), kb_id="default",
        filename="original.txt", size_bytes=55, uploaded_by=uuid4(),
        content_sha256="a" * 64,
    )
    jobs.enqueue(submission, capacity=1)
    claim = jobs.claim_next(lease_seconds=60)
    assert claim is not None
    path = tmp_path / "saved.txt"
    path.write_text("abcdefghij\n" * 5, encoding="utf-8")
    store = ChromaVectorStore(tmp_path / "chroma", "job_pipeline")
    index = BM25Index()
    lock = RLock()
    # 每行十字加分隔符，预留分隔符空间，避免触发字符级二次切分。
    chunker = RecursiveDocumentChunker(chunk_size=11, chunk_overlap=0)
    deletion = IngestionPipeline(
        chunker, embedder, store, parser_resolver=ParserFactory(),
        keyword_index=index, mutation_lock=lock,
    )
    pipeline = JobIngestionPipeline(
        chunker, embedder, store, index, ParserFactory(), deletion,
        embedding_batch_size=2, mutation_lock=lock,
    )
    return pipeline, store, index, jobs, claim, path


def test_staged_pipeline_reports_real_batches_and_publishes_only_indexes(tmp_path: Path) -> None:
    """总数来自真实分块，进度必须在批次落盘之后，流水线不能自行改成功。

    Args:
        tmp_path: 隔离目录。
    """
    embedder = BatchEmbedder()
    pipeline, store, index, jobs, claim, path = _setup(tmp_path, embedder)
    observations = []

    def progress(value: IngestionProgress) -> None:
        jobs.advance(claim, **value.model_dump())
        observations.append((
            value.stage, value.completed_batches, value.total_batches,
            store.count(),
        ))
        assert jobs.unpublished_document_ids() == (claim.job.doc_id,)

    count = pipeline.ingest_job(path, claim.submission, progress)
    assert count == 5
    assert observations == [
        ("parsing", 0, 0, 0), ("chunking", 0, 0, 0),
        ("embedding", 0, 3, 0), ("embedding", 1, 3, 2),
        ("embedding", 2, 3, 4), ("embedding", 3, 3, 5),
        ("indexing", 0, 0, 5),
    ]
    assert [len(batch) for batch in embedder.batches] == [2, 2, 1]
    chunks = store.list_chunks()
    assert {chunk.filename for chunk in chunks} == {"original.txt"}
    assert {chunk.doc_id for chunk in chunks} == {str(claim.job.doc_id)}
    assert {chunk.kb_id for chunk in chunks} == {"default"}
    assert len({chunk.doc_updated_at for chunk in chunks}) == 1
    assert chunks[0].doc_updated_at is not None
    assert len(index.search("abcdefghij", 10)) == 5
    assert jobs.get(claim.job.job_id).status == "running"
    assert not list(tmp_path.glob("rag-ingestion-*"))


def test_embedding_failure_compensates_partial_chunks_and_preserves_old_bm25(tmp_path: Path) -> None:
    """第二批失败不能留下第一批，也不能清掉其他文档的派生索引。

    Args:
        tmp_path: 隔离目录。
    """
    embedder = BatchEmbedder(fail_on=2)
    pipeline, store, index, jobs, claim, path = _setup(tmp_path, embedder)
    old = DocumentChunk(
        id="old:0", doc_id=str(uuid4()), filename="old.txt", page=None,
        text="previous evidence", chunk_index=0,
    )
    store.add_chunks([old], [[0.0, 1.0]])
    index.rebuild([old])

    def progress(value: IngestionProgress) -> None:
        jobs.advance(claim, **value.model_dump())

    with pytest.raises(EmbeddingError):
        pipeline.ingest_job(path, claim.submission, progress)
    assert [chunk.id for chunk in store.list_chunks()] == ["old:0"]
    assert [item.chunk_id for item in index.search("previous", 1)] == ["old:0"]
    assert jobs.get(claim.job.job_id).stage == "embedding"
    assert jobs.unpublished_document_ids() == (claim.job.doc_id,)
    assert not list(tmp_path.glob("rag-ingestion-*"))


def test_empty_document_does_not_embed_or_report_success(tmp_path: Path) -> None:
    """合法非空字节可能没有正文，不能当作成功的零块文档。

    Args:
        tmp_path: 隔离目录。
    """
    embedder = BatchEmbedder()
    pipeline, store, _, jobs, claim, path = _setup(tmp_path, embedder)
    path.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        pipeline.ingest_job(path, claim.submission, lambda progress: None)
    assert embedder.batches == []
    assert store.count() == 0
    assert jobs.get(claim.job.job_id).status == "running"


def test_progress_failure_stops_before_next_embedding_batch(tmp_path: Path) -> None:
    """登记故障或租约失效后不能继续收费，已写块需补偿。

    Args:
        tmp_path: 隔离目录。
    """
    embedder = BatchEmbedder()
    pipeline, store, _, _, claim, path = _setup(tmp_path, embedder)

    def progress(value: IngestionProgress) -> None:
        if value.stage == "embedding" and value.completed_batches == 1:
            raise RuntimeError("lease revoked")

    with pytest.raises(RuntimeError, match="lease revoked"):
        pipeline.ingest_job(path, claim.submission, progress)
    assert len(embedder.batches) == 1
    assert store.count() == 0
    assert not list(tmp_path.glob("rag-ingestion-*"))


def test_cancellation_check_stops_within_chunking_before_any_embedding(tmp_path: Path) -> None:
    """只在阶段切换检查停止不够，长文档分块循环内也必须可协作退出。

    Args:
        tmp_path: 隔离目录。
    """
    embedder = BatchEmbedder()
    pipeline, store, _, _, claim, path = _setup(tmp_path, embedder)
    stage = "parsing"

    def progress(value: IngestionProgress) -> None:
        nonlocal stage
        stage = value.stage

    def check_active() -> None:
        if stage == "chunking":
            raise RuntimeError("shutdown during chunking")

    with pytest.raises(RuntimeError, match="shutdown during chunking"):
        pipeline.ingest_job(path, claim.submission, progress, check_active=check_active)
    assert embedder.batches == []
    assert store.count() == 0
