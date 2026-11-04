# -*- coding: utf-8 -*-
"""持久任务执行、租约心跳、旧部分索引恢复及收费边界测试。"""

from datetime import timedelta
from pathlib import Path
from threading import Event
from threading import RLock
from threading import Thread
from time import monotonic
from uuid import uuid4

import pytest

from app.application.errors import IngestionWorkerUnavailableError
from app.application.errors import IngestionJobConflictError
from app.application.ingestion.job_pipeline import JobIngestionPipeline
from app.application.ingestion.job_runner import IngestionJobRunner
from app.application.ingestion.jobs import IngestionJobService
from app.application.ingestion.pipeline import IngestionPipeline
from app.domain.documents import DocumentChunk
from app.infrastructure.chunking.recursive import RecursiveDocumentChunker
from app.infrastructure.indexes.bm25 import BM25Index
from app.infrastructure.parsing.factory import ParserFactory
from app.infrastructure.persistence.ingestion_execution import LocalIngestionExecutionGuard
from app.infrastructure.persistence.ingestion_jobs import SQLiteIngestionJobRepository
from app.infrastructure.persistence.ingestion_sources import LocalIngestionSources
from app.infrastructure.persistence.knowledge_bases import SQLiteKnowledgeBaseRepository
from app.infrastructure.vectorstores.chroma import ChromaVectorStore
from tests.test_ingestion_job_repository import Clock
from tests.test_job_ingestion_pipeline import BatchEmbedder


def _setup(tmp_path: Path, embedder: BatchEmbedder, *, clock=None, lease_seconds=30):
    path = tmp_path / "catalog.sqlite3"
    catalog = SQLiteKnowledgeBaseRepository(path)
    options = {}
    if clock is not None:
        options["clock"] = clock
    jobs = SQLiteIngestionJobRepository(path, **options)
    sources = LocalIngestionSources(tmp_path / "docs")
    service = IngestionJobService(
        catalog, jobs, sources, ParserFactory(), max_upload_size=1024, capacity=1,
    )
    queued = service.submit("default", "guide.txt", b"abcdefghij\n" * 5, uuid4())
    store = ChromaVectorStore(tmp_path / "chroma", "runner_test")
    index = BM25Index()
    lock = RLock()
    chunker = RecursiveDocumentChunker(11, 0)
    deletion = IngestionPipeline(
        chunker, embedder, store, parser_resolver=ParserFactory(),
        keyword_index=index, mutation_lock=lock,
    )
    pipeline = JobIngestionPipeline(
        chunker, embedder, store, index, ParserFactory(), deletion,
        embedding_batch_size=2, mutation_lock=lock,
    )
    guard = LocalIngestionExecutionGuard(tmp_path / "worker.lock")
    runner = IngestionJobRunner(
        jobs, sources, lambda: pipeline, lambda: deletion, guard,
        lease_seconds=lease_seconds, heartbeat_seconds=0.05,
        mutation_lock=lock,
    )
    return runner, jobs, queued, store, index, guard


def test_runner_finishes_actual_job_then_reports_empty_queue(tmp_path: Path) -> None:
    """没有完成真实索引不能返回succeeded，空队列不应执行模型。

    Args:
        tmp_path: 隔离源文件、任务和索引。
    """
    embedder = BatchEmbedder()
    runner, jobs, queued, store, index, _ = _setup(tmp_path, embedder)
    assert runner.run_next(Event()) is True
    result = jobs.get(queued.job_id)
    assert result.status == "succeeded"
    assert result.chunks_count == 5
    assert result.progress == 1
    assert jobs.unpublished_document_ids() == ()
    assert store.count() == 5
    assert len(index.search("abcdefghij", 10)) == 5
    assert runner.run_next(Event()) is False
    assert len(embedder.batches) == 3


def test_recovered_job_removes_old_partial_chunks_before_reembedding(tmp_path: Path) -> None:
    """跳过恢复清理会让Chroma重复ID保留旧文本，此测试检查真实正文而非次数。

    Args:
        tmp_path: 隔离持久文件和索引。
    """
    clock = Clock()
    runner, jobs, queued, store, index, _ = _setup(tmp_path, BatchEmbedder(), clock=clock)
    old_claim = jobs.claim_next(lease_seconds=30)
    assert old_claim is not None
    partial = DocumentChunk(
        id=f"{queued.doc_id}:0", doc_id=str(queued.doc_id), filename="guide.txt",
        text="BAD OLD PARTIAL", chunk_index=0, page=None,
    )
    store.add_chunks([partial], [[0.0, 1.0]])
    index.rebuild([partial])
    clock.now += timedelta(seconds=31)
    assert runner.run_next(Event()) is True
    assert jobs.get(queued.job_id).status == "succeeded"
    assert {chunk.text for chunk in store.list_chunks()} == {"abcdefghij"}
    assert index.search("BAD", 1) == []
    with pytest.raises(IngestionJobConflictError):
        jobs.fail(old_claim, code="interrupted")


def test_source_corruption_fails_without_embedding_or_exposing_error(tmp_path: Path) -> None:
    """源文件被改后不能收费，也不能把本地路径或摘要作为错误返回。

    Args:
        tmp_path: 隔离持久文件和索引。
    """
    embedder = BatchEmbedder()
    runner, jobs, queued, store, _, _ = _setup(tmp_path, embedder)
    (tmp_path / "docs" / f"{queued.doc_id}_guide.txt").write_bytes(b"changed")
    assert runner.run_next(Event()) is True
    result = jobs.get(queued.job_id)
    assert result.status == "failed"
    assert result.error.code == "source_unavailable"
    assert str(tmp_path) not in result.model_dump_json()
    assert embedder.batches == []
    assert store.count() == 0


def test_failed_embedding_is_terminal_and_its_private_message_is_not_saved(tmp_path: Path) -> None:
    """普通失败不能无限自动重试和重复计费，且只持久保存安全分类。

    Args:
        tmp_path: 隔离持久文件和索引。
    """
    embedder = BatchEmbedder(fail_on=2)
    runner, jobs, queued, store, _, _ = _setup(tmp_path, embedder)
    assert runner.run_next(Event()) is True
    result = jobs.get(queued.job_id)
    assert result.status == "failed"
    assert result.error.code == "embedding_failed"
    assert "private upstream" not in result.model_dump_json()
    assert store.count() == 0
    assert runner.run_next(Event()) is False
    assert len(embedder.batches) == 2


def test_busy_execution_guard_does_not_claim_or_embed(tmp_path: Path) -> None:
    """同一运行时的另一个执行器不能抢占仍在工作的进程的任务。

    Args:
        tmp_path: 隔离持久文件和索引。
    """
    embedder = BatchEmbedder()
    runner, jobs, queued, _, _, guard = _setup(tmp_path, embedder)
    with guard.hold():
        with pytest.raises(IngestionWorkerUnavailableError):
            runner.run_next(Event())
    assert jobs.get(queued.job_id).status == "queued"
    assert embedder.batches == []


def test_stop_after_first_batch_keeps_job_recoverable_without_more_calls(tmp_path: Path) -> None:
    """停止服务不能开始第二批；未完成执行留下可接管状态，不伪报成功。

    Args:
        tmp_path: 隔离持久文件和索引。
    """
    stop = Event()

    class StoppingEmbedder(BatchEmbedder):
        """第一批返回时模拟服务关闭。"""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """执行一次离线向量化后发出停止信号。

            Args:
                texts: 实际批次。

            Returns:
                离线向量。
            """
            result = super().embed_documents(texts)
            stop.set()
            return result

    embedder = StoppingEmbedder()
    runner, jobs, queued, store, _, _ = _setup(tmp_path, embedder)
    assert runner.run_next(stop) is True
    assert len(embedder.batches) == 1
    assert jobs.get(queued.job_id).status == "running"
    assert jobs.unpublished_document_ids() == (queued.doc_id,)
    assert store.count() == 0


def test_heartbeat_prevents_claim_takeover_while_embedding_is_blocked(tmp_path: Path) -> None:
    """超过初始租约的模型等待仍需续约，不能让第二个worker认领同一任务。

    Args:
        tmp_path: 隔离持久文件和索引。
    """
    entered = Event()
    release = Event()

    class BlockingEmbedder(BatchEmbedder):
        """只阻塞首批，模拟可控制的远程等待。"""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """在主测试释放后返回向量。

            Args:
                texts: 真实批次。

            Returns:
                离线向量。
            """
            if not self.batches:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release model")
            return super().embed_documents(texts)

    runner, jobs, queued, _, _, _ = _setup(tmp_path, BlockingEmbedder(), lease_seconds=1)
    outcomes = []
    thread = Thread(target=lambda: outcomes.append(runner.run_next(Event())))
    thread.start()
    try:
        assert entered.wait(2)
        started = monotonic()
        while monotonic() - started < 1.1:
            assert jobs.claim_next(lease_seconds=1) is None
            Event().wait(0.05)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert outcomes == [True]
    assert jobs.get(queued.job_id).status == "succeeded"
