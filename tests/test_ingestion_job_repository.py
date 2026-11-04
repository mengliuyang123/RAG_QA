# -*- coding: utf-8 -*-
"""用真实SQLite验证有界任务登记、租约和发布状态，不联网。"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from app.application.errors import IngestionJobConflictError
from app.application.errors import IngestionJobNotFoundError
from app.application.errors import IngestionJobStorageError
from app.application.errors import IngestionQueueFullError
from app.application.errors import KnowledgeBaseNotFoundError
from app.domain.ingestion_jobs import IngestionJobSubmission
from app.domain.knowledge_bases import KnowledgeBase
from app.infrastructure.persistence.ingestion_jobs import SQLiteIngestionJobRepository
from app.infrastructure.persistence.ingestion_jobs import has_pending_ingestion_jobs
from app.infrastructure.persistence.knowledge_bases import SQLiteKnowledgeBaseRepository


class Clock:
    """可推进的UTC时钟，租约测试不实际等待。"""

    def __init__(self) -> None:
        """固定本测试的初始时间。

        Returns:
            None。
        """
        self.now = datetime(2026, 10, 23, tzinfo=UTC)

    def __call__(self) -> datetime:
        """返回当前测试时间。

        Returns:
            带时区时间。
        """
        return self.now


def _repository(tmp_path: Path, clock: Clock) -> SQLiteIngestionJobRepository:
    path = tmp_path / "catalog.sqlite3"
    catalog = SQLiteKnowledgeBaseRepository(path)
    catalog.create(KnowledgeBase(id="ops", name="运维"))
    return SQLiteIngestionJobRepository(path, clock=clock)


def _submission(kb_id: str = "ops") -> IngestionJobSubmission:
    return IngestionJobSubmission(
        job_id=f"job_{uuid4().hex}", doc_id=uuid4(), kb_id=kb_id,
        filename="部署说明.md", size_bytes=14, uploaded_by=uuid4(),
        content_sha256="a" * 64,
    )


def test_queued_job_survives_reopen_and_has_no_body_or_lease(tmp_path: Path) -> None:
    """缺少真实提交会在重开后丢任务，不能用内存字典假装持久化。

    Args:
        tmp_path: 隔离目录。
    """
    clock = Clock()
    repository = _repository(tmp_path, clock)
    submission = _submission()
    queued = repository.enqueue(submission, capacity=2)
    reopened = SQLiteIngestionJobRepository(tmp_path / "catalog.sqlite3", clock=clock)
    assert reopened.get(queued.job_id) == queued
    assert queued.status == "queued"
    assert queued.stage == "parsing"
    assert queued.progress == 0
    assert queued.chunks_count is None
    assert queued.error is None
    public = queued.model_dump(mode="json")
    assert public["kb_id"] == "ops"
    assert "content_sha256" not in public
    assert "uploaded_by" not in public
    assert "lease_token" not in public
    assert "saved_path" not in public
    assert repository.unpublished_document_ids() == (submission.doc_id,)


def test_admission_is_bounded_even_for_concurrent_connections(tmp_path: Path) -> None:
    """容量检查移出写事务会使两条并发入队都成功。

    Args:
        tmp_path: 隔离目录。
    """
    clock = Clock()
    _repository(tmp_path, clock)
    submissions = (_submission(), _submission())

    def enqueue(submission: IngestionJobSubmission) -> str:
        repository = SQLiteIngestionJobRepository(tmp_path / "catalog.sqlite3", clock=clock)
        try:
            return repository.enqueue(submission, capacity=1).status
        except IngestionQueueFullError:
            return "queue_full"

    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(enqueue, submissions))
    assert sorted(outcomes) == ["queue_full", "queued"]


def test_only_one_worker_claims_each_queued_job(tmp_path: Path) -> None:
    """认领必须在持久化事务内比较状态，不允许两个worker各拿一份。

    Args:
        tmp_path: 隔离目录。
    """
    clock = Clock()
    repository = _repository(tmp_path, clock)
    queued = repository.enqueue(_submission(), capacity=1)

    def claim(_: int) -> object:
        store = SQLiteIngestionJobRepository(tmp_path / "catalog.sqlite3", clock=clock)
        return store.claim_next(lease_seconds=30)

    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(claim, (1, 2)))
    claimed = []
    for outcome in outcomes:
        if outcome is not None:
            claimed.append(outcome)
    assert len(claimed) == 1
    assert claimed[0].job.job_id == queued.job_id
    assert claimed[0].job.status == "running"
    assert claimed[0].attempt == 1
    assert claimed[0].recovered is False
    assert "lease_token" not in claimed[0].model_dump()


def test_expired_claim_restarts_with_new_fence_and_rejects_old_updates(tmp_path: Path) -> None:
    """旧worker不能在被接管后把部分索引伪报为成功。

    Args:
        tmp_path: 隔离目录。
    """
    clock = Clock()
    repository = _repository(tmp_path, clock)
    queued = repository.enqueue(_submission(), capacity=1)
    first = repository.claim_next(lease_seconds=30)
    assert first is not None
    repository.advance(first, stage="embedding", completed_batches=1, total_batches=2)
    clock.now += timedelta(seconds=31)
    second = repository.claim_next(lease_seconds=30)
    assert second is not None
    assert second.attempt == 2
    assert second.recovered is True
    assert second.lease_token != first.lease_token
    assert second.job.progress == 0
    assert second.job.stage == "parsing"
    with pytest.raises(IngestionJobConflictError):
        repository.advance(first, stage="indexing")
    with pytest.raises(IngestionJobConflictError):
        repository.fail(first, code="embedding_failed")
    with pytest.raises(IngestionJobConflictError):
        repository.renew(first, lease_seconds=30)
    with pytest.raises(IngestionJobConflictError):
        repository.succeed(first, chunks_count=5)
    assert repository.get(queued.job_id).status == "running"
    assert repository.unpublished_document_ids() == (queued.doc_id,)


def test_batch_progress_and_publication_require_full_index_commit(tmp_path: Path) -> None:
    """未提交完整索引不能发布，成功后不能再失败或被认领。

    Args:
        tmp_path: 隔离目录。
    """
    clock = Clock()
    repository = _repository(tmp_path, clock)
    queued = repository.enqueue(_submission(), capacity=1)
    claim = repository.claim_next(lease_seconds=30)
    assert claim is not None
    with pytest.raises(IngestionJobConflictError):
        repository.succeed(claim, chunks_count=5)
    repository.advance(claim, stage="chunking")
    half = repository.advance(claim, stage="embedding", completed_batches=1, total_batches=2)
    assert 0 < half.progress < 1
    full = repository.advance(claim, stage="embedding", completed_batches=2, total_batches=2)
    assert half.progress < full.progress < 1
    with pytest.raises(IngestionJobConflictError):
        repository.advance(claim, stage="embedding", completed_batches=1, total_batches=2)
    with pytest.raises(IngestionJobConflictError):
        repository.advance(claim, stage="parsing")
    repository.advance(claim, stage="indexing")
    with pytest.raises(ValueError):
        repository.succeed(claim, chunks_count=0)
    succeeded = repository.succeed(claim, chunks_count=5)
    assert succeeded.status == "succeeded"
    assert succeeded.progress == 1
    assert succeeded.chunks_count == 5
    assert succeeded.error is None
    assert repository.unpublished_document_ids() == ()
    assert repository.claim_next(lease_seconds=30) is None
    with pytest.raises(IngestionJobConflictError):
        repository.fail(claim, code="indexing_failed")
    assert repository.get(queued.job_id) == succeeded
    assert repository.enqueue(_submission(), capacity=1).status == "queued"


def test_failed_job_keeps_unpublished_identity_but_releases_queue_slot(tmp_path: Path) -> None:
    """失败补偿也可能失败，终态释放队列不等于允许遗留块被检索。

    Args:
        tmp_path: 隔离目录。
    """
    clock = Clock()
    repository = _repository(tmp_path, clock)
    queued = repository.enqueue(_submission(), capacity=1)
    claim = repository.claim_next(lease_seconds=30)
    assert claim is not None
    failed = repository.fail(claim, code="embedding_failed")
    assert failed.status == "failed"
    assert failed.error.code == "embedding_failed"
    assert failed.error.message
    assert failed.chunks_count is None
    assert repository.unpublished_document_ids() == (queued.doc_id,)
    assert repository.claim_next(lease_seconds=30) is None
    assert repository.enqueue(_submission(), capacity=1).status == "queued"
    with pytest.raises(ValueError):
        repository.fail(claim, code="API key=private upstream text")


def test_renewal_keeps_claim_live_without_inventing_progress(tmp_path: Path) -> None:
    """心跳不应推进进度，过期token即使未接管也不再拥有写权。

    Args:
        tmp_path: 隔离目录。
    """
    clock = Clock()
    repository = _repository(tmp_path, clock)
    queued = repository.enqueue(_submission(), capacity=1)
    claim = repository.claim_next(lease_seconds=30)
    assert claim is not None
    clock.now += timedelta(seconds=20)
    repository.renew(claim, lease_seconds=30)
    clock.now += timedelta(seconds=20)
    assert repository.claim_next(lease_seconds=30) is None
    assert repository.get(queued.job_id).progress == 0
    clock.now += timedelta(seconds=11)
    with pytest.raises(IngestionJobConflictError):
        repository.advance(claim, stage="chunking")


def test_renewal_cannot_shorten_already_granted_lease(tmp_path: Path) -> None:
    """短心跳间隔不能缩短原执行权，导致仍工作的任务被提前接管。

    Args:
        tmp_path: 隔离目录。
    """
    clock = Clock()
    repository = _repository(tmp_path, clock)
    repository.enqueue(_submission(), capacity=1)
    claim = repository.claim_next(lease_seconds=30)
    assert claim is not None
    clock.now += timedelta(seconds=2)
    repository.renew(claim, lease_seconds=5)
    clock.now += timedelta(seconds=6)
    assert repository.claim_next(lease_seconds=30) is None


def test_incomplete_batches_cannot_enter_indexing_or_change_total(tmp_path: Path) -> None:
    """首批完成不意味着全部完成，总数变化也不能掩盖未完成批次。

    Args:
        tmp_path: 隔离目录。
    """
    clock = Clock()
    repository = _repository(tmp_path, clock)
    repository.enqueue(_submission(), capacity=1)
    claim = repository.claim_next(lease_seconds=30)
    assert claim is not None
    repository.advance(claim, stage="embedding", completed_batches=1, total_batches=2)
    with pytest.raises(IngestionJobConflictError):
        repository.advance(claim, stage="indexing")
    with pytest.raises(IngestionJobConflictError):
        repository.advance(claim, stage="embedding", completed_batches=1, total_batches=1)


def test_corrupt_published_status_fails_closed(tmp_path: Path) -> None:
    """损坏的成功标记不能绕过发布检查，登记故障必须阻止召回。

    Args:
        tmp_path: 隔离目录。
    """
    clock = Clock()
    repository = _repository(tmp_path, clock)
    queued = repository.enqueue(_submission(), capacity=1)
    with sqlite3.connect(tmp_path / "catalog.sqlite3") as connection:
        connection.execute(
            "UPDATE ingestion_jobs SET status='succeeded' WHERE job_id=?",
            (queued.job_id,),
        )
    with pytest.raises(IngestionJobStorageError):
        repository.get(queued.job_id)
    with pytest.raises(IngestionJobStorageError):
        repository.unpublished_document_ids()


def test_unknown_kb_duplicate_identity_and_bad_progress_do_not_mutate(tmp_path: Path) -> None:
    """身份和计数非法时不能靠覆盖已有行消除冲突。

    Args:
        tmp_path: 隔离目录。
    """
    clock = Clock()
    repository = _repository(tmp_path, clock)
    with pytest.raises(KnowledgeBaseNotFoundError):
        repository.enqueue(_submission("missing"), capacity=1)
    submission = _submission()
    queued = repository.enqueue(submission, capacity=2)
    with pytest.raises(IngestionJobConflictError):
        repository.enqueue(submission, capacity=2)
    claim = repository.claim_next(lease_seconds=30)
    assert claim is not None
    for completed, total in ((-1, 2), (3, 2), (1, 0)):
        with pytest.raises(ValueError):
            repository.advance(claim, stage="embedding", completed_batches=completed, total_batches=total)
    assert repository.get(queued.job_id).progress == 0
    with pytest.raises(IngestionJobNotFoundError):
        repository.get(f"job_{uuid4().hex}")


def test_unpublished_lookup_fails_closed_and_parent_cannot_be_orphaned(tmp_path: Path) -> None:
    """登记损坏不能被解读成没有隐藏文档，目录也不能遗留悬空任务。

    Args:
        tmp_path: 隔离目录。
    """
    clock = Clock()
    repository = _repository(tmp_path, clock)
    repository.enqueue(_submission(), capacity=1)
    with sqlite3.connect(tmp_path / "catalog.sqlite3") as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM knowledge_bases WHERE id = 'ops'")
        connection.execute("DROP TABLE ingestion_jobs")
    with pytest.raises(IngestionJobStorageError):
        repository.unpublished_document_ids()


@pytest.mark.parametrize("filename", ["../secret.md", "a/b.md", "a\\b.md", "..", "a\x00.md"])
def test_job_identity_cannot_carry_uncontrolled_paths(filename: str) -> None:
    """持久化任务不能接受目录穿越，恢复时也不能相信历史文件路径。

    Args:
        filename: 非法候选文件名。
    """
    with pytest.raises(ValueError):
        IngestionJobSubmission.model_validate({**_submission().model_dump(), "filename": filename})


def test_idle_worker_probe_is_read_only_and_understands_escaped_paths(tmp_path: Path) -> None:
    """空闲探测不能创建目录表，URI中的问号和中文也不能变成SQLite参数。

    Args:
        tmp_path: 未创建的隔离路径。
    """
    path = tmp_path / "空 间?mode=rw" / "catalog.sqlite3"
    assert has_pending_ingestion_jobs(path) is False
    assert not path.parent.exists()
    catalog = SQLiteKnowledgeBaseRepository(path)
    assert has_pending_ingestion_jobs(path) is False
    catalog.create(KnowledgeBase(id="ops", name="运维"))
    repository = SQLiteIngestionJobRepository(path)
    repository.enqueue(_submission(), capacity=1)
    before = path.stat().st_mtime_ns
    assert has_pending_ingestion_jobs(path) is True
    assert path.stat().st_mtime_ns == before
    claim = repository.claim_next(lease_seconds=30)
    assert claim is not None
    repository.fail(claim, code="interrupted")
    assert has_pending_ingestion_jobs(path) is False
