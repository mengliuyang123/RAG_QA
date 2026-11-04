# -*- coding: utf-8 -*-
"""真实目录、源文件和任务登记的入队应用边界。"""

from pathlib import Path
from uuid import uuid4

import pytest

from app.application.errors import EmptyDocumentError
from app.application.errors import FileTooLargeError
from app.application.errors import IngestionJobStorageError
from app.application.errors import IngestionQueueFullError
from app.application.errors import KnowledgeBaseNotFoundError
from app.application.errors import UnsupportedFileTypeError
from app.application.ingestion.jobs import IngestionJobService
from app.infrastructure.parsing.factory import ParserFactory
from app.infrastructure.persistence.ingestion_jobs import SQLiteIngestionJobRepository
from app.infrastructure.persistence.ingestion_sources import LocalIngestionSources
from app.infrastructure.persistence.knowledge_bases import SQLiteKnowledgeBaseRepository


def _service(tmp_path: Path, *, capacity: int = 1):
    catalog = SQLiteKnowledgeBaseRepository(tmp_path / "catalog.sqlite3")
    jobs = SQLiteIngestionJobRepository(tmp_path / "catalog.sqlite3")
    service = IngestionJobService(
        catalog, jobs, LocalIngestionSources(tmp_path / "docs"), ParserFactory(),
        max_upload_size=32, capacity=capacity,
    )
    return service, jobs


def test_submit_persists_before_queue_and_uses_actual_target_kb(tmp_path: Path) -> None:
    """队列里的任务必须已有完整源文件；这里只入队，不调用模型或解析正文。

    Args:
        tmp_path: 隔离目录。
    """
    service, jobs = _service(tmp_path)
    queued = service.submit("default", "C:\\fakepath\\guide.md", b"payload", uuid4())
    assert queued.status == "queued"
    assert queued.kb_id == "default"
    assert queued.filename == "guide.md"
    assert service.get(queued.job_id) == jobs.get(queued.job_id)
    assert (tmp_path / "docs" / f"{queued.doc_id}_guide.md").read_bytes() == b"payload"


def test_queue_full_removes_only_this_unaccepted_source(tmp_path: Path) -> None:
    """容量拒绝不能留下第二份无任务的上传，也不能清除首个队列源文件。

    Args:
        tmp_path: 隔离目录。
    """
    service, _ = _service(tmp_path)
    queued = service.submit("default", "first.md", b"first", uuid4())
    with pytest.raises(IngestionQueueFullError):
        service.submit("default", "second.md", b"second", uuid4())
    assert [path.name for path in (tmp_path / "docs").iterdir()] == [f"{queued.doc_id}_first.md"]


@pytest.mark.parametrize("kb_id,filename,content,error", [
    ("missing", "ok.md", b"a", KnowledgeBaseNotFoundError),
    ("default", "bad.exe", b"a", UnsupportedFileTypeError),
    ("default", "ok.md", b"", EmptyDocumentError),
    ("default", "ok.md", b"a" * 33, FileTooLargeError),
])
def test_invalid_submission_does_not_write_source(
    tmp_path: Path, kb_id: str, filename: str, content: bytes, error: type[Exception],
) -> None:
    """失败输入必须在文件写入之前被拒绝。

    Args:
        tmp_path: 隔离目录。
        kb_id: 请求库。
        filename: 请求名称。
        content: 请求字节。
        error: 稳定应用错误。
    """
    service, jobs = _service(tmp_path)
    with pytest.raises(error):
        service.submit(kb_id, filename, content, uuid4())
    assert not (tmp_path / "docs").exists()
    assert jobs.unpublished_document_ids() == ()


def test_unknown_commit_outcome_keeps_source_for_possible_committed_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """数据库提交后响应丢失时不能删源文件，把已入队任务变成必失败。

    Args:
        tmp_path: 隔离目录。
        monkeypatch: 模拟一次真实提交之后的连接错误。
    """
    service, jobs = _service(tmp_path)
    original = jobs.enqueue

    def lost_response(submission, *, capacity: int):
        original(submission, capacity=capacity)
        raise IngestionJobStorageError("commit response lost")

    monkeypatch.setattr(jobs, "enqueue", lost_response)
    with pytest.raises(IngestionJobStorageError):
        service.submit("default", "survives.md", b"valid", uuid4())
    claim = jobs.claim_next(lease_seconds=30)
    assert claim is not None
    assert (tmp_path / "docs" / f"{claim.job.doc_id}_survives.md").read_bytes() == b"valid"
