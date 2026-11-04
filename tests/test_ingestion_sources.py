# -*- coding: utf-8 -*-
"""持久源文件完整性、私有快照与入队边界，不碰真实用户文件。"""

from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
from uuid import uuid4

import pytest

from app.application.errors import IngestionSourceError
from app.domain.ingestion_jobs import IngestionJobSubmission
from app.infrastructure.persistence.ingestion_sources import LocalIngestionSources


CONTENT = "原始文件正文\n".encode()


def _submission() -> IngestionJobSubmission:
    return IngestionJobSubmission(
        job_id=f"job_{uuid4().hex}", doc_id=uuid4(), kb_id="default",
        filename="guide.md", size_bytes=len(CONTENT), uploaded_by=uuid4(),
        content_sha256=sha256(CONTENT).hexdigest(),
    )


def test_source_survives_reopen_and_snapshot_cannot_change_with_original(tmp_path: Path) -> None:
    """仅保存内存或校验后直接解析可变原路径都会破坏恢复与完整性。

    Args:
        tmp_path: 隔离源目录。
    """
    sources = LocalIngestionSources(tmp_path / "docs")
    submission = _submission()
    sources.persist(submission, CONTENT)
    original = tmp_path / "docs" / f"{submission.doc_id}_guide.md"
    assert original.read_bytes() == CONTENT
    assert original.stat().st_mode & 0o077 == 0
    reopened = LocalIngestionSources(tmp_path / "docs")
    with reopened.verified_snapshot(submission) as snapshot:
        assert snapshot.read_bytes() == CONTENT
        original.write_bytes(b"changed after verification")
        assert snapshot.read_bytes() == CONTENT
        assert snapshot.suffix == ".md"
    assert not snapshot.exists()
    with pytest.raises(IngestionSourceError):
        with reopened.verified_snapshot(submission):
            pytest.fail("改动的源文件不得进入流水线")


def test_source_never_overwrites_existing_identity_and_discard_is_exact(tmp_path: Path) -> None:
    """UUID碰撞或重复请求不能覆盖已有文件，清理也不能扩大到其他文档。

    Args:
        tmp_path: 隔离目录。
    """
    sources = LocalIngestionSources(tmp_path / "docs")
    submission = _submission()
    sources.persist(submission, CONTENT)
    sentinel = tmp_path / "docs" / "keep.txt"
    sentinel.write_bytes(b"keep")
    with pytest.raises(IngestionSourceError):
        sources.persist(submission, CONTENT)
    with sources.verified_snapshot(submission) as snapshot:
        assert snapshot.read_bytes() == CONTENT
    sources.discard(submission)
    assert sentinel.read_bytes() == b"keep"
    assert not (tmp_path / "docs" / f"{submission.doc_id}_guide.md").exists()


def test_recovery_does_not_follow_source_symlink(tmp_path: Path) -> None:
    """受控文件名也不能让恢复读取指向其他目录的符号链接。

    Args:
        tmp_path: 受控目录和外部哨兵。
    """
    directory = tmp_path / "docs"
    directory.mkdir()
    external = tmp_path / "external.md"
    external.write_bytes(CONTENT)
    submission = _submission()
    (directory / f"{submission.doc_id}_guide.md").symlink_to(external)
    with pytest.raises(IngestionSourceError):
        with LocalIngestionSources(directory).verified_snapshot(submission):
            pytest.fail("来源链接不得进入解析")
    assert external.read_bytes() == CONTENT


def test_wrong_size_or_digest_is_rejected_before_source_creation(tmp_path: Path) -> None:
    """完整性资料不一致的上传不得留出可执行源文件。

    Args:
        tmp_path: 尚未创建的受控目录。
    """
    directory = tmp_path / "docs"
    with pytest.raises(IngestionSourceError):
        LocalIngestionSources(directory).persist(_submission(), b"wrong")
    assert not directory.exists()


def test_parser_io_error_is_not_misclassified_as_bad_source(tmp_path: Path) -> None:
    """上下文内部的流水线异常不能被源文件适配器错误包装。

    Args:
        tmp_path: 隔离源目录。
    """
    sources = LocalIngestionSources(tmp_path / "docs")
    submission = _submission()
    sources.persist(submission, CONTENT)
    with pytest.raises(OSError, match="parser disk error"):
        with sources.verified_snapshot(submission):
            raise OSError("parser disk error")


def test_recovery_rejects_fifo_without_blocking_open(tmp_path: Path) -> None:
    """恢复不能在fstat之前因打开FIFO永久阻塞，导致无法安全关闭服务。

    Args:
        tmp_path: 隔离受控目录。
    """
    directory = tmp_path / "docs"
    directory.mkdir()
    submission = _submission()
    path = directory / f"{submission.doc_id}_guide.md"
    os.mkfifo(path)

    def attempt() -> None:
        with pytest.raises(IngestionSourceError):
            with LocalIngestionSources(directory).verified_snapshot(submission):
                pytest.fail("FIFO不得作为恢复来源")

    # 另开读写描述符使旧实现也能结束；旧实现需先由超时证明其阻塞。
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(attempt)
        try:
            future.result(timeout=0.5)
        finally:
            if not future.done():
                descriptor = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
                os.close(descriptor)
