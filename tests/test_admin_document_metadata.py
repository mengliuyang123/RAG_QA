# -*- coding: utf-8 -*-
"""管理列表读取真实元数据与任务状态，不初始化模型或泄漏未发布正文。"""

from pathlib import Path
import sqlite3
from uuid import uuid4

import httpx
import pytest

from app.bootstrap.container import build_application_container
from app.domain.documents import DocumentChunk
from app.domain.knowledge_bases import KnowledgeBase
from app.domain.ingestion_jobs import IngestionJobSubmission
from app.main import create_app
from tests.http_auth import allow_test_identity
from tests.test_ingestion_job_container import _settings


@pytest.mark.anyio
async def test_admin_list_uses_metadata_and_real_job_state_without_model_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """列表若调用全量正文／模型、忽略KB或把未发布标成功，会使契约失败。

    Args:
        tmp_path: 所有生产适配器隔离路径。
        monkeypatch: 禁止回退到旧全量正文读取。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    uploader = uuid4()
    queued = root.ingestion_job_service.submit("ops", "guide.TXT", b"payload", uploader)
    legacy_id = uuid4()
    chunks = [
        DocumentChunk(id=f"{legacy_id}:0", doc_id=str(legacy_id), filename="old.md", text="旧私密正文", page=None, chunk_index=0),
        DocumentChunk(id=f"{queued.doc_id}:0", doc_id=str(queued.doc_id), filename="guide.TXT", text="未发布私密正文", page=None, chunk_index=0, kb_id="ops"),
    ]
    root.vector_store.add_chunks(chunks, [[1.0, 0.0], [1.0, 0.0]])

    def forbid_body() -> list[DocumentChunk]:
        raise AssertionError("管理列表不能读取所有正文")

    monkeypatch.setattr(root.vector_store, "list_chunks", forbid_body)
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get("/api/documents")
            assert response.status_code == 200
            assert response.headers["Cache-Control"] == "no-store"
            assert response.json()["total"] == 2
            rows = {row["doc_id"]: row for row in response.json()["documents"]}
            row = rows[str(queued.doc_id)]
            assert row["kb_id"] == "ops"
            assert row["file_type"] == "txt"
            assert row["size_bytes"] == 7
            assert row["uploaded_by"] == str(uploader)
            assert row["ingestion_status"] == "queued"
            assert row["published"] is False
            old = rows[str(legacy_id)]
            assert old["kb_id"] == "default"
            assert old["file_status"] == "missing"
            assert old["created_at"] is None
            assert old["size_bytes"] is None
            assert old["uploaded_by"] is None
            assert old["ingestion_status"] is None
            assert old["published"] is True
            assert "私密正文" not in response.text
            filtered = await client.get("/api/documents", params={"kb_id": "ops"})
            assert filtered.status_code == 200
            assert filtered.json()["total"] == 1
            assert filtered.json()["documents"][0]["doc_id"] == str(queued.doc_id)
            assert (await client.get("/api/documents", params={"kb_id": "none"})).json()["total"] == 0
            assert (await client.get("/api/documents", params={"kb_id": "bad/id"})).status_code == 422
            # 管理员能诊断部分索引，但普通浏览仍不能看到未发布文档。
            browsed = await client.get("/api/knowledge-bases/ops/documents")
            assert browsed.status_code == 200
            assert browsed.json()["documents"] == []
            claim = root.ingestion_job_repository.claim_next(lease_seconds=60)
            root.ingestion_job_repository.advance(claim, stage="embedding", completed_batches=1, total_batches=1)
            root.ingestion_job_repository.advance(claim, stage="indexing")
            root.ingestion_job_repository.succeed(claim, chunks_count=1)
            changed = (await client.get("/api/documents", params={"kb_id": "ops"})).json()["documents"][0]
            assert changed["ingestion_status"] == "succeeded"
            assert changed["published"] is True
            assert "lease_token" not in changed
            assert "content_sha256" not in changed
    finally:
        root.close()


def test_document_job_metadata_is_batched_and_never_exposes_recovery_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """101条任务不应触发101次查询，公开投影也不能包含摘要或执行权。

    Args:
        tmp_path: 真实临时目录数据库。
        monkeypatch: 给真实只读连接安装SQL观察器，不替代数据库结果。

    Returns:
        None。
    """
    root = build_application_container(_settings(tmp_path))
    uploader = uuid4()
    identifiers = []
    try:
        repository = root.ingestion_job_repository
        for _ in range(101):
            identifier = uuid4()
            identifiers.append(identifier)
            repository.enqueue(IngestionJobSubmission(
                job_id=f"job_{uuid4().hex}", doc_id=identifier, kb_id="default",
                filename="guide.md", size_bytes=1, uploaded_by=uploader,
                content_sha256="a" * 64,
            ), capacity=200)
        selects = []
        connect = sqlite3.connect

        def _trace(statement: str) -> None:
            normalized = " ".join(statement.lower().split())
            if normalized.startswith("select") and "from ingestion_jobs" in normalized:
                selects.append(statement)

        def _connect(*args: object, **kwargs: object) -> sqlite3.Connection:
            connection = connect(*args, **kwargs)
            connection.set_trace_callback(_trace)
            return connection

        monkeypatch.setattr(sqlite3, "connect", _connect)
        rows = repository.get_document_jobs(tuple(identifiers))
        assert set(rows) == set(identifiers)
        assert 1 <= len(selects) < 10
        for row in rows.values():
            assert row.uploaded_by == uploader
            assert row.status == "queued"
            assert set(row.model_dump()) == {"doc_id", "kb_id", "filename", "uploaded_by", "status"}
    finally:
        root.close()


@pytest.mark.anyio
@pytest.mark.parametrize("fault", ["status", "identity"])
async def test_admin_list_fails_closed_on_corrupt_task_metadata(tmp_path: Path, fault: str) -> None:
    """损坏任务不得退回“旧文档”并伪称发布，也不能错误关联其他来源。

    Args:
        tmp_path: 真实临时索引及目录。
        fault: 任务完整状态或来源身份损坏。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    queued = root.ingestion_job_service.submit("default", "guide.txt", b"payload", uuid4())
    root.vector_store.add_chunks([DocumentChunk(
        id=f"{queued.doc_id}:0", doc_id=str(queued.doc_id), filename="guide.txt",
        text="私密正文", page=None, chunk_index=0,
    )], [[1.0, 0.0]])
    with sqlite3.connect(settings.knowledge_base_db_path) as connection:
        if fault == "status":
            connection.execute("UPDATE ingestion_jobs SET status='succeeded'")
        else:
            connection.execute("UPDATE ingestion_jobs SET filename='another.txt'")
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get("/api/documents")
            assert response.status_code == (503 if fault == "status" else 500)
            assert "私密正文" not in response.text
    finally:
        root.close()
