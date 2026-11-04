# -*- coding: utf-8 -*-
"""管理员要拿到真实有界正文与引用事实，而不是块数或伪造零占位。"""

import base64
import json
import sqlite3
from contextlib import AsyncExitStack
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from uuid import UUID
from uuid import uuid4

import httpx
import pytest

from app.bootstrap.container import build_application_container
from app.bootstrap.container import RuntimeOptions
from app.application.errors import TraceStorageError
from app.domain.conversations import AgentResult
from app.domain.conversations import ConversationTurn
from app.domain.conversations import TurnCitation
from app.domain.documents import DocumentChunk
from app.domain.knowledge_views import IndexedChunkHeader
from app.infrastructure.conversations.sqlite_repository import SQLiteConversationRepository
from app.main import create_app
from app.interfaces.http.auth import require_user
from tests.http_auth import allow_test_identity
from tests.test_auth_service import PASSWORD
from tests.test_ingestion_job_container import _settings


def _cite(
    repository: SQLiteConversationRepository, doc_id: UUID,
    chunk_ids: list[str | None],
) -> ConversationTurn:
    citations = []
    for position, chunk_id in enumerate(chunk_ids, start=1):
        citations.append(TurnCitation(
            position=position, doc_id=str(doc_id), filename="guide.txt", page=None,
            chunk_id=chunk_id, chunk_text="私密历史引用正文",
        ))
    return repository.save_turn(repository.create(owner_id=uuid4()).id, "私密历史问题", AgentResult(
        answer="私密历史答案", answer_mode="retrieval", retrieval_calls=1,
        citations=citations,
    ), 1.0)


@pytest.mark.anyio
async def test_real_chunk_pages_and_document_counts_use_formal_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """分页必须返回真实正文、支持稀疏顺序和opaque ID，每次正文读取不超7块。

    Args:
        tmp_path: 真实索引／业务库全隔离。
        monkeypatch: 观察真实正文边界并禁止全量读取。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    doc_id = uuid4()
    chunks = []
    for index in range(17):
        chunks.append(DocumentChunk(
            id=f"opaque/{doc_id}/part-{index}", doc_id=str(doc_id),
            filename="guide.txt", text=f"实际正文{index}", page=None, chunk_index=index * 2,
        ))
    root.vector_store.add_chunks(chunks, [[1.0, 0.0]] * len(chunks))
    repository = SQLiteConversationRepository(settings.conversation_db_path)
    _cite(repository, doc_id, [chunks[0].id, chunks[1].id])
    old = _cite(repository, doc_id, [chunks[0].id])
    deleted = _cite(repository, doc_id, [chunks[0].id])
    repository.mark_deleting(deleted.conversation_id)
    with sqlite3.connect(settings.conversation_db_path) as connection:
        connection.execute("UPDATE conversation_turns SET created_at=? WHERE id=?", (
            (datetime.now(UTC) - timedelta(days=8)).isoformat(), str(old.id),
        ))
    reads = []
    read = root.vector_store.read_chunks

    def tracked(headers: tuple[IndexedChunkHeader, ...]) -> tuple[DocumentChunk, ...]:
        reads.append(len(headers))
        return read(headers)

    def forbid_all() -> list[DocumentChunk]:
        raise AssertionError("诊断不能读取全部正文")

    monkeypatch.setattr(root.vector_store, "read_chunks", tracked)
    monkeypatch.setattr(root.vector_store, "list_chunks", forbid_all)
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            path = f"/api/admin/documents/{doc_id}/chunks"
            response = await client.get(path, params={"limit": 10})
            assert response.status_code == 200
            assert response.headers["Cache-Control"] == "no-store"
            first = response.json()
            assert first["total"] == 17
            assert first["published"] is True
            assert [row["text"] for row in first["chunks"]] == [f"实际正文{i}" for i in range(10)]
            assert first["chunks"][0]["citations_7d"] == 1
            assert first["chunks"][1]["citations_7d"] == 1
            assert first["chunks"][2]["citations_7d"] == 0
            second = await client.get(path, params={"limit": 10, "cursor": first["next_cursor"]})
            assert second.status_code == 200
            assert [row["chunk_index"] for row in second.json()["chunks"]] == list(range(20, 34, 2))
            assert second.json()["next_cursor"] is None
            assert sum(reads) == 17
            assert max(reads) <= 7
            document = (await client.get("/api/documents")).json()["documents"][0]
            assert document["citations_7d"] == 2
            assert document["unattributed_citations_7d"] == 0
            # 旧记录能归属文档却不能归属块，不能把未知误报为当前块的0。
            _cite(repository, doc_id, [None])
            partial = (await client.get(path, params={"limit": 3})).json()
            assert partial["unattributed_citations_7d"] == 1
            assert partial["chunks"][0]["citations_7d"] is None
            assert partial["chunks"][0]["known_citations_7d"] == 1
            assert partial["chunks"][2]["citations_7d"] is None
            assert partial["chunks"][2]["known_citations_7d"] == 0
            document = (await client.get("/api/documents")).json()["documents"][0]
            assert document["citations_7d"] == 3
            assert document["unattributed_citations_7d"] == 1
            assert "私密历史" not in response.text
    finally:
        root.close()


@pytest.mark.anyio
async def test_chunk_diagnostics_enforces_cookie_role_and_unpublished_boundary(tmp_path: Path) -> None:
    """后台正文只给真实管理员，普通来源面板也不能借管理员身份读未发布块。

    Args:
        tmp_path: 全部临时索引、认证及业务库。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    job = root.ingestion_job_service.submit("default", "guide.txt", b"payload", uuid4())
    root.vector_store.add_chunks([DocumentChunk(
        id=f"opaque/{job.doc_id}/0", doc_id=str(job.doc_id), filename="guide.txt",
        text="未发布的真实私密正文", page=None, chunk_index=0,
    )], [[1.0, 0.0]])
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    app.dependency_overrides.pop(require_user)
    path = f"/api/admin/documents/{job.doc_id}/chunks"
    try:
        async with AsyncExitStack() as stack:
            anonymous = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test"))
            assert (await anonymous.get(path)).status_code == 401
            for role in ("member", "admin"):
                root.auth_service.create_user(role, PASSWORD, name=role, role=role)
                client = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test"))
                assert (await client.post("/api/auth/login", json={"username": role, "password": PASSWORD})).status_code == 200
                response = await client.get(path)
                if role == "member":
                    assert response.status_code == 403
                    assert "私密正文" not in response.text
                else:
                    assert response.status_code == 200
                    assert response.json()["published"] is False
                    assert response.json()["chunks"][0]["text"] == "未发布的真实私密正文"
                    assert response.json()["chunks"][0]["citations_7d"] == 0
                normal = await client.get(f"/api/documents/{job.doc_id}/chunks/opaque/{job.doc_id}/0")
                assert normal.status_code == 404
                assert "私密正文" not in normal.text
        with sqlite3.connect(settings.trace_db_path) as connection:
            records = connection.execute("SELECT event_json FROM admin_access_audit").fetchall()
        assert len(records) == 1
        event = json.loads(records[0][0])
        assert event["resource_kind"] == "document_chunks"
        assert event["resource_ids"] == [str(job.doc_id)]
        assert "私密正文" not in records[0][0]
    finally:
        root.close()


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["audit", "statistics"])
async def test_diagnostic_storage_failure_never_reads_body_or_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """审计或正式引用故障必须503，且不能继续读取真实正文。

    Args:
        tmp_path: 全部隔离存储。
        monkeypatch: 注入存储故障并禁止正文读取。
        failure: 故障边界。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    identifier = uuid4()
    root.vector_store.add_chunks([DocumentChunk(
        id="opaque/a", doc_id=str(identifier), filename="guide.txt",
        text="不能泄露正文", page=None, chunk_index=0,
    )], [[1.0, 0.0]])
    root.admin_document_chunks

    def _forbid_body(headers: tuple[IndexedChunkHeader, ...]) -> tuple[DocumentChunk, ...]:
        raise AssertionError("故障后不应读取正文")

    monkeypatch.setattr(root.vector_store, "read_chunks", _forbid_body)
    if failure == "audit":
        def _fail_audit(event: object) -> None:
            raise TraceStorageError("内部路径不能泄露")

        monkeypatch.setattr(root.trace_repository, "record_admin_access", _fail_audit)
    else:
        with sqlite3.connect(settings.conversation_db_path) as connection:
            connection.execute("DROP TABLE turn_citations")
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/admin/documents/{identifier}/chunks")
            assert response.status_code == 503
            assert "不能泄露" not in response.text
            assert "内部路径" not in response.text
            if failure == "statistics":
                assert (await client.get("/api/documents")).status_code == 503
    finally:
        root.close()


def test_citation_statistics_runtime_overrides_do_not_open_production(tmp_path: Path) -> None:
    """只覆盖Chroma的评估不能把引用统计回落到生产业务库。

    Args:
        tmp_path: 生产候选和两个实际隔离目录。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    first = build_application_container(settings, RuntimeOptions(chroma_path=tmp_path / "eval", collection_name="first"))
    second = build_application_container(settings, RuntimeOptions(chroma_path=tmp_path / "eval", collection_name="second"))
    explicit = tmp_path / "selected-business.sqlite3"
    third = build_application_container(settings, RuntimeOptions(conversation_db_path=explicit))
    identifier = uuid4()
    try:
        assert first.citation_statistics.documents((identifier,))[identifier].total == 0
        path = next((tmp_path / "eval").rglob("conversations.sqlite3"))
        _cite(SQLiteConversationRepository(path), identifier, ["opaque/a"])
        assert first.citation_statistics.documents((identifier,))[identifier].total == 1
        assert second.citation_statistics.documents((identifier,))[identifier].total == 0
        assert third.citation_statistics.documents((identifier,))[identifier].total == 0
        assert explicit.is_file()
        assert not settings.conversation_db_path.exists()
        assert not settings.agent_checkpoint_db_path.exists()
        assert not settings.chroma_path.exists()
        assert not settings.trace_db_path.exists()
    finally:
        first.close()
        second.close()
        third.close()


@pytest.mark.anyio
async def test_chunk_cursor_is_bound_to_document_and_current_revision(tmp_path: Path) -> None:
    """跨文档复用或元数据改版后续页必须拒绝，而非混合两个来源版本。

    Args:
        tmp_path: 真实隔离索引。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    doc_id = uuid4()
    other_id = uuid4()
    chunks = []
    for index in range(3):
        chunks.append(DocumentChunk(id=f"{doc_id}:{index}", doc_id=str(doc_id), filename="guide.txt", text=f"text{index}", page=None, chunk_index=index))
    chunks.append(DocumentChunk(id=f"{other_id}:0", doc_id=str(other_id), filename="other.txt", text="other", page=None, chunk_index=0))
    root.vector_store.add_chunks(chunks, [[1.0, 0.0]] * len(chunks))
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            path = f"/api/admin/documents/{doc_id}/chunks"
            first = await client.get(path, params={"limit": 1})
            assert first.status_code == 200
            cursor = first.json()["next_cursor"]
            position = json.loads(base64.urlsafe_b64decode(cursor))
            position["last_index"] = 999
            forged = base64.urlsafe_b64encode(json.dumps(position).encode()).decode()
            assert (await client.get(path, params={"cursor": forged})).status_code == 422
            assert (await client.get(f"/api/admin/documents/{other_id}/chunks", params={"cursor": cursor})).status_code == 422
            for params in ({"limit": 101}, {"cursor": "bad"}):
                assert (await client.get(path, params=params)).status_code == 422
            root.vector_store.add_chunks([DocumentChunk(id=f"{doc_id}:3", doc_id=str(doc_id), filename="guide.txt", text="new", page=None, chunk_index=3)], [[1.0, 0.0]])
            changed = await client.get(path, params={"cursor": cursor})
            assert changed.status_code == 409
            assert changed.json()["error"]["code"] == "document_changed"
            assert changed.headers["Cache-Control"] == "no-store"
            assert (await client.get(f"/api/admin/documents/{uuid4()}/chunks")).status_code == 404
    finally:
        root.close()
