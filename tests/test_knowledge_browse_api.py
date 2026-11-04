# -*- coding: utf-8 -*-
"""只读文档分页、过滤、引用上下文与预设建议的真实HTTP契约。"""

from datetime import UTC
from datetime import datetime
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import os
from urllib.parse import quote
from uuid import UUID
from uuid import uuid4

import httpx
import pytest

from app.domain.documents import DocumentChunk
from app.domain.ingestion_jobs import IngestionJobSubmission
from app.domain.knowledge_bases import KnowledgeBase
from app.application.errors import KnowledgeContentError
from tests.test_auth_service import PASSWORD
from tests.test_unified_query_api import _app


DOC = UUID("11111111-1111-4111-8111-111111111111")
SECOND = UUID("22222222-2222-4222-8222-222222222222")


@asynccontextmanager
async def _client(app: object) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://test",
    ) as client:
        await client.post("/api/auth/login", json={"username": "member", "password": PASSWORD})
        yield client


@pytest.mark.anyio
async def test_member_document_browse_filters_before_cursor_pagination(tmp_path: Path) -> None:
    """分页前不分库、按chunk计数或公开源文件运维字段都会失败。

    Args:
        tmp_path: 真实索引和认证数据的隔离目录。
    """
    app, _, quality = _app(tmp_path, legacy_identity=False)
    root = app.state.container
    root.auth_service.create_user("member", PASSWORD, name="成员", role="member")
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    updated = datetime(2026, 10, 21, tzinfo=UTC)
    chunks = [
        DocumentChunk(id="one", doc_id=str(DOC), filename="部署_Plan%.MD", text="PRIVATE_BODY",
                      chunk_index=0, page=None, kb_id="ops", doc_updated_at=updated),
        DocumentChunk(id="two", doc_id=str(DOC), filename="部署_Plan%.MD", text="PRIVATE_BODY",
                      chunk_index=1, page=None, kb_id="ops", doc_updated_at=updated),
        DocumentChunk(id="three", doc_id=str(SECOND), filename="部署-B.md", text="PRIVATE_BODY",
                      chunk_index=0, page=None, kb_id="ops"),
    ]
    root.vector_store.add_chunks(chunks, [[1.0, 0.0] for _ in chunks])
    try:
        async with _client(app) as client:
            first = await client.get("/api/knowledge-bases/ops/documents", params={"limit": 1})
            assert first.status_code == 200
            assert first.headers["Cache-Control"] == "no-store"
            payload = first.json()
            assert payload["documents"][0]["doc_id"] == str(DOC)
            assert payload["documents"][0]["chunks_count"] == 2
            assert payload["documents"][0]["file_type"] == "md"
            assert payload["documents"][0]["updated_at"] == "2026-10-21T00:00:00Z"
            assert payload["documents"][0]["is_stale"] is False
            cursor = payload["next_cursor"]
            assert cursor
            second = await client.get("/api/knowledge-bases/ops/documents", params={"cursor": cursor, "limit": 1})
            assert [doc["doc_id"] for doc in second.json()["documents"]] == [str(SECOND)]
            assert second.json()["documents"][0]["updated_at"] is None
            assert second.json()["next_cursor"] is None
            filtered = await client.get("/api/knowledge-bases/ops/documents", params={"q": "_PLAN%"})
            assert [doc["doc_id"] for doc in filtered.json()["documents"]] == [str(DOC)]
            for params in ({"cursor": "bad"}, {"cursor": cursor, "q": "different"}, {"limit": 101}):
                assert (await client.get("/api/knowledge-bases/ops/documents", params=params)).status_code == 422
            assert (await client.get("/api/knowledge-bases/default/documents", params={"cursor": cursor})).status_code == 422
            assert (await client.get("/api/knowledge-bases/missing/documents")).status_code == 404
            for private in ("PRIVATE_BODY", "file_status", "ingestion_status", "created_at"):
                assert private not in first.text
            assert quality.evidence_calls == 0
    finally:
        root.close()


@pytest.mark.anyio
async def test_pending_document_is_hidden_from_browse_counts_context_and_retrieval(
    tmp_path: Path,
) -> None:
    """只过滤向量召回不够；未发布正文也不能经浏览或原文窗口泄漏。

    Args:
        tmp_path: 含部分写入索引的隔离应用。
    """
    app, chunk, _ = _app(tmp_path, legacy_identity=False)
    root = app.state.container
    member = root.auth_service.create_user("member", PASSWORD, name="成员", role="member")
    try:
        root.ingestion_job_repository.enqueue(IngestionJobSubmission(
            job_id=f"job_{uuid4().hex}", doc_id=UUID(chunk.doc_id), kb_id="default",
            filename=chunk.filename, size_bytes=10, uploaded_by=member.id,
            content_sha256="c" * 64,
        ), capacity=1)
        async with _client(app) as client:
            cards = await client.get("/api/knowledge-bases")
            assert cards.status_code == 200
            assert cards.json()["knowledge_bases"][0]["documents_count"] == 0
            documents = await client.get("/api/knowledge-bases/default/documents")
            assert documents.json()["documents"] == []
            target = quote(chunk.id, safe="")
            path = f"/api/documents/{chunk.doc_id}/chunks/{target}"
            assert (await client.get(path)).status_code == 404
            assert root.retrieval_service.retrieve("needle", "hybrid", 1) == []
            claim = root.ingestion_job_repository.claim_next(lease_seconds=30)
            assert claim is not None
            root.ingestion_job_repository.advance(
                claim, stage="embedding", completed_batches=1, total_batches=1,
            )
            root.ingestion_job_repository.advance(claim, stage="indexing")
            root.ingestion_job_repository.succeed(claim, chunks_count=1)
            cards = await client.get("/api/knowledge-bases")
            assert cards.json()["knowledge_bases"][0]["documents_count"] == 1
            assert (await client.get(path)).status_code == 200
            documents = await client.get("/api/knowledge-bases/default/documents")
            assert [item["doc_id"] for item in documents.json()["documents"]] == [chunk.doc_id]
            assert [item.chunk_id for item in root.retrieval_service.retrieve("needle", "hybrid", 1)] == [chunk.id]
    finally:
        root.close()


@pytest.mark.anyio
async def test_context_uses_neighbors_and_rejects_other_document_chunk(tmp_path: Path) -> None:
    """ID不能解析index，跨文档ID不能读正文，未知资源不能猜测成410。

    Args:
        tmp_path: 隔离文档和真实索引。
    """
    app, other, _ = _app(tmp_path, legacy_identity=False)
    root = app.state.container
    root.auth_service.create_user("member", PASSWORD, name="成员", role="member")
    updated = datetime(2026, 10, 21, tzinfo=UTC)
    chunks = []
    for index, identifier in ((0, "opaque-first"), (10, "opaque/target"), (20, "opaque-last")):
        chunks.append(DocumentChunk(id=identifier, doc_id=str(DOC), filename="guide.md",
                                    text=f"正文{index}", chunk_index=index, page=None,
                                    doc_updated_at=updated))
    root.vector_store.add_chunks(chunks, [[1.0, 0.0] for _ in chunks])
    try:
        async with _client(app) as client:
            target = quote("opaque/target", safe="")
            path = f"/api/documents/{DOC}/chunks/{target}"
            response = await client.get(path)
            assert response.status_code == 200
            body = response.json()
            assert body["doc_updated_at"] == "2026-10-21T00:00:00Z"
            assert [item["chunk_index"] for item in body["chunks"]] == [0, 10, 20]
            assert [item["is_target"] for item in body["chunks"]] == [False, True, False]
            assert "needle" not in response.text
            only = await client.get(path, params={"context": 0})
            assert [item["chunk_id"] for item in only.json()["chunks"]] == ["opaque/target"]
            assert (await client.get(path, params={"context": 4})).status_code == 422
            assert (await client.get(f"/api/documents/{other.doc_id}/chunks/{target}")).status_code == 404
            assert (await client.get(f"/api/documents/{uuid4()}/chunks/unknown")).status_code == 404
    finally:
        root.close()


@pytest.mark.anyio
async def test_confirmed_removed_document_is_gone_but_failed_delete_keeps_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """删除事实必须持久保存；只有删除意图、仍有正文时不能错误返回410。

    Args:
        tmp_path: 隔离业务索引与目录。
        monkeypatch: 模拟一次未改变索引的删除失败。
    """
    app, chunk, _ = _app(tmp_path, legacy_identity=False)
    root = app.state.container
    root.auth_service.create_user("member", PASSWORD, name="成员", role="member")
    real_delete = root.ingestion_pipeline.delete_document

    def failed(identifier: str) -> None:
        raise RuntimeError("PRIVATE_DELETE_FAILURE")

    try:
        async with _client(app) as client:
            path = f"/api/documents/{chunk.doc_id}/chunks/{chunk.id}"
            assert (await client.get(path)).status_code == 200
            monkeypatch.setattr(root.ingestion_pipeline, "delete_document", failed)
            with pytest.raises(RuntimeError):
                root.document_service.delete_document(UUID(chunk.doc_id))
            assert (await client.get(path)).status_code == 200
            monkeypatch.setattr(root.ingestion_pipeline, "delete_document", real_delete)
            root.document_service.delete_document(UUID(chunk.doc_id))
            gone = await client.get(path)
            assert gone.status_code == 410
            assert gone.json()["error"]["code"] == "document_deleted"
            assert "PRIVATE_DELETE_FAILURE" not in gone.text
            unknown = await client.get(f"/api/documents/{uuid4()}/chunks/unknown")
            assert unknown.status_code == 404
            from app.infrastructure.persistence.knowledge_registry import SQLiteKnowledgeReadRegistry
            reopened = SQLiteKnowledgeReadRegistry(tmp_path / "catalog.sqlite3")
            assert reopened.was_deleted(UUID(chunk.doc_id)) is True
    finally:
        root.close()


@pytest.mark.anyio
async def test_upload_records_version_time_not_current_file_mtime(tmp_path: Path) -> None:
    """新文档时间不能始终null或从可被外部修改的源文件mtime猜测。

    Args:
        tmp_path: 文件和索引的隔离目录。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    root = app.state.container
    root.auth_service.create_user("member", PASSWORD, name="成员", role="member")
    try:
        before = datetime.now(UTC)
        uploaded = root.upload_service.save_and_parse("fresh.md", b"# SQLite\n\nPersistent state.")
        after = datetime.now(UTC)
        source = tmp_path / "docs" / f"{uploaded.doc_id}_fresh.md"
        os.utime(source, (946684800, 946684800))
        async with _client(app) as client:
            response = await client.get("/api/knowledge-bases/default/documents", params={"q": "fresh"})
            assert response.status_code == 200
            value = response.json()["documents"][0]["updated_at"]
            assert value is not None
            updated = datetime.fromisoformat(value.replace("Z", "+00:00"))
            assert before <= updated <= after
    finally:
        root.close()


@pytest.mark.anyio
async def test_curated_questions_require_admin_and_never_publish_private_history(tmp_path: Path) -> None:
    """预设建议必须真实持久化，不能采集历史提问、忽略范围或允许成员发布。

    Args:
        tmp_path: 全部业务数据的临时目录。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    root = app.state.container
    root.auth_service.create_user("member", PASSWORD, name="成员", role="member")
    root.auth_service.create_user("admin", PASSWORD, name="管理员", role="admin")
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    path = "/api/admin/knowledge-bases/ops/suggested-questions"
    body = {"questions": [" 服务启动失败怎么办 ", "如何查看日志"]}
    try:
        async with _client(app) as member:
            member.headers["X-CSRF-Token"] = member.cookies.get("rag_csrf")
            created = await member.post("/api/conversations")
            private_turn = await member.post(
                f"/api/conversations/{created.json()['id']}/query",
                json={"question": "PRIVATE_HISTORY_NOT_FOR_HOMEPAGE"},
            )
            assert private_turn.status_code == 200
            assert (await member.get("/api/suggested-questions")).json() == {"questions": []}
            assert (await member.put(path, json=body)).status_code == 403
            assert (await member.get("/api/suggested-questions")).json() == {"questions": []}
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as admin:
                await admin.post("/api/auth/login", json={"username": "admin", "password": PASSWORD})
                assert (await admin.put(path, json=body)).status_code == 403
                assert (await member.get("/api/suggested-questions")).json() == {"questions": []}
                admin.headers["X-CSRF-Token"] = admin.cookies.get("rag_csrf")
                written = await admin.put(path, json=body)
                assert written.status_code == 200
                assert [q["text"] for q in written.json()["questions"]] == ["服务启动失败怎么办", "如何查看日志"]
                from app.infrastructure.persistence.knowledge_registry import SQLiteKnowledgeReadRegistry
                reopened = SQLiteKnowledgeReadRegistry(tmp_path / "catalog.sqlite3")
                assert [q.id for q in reopened.list_curated("ops", 50)] == [q["id"] for q in written.json()["questions"]]
                retry = await admin.put(path, json=body)
                assert retry.json() == written.json()
                for invalid in ({"questions": [" "]}, {"questions": ["same", "same"]}, {"questions": ["x"], "source": "popular"}):
                    assert (await admin.put(path, json=invalid)).status_code == 422
                missing = await admin.put("/api/admin/knowledge-bases/missing/suggested-questions", json=body)
                assert missing.status_code == 404
                shown = await member.get("/api/suggested-questions", params={"kb_id": "ops", "limit": 1})
                assert shown.status_code == 200
                assert shown.headers["Cache-Control"] == "no-store"
                assert len(shown.json()["questions"]) == 1
                assert shown.json()["questions"][0]["source"] == "curated"
                assert "PRIVATE_HISTORY_NOT_FOR_HOMEPAGE" not in shown.text
                assert (await member.get("/api/suggested-questions", params={"kb_id": "default"})).json() == {"questions": []}
                assert (await member.get("/api/suggested-questions", params={"kb_id": "missing"})).status_code == 404
                cleared = await admin.put(path, json={"questions": []})
                assert cleared.json() == {"questions": []}
                assert (await member.get("/api/suggested-questions")).json() == {"questions": []}
    finally:
        root.close()


@pytest.mark.anyio
@pytest.mark.parametrize("path", [
    "/api/knowledge-bases/default/documents",
    f"/api/documents/{DOC}/chunks/unknown",
    "/api/suggested-questions",
])
async def test_read_apis_reject_anonymous_before_reading_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str,
) -> None:
    """不能只保护运维列表，而把新增成员读取接口暴露给匿名用户。

    Args:
        tmp_path: 隔离数据。
        monkeypatch: 禁止访问实际内容读取边界。
        path: 三种新增只读入口。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("匿名请求不得读取索引内容")

    monkeypatch.setattr(app.state.container.vector_store._collection, "get", forbidden)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
            assert (await client.get(path)).status_code == 401
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_max_context_is_seven_chunks_and_storage_errors_are_not_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """上下文不能全量读正文；存储故障不得伪装成404/410。

    Args:
        tmp_path: 真实索引隔离目录。
        monkeypatch: 验证SDK读取范围并模拟后续故障。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    root = app.state.container
    root.auth_service.create_user("member", PASSWORD, name="成员", role="member")
    chunks = []
    for index in range(9):
        chunks.append(DocumentChunk(
            id=f"opaque-{index}", doc_id=str(DOC), filename="guide.md",
            text=f"段{index}", chunk_index=index, page=index + 1,
        ))
    root.vector_store.add_chunks(chunks, [[1.0, 0.0] for _ in chunks])
    original = root.vector_store._collection.get

    def bounded(*args: object, **kwargs: object):
        if "documents" in kwargs.get("include", []):
            if set(kwargs.get("ids", [])) != {f"opaque-{index}" for index in range(1, 8)}:
                raise AssertionError("正文读取不应超出选定的七块")
        return original(*args, **kwargs)

    monkeypatch.setattr(root.vector_store._collection, "get", bounded)
    try:
        async with _client(app) as client:
            response = await client.get(f"/api/documents/{DOC}/chunks/opaque-4", params={"context": 3})
            assert response.status_code == 200
            assert [chunk["chunk_index"] for chunk in response.json()["chunks"]] == list(range(1, 8))
            assert response.json()["chunks"][3]["page"] == 5

            def unavailable(*args: object, **kwargs: object) -> None:
                raise RuntimeError("PRIVATE_STORE_DETAILS")

            monkeypatch.setattr(root.vector_store._collection, "get", unavailable)
            failed = await client.get(f"/api/documents/{DOC}/chunks/opaque-4")
            assert failed.status_code == 503
            assert "PRIVATE_STORE_DETAILS" not in failed.text
    finally:
        root.close()


@pytest.mark.anyio
async def test_failed_deletion_record_prevents_index_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不能先删除索引，等删除记录写入失败才发现来源无法可靠回显。

    Args:
        tmp_path: 隔离存储。
        monkeypatch: 在登记写边界模拟一次持久化故障。
    """
    app, chunk, _ = _app(tmp_path)
    root = app.state.container
    root.document_service

    def failed(doc_id: UUID, kb_id: str) -> None:
        raise KnowledgeContentError("PRIVATE_REGISTRY_ERROR")

    monkeypatch.setattr(root.knowledge_read_registry, "record_deletion", failed)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete(f"/api/documents/{chunk.doc_id}")
        assert response.status_code == 503
        assert "PRIVATE_REGISTRY_ERROR" not in response.text
        assert root.vector_store.list_chunk_headers(UUID(chunk.doc_id))[0].chunk_id == chunk.id
        assert root.knowledge_read_registry.was_deleted(UUID(chunk.doc_id)) is False
    finally:
        root.close()
