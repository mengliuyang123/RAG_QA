# -*- coding: utf-8 -*-
"""知识库删除必须清真实索引与源，保留历史并可重试部分失败。"""

import json
import sqlite3
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
from threading import Event
from threading import Thread
from uuid import uuid4

import httpx
import pytest
import anyio

from app.bootstrap.container import build_application_container
from app.bootstrap.container import RuntimeOptions
from app.bootstrap.factories import AdapterFactories
from app.domain.documents import DocumentChunk
from app.domain.knowledge_bases import KnowledgeBase
from app.domain.suggestions import CuratedQuestionsUpdate
from app.main import create_app
from app.application.errors import TraceStorageError
from app.application.errors import VectorStoreError
from app.interfaces.http.auth import require_user
from app.infrastructure.persistence.knowledge_deletion import SQLiteKnowledgeDeletionRepository
from tests.http_auth import allow_test_identity
from tests.test_admin_chunk_diagnostics import _cite
from tests.test_ingestion_job_container import _settings
from tests.test_job_ingestion_pipeline import BatchEmbedder
from tests.test_auth_service import PASSWORD
from app.domain.knowledge_deletion import KnowledgeDeletionPlan
from app.infrastructure.conversations.sqlite_repository import SQLiteConversationRepository


@pytest.mark.anyio
async def test_force_delete_removes_only_its_documents_and_keeps_formal_history(tmp_path: Path) -> None:
    """只删目录、误删别库、丢任务／会话或让预设泄露都会失败。

    Args:
        tmp_path: 全部数据隔离，向量化为确定性离线端口。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    factories = replace(AdapterFactories(), embedding_provider=lambda *args: BatchEmbedder())
    root = build_application_container(settings, factories=factories)
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    root.knowledge_browse_service.replace_questions("ops", CuratedQuestionsUpdate(questions=("公开运维问题",)))
    job = root.ingestion_job_service.submit("ops", "guide.txt", b"abcdefghij\n" * 3, uuid4())
    assert root.ingestion_job_runner.run_next(Event()) is True
    other = uuid4()
    root.vector_store.add_chunks([DocumentChunk(id=f"{other}:0", doc_id=str(other), filename="other.txt", text="别库正文", page=None, chunk_index=0)], [[1.0, 0.0]])
    chunks = root.vector_store.list_chunks()
    own_chunk = next(chunk for chunk in chunks if chunk.doc_id == str(job.doc_id))
    repository = SQLiteConversationRepository(settings.conversation_db_path)
    turn = _cite(repository, job.doc_id, [own_chunk.id])
    source = settings.docs_path / f"{job.doc_id}_guide.txt"
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            path = "/api/admin/knowledge-bases/ops"
            refused = await client.delete(path)
            assert refused.status_code == 409
            assert refused.json()["error"]["code"] == "knowledge_base_not_empty"
            assert source.is_file()
            assert root.knowledge_base_repository.get("ops").id == "ops"
            deleted = await client.delete(path, params={"force": True})
            assert deleted.status_code == 200
            assert deleted.json() == {"kb_id": "ops", "deleted": True, "documents_deleted": 1}
            assert deleted.headers["Cache-Control"] == "no-store"
            assert not source.exists()
            assert {chunk.doc_id for chunk in root.vector_store.list_chunks()} == {str(other)}
            assert root.keyword_index.search("abcdefghij", 5, kb_ids=("ops",)) == []
            assert root.ingestion_job_service.get(job.job_id).status == "succeeded"
            assert repository.get(turn.conversation_id).turns[0].citations[0].chunk_id == own_chunk.id
            assert (await client.get(f"/api/documents/{job.doc_id}/chunks/{own_chunk.id}")).status_code == 410
            listed = (await client.get("/api/knowledge-bases")).json()
            assert [row["id"] for row in listed["knowledge_bases"]] == ["default"]
            assert (await client.get("/api/suggested-questions")).json()["questions"] == []
            assert (await client.patch(path, json={"name": "恢复"})).status_code == 404
            assert (await client.post("/api/admin/knowledge-bases", json={"id": "ops", "name": "重用"})).status_code == 409
            repeated = await client.delete(path, params={"force": True})
            assert repeated.status_code == 200
            assert repeated.json() == deleted.json()
            with sqlite3.connect(settings.trace_db_path) as connection:
                events = connection.execute("SELECT event_json FROM admin_access_audit").fetchall()
            attempts = []
            for row in events:
                event = json.loads(row[0])
                if event["resource_kind"] == "knowledge_base":
                    attempts.append(event)
            assert len(attempts) == 2
            assert all(event["action"] == "delete_attempt" for event in attempts)
            assert all(event["knowledge_base_id"] == "ops" for event in attempts)
    finally:
        root.close()


@pytest.mark.anyio
async def test_empty_base_delete_and_default_protection_need_no_models(tmp_path: Path) -> None:
    """空库实删后不能重新出现，default不能伪称删除后初始化又重建。

    Args:
        tmp_path: 无模型凭据的完整隔离目录。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    root.knowledge_base_repository.create(KnowledgeBase(id="empty", name="空库"))
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete("/api/admin/knowledge-bases/empty")
            assert response.status_code == 200
            assert response.json()["documents_deleted"] == 0
            protected = await client.delete("/api/admin/knowledge-bases/default", params={"force": True})
            assert protected.status_code == 409
            assert protected.json()["error"]["code"] == "knowledge_base_protected"
            assert (await client.delete("/api/admin/knowledge-bases/missing")).status_code == 404
    finally:
        root.close()
    reopened = build_application_container(settings)
    try:
        assert [row.id for row in reopened.knowledge_base_repository.list()] == ["default"]
    finally:
        reopened.close()


@pytest.mark.anyio
@pytest.mark.parametrize("running", [False, True])
async def test_pending_job_blocks_force_delete_without_any_index(tmp_path: Path, running: bool) -> None:
    """只检查已入库文档会漏掉零索引queued/running并破坏可恢复源文件。

    Args:
        tmp_path: 实际临时任务库与源文件。
        running: 是否已经认领但尚未产生索引。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    job = root.ingestion_job_service.submit("ops", "guide.txt", b"source", uuid4())
    if running:
        assert root.ingestion_job_repository.claim_next(lease_seconds=60) is not None
        with sqlite3.connect(settings.knowledge_base_db_path) as connection:
            connection.execute("UPDATE ingestion_jobs SET lease_expires_at='2000-01-01T00:00:00+00:00'")
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for force in (False, True):
                response = await client.delete("/api/admin/knowledge-bases/ops", params={"force": force})
                assert response.status_code == 409
                assert response.json()["error"]["code"] == "knowledge_base_ingestion_in_progress"
            assert (settings.docs_path / f"{job.doc_id}_guide.txt").read_bytes() == b"source"
            assert root.ingestion_job_service.get(job.job_id).status == ("running" if running else "queued")
            assert root.knowledge_base_repository.get("ops").id == "ops"
    finally:
        root.close()


@pytest.mark.anyio
async def test_running_worker_rejects_kb_delete_before_waiting_for_document_lock(tmp_path: Path) -> None:
    """worker已持变更锁时，KB删除必须立即拒绝而非任务完成后偷偷整库删除。

    Args:
        tmp_path: 全部真实临时存储。

    Returns:
        None。
    """
    entered = Event()
    release = Event()

    class BlockingEmbedder(BatchEmbedder):
        """只暂停外部向量化边界，其他生命周期均真实执行。"""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """让请求能在真实持锁期间验证前置检查。

            Args:
                texts: 实际分块文本。

            Returns:
                恢复后确定性离线向量。
            """
            entered.set()
            assert release.wait(4)
            return super().embed_documents(texts)

    settings = _settings(tmp_path)
    root = build_application_container(settings, factories=replace(AdapterFactories(), embedding_provider=lambda *args: BlockingEmbedder()))
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    job = root.ingestion_job_service.submit("ops", "guide.txt", b"abcdefghij\n" * 3, uuid4())
    worker = Thread(target=root.ingestion_job_runner.run_next, args=(Event(),))
    worker.start()
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        assert entered.wait(2)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with anyio.fail_after(0.8):
                response = await client.delete("/api/admin/knowledge-bases/ops", params={"force": True})
            assert response.status_code == 409
            assert root.ingestion_job_service.get(job.job_id).status == "running"
    finally:
        release.set()
        worker.join(5)
        assert not worker.is_alive()
        root.close()


@pytest.mark.anyio
async def test_force_delete_cleans_failed_job_source_with_no_index(tmp_path: Path) -> None:
    """只遍历索引会遗留失败任务的已登记源文件，不能把该库当作空库。

    Args:
        tmp_path: 临时真实任务与文件。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    job = root.ingestion_job_service.submit("ops", "guide.txt", b"source", uuid4())
    claim = root.ingestion_job_repository.claim_next(lease_seconds=60)
    root.ingestion_job_repository.fail(claim, code="parse_failed")
    source = settings.docs_path / f"{job.doc_id}_guide.txt"
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.delete("/api/admin/knowledge-bases/ops")).status_code == 409
            response = await client.delete("/api/admin/knowledge-bases/ops", params={"force": True})
            assert response.status_code == 200
            assert response.json()["documents_deleted"] == 1
            assert not source.exists()
            assert root.ingestion_job_service.get(job.job_id).status == "failed"
            assert root.ingestion_job_repository.claim_next(lease_seconds=60) is None
    finally:
        root.close()


@pytest.mark.anyio
async def test_kb_delete_audit_failure_keeps_catalog_index_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """审计失败不能先把库隐藏、改变计划或删除任何正文资源。

    Args:
        tmp_path: 隔离索引及目录。
        monkeypatch: 只注入审计存储故障。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    root.knowledge_base_repository.create(KnowledgeBase(id="empty", name="空库"))

    def _fail(event: object) -> None:
        raise TraceStorageError("内部路径不能泄露")

    monkeypatch.setattr(root.trace_repository, "record_admin_access", _fail)
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete("/api/admin/knowledge-bases/empty")
            assert response.status_code == 503
            assert "内部路径" not in response.text
            assert root.knowledge_base_repository.get("empty").id == "empty"
            with sqlite3.connect(settings.knowledge_base_db_path) as connection:
                assert connection.execute("SELECT COUNT(*) FROM knowledge_deletion_plans").fetchone()[0] == 0
    finally:
        root.close()


@pytest.mark.anyio
async def test_kb_delete_enforces_real_cookie_role_and_csrf(tmp_path: Path) -> None:
    """前端按钮隐藏或仅检查角色都不能替代后端Cookie与CSRF验证。

    Args:
        tmp_path: 隔离认证及目录。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    root.knowledge_base_repository.create(KnowledgeBase(id="empty", name="空库"))
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    app.dependency_overrides.pop(require_user)
    path = "/api/admin/knowledge-bases/empty"
    try:
        async with AsyncExitStack() as stack:
            anonymous = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test"))
            assert (await anonymous.delete(path)).status_code == 401
            for role in ("member", "admin"):
                root.auth_service.create_user(role, PASSWORD, name=role, role=role)
                client = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test"))
                assert (await client.post("/api/auth/login", json={"username": role, "password": PASSWORD})).status_code == 200
                assert (await client.delete(path)).status_code == 403
                client.headers["X-CSRF-Token"] = client.cookies["rag_csrf"]
                response = await client.delete(path)
                assert response.status_code == (200 if role == "admin" else 403)
    finally:
        root.close()


@pytest.mark.anyio
async def test_runtime_delete_uses_its_own_catalog_sources_and_audit(tmp_path: Path) -> None:
    """复用通用DB路径而非原catalog派生规则，会删失败或误用另一套目录。

    Args:
        tmp_path: 所有生产哨兵及实际runtime都在临时目录。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    runtime = tmp_path / "runtime"
    root = build_application_container(settings, RuntimeOptions(chroma_path=runtime, collection_name="delete-test"))
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    job = root.ingestion_job_service.submit("ops", "guide.txt", b"source", uuid4())
    claim = root.ingestion_job_repository.claim_next(lease_seconds=60)
    assert claim is not None
    root.ingestion_job_repository.fail(claim, code="parse_failed")
    root.vector_store.add_chunks([DocumentChunk(id=f"{job.doc_id}:0", doc_id=str(job.doc_id), kb_id="ops", filename="guide.txt", text="真实部分索引", page=None, chunk_index=0)], [[1.0, 0.0]])
    settings.docs_path.mkdir(parents=True)
    sentinel = settings.docs_path / f"{job.doc_id}_guide.txt"
    sentinel.write_bytes(b"production sentinel")
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete("/api/admin/knowledge-bases/ops", params={"force": True})
            assert response.status_code == 200
            assert response.json()["documents_deleted"] == 1
            assert list(runtime.rglob(f"{job.doc_id}_guide.txt")) == []
            assert sentinel.read_bytes() == b"production sentinel"
            assert root.ingestion_job_service.get(job.job_id).status == "failed"
            assert not settings.knowledge_base_db_path.exists()
            assert not settings.trace_db_path.exists()
            assert not settings.conversation_db_path.exists()
    finally:
        root.close()


@pytest.mark.anyio
async def test_delete_refuses_index_document_owned_by_other_kb_task(tmp_path: Path) -> None:
    """损坏索引把别库docUUID归到目标库时，不能把别库真实源文件也删掉。

    Args:
        tmp_path: 全部临时真实存储。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    job = root.ingestion_job_service.submit("default", "guide.txt", b"other source", uuid4())
    claim = root.ingestion_job_repository.claim_next(lease_seconds=60)
    root.ingestion_job_repository.fail(claim, code="parse_failed")
    root.vector_store.add_chunks([DocumentChunk(id=f"{job.doc_id}:0", doc_id=str(job.doc_id), kb_id="ops", filename="guide.txt", text="错误归属索引", page=None, chunk_index=0)], [[1.0, 0.0]])
    source = settings.docs_path / f"{job.doc_id}_guide.txt"
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete("/api/admin/knowledge-bases/ops", params={"force": True})
            assert response.status_code == 500
            assert source.read_bytes() == b"other source"
            assert root.vector_store.list_chunks()
            assert root.knowledge_base_repository.get("ops").id == "ops"
    finally:
        root.close()


@pytest.mark.anyio
async def test_force_delete_resumes_source_cleanup_after_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """索引已删而源清理失败时，重启后必须根据原计划继续，不丢失源身份。

    Args:
        tmp_path: 真实临时源、Chroma及SQLite。
        monkeypatch: 只模拟目标文件的unlink权限故障。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    identifier = uuid4()
    root.vector_store.add_chunks([DocumentChunk(id=f"{identifier}:0", doc_id=str(identifier), kb_id="ops", filename="guide.txt", text="旧索引正文", page=None, chunk_index=0)], [[1.0, 0.0]])
    source = settings.docs_path / f"{identifier}_guide.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    unlink = Path.unlink

    def _unlink(path: Path, missing_ok: bool = False) -> None:
        if path == source:
            raise PermissionError("不能泄露路径")
        unlink(path, missing_ok=missing_ok)

    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", _unlink)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.delete("/api/admin/knowledge-bases/ops", params={"force": True})
                assert response.status_code == 500
                assert response.json()["error"]["code"] == "knowledge_base_delete_partial"
                assert "不能泄露路径" not in response.text
                assert root.vector_store.list_chunks() == []
                assert source.is_file()
                assert [row.id for row in root.knowledge_base_repository.list()] == ["default"]
                assert (await client.delete("/api/admin/knowledge-bases/ops")).status_code == 409
    finally:
        root.close()
    reopened = build_application_container(settings)
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: reopened))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete("/api/admin/knowledge-bases/ops", params={"force": True})
            assert response.status_code == 200
            assert response.json()["documents_deleted"] == 1
            assert not source.exists()
            assert reopened.knowledge_read_registry.was_deleted(identifier)
    finally:
        reopened.close()


@pytest.mark.anyio
async def test_enqueue_between_prepare_and_freeze_blocks_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """应用预检不够，真实竞争任务提交后SQL冻结仍应409而不是丢失源文件。

    Args:
        tmp_path: 真实临时源文件与队列。
        monkeypatch: 在原SQL提交前安排一次真实入队，不替代原实现。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    begin = SQLiteKnowledgeDeletionRepository.begin
    admitted = []

    def _begin(
        repository: SQLiteKnowledgeDeletionRepository,
        plan: KnowledgeDeletionPlan,
    ) -> KnowledgeDeletionPlan:
        admitted.append(root.ingestion_job_service.submit("ops", "guide.txt", b"concurrent source", uuid4()))
        return begin(repository, plan)

    monkeypatch.setattr(SQLiteKnowledgeDeletionRepository, "begin", _begin)
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete("/api/admin/knowledge-bases/ops", params={"force": True})
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "knowledge_base_ingestion_in_progress"
            assert root.knowledge_base_repository.get("ops").id == "ops"
            job = admitted[0]
            assert root.ingestion_job_service.get(job.job_id).status == "queued"
            assert (settings.docs_path / f"{job.doc_id}_guide.txt").read_bytes() == b"concurrent source"
    finally:
        root.close()


@pytest.mark.anyio
async def test_force_index_failure_keeps_source_and_restores_bm25_then_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """跨存储失败不能伪称成功或丢失BM25上一代，恢复后同请求必须能完成。

    Args:
        tmp_path: 真实临时Chroma／BM25和源文件。
        monkeypatch: 只注入向量存储删除故障。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    identifier = uuid4()
    root.vector_store.add_chunks([DocumentChunk(id=f"{identifier}:0", doc_id=str(identifier), kb_id="ops", filename="guide.txt", text="needle 实际正文", page=None, chunk_index=0)], [[1.0, 0.0]])
    source = settings.docs_path / f"{identifier}_guide.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    assert root.keyword_index.search("needle", 5, kb_ids=("ops",))

    def _fail(doc_id: str) -> None:
        raise VectorStoreError("内部故障路径")

    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with monkeypatch.context() as patch:
                patch.setattr(root.vector_store, "delete_document", _fail)
                response = await client.delete("/api/admin/knowledge-bases/ops", params={"force": True})
                assert response.status_code == 500
                assert response.json()["error"]["code"] == "knowledge_base_delete_partial"
                assert "内部故障路径" not in response.text
                assert source.read_bytes() == b"source"
                assert root.vector_store.list_chunks()
                assert root.keyword_index.search("needle", 5, kb_ids=("ops",))
            response = await client.delete("/api/admin/knowledge-bases/ops", params={"force": True})
            assert response.status_code == 200
            assert not source.exists()
            assert root.keyword_index.search("needle", 5, kb_ids=("ops",)) == []
    finally:
        root.close()


@pytest.mark.anyio
async def test_unsafe_source_fails_before_any_document_or_plan_changes(tmp_path: Path) -> None:
    """整库预检必须先完成，不能删除前一篇后才发现后一篇是链接。

    Args:
        tmp_path: 链接及目标均为临时测试文件，不接触生产。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    identifiers = sorted((uuid4(), uuid4()), key=str)
    chunks = []
    for identifier in identifiers:
        chunks.append(DocumentChunk(id=f"{identifier}:0", doc_id=str(identifier), kb_id="ops", filename="guide.txt", text="真实正文", page=None, chunk_index=0))
    root.vector_store.add_chunks(chunks, [[1.0, 0.0]] * 2)
    settings.docs_path.mkdir(parents=True, exist_ok=True)
    first = settings.docs_path / f"{identifiers[0]}_guide.txt"
    first.write_bytes(b"first source")
    target = tmp_path / "do-not-delete.txt"
    target.write_bytes(b"protected target")
    second = settings.docs_path / f"{identifiers[1]}_guide.txt"
    second.symlink_to(target)
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.delete("/api/admin/knowledge-bases/ops", params={"force": True})).status_code == 500
            assert first.read_bytes() == b"first source"
            assert target.read_bytes() == b"protected target"
            assert second.is_symlink()
            assert len(root.vector_store.list_chunks()) == 2
            assert root.knowledge_base_repository.get("ops").id == "ops"
    finally:
        root.close()


@pytest.mark.anyio
async def test_corrupt_job_is_not_treated_as_empty_kb(tmp_path: Path) -> None:
    """损坏状态不能被忽略成空库或终态，必须阻止所有清理。

    Args:
        tmp_path: 真实临时任务与源文件。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    job = root.ingestion_job_service.submit("ops", "guide.txt", b"source", uuid4())
    with sqlite3.connect(settings.knowledge_base_db_path) as connection:
        connection.execute("UPDATE ingestion_jobs SET status='succeeded'")
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete("/api/admin/knowledge-bases/ops", params={"force": True})
            assert response.status_code == 503
            assert (settings.docs_path / f"{job.doc_id}_guide.txt").read_bytes() == b"source"
            assert root.knowledge_base_repository.get("ops").id == "ops"
    finally:
        root.close()


@pytest.mark.anyio
async def test_bulk_partial_delete_keeps_original_two_document_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """第二篇失败时第一篇不虚假回滚，重试仍按原两篇清理并返回原计数。

    Args:
        tmp_path: 两篇真实文档及源文件。
        monkeypatch: 只对第二个实际向量删除注入故障。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    identifiers = sorted((uuid4(), uuid4()), key=str)
    chunks = []
    sources = []
    settings.docs_path.mkdir(parents=True, exist_ok=True)
    for identifier in identifiers:
        chunks.append(DocumentChunk(id=f"{identifier}:0", doc_id=str(identifier), kb_id="ops", filename="guide.txt", text="needle 真正文", page=None, chunk_index=0))
        source = settings.docs_path / f"{identifier}_guide.txt"
        source.write_bytes(b"source")
        sources.append(source)
    root.vector_store.add_chunks(chunks, [[1.0, 0.0]] * 2)
    delete = root.vector_store.delete_document

    def _delete(doc_id: str) -> None:
        if doc_id == str(identifiers[1]):
            raise VectorStoreError("second index failed")
        delete(doc_id)

    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with monkeypatch.context() as patch:
                patch.setattr(root.vector_store, "delete_document", _delete)
                response = await client.delete("/api/admin/knowledge-bases/ops", params={"force": True})
                assert response.status_code == 500
                assert not sources[0].exists()
                assert sources[1].exists()
                assert {chunk.doc_id for chunk in root.vector_store.list_chunks()} == {str(identifiers[1])}
            response = await client.delete("/api/admin/knowledge-bases/ops", params={"force": True})
            assert response.status_code == 200
            assert response.json()["documents_deleted"] == 2
            assert all(not source.exists() for source in sources)
            assert root.vector_store.list_chunks() == []
            assert root.keyword_index.search("needle", 5, kb_ids=("ops",)) == []
    finally:
        root.close()
