# -*- coding: utf-8 -*-
"""真实生命周期后台入库、202、进度查询及身份校验的离线契约。"""

from dataclasses import replace
from pathlib import Path
from threading import Event
import sqlite3

import anyio
import httpx
import pytest

from app.bootstrap.container import build_application_container
from app.bootstrap.factories import AdapterFactories
from app.config import Settings
from app.domain.knowledge_bases import KnowledgeBase
from app.domain.documents import DocumentChunk
from app.main import create_app
from tests.test_auth_service import PASSWORD
from tests.test_job_ingestion_pipeline import BatchEmbedder


def _app(tmp_path: Path, embedder: BatchEmbedder, *, enabled: bool = True):
    settings = Settings(
        _env_file=None, ingestion_worker_enabled=enabled,
        docs_path=tmp_path / "docs", chroma_path=tmp_path / "chroma",
        knowledge_base_db_path=tmp_path / "catalog.sqlite3",
        trace_db_path=tmp_path / "traces.sqlite3",
        auth_db_path=tmp_path / "auth.sqlite3",
        conversation_db_path=tmp_path / "conversations.sqlite3",
        agent_checkpoint_db_path=tmp_path / "checkpoints.sqlite3",
        http_allowed_hosts=("test",), chunk_size=11, chunk_overlap=0,
        embedding_batch_size=2, max_upload_size=1024,
        ingestion_queue_capacity=1, ingestion_poll_seconds=0.05,
    )
    root = build_application_container(settings, factories=replace(
        AdapterFactories(), embedding_provider=lambda *args: embedder,
    ))
    app = create_app(settings, container_factory=lambda settings: root)
    root.auth_service.create_user("admin", PASSWORD, name="管理员", role="admin")
    root.auth_service.create_user("member", PASSWORD, name="成员", role="member")
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    return app, root


async def _login(client: httpx.AsyncClient, username: str = "admin") -> dict[str, str]:
    response = await client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200
    return {"X-CSRF-Token": client.cookies["rag_csrf"]}


@pytest.mark.anyio
async def test_upload_returns_202_while_worker_runs_and_health_remains_responsive(tmp_path: Path) -> None:
    """真正异步才会在向量服务仍阻塞时返回202，进度查询不能等待写锁。

    Args:
        tmp_path: 隔离完整应用。
    """
    entered = Event()
    release = Event()

    class BlockingEmbedder(BatchEmbedder):
        """阻塞首批，允许验证202和进度而非仅检查代码用了线程池。"""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """等测试释放后才完成首批。

            Args:
                texts: 实际分块。

            Returns:
                固定离线向量。
            """
            if not self.batches:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test release timeout")
            return super().embed_documents(texts)

    app, root = _app(tmp_path, BlockingEmbedder())
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
                headers = await _login(client)
                with anyio.fail_after(2):
                    response = await client.post(
                        "/api/admin/knowledge-bases/ops/documents", headers=headers,
                        files={"file": ("guide.txt", b"abcdefghij\n" * 5, "text/plain")},
                    )
                assert response.status_code == 202
                queued = response.json()
                assert queued["status"] == "queued"
                assert set(queued) == {"job_id", "doc_id", "status"}
                assert await anyio.to_thread.run_sync(entered.wait, 2)
                path = f"/api/admin/ingestion-jobs/{queued['job_id']}"
                with anyio.fail_after(2):
                    health = await client.get("/api/health")
                    progress = await client.get(path)
                assert health.status_code == 200
                assert progress.json()["status"] == "running"
                assert progress.json()["stage"] == "embedding"
                assert 0 < progress.json()["progress"] < 1
                assert progress.headers["Cache-Control"] == "no-store"
                rejected = await client.post(
                    "/api/admin/knowledge-bases/ops/documents", headers=headers,
                    files={"file": ("second.txt", b"second")},
                )
                assert rejected.status_code == 503
                release.set()
                with anyio.fail_after(3):
                    while True:
                        progress = await client.get(path)
                        if progress.json()["status"] in {"succeeded", "failed"}:
                            break
                        await anyio.sleep(0.02)
                assert progress.json()["status"] == "succeeded"
                assert progress.json()["chunks_count"] == 5
                assert {chunk.kb_id for chunk in root.vector_store.list_chunks()} == {"ops"}
    finally:
        release.set()
        root.close()


@pytest.mark.anyio
async def test_job_api_requires_admin_and_csrf_before_writing_source(tmp_path: Path) -> None:
    """上传和轮询不能绕过角色，匿名／无CSRF不得先保存文件。

    Args:
        tmp_path: 隔离完整应用。
    """
    app, root = _app(tmp_path, BatchEmbedder())
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
                path = "/api/admin/knowledge-bases/ops/documents"
                assert (await client.post(path, files={"file": ("a.txt", b"a")})).status_code == 401
                assert (await client.get("/api/admin/ingestion-jobs/job_missing")).status_code == 401
                headers = await _login(client, "member")
                assert (await client.post(path, files={"file": ("a.txt", b"a")}, headers=headers)).status_code == 403
                assert (await client.get("/api/admin/ingestion-jobs/job_missing")).status_code == 403
                await _login(client)
                assert (await client.post(path, files={"file": ("a.txt", b"a")})).status_code == 403
                assert not (tmp_path / "docs").exists()
    finally:
        root.close()


@pytest.mark.anyio
async def test_startup_processes_persisted_queue_without_new_upload(tmp_path: Path) -> None:
    """若只在上传请求后唤起worker，重启后已有queued任务会永久停住。

    Args:
        tmp_path: 可重开的同一持久目录。
    """
    app, root = _app(tmp_path, BatchEmbedder())
    queued = root.ingestion_job_service.submit("ops", "queued.txt", b"payload", root.auth_service.create_user(
        "otheradmin", PASSWORD, name="另一管理员", role="admin",
    ).id)
    root.close()
    replacement = build_application_container(root.settings, factories=replace(
        AdapterFactories(), embedding_provider=lambda *args: BatchEmbedder(),
    ))
    restarted = create_app(root.settings, container_factory=lambda settings: replacement)
    try:
        async with restarted.router.lifespan_context(restarted):
            with anyio.fail_after(3):
                while replacement.ingestion_job_service.get(queued.job_id).status != "succeeded":
                    await anyio.sleep(0.02)
            assert replacement.vector_store.count() == 1
    finally:
        replacement.close()


@pytest.mark.anyio
async def test_failed_progress_and_rejected_inputs_have_safe_http_responses(tmp_path: Path) -> None:
    """异步失败从轮询读取；非法输入不能先写文件或暴露上游异常。

    Args:
        tmp_path: 隔离完整应用。
    """
    app, root = _app(tmp_path, BatchEmbedder(fail_on=1))
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
                headers = await _login(client)
                for path, filename, content, code in (
                    ("ops", "a.exe", b"a", 415),
                    ("ops", "a.txt", b"", 422),
                    ("ops", "a.txt", b"a" * 1025, 413),
                    ("missing", "a.txt", b"a", 404),
                ):
                    response = await client.post(
                        f"/api/admin/knowledge-bases/{path}/documents", headers=headers,
                        files={"file": (filename, content)},
                    )
                    assert response.status_code == code
                assert not (tmp_path / "docs").exists()
                missing = await client.get("/api/admin/ingestion-jobs/job_missing")
                assert missing.status_code == 404
                assert missing.json()["error"]["code"] == "ingestion_job_not_found"
                response = await client.post(
                    "/api/admin/knowledge-bases/ops/documents", headers=headers,
                    files={"file": ("fails.txt", b"payload")},
                )
                assert response.status_code == 202
                path = f"/api/admin/ingestion-jobs/{response.json()['job_id']}"
                with anyio.fail_after(3):
                    while True:
                        progress = await client.get(path)
                        if progress.json()["status"] == "failed":
                            break
                        await anyio.sleep(0.02)
                assert progress.json()["error"]["code"] == "embedding_failed"
                assert "private upstream" not in progress.text
                assert "lease_token" not in progress.text
                assert "content_sha256" not in progress.text
    finally:
        root.close()


@pytest.mark.anyio
async def test_disabled_worker_refuses_202_and_legacy_upload_keeps_201(tmp_path: Path) -> None:
    """未启动任务系统不能假接收202；旧路径仍可同步完成入库。

    Args:
        tmp_path: 隔离完整应用。
    """
    app, root = _app(tmp_path, BatchEmbedder(), enabled=False)
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
                headers = await _login(client)
                response = await client.post(
                    "/api/admin/knowledge-bases/ops/documents", headers=headers,
                    files={"file": ("a.txt", b"payload")},
                )
                assert response.status_code == 503
                assert response.json()["error"]["code"] == "ingestion_unavailable"
                assert not (tmp_path / "docs").exists()
                legacy = await client.post(
                    "/api/upload", headers=headers, files={"file": ("legacy.txt", b"payload")},
                )
                assert legacy.status_code == 201
                assert set(legacy.json()) == {"doc_id", "filename", "chunks_count"}
                assert legacy.json()["chunks_count"] == 1
    finally:
        root.close()


@pytest.mark.anyio
async def test_startup_recovers_expired_running_job_and_replaces_partial_text(tmp_path: Path) -> None:
    """重启不仅消费queued，还需接管过期running并清理相同ID的旧正文。

    Args:
        tmp_path: 重启前后的同一隔离目录。
    """
    app, root = _app(tmp_path, BatchEmbedder())
    admin = root.auth_service.create_user("uploader", PASSWORD, name="上传者", role="admin")
    queued = root.ingestion_job_service.submit("ops", "queued.txt", b"new text", admin.id)
    claim = root.ingestion_job_repository.claim_next(lease_seconds=60)
    assert claim is not None
    root.vector_store.add_chunks([DocumentChunk(
        id=f"{queued.doc_id}:0", doc_id=str(queued.doc_id), kb_id="ops",
        filename="queued.txt", text="OLD PRIVATE PARTIAL", chunk_index=0, page=None,
    )], [[1.0, 0.0]])
    root.close()
    with sqlite3.connect(tmp_path / "catalog.sqlite3") as connection:
        connection.execute(
            "UPDATE ingestion_jobs SET lease_expires_at=? WHERE job_id=?",
            ("2000-01-01T00:00:00+00:00", queued.job_id),
        )
    replacement = build_application_container(root.settings, factories=replace(
        AdapterFactories(), embedding_provider=lambda *args: BatchEmbedder(),
    ))
    restarted = create_app(root.settings, container_factory=lambda settings: replacement)
    try:
        async with restarted.router.lifespan_context(restarted):
            with anyio.fail_after(3):
                while replacement.ingestion_job_service.get(queued.job_id).status != "succeeded":
                    await anyio.sleep(0.02)
            assert [chunk.text for chunk in replacement.vector_store.list_chunks()] == ["new text"]
    finally:
        replacement.close()


@pytest.mark.anyio
async def test_shutdown_joins_worker_before_closing_embedding_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型仍在运行时关闭连接会使任务异常，必须先停止下一批并等当前调用结束。

    Args:
        tmp_path: 隔离应用。
        monkeypatch: 观察真实close开始，不替代其行为。
    """
    entered = Event()
    release = Event()
    closing = Event()

    class ClosingEmbedder(BatchEmbedder):
        """具有真实关闭状态的离线模型连接替身。"""

        def __init__(self) -> None:
            """初始化未关闭连接。

            Returns:
                None。
            """
            super().__init__()
            self.closed = False

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """检查调用返回前连接一直保持打开。

            Args:
                texts: 实际分块。

            Returns:
                固定向量。
            """
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release model")
            assert self.closed is False
            return super().embed_documents(texts)

        def close(self) -> None:
            """标记连接已释放。

            Returns:
                None。
            """
            self.closed = True

    embedder = ClosingEmbedder()
    app, root = _app(tmp_path, embedder)
    jobs = root.ingestion_job_repository
    original = root.ingestion_worker.close

    def observe_close() -> None:
        closing.set()
        original()

    monkeypatch.setattr(root.ingestion_worker, "close", observe_close)

    async def shutdown() -> None:
        await anyio.to_thread.run_sync(root.close)

    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
                headers = await _login(client)
                response = await client.post(
                    "/api/admin/knowledge-bases/ops/documents", headers=headers,
                    files={"file": ("guide.txt", b"abcdefghij\n" * 5)},
                )
                assert response.status_code == 202
                queued = response.json()
                assert await anyio.to_thread.run_sync(entered.wait, 2)
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(shutdown)
                    assert await anyio.to_thread.run_sync(closing.wait, 2)
                    assert embedder.closed is False
                    release.set()
                assert embedder.closed is True
                assert len(embedder.batches) == 1
                assert jobs.get(queued["job_id"]).status == "running"
    finally:
        release.set()
        root.close()


@pytest.mark.anyio
async def test_enabled_idle_worker_keeps_health_and_storage_lazy(tmp_path: Path) -> None:
    """新生命周期不得让仅健康检查的空应用创建目录库或模型连接。

    Args:
        tmp_path: 所有尚未创建的受控目录。
    """
    settings = Settings(
        _env_file=None, ingestion_worker_enabled=True, ingestion_poll_seconds=0.05,
        docs_path=tmp_path / "docs", chroma_path=tmp_path / "chroma",
        knowledge_base_db_path=tmp_path / "catalog.sqlite3",
        trace_db_path=tmp_path / "traces.sqlite3", http_allowed_hosts=("test",),
        llm_api_key=None,
        embedding_api_key=None,
        openai_api_key=None,
    )
    root = build_application_container(settings)
    app = create_app(settings, container_factory=lambda settings: root)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
            assert (await client.get("/api/health")).status_code == 200
            await anyio.sleep(0.12)
            assert not settings.knowledge_base_db_path.exists()
            assert not settings.chroma_path.exists()
            assert not settings.docs_path.exists()
