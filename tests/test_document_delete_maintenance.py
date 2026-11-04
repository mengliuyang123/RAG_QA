# -*- coding: utf-8 -*-
"""删除只依赖本地索引，且不能与仍能继续写入的持久任务竞争。"""

from dataclasses import replace
from pathlib import Path
from threading import Event
from threading import Thread
from uuid import uuid4

import httpx
import pytest
import anyio

from app.bootstrap.container import build_application_container
from app.bootstrap.factories import AdapterFactories
from app.domain.documents import DocumentChunk
from app.application.errors import VectorStoreError
from app.main import create_app
from tests.http_auth import allow_test_identity
from tests.test_ingestion_job_container import _settings
from tests.test_job_ingestion_pipeline import BatchEmbedder


@pytest.mark.anyio
async def test_local_delete_never_constructs_embedding_client(tmp_path: Path) -> None:
    """没有模型Key时仍应删除源文件／Chroma／预建BM25，不偷偷访问模型。

    Args:
        tmp_path: 真实临时索引与源文件目录。

    Returns:
        None。
    """
    def forbid_embedding(*args: object, **kwargs: object) -> object:
        raise AssertionError("删除不得装配Embedding客户端")

    settings = _settings(tmp_path)
    root = build_application_container(settings, factories=replace(
        AdapterFactories(), embedding_provider=forbid_embedding,
    ))
    doc_id = uuid4()
    root.vector_store.add_chunks([DocumentChunk(
        id=f"{doc_id}:0", doc_id=str(doc_id), filename="guide.txt",
        text="needle 配置步骤", page=None, chunk_index=0,
    )], [[1.0, 0.0]])
    settings.docs_path.mkdir(parents=True)
    source = settings.docs_path / f"{doc_id}_guide.txt"
    source.write_bytes(b"source")
    assert root.keyword_index.search("needle", 5)
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.delete(f"/api/documents/{doc_id}")
            assert response.status_code == 200
            assert response.json() == {"doc_id": str(doc_id), "deleted": True}
            assert root.vector_store.list_chunks() == []
            assert root.keyword_index.search("needle", 5) == []
            assert not source.exists()
    finally:
        root.close()


@pytest.mark.anyio
async def test_running_worker_rejects_delete_before_waiting_for_mutation_lock(tmp_path: Path) -> None:
    """执行中任务已持全局变更锁时，DELETE仍应立即409且不能等完后偷偷删除。

    Args:
        tmp_path: 真实临时队列、源文件与索引。

    Returns:
        None。
    """
    entered = Event()
    release = Event()

    class BlockingEmbedder(BatchEmbedder):
        """仅阻塞真实向量化端口，保留任务、锁、进度与索引实现。"""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """暂停实际向量批次，便于验证锁外任务预检。

            Args:
                texts: 真实分块。

            Returns:
                恢复执行后的离线向量。
            """
            entered.set()
            assert release.wait(3)
            return super().embed_documents(texts)

    settings = _settings(tmp_path)
    root = build_application_container(settings, factories=replace(
        AdapterFactories(), embedding_provider=lambda *args: BlockingEmbedder(),
    ))
    job = root.ingestion_job_service.submit("default", "guide.txt", b"abcdefghij\n" * 3, uuid4())
    worker = Thread(target=root.ingestion_job_runner.run_next, args=(Event(),))
    worker.start()
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 2)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            with anyio.fail_after(0.8):
                response = await client.delete(f"/api/documents/{job.doc_id}")
            assert response.status_code == 409
        assert root.ingestion_job_service.get(job.job_id).status == "running"
        assert (settings.docs_path / f"{job.doc_id}_guide.txt").is_file()
        assert root.knowledge_read_registry.was_deleted(job.doc_id) is False
    finally:
        release.set()
        await anyio.to_thread.run_sync(worker.join, 3)
        assert not worker.is_alive()
        root.close()


@pytest.mark.anyio
async def test_failed_partial_job_can_be_deleted_without_reviving_task(tmp_path: Path) -> None:
    """终态failed不会自动重试，可清理其部分索引，不能把失败改写成成功／取消。

    Args:
        tmp_path: 真实临时队列、源文件与索引。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    job = root.ingestion_job_service.submit("default", "guide.txt", b"payload", uuid4())
    claim = root.ingestion_job_repository.claim_next(lease_seconds=60)
    root.ingestion_job_repository.fail(claim, code="embedding_failed")
    root.vector_store.add_chunks([DocumentChunk(
        id=f"{job.doc_id}:0", doc_id=str(job.doc_id), filename="guide.txt",
        text="needle 部分索引", page=None, chunk_index=0,
    )], [[1.0, 0.0]])
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.delete(f"/api/documents/{job.doc_id}")
            assert response.status_code == 200
        assert root.vector_store.list_chunks() == []
        assert not (settings.docs_path / f"{job.doc_id}_guide.txt").exists()
        assert root.ingestion_job_service.get(job.job_id).status == "failed"
        assert root.ingestion_job_repository.claim_next(lease_seconds=60) is None
    finally:
        root.close()


@pytest.mark.anyio
async def test_vector_delete_failure_restores_bm25_and_keeps_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本地删除失败不能让BM25仍处于过滤后状态，也不能提前删除源文件。

    Args:
        tmp_path: 真实临时源文件／索引。
        monkeypatch: 仅替代向量存储的失败提交，不替代BM25或业务流程。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    doc_id = uuid4()
    chunk = DocumentChunk(
        id=f"{doc_id}:0", doc_id=str(doc_id), filename="guide.txt",
        text="needle 配置步骤", page=None, chunk_index=0,
    )
    root.vector_store.add_chunks([chunk], [[1.0, 0.0]])
    settings.docs_path.mkdir(parents=True)
    source = settings.docs_path / f"{doc_id}_guide.txt"
    source.write_bytes(b"source")
    assert root.keyword_index.search("needle", 5)[0].chunk_id == chunk.id

    def fail(identifier: str) -> None:
        raise VectorStoreError("private vector backend detail")

    monkeypatch.setattr(root.vector_store, "delete_document", fail)
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.delete(f"/api/documents/{doc_id}")
            assert response.status_code == 502
            assert "private" not in response.text
        assert source.read_bytes() == b"source"
        assert root.vector_store.list_chunks() == [chunk]
        assert root.keyword_index.search("needle", 5)[0].chunk_id == chunk.id
        assert root.knowledge_browse_service.get_context(doc_id, chunk.id).chunks[0].text == chunk.text
    finally:
        root.close()


@pytest.mark.anyio
@pytest.mark.parametrize("running", [False, True])
@pytest.mark.parametrize("indexed", [False, True])
async def test_pending_job_blocks_delete_without_mutating_any_resource(
    tmp_path: Path, running: bool, indexed: bool,
) -> None:
    """queued／running包括尚无块的任务也必须409，不能404或误删后重建。

    Args:
        tmp_path: 真实临时任务／索引／源文件目录。
        running: 是否已认领执行。
        indexed: 是否已有崩溃／执行中部分索引。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    job = root.ingestion_job_service.submit("default", "guide.txt", b"payload", uuid4())
    if running:
        root.ingestion_job_repository.claim_next(lease_seconds=60)
    if indexed:
        root.vector_store.add_chunks([DocumentChunk(
            id=f"{job.doc_id}:0", doc_id=str(job.doc_id), filename="guide.txt",
            text="needle 部分索引", page=None, chunk_index=0,
        )], [[1.0, 0.0]])
    before = root.vector_store.list_chunks()
    source = settings.docs_path / f"{job.doc_id}_guide.txt"
    app = allow_test_identity(create_app(settings=settings, container_factory=lambda settings: root))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.delete(f"/api/documents/{job.doc_id}")
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "document_ingestion_in_progress"
            assert source.read_bytes() == b"payload"
            assert root.vector_store.list_chunks() == before
            assert root.ingestion_job_service.get(job.job_id).status == ("running" if running else "queued")
            assert root.knowledge_read_registry.was_deleted(job.doc_id) is False
    finally:
        root.close()
