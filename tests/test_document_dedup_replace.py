# -*- coding: utf-8 -*-
"""documents 登记：上传查重、同名替换不空档、按 doc_id 删除与批量删除。"""

from dataclasses import replace
from pathlib import Path
from threading import Event
from uuid import UUID
from uuid import uuid4

import httpx
import psycopg
import pytest

from app.application.errors import DocumentAlreadyExistsError
from app.application.errors import DocumentNameConflictError
from app.application.errors import DocumentReplaceConflictError
from app.application.retrieval.service import RetrievalService
from app.application.retrieval.service import VectorRetriever
from app.bootstrap.container import build_application_container
from app.bootstrap.factories import AdapterFactories
from app.domain.documents import DocumentChunk
from app.main import create_app
from tests.http_auth import allow_test_identity
from tests.test_ingestion_job_container import _settings
from tests.test_job_ingestion_pipeline import BatchEmbedder
from tests.test_retriever import FakeQueryEmbedder


ADMIN = UUID("99999999-9999-4999-8999-999999999999")


def _root(tmp_path: Path, embedder: BatchEmbedder | None = None):
    selected = embedder or BatchEmbedder()
    return build_application_container(_settings(tmp_path), factories=replace(
        AdapterFactories(), embedding_provider=lambda *args: selected,
    ))


def _sources(root) -> Path:
    return root._job_sources_path()


def _ingest(root, filename: str, content: bytes, **options):
    """提交并同步执行一个入库任务，返回任务。"""
    job = root.ingestion_job_service.submit("default", filename, content, ADMIN, **options)
    assert root.ingestion_job_runner.run_next(Event()) is True
    return root.ingestion_job_service.get(job.job_id), job


def _retrieval(root) -> RetrievalService:
    return RetrievalService(
        vector_retriever=VectorRetriever(root.vector_store, FakeQueryEmbedder()),
        publication=root.ingestion_job_repository,
    )


def _doc_ids(results) -> set[str]:
    return {result.doc_id for result in results}


def test_documents_table_enforces_live_unique_sha_per_kb(tmp_path: Path) -> None:
    """(kb_id, sha256) 只对未删除文档唯一；删除后可再次登记相同内容。"""
    root = _root(tmp_path)
    try:
        database = root.database
        root.knowledge_base_repository
        row = "INSERT INTO documents (doc_id,kb_id,filename,sha256,status,uploaded_at,deleted_at) " \
              "VALUES (%s,'default','a.md',%s,'ready',now(),%s)"
        digest = "b" * 64
        with database.connection() as connection:
            connection.execute(row, (str(uuid4()), digest, None))
        with pytest.raises(psycopg.errors.UniqueViolation):
            with database.connection() as connection:
                connection.execute(row, (str(uuid4()), digest, None))
        with database.connection() as connection:
            connection.execute("UPDATE documents SET deleted_at=now() WHERE sha256=%s", (digest,))
            connection.execute(row, (str(uuid4()), digest, None))
            assert connection.execute(
                "SELECT version FROM schema_migrations WHERE version=6",
            ).fetchone() is not None
    finally:
        root.close()


def test_identical_content_returns_existing_without_ingesting(tmp_path: Path) -> None:
    """同库同 sha256：返回已有文档，不再保存源文件、不再入队。"""
    root = _root(tmp_path)
    try:
        first, _ = _ingest(root, "guide.txt", b"abcdefghij\n" * 3)
        before = set(_sources(root).iterdir())
        with pytest.raises(DocumentAlreadyExistsError) as captured:
            root.ingestion_job_service.submit("default", "renamed.txt", b"abcdefghij\n" * 3, ADMIN)
        assert captured.value.existing.doc_id == first.doc_id
        assert captured.value.existing.status == "ready"
        assert set(_sources(root).iterdir()) == before
        assert root.ingestion_job_repository.claim_next(lease_seconds=30) is None
    finally:
        root.close()


def test_name_conflict_requires_a_choice_and_keep_both_adds_another(tmp_path: Path) -> None:
    """同名不同内容：未选择时拒绝且不留源文件；另存为一篇则两篇并存。"""
    root = _root(tmp_path)
    try:
        first, _ = _ingest(root, "guide.txt", b"version one\n")
        before = set(_sources(root).iterdir())
        with pytest.raises(DocumentNameConflictError) as captured:
            root.ingestion_job_service.submit("default", "guide.txt", b"version two\n", ADMIN)
        assert captured.value.existing.doc_id == first.doc_id
        assert set(_sources(root).iterdir()) == before

        second, _ = _ingest(root, "guide.txt", b"version two\n", conflict="keep_both")
        assert second.status == "succeeded"
        live = root.document_registry.list_live("default")
        assert {record.doc_id for record in live} == {first.doc_id, second.doc_id}
        assert {record.status for record in live} == {"ready"}
    finally:
        root.close()


def test_replace_never_leaves_a_retrieval_gap(tmp_path: Path) -> None:
    """替换：新版本入库期间检索只见旧版本，发布后只见新版本，任何时刻都不为空。"""
    observed: list[set[str]] = []
    holder: dict[str, object] = {}

    class ObservingEmbedder(BatchEmbedder):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            # 每批之前检索一次：此时新版本可能已有部分块写入但未发布。
            if "root" in holder:
                observed.append(_doc_ids(_retrieval(holder["root"]).retrieve("needle", top_k=20)))
            return super().embed_documents(texts)

    root = _root(tmp_path, ObservingEmbedder())
    try:
        old, _ = _ingest(root, "guide.txt", b"old needle\n" * 2)
        old_source = _sources(root) / f"{old.doc_id}_guide.txt"
        assert old_source.exists()
        holder["root"] = root
        new, _ = _ingest(
            root, "guide.txt", b"new needle\n" * 6, conflict="replace", replace_doc_id=old.doc_id,
        )
        assert new.status == "succeeded"
        assert len(observed) >= 2
        assert all(seen == {str(old.doc_id)} for seen in observed)
        after = _doc_ids(_retrieval(root).retrieve("needle", top_k=20))
        assert after == {str(new.doc_id)}
        assert root.vector_store.list_chunk_headers(old.doc_id) == ()
        assert root.document_registry.get(old.doc_id) is None
        assert root.document_registry.get(new.doc_id).status == "ready"
        assert not old_source.exists()
        assert root.knowledge_read_registry.was_deleted(old.doc_id) is True
    finally:
        root.close()


def test_publish_between_snapshot_and_search_is_retried(tmp_path: Path) -> None:
    """替换在检索快照之后、查询之前提交时，按新快照重查而不是返回空结果。"""
    root = _root(tmp_path)
    try:
        old, _ = _ingest(root, "guide.txt", b"old needle\n")
        job = root.ingestion_job_service.submit(
            "default", "guide.txt", b"new needle\n", ADMIN, conflict="replace",
        )
        claim = root.ingestion_job_repository.claim_next(lease_seconds=60)
        chunk = DocumentChunk(
            id=f"{job.doc_id}:0", doc_id=str(job.doc_id), filename="guide.txt",
            text="new needle", page=None, chunk_index=0, kb_id="default",
        )
        root.vector_store.add_chunks([chunk], [[1.0, 0.0]])
        root.ingestion_job_repository.advance(claim, stage="embedding", completed_batches=1, total_batches=1)
        root.ingestion_job_repository.advance(claim, stage="indexing")

        class PublishingEmbedder(FakeQueryEmbedder):
            def embed_query(self, text: str) -> list[float]:
                # 查询向量远程等待期间，替换版本提交并删除旧版本。
                if not self.queries:
                    root.ingestion_job_repository.succeed(claim, chunks_count=1)
                return super().embed_query(text)

        embedder = PublishingEmbedder()
        service = RetrievalService(
            vector_retriever=VectorRetriever(root.vector_store, embedder),
            publication=root.ingestion_job_repository,
        )
        results = service.retrieve("needle", top_k=5)
        assert _doc_ids(results) == {str(job.doc_id)}
        assert len(embedder.queries) == 2
        assert root.vector_store.list_chunk_headers(old.doc_id) == ()
    finally:
        root.close()


def test_failed_replacement_keeps_old_version_untouched(tmp_path: Path) -> None:
    """新版本入库失败时旧版本的分块、登记与源文件保持不变。"""
    root = _root(tmp_path, BatchEmbedder(fail_on=2))
    try:
        old, _ = _ingest(root, "guide.txt", b"old needle\n")
        old_source = _sources(root) / f"{old.doc_id}_guide.txt"
        before = root.vector_store.list_chunk_headers(old.doc_id)
        new, _ = _ingest(
            root, "guide.txt", b"new needle\n" * 6, conflict="replace", replace_doc_id=old.doc_id,
        )
        assert new.status == "failed"
        assert root.vector_store.list_chunk_headers(old.doc_id) == before
        assert root.vector_store.list_chunk_headers(new.doc_id) == ()
        assert root.document_registry.get(old.doc_id).status == "ready"
        assert root.document_registry.get(new.doc_id).status == "failed"
        assert old_source.exists()
        assert _doc_ids(_retrieval(root).retrieve("needle", top_k=20)) == {str(old.doc_id)}
    finally:
        root.close()


def test_replace_target_still_ingesting_is_rejected(tmp_path: Path) -> None:
    """替换目标仍在入库时拒绝，避免两个任务竞争同一旧版本。"""
    root = _root(tmp_path)
    try:
        pending = root.ingestion_job_service.submit("default", "guide.txt", b"first\n", ADMIN)
        with pytest.raises(DocumentReplaceConflictError):
            root.ingestion_job_service.submit(
                "default", "guide.txt", b"second\n", ADMIN, conflict="replace",
                replace_doc_id=pending.doc_id,
            )
    finally:
        root.close()


@pytest.mark.anyio
async def test_upload_api_reports_exists_conflict_and_accepts_choice(tmp_path: Path) -> None:
    """HTTP：相同内容 200 exists；同名不同内容 409 带已有文档；带选择则 202。"""
    root = _root(tmp_path)
    root.ingestion_worker.ensure_running = lambda: None
    root.ingestion_worker.wake = lambda: None
    app = allow_test_identity(create_app(settings=_settings(tmp_path), container_factory=lambda settings: root))
    path = "/api/admin/knowledge-bases/default/documents"
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = await client.post(path, files={"file": ("guide.txt", b"version one\n", "text/plain")})
            assert first.status_code == 202
            assert set(first.json()) == {"job_id", "doc_id", "status"}
            same = await client.post(path, files={"file": ("copy.txt", b"version one\n", "text/plain")})
            assert same.status_code == 200
            assert same.json()["status"] == "exists"
            assert same.json()["job_id"] is None
            assert same.json()["existing"]["doc_id"] == first.json()["doc_id"]
            assert same.json()["existing"]["filename"] == "guide.txt"

            conflict = await client.post(path, files={"file": ("guide.txt", b"version two\n", "text/plain")})
            assert conflict.status_code == 409
            error = conflict.json()["error"]
            assert error["code"] == "document_name_conflict"
            assert error["details"]["existing"]["doc_id"] == first.json()["doc_id"]
            assert str(tmp_path) not in conflict.text

            kept = await client.post(
                path, data={"on_conflict": "keep_both"},
                files={"file": ("guide.txt", b"version two\n", "text/plain")},
            )
            assert kept.status_code == 202
            busy = await client.post(
                path, data={"on_conflict": "replace", "replace_doc_id": first.json()["doc_id"]},
                files={"file": ("guide.txt", b"version three\n", "text/plain")},
            )
            assert busy.status_code == 409
            assert busy.json()["error"]["code"] == "document_replace_conflict"
            invalid = await client.post(
                path, data={"on_conflict": "overwrite"},
                files={"file": ("guide.txt", b"version four\n", "text/plain")},
            )
            assert invalid.status_code == 422
    finally:
        root.close()


@pytest.mark.anyio
async def test_failed_document_without_chunks_can_be_listed_and_deleted(tmp_path: Path) -> None:
    """入库失败（没有分块）的文档出现在管理列表中，并能按 doc_id 删除源文件与登记。"""
    root = _root(tmp_path, BatchEmbedder(fail_on=1))
    app = allow_test_identity(create_app(settings=_settings(tmp_path), container_factory=lambda settings: root))
    try:
        failed, _ = _ingest(root, "broken.txt", b"some text\n")
        assert failed.status == "failed"
        source = _sources(root) / f"{failed.doc_id}_broken.txt"
        assert source.exists()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            listed = (await client.get("/api/documents", params={"kb_id": "default"})).json()["documents"]
            entry = next(item for item in listed if item["doc_id"] == str(failed.doc_id))
            assert entry["chunks_count"] == 0
            assert entry["ingestion_status"] == "failed"
            response = await client.delete(f"/api/documents/{failed.doc_id}")
            assert response.status_code == 200
            assert response.json() == {"doc_id": str(failed.doc_id), "deleted": True}
            again = await client.delete(f"/api/documents/{failed.doc_id}")
            assert again.status_code == 404
        assert not source.exists()
        assert root.document_registry.get(failed.doc_id) is None
        # 删除后可以重新上传同样的内容。
        retry = root.ingestion_job_service.submit("default", "broken.txt", b"some text\n", ADMIN)
        assert retry.doc_id != failed.doc_id
    finally:
        root.close()


@pytest.mark.anyio
async def test_batch_delete_removes_chunks_records_and_sources_by_doc_id(tmp_path: Path) -> None:
    """批量删除：登记文档、旧版未登记文档与失败文档一起删，未知 id 单独报告。"""
    root = _root(tmp_path, BatchEmbedder())
    app = allow_test_identity(create_app(settings=_settings(tmp_path), container_factory=lambda settings: root))
    try:
        ready, _ = _ingest(root, "ready.txt", b"needle ready\n")
        legacy = uuid4()
        root.vector_store.add_chunks([DocumentChunk(
            id=f"{legacy}:0", doc_id=str(legacy), filename="legacy.txt",
            text="needle legacy", page=None, chunk_index=0,
        )], [[1.0, 0.0]])
        _sources(root).mkdir(parents=True, exist_ok=True)
        legacy_source = _sources(root) / f"{legacy}_legacy.txt"
        legacy_source.write_bytes(b"legacy")
        keep, _ = _ingest(root, "keep.txt", b"needle keep\n")
        unknown = uuid4()

        statements: list[str] = []
        from tests.postgres_support import observe_queries
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with observe_queries(root.database, statements.append):
                response = await client.post("/api/documents/batch-delete", json={
                    "doc_ids": [str(ready.doc_id), str(legacy), str(unknown), str(ready.doc_id)],
                })
            assert response.status_code == 200
            body = response.json()
            assert sorted(body["deleted"]) == sorted([str(ready.doc_id), str(legacy)])
            assert body["not_found"] == [str(unknown)]
            empty = await client.post("/api/documents/batch-delete", json={"doc_ids": []})
            assert empty.status_code == 422
        # 不再读取整个向量库：没有不带 doc_id 条件的全库分块扫描。
        scans = [s for s in statements if "FROM vector_chunks" in s and "doc_id" not in s]
        assert scans == []
        assert any("DELETE FROM vector_chunks WHERE doc_id = ANY" in s for s in statements)
        assert root.vector_store.list_chunk_headers(ready.doc_id) == ()
        assert root.vector_store.list_chunk_headers(legacy) == ()
        assert root.vector_store.list_chunk_headers(keep.doc_id) != ()
        assert not (_sources(root) / f"{ready.doc_id}_ready.txt").exists()
        assert not legacy_source.exists()
        assert (_sources(root) / f"{keep.doc_id}_keep.txt").exists()
        assert root.document_registry.get(ready.doc_id) is None
        assert root.document_registry.get(keep.doc_id) is not None
    finally:
        root.close()


@pytest.mark.anyio
async def test_batch_delete_with_an_ingesting_document_deletes_nothing(tmp_path: Path) -> None:
    """批量里有文档仍在入库时整批 409，其余文档的分块、登记和源文件都不动；超过 100 个 422。"""
    root = _root(tmp_path, BatchEmbedder())
    app = allow_test_identity(create_app(settings=_settings(tmp_path), container_factory=lambda settings: root))
    try:
        ready, _ = _ingest(root, "ready.txt", b"needle ready\n")
        pending = root.ingestion_job_service.submit("default", "pending.txt", b"still queued\n", ADMIN)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            busy = await client.post("/api/documents/batch-delete", json={
                "doc_ids": [str(ready.doc_id), str(pending.doc_id)],
            })
            assert busy.status_code == 409
            assert busy.json()["error"]["code"] == "document_ingestion_in_progress"
            too_many = await client.post("/api/documents/batch-delete", json={
                "doc_ids": [str(uuid4()) for _ in range(101)],
            })
            assert too_many.status_code == 422
        assert root.vector_store.list_chunk_headers(ready.doc_id) != ()
        assert root.document_registry.get(ready.doc_id) is not None
        assert (_sources(root) / f"{ready.doc_id}_ready.txt").exists()
    finally:
        root.close()
