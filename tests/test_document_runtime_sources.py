# -*- coding: utf-8 -*-
"""隔离索引的上传、任务与维护必须使用同一源目录。"""

from dataclasses import replace
from pathlib import Path
from threading import Event
from uuid import uuid4

from app.bootstrap.container import RuntimeOptions
from app.bootstrap.container import build_application_container
from app.bootstrap.factories import AdapterFactories
from tests.test_ingestion_job_container import _settings
from tests.test_job_ingestion_pipeline import BatchEmbedder


def test_runtime_delete_removes_own_source_and_preserves_production_sentinel(tmp_path: Path) -> None:
    """删除隔离索引时不得删除同名生产源文件或遗留自己的真实源文件。

    Args:
        tmp_path: 两套受控文件均在临时目录。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    factories = replace(AdapterFactories(), embedding_provider=lambda *args: BatchEmbedder())
    root = build_application_container(settings, RuntimeOptions(
        chroma_path=tmp_path / "isolated", collection_name="runtime-docs",
    ), factories=factories)
    try:
        job = root.ingestion_job_service.submit("default", "guide.txt", b"abcdefghij\n" * 3, uuid4())
        assert root.ingestion_job_runner.run_next(Event()) is True
        settings.docs_path.mkdir(parents=True)
        sentinel = settings.docs_path / f"{job.doc_id}_guide.txt"
        sentinel.write_bytes(b"production sentinel")
        sources = list((tmp_path / "isolated").rglob(f"{job.doc_id}_guide.txt"))
        assert len(sources) == 1
        root.document_service.delete_document(job.doc_id)
        assert sentinel.read_bytes() == b"production sentinel"
        assert not sources[0].exists()
        assert root.vector_store.list_chunks() == []
    finally:
        root.close()


def test_runtime_legacy_upload_uses_job_source_directory(tmp_path: Path) -> None:
    """旧201入口也必须跟随runtime，不能把新任务隔离而仍上传到生产目录。

    Args:
        tmp_path: 隔离目录。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    factories = replace(AdapterFactories(), embedding_provider=lambda *args: BatchEmbedder())
    root = build_application_container(settings, RuntimeOptions(
        chroma_path=tmp_path / "isolated", collection_name="legacy-docs",
    ), factories=factories)
    try:
        result = root.upload_service.save_and_parse("guide.txt", b"abcdefghij\n" * 3)
        assert not settings.docs_path.exists()
        assert len(list((tmp_path / "isolated").rglob(f"{result.doc_id}_guide.txt"))) == 1
        assert root.document_service.list_documents()[0].file_status.value == "available"
    finally:
        root.close()
