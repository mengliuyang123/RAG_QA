# -*- coding: utf-8 -*-
"""组合根任务服务的惰性装配、真实kb归属及运行时源文件隔离。"""

from dataclasses import replace
from pathlib import Path
from threading import Event
from uuid import uuid4

from app.bootstrap.container import RuntimeOptions
from app.bootstrap.container import build_application_container
from app.bootstrap.factories import AdapterFactories
from app.config import Settings
from app.domain.knowledge_bases import KnowledgeBase
from tests.test_job_ingestion_pipeline import BatchEmbedder


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None, docs_path=tmp_path / "production-docs",
        chroma_path=tmp_path / "production-chroma",
        knowledge_base_db_path=tmp_path / "production.sqlite3",
        conversation_db_path=tmp_path / "conversations.sqlite3",
        trace_db_path=tmp_path / "traces.sqlite3",
        auth_db_path=tmp_path / "auth.sqlite3",
        agent_checkpoint_db_path=tmp_path / "checkpoints.sqlite3",
        llm_api_key=None,
        embedding_api_key=None,
        openai_api_key=None,
        reranker_base_url=None, chunk_size=11, chunk_overlap=0,
        embedding_batch_size=2,
    )


def test_submit_and_read_job_do_not_initialize_models_or_indexes(tmp_path: Path) -> None:
    """前台入队不能因缺模型凭据失败，且不应在返回202前执行解析。

    Args:
        tmp_path: 隔离目录。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings)
    try:
        service = root.ingestion_job_service
        queued = service.submit("default", "guide.txt", b"payload", uuid4())
        assert queued.status == "queued"
        assert service is root.ingestion_job_service
        assert service.get(queued.job_id).status == "queued"
        assert not settings.chroma_path.exists()
        assert (settings.docs_path / f"{queued.doc_id}_guide.txt").is_file()
    finally:
        root.close()


def test_root_executes_job_in_requested_kb_and_keeps_legacy_file_management(tmp_path: Path) -> None:
    """只创建了纯模块但生产组合根未接线，或遗失kb_id，都会使此测试失败。

    Args:
        tmp_path: 全部受控目录。
    """
    settings = _settings(tmp_path)
    factories = replace(AdapterFactories(), embedding_provider=lambda *args: BatchEmbedder())
    root = build_application_container(settings, factories=factories)
    try:
        root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
        queued = root.ingestion_job_service.submit("ops", "guide.txt", b"abcdefghij\n" * 5, uuid4())
        assert root.ingestion_job_runner.run_next(Event()) is True
        result = root.ingestion_job_service.get(queued.job_id)
        assert result.status == "succeeded"
        assert result.chunks_count == 5
        chunks = root.vector_store.list_chunks()
        assert {chunk.kb_id for chunk in chunks} == {"ops"}
        assert len(root.retrieval_service.retrieve("abcdefghij", "hybrid", 5, kb_ids=("ops",))) == 5
        assert root.retrieval_service.retrieve("abcdefghij", "hybrid", 5, kb_ids=("default",)) == []
        summary = root.document_service.list_documents()
        assert len(summary) == 1
        assert summary[0].filename == "guide.txt"
        assert summary[0].file_status.value == "available"
    finally:
        root.close()


def test_runtime_job_sources_never_write_production_docs(tmp_path: Path) -> None:
    """评估只覆盖Chroma路径时也必须同时隔离任务源文件，不能落生产目录。

    Args:
        tmp_path: 生产哨兵和独立评估目录。
    """
    settings = _settings(tmp_path)
    first = build_application_container(settings, RuntimeOptions(
        chroma_path=tmp_path / "evaluation", collection_name="first",
    ))
    second = build_application_container(settings, RuntimeOptions(
        chroma_path=tmp_path / "evaluation", collection_name="second",
    ))
    try:
        queued = first.ingestion_job_service.submit("default", "guide.md", b"payload", uuid4())
        assert first.ingestion_job_service.get(queued.job_id).status == "queued"
        assert second.ingestion_job_repository.unpublished_document_ids() == ()
        assert not settings.docs_path.exists()
        assert not settings.knowledge_base_db_path.exists()
        assert not settings.chroma_path.exists()
        assert len(list((tmp_path / "evaluation").rglob(f"{queued.doc_id}_guide.md"))) == 1
    finally:
        first.close()
        second.close()
