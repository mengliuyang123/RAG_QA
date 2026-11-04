# -*- coding: utf-8 -*-
"""组合根必须惰性装配目录，且评估不能读写生产的知识库元数据。"""

from pathlib import Path
from uuid import uuid4

from app.bootstrap.container import RuntimeOptions
from app.bootstrap.container import build_application_container
from app.config import Settings
from app.domain.knowledge_bases import KnowledgeBase
from app.domain.suggestions import CuratedQuestionsUpdate
from app.domain.ingestion_jobs import IngestionJobSubmission


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None, knowledge_base_db_path=tmp_path / "production.sqlite3",
        llm_api_key=None,
        embedding_api_key=None,
        openai_api_key=None,
        reranker_base_url=None,
    )


def test_catalog_is_lazy_shared_without_models(tmp_path: Path) -> None:
    """目录使用不应构造生成模型或向量模型。

    Args:
        tmp_path: 隔离目录。
    """
    settings = _settings(tmp_path)
    container = build_application_container(settings)
    try:
        assert not settings.knowledge_base_db_path.exists()
        first = container.knowledge_base_repository
        assert first is container.knowledge_base_repository
        assert [item.id for item in first.list()] == ["default"]
        assert settings.knowledge_base_db_path.exists()
    finally:
        container.close()


def test_runtime_catalog_isolates_production_and_each_collection(
    tmp_path: Path,
) -> None:
    """已有评估RuntimeOptions只覆盖Chroma时，也不得隐式打开生产目录。

    Args:
        tmp_path: 隔离生产和评估目录。
    """
    settings = _settings(tmp_path)
    production = build_application_container(settings)
    first = build_application_container(
        settings, RuntimeOptions(
            chroma_path=tmp_path / "eval", collection_name="first",
        ),
    )
    second = build_application_container(
        settings, RuntimeOptions(
            chroma_path=tmp_path / "eval", collection_name="second",
        ),
    )
    try:
        production.knowledge_base_repository.create(
            KnowledgeBase(id="ops", name="生产"),
        )
        first.knowledge_base_repository.create(
            KnowledgeBase(id="dev", name="评估"),
        )
        production_ids = [
            kb.id for kb in production.knowledge_base_repository.list()
        ]
        first_ids = [kb.id for kb in first.knowledge_base_repository.list()]
        second_ids = [kb.id for kb in second.knowledge_base_repository.list()]
        assert production_ids == ["default", "ops"]
        assert first_ids == ["default", "dev"]
        assert second_ids == ["default"]
    finally:
        production.close()
        first.close()
        second.close()


def test_runtime_can_explicitly_select_catalog_database(tmp_path: Path) -> None:
    """显式隔离路径必须优先于自动派生路径。

    Args:
        tmp_path: 隔离目录。
    """
    settings = _settings(tmp_path)
    selected = tmp_path / "selected.sqlite3"
    container = build_application_container(
        settings, RuntimeOptions(knowledge_base_db_path=selected),
    )
    try:
        assert container.knowledge_base_repository.get("default").id == "default"
        assert selected.exists()
        assert not settings.knowledge_base_db_path.exists()
    finally:
        container.close()


def test_runtime_suggestions_and_deletions_do_not_open_production_catalog(tmp_path: Path) -> None:
    """新登记不能因为目录本身已隔离，却回落到生产建议或删除记录。

    Args:
        tmp_path: 生产哨兵和两个独立评估目录。
    """
    settings = _settings(tmp_path)
    first = build_application_container(settings, RuntimeOptions(
        chroma_path=tmp_path / "eval", collection_name="first",
    ))
    second = build_application_container(settings, RuntimeOptions(
        chroma_path=tmp_path / "eval", collection_name="second",
    ))
    identifier = uuid4()
    try:
        first.knowledge_read_registry.replace_curated(
            "default", CuratedQuestionsUpdate(questions=("评估预设问题",)),
        )
        first.knowledge_read_registry.record_deletion(identifier, "default")
        assert len(first.knowledge_read_registry.list_curated(None, 3)) == 1
        assert second.knowledge_read_registry.list_curated(None, 3) == ()
        assert second.knowledge_read_registry.was_deleted(identifier) is False
        assert not settings.knowledge_base_db_path.exists()
    finally:
        first.close()
        second.close()


def test_runtime_ingestion_jobs_are_lazy_and_collection_isolated(tmp_path: Path) -> None:
    """任务登记不能误写生产目录或触发没有凭据的模型工厂。

    Args:
        tmp_path: 生产哨兵和两个独立评估目录。
    """
    settings = _settings(tmp_path)
    first = build_application_container(settings, RuntimeOptions(
        chroma_path=tmp_path / "eval", collection_name="first",
    ))
    second = build_application_container(settings, RuntimeOptions(
        chroma_path=tmp_path / "eval", collection_name="second",
    ))
    try:
        assert not settings.knowledge_base_db_path.exists()
        submission = IngestionJobSubmission(
            job_id=f"job_{uuid4().hex}", doc_id=uuid4(), kb_id="default",
            filename="test.md", size_bytes=14, uploaded_by=uuid4(),
            content_sha256="b" * 64,
        )
        first.ingestion_job_repository.enqueue(submission, capacity=1)
        assert first.ingestion_job_repository is first.ingestion_job_repository
        assert first.ingestion_job_repository.unpublished_document_ids() == (submission.doc_id,)
        assert second.ingestion_job_repository.unpublished_document_ids() == ()
        assert not settings.knowledge_base_db_path.exists()
    finally:
        first.close()
        second.close()
