# -*- coding: utf-8 -*-
"""按新分块规则原地重建已发布文档（document-chunking：按新规则重新入库现有文档）。"""

from dataclasses import replace
from pathlib import Path
from threading import Event
from uuid import UUID

import pytest

from app.application.errors import EmbeddingError
from app.bootstrap.container import build_application_container
from app.bootstrap.factories import AdapterFactories
from app.interfaces.cli import reindex as cli
from tests.test_ingestion_job_container import _settings
from tests.test_job_ingestion_pipeline import BatchEmbedder


ADMIN = UUID("99999999-9999-4999-8999-999999999999")
MARKDOWN = "# 报销\n## 差旅\n" + "住宿每晚五百元。" * 3 + "\n## 招待\n招待另行审批。\n"


class SwitchableEmbedder(BatchEmbedder):
    """可在重建时切换为失败，模拟向量服务不可用。"""

    def __init__(self) -> None:
        super().__init__()
        self.broken = False

    def embed_documents(self, texts):
        if self.broken:
            raise EmbeddingError("offline")
        return super().embed_documents(texts)


@pytest.fixture
def root(tmp_path: Path):
    embedder = SwitchableEmbedder()
    settings = _settings(tmp_path).model_copy(update={"chunk_size": 20, "parent_chunk_size": 60})
    container = build_application_container(settings, factories=replace(
        AdapterFactories(), embedding_provider=lambda *args: embedder,
    ))
    container.test_embedder = embedder
    try:
        yield container
    finally:
        container.close()


def _ingest(root, filename: str = "制度.md", content: str = MARKDOWN):
    job = root.ingestion_job_service.submit("default", filename, content.encode("utf-8"), ADMIN)
    assert root.ingestion_job_runner.run_next(Event()) is True
    done = root.ingestion_job_service.get(job.job_id)
    assert done.status == "succeeded"
    return str(done.doc_id)


def _make_legacy(root, doc_id: str) -> None:
    """把一篇文档的块改成本次改动前的格式：没有父块序号和标题路径。"""
    with root.database.connection() as connection:
        connection.execute(
            "UPDATE vector_chunks SET metadata = metadata - 'parent_index' - 'heading_path' "
            "WHERE doc_id=%s", (doc_id,),
        )


def _chunks(root, doc_id: str):
    return [c for c in root.vector_store.list_chunks() if c.doc_id == doc_id]


def test_plan_lists_only_legacy_documents_by_default(root) -> None:
    legacy, current = _ingest(root, "a.md"), _ingest(root, "b.md", MARKDOWN + "补充。")
    _make_legacy(root, legacy)

    service = root.document_reindex_service

    assert [str(c.doc_id) for c in service.plan().candidates] == [legacy]
    assert {str(c.doc_id) for c in service.plan(include_current=True).candidates} == {legacy, current}
    assert service.plan(("other",)).candidates == ()
    assert len(service.plan(include_current=True, limit=1).candidates) == 1


def test_rebuild_replaces_chunks_in_place_with_new_format(root) -> None:
    doc_id = _ingest(root)
    _make_legacy(root, doc_id)
    assert all(c.parent_index is None for c in _chunks(root, doc_id))
    service = root.document_reindex_service

    [candidate] = service.plan().candidates
    result = service.rebuild(candidate)

    assert result.outcome == "rebuilt"
    chunks = _chunks(root, doc_id)
    assert len(chunks) == result.chunks > 0
    assert all(c.parent_index is not None for c in chunks)
    assert ("报销", "差旅") in {c.heading_path for c in chunks}
    assert {c.kb_id for c in chunks} == {"default"}
    assert service.plan().candidates == ()
    hits = root.retrieval_service.retrieve("住宿", "hybrid", 5, kb_ids=("default",))
    assert hits and {h.doc_id for h in hits} == {doc_id}


def test_missing_source_is_skipped_and_old_chunks_stay(root) -> None:
    doc_id = _ingest(root)
    _make_legacy(root, doc_id)
    before = _chunks(root, doc_id)
    for path in root._job_sources_path().iterdir():
        if path.is_file():
            path.unlink()

    [candidate] = root.document_reindex_service.plan().candidates
    result = root.document_reindex_service.rebuild(candidate)

    assert result.outcome == "skipped"
    assert "源文件" in result.reason
    assert _chunks(root, doc_id) == before


def test_embedding_failure_keeps_old_chunks(root) -> None:
    doc_id = _ingest(root)
    _make_legacy(root, doc_id)
    before = _chunks(root, doc_id)
    root.test_embedder.broken = True

    [candidate] = root.document_reindex_service.plan().candidates
    result = root.document_reindex_service.rebuild(candidate)

    assert result.outcome == "failed"
    assert _chunks(root, doc_id) == before


def test_document_deleted_during_rebuild_is_not_resurrected(root) -> None:
    doc_id = _ingest(root)
    _make_legacy(root, doc_id)
    [candidate] = root.document_reindex_service.plan().candidates
    with root.database.connection() as connection:
        connection.execute("UPDATE documents SET deleted_at=now() WHERE doc_id=%s", (doc_id,))
        connection.execute("DELETE FROM vector_chunks WHERE doc_id=%s", (doc_id,))

    result = root.document_reindex_service.rebuild(candidate)

    assert result.outcome == "skipped"
    assert _chunks(root, doc_id) == []


def test_cli_dry_run_calls_no_model_and_writes_nothing(root, monkeypatch, capsys) -> None:
    doc_id = _ingest(root)
    _make_legacy(root, doc_id)
    before = _chunks(root, doc_id)
    calls = len(root.test_embedder.batches)
    monkeypatch.setattr(cli, "build_application_container", lambda settings: root)
    monkeypatch.setattr(root, "close", lambda: None)

    assert cli.main(["--dry-run"], settings=object()) == 0

    output = capsys.readouterr().out
    assert "待重建 1 篇" in output and "制度.md" in output and "试运行" in output
    assert len(root.test_embedder.batches) == calls
    assert _chunks(root, doc_id) == before


def test_cli_rebuilds_and_reports(root, monkeypatch, capsys) -> None:
    doc_id = _ingest(root)
    _make_legacy(root, doc_id)
    monkeypatch.setattr(cli, "build_application_container", lambda settings: root)
    monkeypatch.setattr(root, "close", lambda: None)

    assert cli.main(["--kb", "default"], settings=object()) == 0

    output = capsys.readouterr().out
    assert "完成 [default] 制度.md" in output
    assert "完成 1 篇，跳过 0 篇，失败 0 篇" in output
    assert all(c.parent_index is not None for c in _chunks(root, doc_id))


def test_cli_rejects_non_positive_limit(capsys) -> None:
    assert cli.main(["--limit", "0"], settings=object()) == 2
