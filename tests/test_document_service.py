# -*- coding: utf-8 -*-
"""文档管理服务的聚合、路径约束和删除一致性测试。"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from threading import RLock
from uuid import UUID

import pytest

from app.application.documents.service import DocumentService
from app.application.errors import (
    DocumentDeletePartialError,
    DocumentNotFoundError,
    DocumentStorageCorruptionError,
)
from app.domain.documents import DocumentChunk
from app.infrastructure.persistence.repair_log import RepairLog


DOCUMENT_ID = "11111111-1111-4111-8111-111111111111"
SECOND_DOCUMENT_ID = "22222222-2222-4222-8222-222222222222"


def _chunk(
    doc_id: str,
    filename: str,
    index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        id=f"{doc_id}:{index}",
        doc_id=doc_id,
        text=f"chunk-{index}",
        filename=filename,
        page=None,
        chunk_index=index,
    )


class FakeStore:
    """返回测试预置文本块的只读向量存储。"""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self.list_calls = 0

    def list_chunks(self) -> list[DocumentChunk]:
        """记录快照调用并返回文本块副本。"""
        self.list_calls += 1
        return list(self.chunks)


class FakePipeline:
    """记录文档删除并可注入索引删除失败。"""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.deleted: list[str] = []

    def delete_document(self, doc_id: str) -> None:
        """记录删除顺序，并按需抛出索引错误。"""
        self.deleted.append(doc_id)
        if self.error is not None:
            raise self.error


def _service(
    tmp_path: Path,
    chunks: list[DocumentChunk],
    pipeline: FakePipeline | None = None,
):
    selected_pipeline = pipeline or FakePipeline()
    repair_path = tmp_path / "repair" / "document_deletions.jsonl"
    service = DocumentService(
        docs_path=tmp_path / "docs",
        vector_store=FakeStore(chunks),
        ingestion_pipeline=selected_pipeline,
        repair_log=RepairLog(
            repair_path,
            now=lambda: datetime(
                2026,
                8,
                20,
                1,
                2,
                3,
                tzinfo=timezone.utc,
            ),
        ),
    )
    return service, selected_pipeline, repair_path


def test_concurrent_delete_does_not_report_false_file_cleanup_failure(
    tmp_path: Path,
) -> None:
    """共享锁必须覆盖快照检查，第二次删除应观察已提交的新状态。

    Args:
        tmp_path: pytest 临时目录。
    """
    lock = RLock()
    entered = Event()
    release = Event()
    second_started = Event()
    second_lookup = Event()

    class RecordingStore(FakeStore):
        """标记第二次语料快照读取的存储。"""

        def list_chunks(self) -> list[DocumentChunk]:
            """记录第二次快照读取是否越过共享锁。

            Returns:
                当前文本块快照。
            """
            if self.list_calls > 0:
                second_lookup.set()
            return super().list_chunks()

    store = RecordingStore([_chunk(DOCUMENT_ID, "notes.md")])

    class HoldingPipeline(FakePipeline):
        """让第一次删除暂停，以触发并发请求的流水线。"""

        def delete_document(self, doc_id: str) -> None:
            """在第一个删除中暂停，以重现第二个请求读取旧状态。

            Args:
                doc_id: 要删除的文档标识。

            Returns:
                None。
            """
            with lock:
                if not entered.is_set():
                    entered.set()
                    assert release.wait(3)
                super().delete_document(doc_id)
                store.chunks.clear()

    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / f"{DOCUMENT_ID}_notes.md"
    source.write_text("source", encoding="utf-8")
    repair_path = tmp_path / "repair.jsonl"
    service = DocumentService(
        docs_path=docs, vector_store=store,
        ingestion_pipeline=HoldingPipeline(), repair_log=RepairLog(repair_path),
        mutation_lock=lock,
    )

    def second_delete() -> object:
        second_started.set()
        return service.delete_document(UUID(DOCUMENT_ID))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.delete_document, UUID(DOCUMENT_ID))
        assert entered.wait(2)
        second = pool.submit(second_delete)
        try:
            assert second_started.wait(2)
            assert not second_lookup.wait(0.1)
        finally:
            release.set()
        assert first.result(timeout=2).deleted is True
        with pytest.raises(DocumentNotFoundError):
            second.result(timeout=2)
    assert second_lookup.is_set()
    assert not source.exists()
    assert not repair_path.exists()


def test_list_documents_groups_sorts_and_reports_file_state(
    tmp_path: Path,
) -> None:
    """列表应按 doc_id 聚合并把可用文件 mtime 转成 UTC。"""
    chunks = [
        _chunk(DOCUMENT_ID, "notes.md", 0),
        _chunk(DOCUMENT_ID, "notes.md", 1),
        _chunk(SECOND_DOCUMENT_ID, "missing.txt", 0),
    ]
    service, _, _ = _service(tmp_path, chunks)
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    source_path = docs_path / f"{DOCUMENT_ID}_notes.md"
    source_path.write_text("source", encoding="utf-8")
    timestamp = datetime(
        2026,
        8,
        20,
        1,
        2,
        3,
        tzinfo=timezone.utc,
    ).timestamp()
    os.utime(source_path, (timestamp, timestamp))

    documents = service.list_documents()

    assert [str(document.doc_id) for document in documents] == [
        DOCUMENT_ID,
        SECOND_DOCUMENT_ID,
    ]
    assert documents[0].model_dump(mode="json") == {
        "doc_id": DOCUMENT_ID,
        "filename": "notes.md",
        "chunks_count": 2,
        "created_at": "2026-08-20T01:02:03Z",
        "file_status": "available",
        "kb_id": None,
        "file_type": None,
        "size_bytes": None,
        "uploaded_by": None,
        "ingestion_status": None,
        "published": None,
        "citations_7d": None,
        "unattributed_citations_7d": None,
    }
    assert documents[1].created_at is None
    assert documents[1].file_status.value == "missing"


def test_list_documents_accepts_missing_docs_directory(
    tmp_path: Path,
) -> None:
    """源文件目录尚未创建时，安全元数据应显示为 missing。"""
    service, _, _ = _service(
        tmp_path,
        [_chunk(DOCUMENT_ID, "notes.md")],
    )

    documents = service.list_documents()

    assert len(documents) == 1
    assert documents[0].created_at is None
    assert documents[0].file_status.value == "missing"


def test_list_documents_rejects_multiple_filenames_for_one_doc_id(
    tmp_path: Path,
) -> None:
    """同一 doc_id 出现多个 filename 应判定为存储损坏。"""
    service, _, _ = _service(
        tmp_path,
        [
            _chunk(DOCUMENT_ID, "first.md", 0),
            _chunk(DOCUMENT_ID, "second.md", 1),
        ],
    )

    with pytest.raises(DocumentStorageCorruptionError):
        service.list_documents()


@pytest.mark.parametrize(
    "filename",
    ["../escape.md", "/tmp/escape.md", r"C:\escape.md"],
)
def test_list_documents_rejects_malformed_saved_filename(
    tmp_path: Path,
    filename: str,
) -> None:
    """向量元数据中的目录或绝对路径不得用于文件访问。"""
    service, _, _ = _service(
        tmp_path,
        [_chunk(DOCUMENT_ID, filename)],
    )

    with pytest.raises(DocumentStorageCorruptionError):
        service.list_documents()


def test_list_documents_rejects_symlink_escape(tmp_path: Path) -> None:
    """保存路径是越界 symlink 时不得 stat 外部目标。"""
    service, _, _ = _service(
        tmp_path,
        [_chunk(DOCUMENT_ID, "notes.md")],
    )
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("do not touch", encoding="utf-8")
    saved_path = docs_path / f"{DOCUMENT_ID}_notes.md"
    saved_path.symlink_to(outside)

    with pytest.raises(DocumentStorageCorruptionError):
        service.list_documents()

    assert outside.read_text(encoding="utf-8") == "do not touch"


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_nonregular_source_is_rejected_before_index_deletion(tmp_path: Path, kind: str) -> None:
    """源文件必须是本身份的普通文件，不能resolve后误删另一个文档。

    Args:
        tmp_path: 全部受控临时路径。
        kind: 三类不安全源文件。

    Returns:
        None。
    """
    service, pipeline, _ = _service(tmp_path, [_chunk(DOCUMENT_ID, "notes.md")])
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "another.md"
    target.write_text("keep other document", encoding="utf-8")
    source = docs / f"{DOCUMENT_ID}_notes.md"
    if kind == "symlink":
        source.symlink_to(target)
    elif kind == "directory":
        source.mkdir()
    else:
        os.mkfifo(source)
    with pytest.raises(DocumentStorageCorruptionError):
        service.delete_document(UUID(DOCUMENT_ID))
    assert pipeline.deleted == []
    assert target.read_text(encoding="utf-8") == "keep other document"


def test_delete_document_removes_index_before_exact_source_file(
    tmp_path: Path,
) -> None:
    """成功删除应先调用索引流水线，再清理匹配的源文件。"""
    events: list[str] = []

    class OrderedPipeline(FakePipeline):
        def delete_document(self, doc_id: str) -> None:
            """记录索引提交时源文件仍然存在。"""
            source_path = (
                tmp_path / "docs" / f"{DOCUMENT_ID}_notes.md"
            )
            assert source_path.exists()
            events.append("index")
            super().delete_document(doc_id)

    pipeline = OrderedPipeline()
    service, _, repair_path = _service(
        tmp_path,
        [_chunk(DOCUMENT_ID, "notes.md")],
        pipeline,
    )
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    source_path = docs_path / f"{DOCUMENT_ID}_notes.md"
    source_path.write_text("source", encoding="utf-8")

    result = service.delete_document(UUID(DOCUMENT_ID))
    events.append("file")

    assert result.model_dump(mode="json") == {
        "doc_id": DOCUMENT_ID,
        "deleted": True,
    }
    assert events == ["index", "file"]
    assert pipeline.deleted == [DOCUMENT_ID]
    assert not source_path.exists()
    assert not repair_path.exists()


def test_delete_missing_document_does_not_mutate_index(
    tmp_path: Path,
) -> None:
    """未知 doc_id 应返回领域 404，且不调用删除流水线。"""
    service, pipeline, repair_path = _service(tmp_path, [])

    with pytest.raises(DocumentNotFoundError):
        service.delete_document(UUID(DOCUMENT_ID))

    assert pipeline.deleted == []
    assert not repair_path.exists()


def test_index_delete_failure_preserves_source_and_repair_log(
    tmp_path: Path,
) -> None:
    """BM25 或 Chroma 删除失败时，源文件和 repair log 都不应改变。"""
    pipeline = FakePipeline(RuntimeError("private vector error"))
    service, _, repair_path = _service(
        tmp_path,
        [_chunk(DOCUMENT_ID, "notes.md")],
        pipeline,
    )
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    source_path = docs_path / f"{DOCUMENT_ID}_notes.md"
    source_path.write_text("secret source", encoding="utf-8")

    with pytest.raises(RuntimeError, match="private vector error"):
        service.delete_document(UUID(DOCUMENT_ID))

    assert source_path.read_text(encoding="utf-8") == "secret source"
    assert not repair_path.exists()


def test_file_cleanup_failure_writes_sanitized_durable_repair_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """索引已删但文件失败时应写一条不含敏感信息的 repair 记录。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest 属性替换工具。
    """
    service, pipeline, repair_path = _service(
        tmp_path,
        [_chunk(DOCUMENT_ID, "notes.md")],
    )
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    source_path = docs_path / f"{DOCUMENT_ID}_notes.md"
    source_path.write_text("secret source", encoding="utf-8")
    real_unlink = Path.unlink

    def failing_unlink(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        if path == source_path:
            raise PermissionError("raw secret API_KEY=abc")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with pytest.raises(DocumentDeletePartialError) as captured:
        service.delete_document(UUID(DOCUMENT_ID))

    assert captured.value.repair_required is True
    assert pipeline.deleted == [DOCUMENT_ID]
    record = json.loads(
        repair_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert record == {
        "doc_id": DOCUMENT_ID,
        "filename": "notes.md",
        "occurred_at": "2026-08-20T01:02:03Z",
        "error_category": "source_file_cleanup_failed",
    }
    serialized = json.dumps(record, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert "secret source" not in serialized
    assert "API_KEY" not in serialized
    assert "raw secret" not in serialized
