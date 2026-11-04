# -*- coding: utf-8 -*-
"""目录统计不读取全文或向量，按稳定文档身份计数而非按chunk计数。"""

from pathlib import Path
from uuid import uuid4

import pytest

from app.domain.documents import DocumentChunk
from app.application.errors import VectorStoreError
from app.infrastructure.vectorstores.chroma import ChromaVectorStore


def test_count_scans_metadata_in_pages_and_deduplicates_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """若退回list_chunks、读向量或只统计第一页／直接数chunk会失败。

    Args:
        tmp_path: 真实Chroma临时目录。
        monkeypatch: 只拦截不允许的读载荷，实际存储仍然运行。
    """
    store = ChromaVectorStore(tmp_path / "chroma", "metadata-counts")
    first, second, third = str(uuid4()), str(uuid4()), str(uuid4())
    chunks = []
    for index in range(1001):
        chunks.append(DocumentChunk(id=f"{first}:{index}", doc_id=first,
                                    filename="large.md", page=None, text="PRIVATE_BODY",
                                    chunk_index=index, kb_id="ops"))
    chunks.extend([
        DocumentChunk(id=f"{second}:0", doc_id=second, filename="second.md",
                      page=None, text="PRIVATE_BODY", chunk_index=0, kb_id="ops"),
        DocumentChunk(id=f"{third}:0", doc_id=third, filename="guide.md",
                      page=None, text="PRIVATE_BODY", chunk_index=0),
    ])
    store.add_chunks(chunks, [[1.0, 0.0] for _ in chunks])
    original = store._collection.get

    def metadata_only(*args: object, **kwargs: object):
        if kwargs.get("include") != ["metadatas"]:
            raise AssertionError("目录统计不能读取正文或向量")
        if not 1 <= kwargs.get("limit", 0) <= 1000:
            raise AssertionError("目录统计必须有界分页")
        return original(*args, **kwargs)

    monkeypatch.setattr(store._collection, "get", metadata_only)
    assert store.count_documents_by_kb() == {"ops": 2, "default": 1}


def test_document_with_inconsistent_kb_cannot_be_counted_twice(tmp_path: Path) -> None:
    """同一UUID被标成两个库时，应报告损坏而不是悄悄显示两篇文档。

    Args:
        tmp_path: 真实Chroma隔离目录。
    """
    store = ChromaVectorStore(tmp_path / "chroma", "inconsistent-metadata")
    identifier = str(uuid4())
    chunks = []
    for index, kb_id in enumerate(("default", "ops")):
        chunks.append(DocumentChunk(
            id=f"{identifier}:{index}", doc_id=identifier,
            filename="guide.md", page=None, text="正文",
            chunk_index=index, kb_id=kb_id,
        ))
    store.add_chunks(chunks, [[1.0, 0.0], [1.0, 0.0]])
    with pytest.raises(VectorStoreError):
        store.count_documents_by_kb()
