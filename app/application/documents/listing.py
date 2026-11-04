# -*- coding: utf-8 -*-
"""管理文档只读元数据，索引正文与模型不进入此用例。"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from datetime import timezone
from pathlib import Path
from pathlib import PurePath

from app.application.documents.sources import safe_source_path
from app.application.documents.sources import source_details
from app.application.documents.citations import CitationStatisticsService
from app.application.errors import DocumentStorageCorruptionError
from app.application.errors import IngestionJobStorageError
from app.application.errors import KnowledgeContentError
from app.application.errors import VectorStoreError
from app.application.ports.ingestion_jobs import DocumentJobReader
from app.application.ports.knowledge_content import KnowledgeContentReader
from app.domain.documents import DocumentSummary


class AdminDocumentListing:
    """仅管理员可用的当前索引管理视图，不把它当普通用户发布清单。"""

    def __init__(
        self, docs_path: Path, content: KnowledgeContentReader,
        jobs: DocumentJobReader, mutation_lock: AbstractContextManager[object],
        statistics: Callable[[], CitationStatisticsService] | None = None,
    ) -> None:
        """保存与文档维护相同的源路径、元数据与一致性锁。

        Args:
            docs_path: 本runtime的实际源目录。
            content: 仅读取元数据头的端口。
            jobs: 批量读取管理任务资料的端口。
            mutation_lock: 与入库／删除共用的锁。
            statistics: 惰性获取当前业务库统计，空索引不触发建库。

        Returns:
            None。
        """
        self._docs_path = docs_path
        self._content = content
        self._jobs = jobs
        self._mutation_lock = mutation_lock
        self._statistics = statistics

    def list_documents(self, kb_id: str | None = None) -> list[DocumentSummary]:
        """保留旧列表和mtime排序，批量附加真实任务字段。

        Args:
            kb_id: 元数据召回前限定知识库，None为全部索引管理视图。

        Returns:
            包含实际文件及发布状态的不可变文档摘要。
        """
        with self._mutation_lock:
            try:
                headers = self._content.list_document_headers(kb_id)
                jobs = self._jobs.get_document_jobs(tuple(header.doc_id for header in headers))
            except (VectorStoreError, IngestionJobStorageError) as error:
                raise KnowledgeContentError("管理文档元数据暂时不可用") from error
            totals = {}
            if headers and self._statistics is not None:
                totals = self._statistics().documents(tuple(header.doc_id for header in headers))
            documents = []
            for header in headers:
                citations = totals.get(header.doc_id)
                job = jobs.get(header.doc_id)
                if job is not None:
                    if job.kb_id != header.kb_id or job.filename != header.filename:
                        raise DocumentStorageCorruptionError("文档任务与索引身份不一致")
                path = safe_source_path(self._docs_path, header.doc_id, header.filename)
                file_status, created_at, size_bytes = source_details(path)
                documents.append(DocumentSummary(
                    doc_id=header.doc_id, filename=header.filename,
                    chunks_count=header.chunks_count, created_at=created_at,
                    file_status=file_status, kb_id=header.kb_id,
                    file_type=PurePath(header.filename).suffix.lstrip(".").lower(),
                    size_bytes=size_bytes,
                    uploaded_by=job.uploaded_by if job is not None else None,
                    ingestion_status=job.status if job is not None else None,
                    published=job is None or job.status == "succeeded",
                    citations_7d=citations.total if citations is not None else None,
                    unattributed_citations_7d=citations.unattributed if citations is not None else None,
                ))
        documents.sort(key=lambda item: str(item.doc_id))
        minimum = datetime.min.replace(tzinfo=timezone.utc)
        documents.sort(key=lambda item: item.created_at or minimum, reverse=True)
        return documents
