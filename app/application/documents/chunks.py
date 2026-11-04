# -*- coding: utf-8 -*-
"""管理员有界正文诊断，与普通来源窗口的发布权限分开。"""

import base64
import binascii
import hashlib
import json
from contextlib import AbstractContextManager
from uuid import UUID

from app.application.documents.citations import CitationStatisticsService
from app.application.errors import DocumentNotFoundError
from app.application.errors import DocumentRevisionChangedError
from app.application.errors import IngestionJobStorageError
from app.application.errors import KnowledgeContentError
from app.application.errors import VectorStoreError
from app.application.ports.ingestion_jobs import DocumentJobReader
from app.application.ports.knowledge_content import KnowledgeContentReader
from app.application.tracing.auditing import AdminAuditService
from app.domain.document_diagnostics import DiagnosticChunk
from app.domain.document_diagnostics import DiagnosticChunkPage
from app.domain.document_diagnostics import DiagnosticCursor
from app.domain.knowledge_views import IndexedChunkHeader


class AdminDocumentChunksService:
    """先选元数据页并审计，再按最多七块读取当前真实正文。"""

    def __init__(
        self, content: KnowledgeContentReader, jobs: DocumentJobReader,
        statistics: CitationStatisticsService, audit: AdminAuditService,
        mutation_lock: AbstractContextManager[object],
    ) -> None:
        """保存同运行的只读端口与共享文档变更锁。

        Args:
            content: 支持有界正文读取的索引端口。
            jobs: 当前任务与发布状态。
            statistics: 正式会话引用统计。
            audit: 必须成功落库的访问审计。
            mutation_lock: 与入库和删除共用的变更锁。

        Returns:
            None。
        """
        self._content = content
        self._jobs = jobs
        self._statistics = statistics
        self._audit = audit
        self._mutation_lock = mutation_lock

    def list(
        self, actor_id: UUID, doc_id: UUID, *, limit: int = 20,
        cursor: str | None = None,
    ) -> DiagnosticChunkPage:
        """返回经审计的真实页，旧引用未知归属不伪造为零。

        Args:
            actor_id: HTTP已认证的管理员身份。
            doc_id: 目标文档UUID。
            limit: 每页1至100块。
            cursor: 绑定文档及全部元数据修订的上页游标。

        Returns:
            当前索引正文、实际发布状态和引用事实。
        """
        if not 1 <= limit <= 100:
            raise ValueError("分块页大小无效")
        position = _decode_cursor(cursor, doc_id)
        with self._mutation_lock:
            try:
                headers = self._content.list_chunk_headers(doc_id)
                if not headers:
                    raise DocumentNotFoundError("当前索引中无此文档")
                revision = _revision(headers)
                if position is not None and position.revision != revision:
                    raise DocumentRevisionChangedError("来源已变化，请重新加载首页")
                if position is not None:
                    indexes = {header.chunk_index for header in headers}
                    if position.last_index not in indexes:
                        raise ValueError("分块游标未指向实际块")
                first = headers[0]
                job = self._jobs.get_document_jobs((doc_id,)).get(doc_id)
                if job is not None:
                    if job.kb_id != first.kb_id or job.filename != first.filename:
                        raise KnowledgeContentError("文档任务与索引身份不一致")
                candidates = []
                for header in headers:
                    if position is None or header.chunk_index > position.last_index:
                        candidates.append(header)
                selected = tuple(candidates[:limit])
                counts = self._statistics.chunks(doc_id, tuple(header.chunk_id for header in selected))
                # 诊断允许读取未发布正文，因此在任何正文读取前落访问审计。
                self._audit.record(actor_id, "read", "document_chunks", (doc_id,))
                actual = []
                for offset in range(0, len(selected), 7):
                    actual.extend(self._content.read_chunks(selected[offset:offset + 7]))
            except (VectorStoreError, IngestionJobStorageError) as error:
                raise KnowledgeContentError("管理分块暂时不可用") from error
        known = {count.chunk_id: count.count for count in counts.counts}
        chunks = []
        for chunk in actual:
            value = known[chunk.id]
            chunks.append(DiagnosticChunk(
                chunk_id=chunk.id, chunk_index=chunk.chunk_index,
                page=chunk.page, text=chunk.text, known_citations_7d=value,
                citations_7d=value if counts.unattributed == 0 else None,
            ))
        next_cursor = None
        if len(candidates) > limit:
            next_cursor = base64.urlsafe_b64encode(DiagnosticCursor(
                doc_id=doc_id, revision=revision,
                last_index=selected[-1].chunk_index,
            ).model_dump_json().encode()).decode()
        return DiagnosticChunkPage(
            doc_id=doc_id, kb_id=first.kb_id, filename=first.filename,
            doc_updated_at=first.doc_updated_at,
            published=job is None or job.status == "succeeded",
            chunks=tuple(chunks), total=len(headers), next_cursor=next_cursor,
            unattributed_citations_7d=counts.unattributed,
            window_started_at=counts.window_started_at,
            window_ended_at=counts.window_ended_at,
        )


def _decode_cursor(cursor: str | None, doc_id: UUID) -> DiagnosticCursor | None:
    if cursor is None:
        return None
    if len(cursor) > 2048:
        raise ValueError("分块游标过长")
    try:
        raw = base64.b64decode(cursor, altchars=b"-_", validate=True)
        position = DiagnosticCursor.model_validate_json(raw)
    except (ValueError, binascii.Error) as error:
        raise ValueError("分块游标无效") from error
    if position.doc_id != doc_id:
        raise ValueError("分块游标不属于此文档")
    return position


def _revision(headers: tuple[IndexedChunkHeader, ...]) -> str:
    values = [header.model_dump(mode="json") for header in headers]
    raw = json.dumps(values, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()
