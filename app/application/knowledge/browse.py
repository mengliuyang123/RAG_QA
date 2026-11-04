# -*- coding: utf-8 -*-
"""只读文档分页和来源窗口，范围与原文加载由程序确定。"""

import base64
import binascii
import hashlib
import json
from contextlib import AbstractContextManager
from pathlib import PurePath
from threading import RLock
from uuid import UUID

from app.application.errors import ChunkNotFoundError
from app.application.errors import DocumentGoneError
from app.application.errors import KnowledgeContentError
from app.application.errors import VectorStoreError
from app.application.errors import IngestionJobStorageError
from app.application.ports.knowledge_bases import KnowledgeBaseRepository
from app.application.ports.knowledge_content import KnowledgeContentReader
from app.application.ports.document_lifecycle import DocumentLifecycle
from app.application.ports.suggestions import CuratedQuestionRepository
from app.application.ports.ingestion_jobs import DocumentPublication
from app.domain.knowledge_views import ContextChunk
from app.domain.knowledge_views import DocumentBrowseCursor
from app.domain.knowledge_views import DocumentContext
from app.domain.knowledge_views import KnowledgeDocument
from app.domain.knowledge_views import KnowledgeDocumentPage
from app.domain.suggestions import CuratedQuestionsUpdate
from app.domain.suggestions import SuggestedQuestions


class KnowledgeBrowseService:
    """使用权威目录和同一文档锁，不让客户端通过ID绕过范围校验。"""

    def __init__(
        self, catalog: KnowledgeBaseRepository, content: KnowledgeContentReader,
        lifecycle: DocumentLifecycle, suggestions: CuratedQuestionRepository,
        mutation_lock: AbstractContextManager[object] | None = None,
        *,
        publication: DocumentPublication | None = None,
    ) -> None:
        """保存只读投影依赖，不接收模型客户端或源文件目录。

        Args:
            catalog: 已注册知识库目录。
            content: 仅读元数据与有限正文的端口。
            lifecycle: 删除前确认过身份的持久记录。
            suggestions: 仅包含明确管理员预设的仓储。
            mutation_lock: 与入库、删除共享的本进程锁。
            publication: 只读发布登记，不能经浏览泄漏部分索引。

        Returns:
            None。
        """
        self._catalog = catalog
        self._content = content
        self._lifecycle = lifecycle
        self._suggestions = suggestions
        self._mutation_lock = mutation_lock or RLock()
        self._publication = publication

    def list_documents(
        self, kb_id: str, *, q: str | None = None,
        limit: int = 20, cursor: str | None = None,
    ) -> KnowledgeDocumentPage:
        """按库与字面文件名子串过滤，再做稳定UUID游标分页。

        Args:
            kb_id: 由请求路径限定的知识库。
            q: 不区分大小写的文件名子串，百分号与下划线不是通配符。
            limit: 每页1–100条。
            cursor: 与知识库及规范搜索词绑定的位置。

        Returns:
            不包含正文、入库状态或源文件状态的文档页。
        """
        if not 1 <= limit <= 100:
            raise ValueError("文档页大小无效")
        query = q.strip().casefold() if q else ""
        fingerprint = hashlib.sha256(json.dumps(
            {"kb_id": kb_id, "q": query}, sort_keys=True,
        ).encode()).hexdigest()
        position = None
        if cursor is not None:
            try:
                raw = base64.b64decode(cursor, altchars=b"-_", validate=True)
                position = DocumentBrowseCursor.model_validate_json(raw)
                if position.fingerprint != fingerprint:
                    raise ValueError("游标条件不一致")
            except (ValueError, binascii.Error) as error:
                raise ValueError("文档游标无效") from error
        with self._mutation_lock:
            self._catalog.get(kb_id)
            unpublished = self._unpublished_documents()
            try:
                headers = self._content.list_document_headers(kb_id)
            except VectorStoreError as error:
                raise KnowledgeContentError("文档浏览暂时不可用") from error
            candidates = []
            for header in headers:
                if header.doc_id in unpublished:
                    continue
                if query not in header.filename.casefold():
                    continue
                if position is not None and str(header.doc_id) <= str(position.last_doc_id):
                    continue
                candidates.append(header)
            candidates.sort(key=lambda header: str(header.doc_id))
            documents = []
            for header in candidates[:limit]:
                documents.append(KnowledgeDocument(
                    doc_id=header.doc_id, filename=header.filename,
                    file_type=PurePath(header.filename).suffix.lstrip(".").lower(),
                    chunks_count=header.chunks_count, updated_at=header.updated_at,
                ))
        next_cursor = None
        if len(candidates) > limit:
            raw = DocumentBrowseCursor(
                last_doc_id=documents[-1].doc_id, fingerprint=fingerprint,
            ).model_dump_json().encode()
            next_cursor = base64.urlsafe_b64encode(raw).decode()
        return KnowledgeDocumentPage(documents=tuple(documents), next_cursor=next_cursor)

    def _unpublished_documents(self) -> frozenset[UUID]:
        if self._publication is None:
            return frozenset()
        try:
            return frozenset(self._publication.unpublished_document_ids())
        except IngestionJobStorageError as error:
            raise KnowledgeContentError("文档发布状态暂时不可用") from error

    def get_context(self, doc_id: UUID, chunk_id: str, context: int = 1) -> DocumentContext:
        """只对匹配文档的实际目标块加载有界邻居正文。

        Args:
            doc_id: 来源文档UUID。
            chunk_id: 不透明目标块ID，不从中解析顺序。
            context: 前后各0–3块。

        Returns:
            最多七块的当前原文窗口；未知资源不猜测已删除。
        """
        if not 0 <= context <= 3:
            raise ValueError("原文窗口大小无效")
        with self._mutation_lock:
            if doc_id in self._unpublished_documents():
                raise ChunkNotFoundError("文档与块组合不存在")
            try:
                headers = self._content.list_chunk_headers(doc_id)
                if not headers and self._lifecycle.was_deleted(doc_id):
                    raise DocumentGoneError("来源文档已删除")
                position = None
                for index, header in enumerate(headers):
                    if header.chunk_id == chunk_id:
                        position = index
                        break
                if position is None:
                    raise ChunkNotFoundError("文档与块组合不存在")
                target = headers[position]
                self._catalog.get(target.kb_id)
                window = headers[max(0, position - context):position + context + 1]
                actual = self._content.read_chunks(window)
            except VectorStoreError as error:
                raise KnowledgeContentError("来源原文暂时不可用") from error
        chunks = []
        for chunk in actual:
            chunks.append(ContextChunk(
                chunk_id=chunk.id, chunk_index=chunk.chunk_index,
                page=chunk.page, text=chunk.text, is_target=chunk.id == chunk_id,
            ))
        return DocumentContext(
            doc_id=doc_id, kb_id=target.kb_id, filename=target.filename,
            doc_updated_at=target.doc_updated_at, chunks=tuple(chunks),
        )

    def suggested_questions(self, kb_id: str | None = None, limit: int = 3) -> SuggestedQuestions:
        """仅返回管理员明确发布的预设，不读取私人会话或反馈。

        Args:
            kb_id: 可选已注册范围。
            limit: 1–20条公开上限。

        Returns:
            稳定有序的公开预设，未配置时为空。
        """
        if not 1 <= limit <= 20:
            raise ValueError("建议问题数量无效")
        if kb_id is not None:
            self._catalog.get(kb_id)
        return SuggestedQuestions(questions=self._suggestions.list_curated(kb_id, limit))

    def replace_questions(self, kb_id: str, update: CuratedQuestionsUpdate) -> SuggestedQuestions:
        """由受保护管理员入口明确发布或清空单库预设。

        Args:
            kb_id: 已注册知识库。
            update: 通过严格校验的明确发布内容。

        Returns:
            已提交预设，不引入自动popular发布。
        """
        with self._mutation_lock:
            self._catalog.get(kb_id)
            return SuggestedQuestions(questions=self._suggestions.replace_curated(kb_id, update))
