# -*- coding: utf-8 -*-
"""持久计划串联KB文档清理，失败后按原范围恢复，不伪称跨存储原子性。"""

import logging
from contextlib import AbstractContextManager
from datetime import UTC
from datetime import datetime
from pathlib import Path
from uuid import UUID

from app.application.documents.service import DocumentService
from app.application.documents.sources import safe_source_path
from app.application.documents.sources import source_details
from app.application.errors import DocumentDeletePartialError
from app.application.errors import DocumentStorageCorruptionError
from app.application.errors import IngestionJobStorageError
from app.application.errors import KnowledgeBaseDeletePartialError
from app.application.errors import KnowledgeBaseDeletionInProgressError
from app.application.errors import KnowledgeBaseIngestionBusyError
from app.application.errors import KnowledgeBaseNotEmptyError
from app.application.errors import KnowledgeBaseProtectedError
from app.application.errors import KnowledgeBaseStorageError
from app.application.errors import KnowledgeContentError
from app.application.errors import VectorStoreError
from app.application.ports.ingestion_jobs import DocumentJobReader
from app.application.ports.knowledge_bases import KnowledgeBaseRepository
from app.application.ports.knowledge_content import KnowledgeContentReader
from app.application.ports.knowledge_deletion import KnowledgeDeletionRepository
from app.application.tracing.auditing import AdminAuditService
from app.domain.documents import DocumentFileStatus
from app.domain.knowledge_bases import DEFAULT_KB_ID
from app.domain.knowledge_deletion import KnowledgeBaseDeleteResult
from app.domain.knowledge_deletion import KnowledgeDeletionItem
from app.domain.knowledge_deletion import KnowledgeDeletionPlan


logger = logging.getLogger(__name__)


class KnowledgeBaseDeletionService:
    """仅由已鉴权管理入口调用，清理范围只来自后端索引与持久任务。"""

    def __init__(
        self, catalog: KnowledgeBaseRepository, content: KnowledgeContentReader,
        jobs: DocumentJobReader, plans: KnowledgeDeletionRepository,
        documents: DocumentService, docs_path: Path,
        mutation_lock: AbstractContextManager[object], audit: AdminAuditService,
    ) -> None:
        """保存同运行、同锁、同源目录的维护能力。

        Args:
            catalog: 权威可用知识库目录。
            content: 真实索引元数据头。
            jobs: 包括无索引源文件的真实任务身份。
            plans: 目录冻结与持久恢复计划。
            documents: 原文档删除与BM25回滚边界。
            docs_path: 当前运行实际源目录。
            mutation_lock: 与入库和文档删除共用的锁。
            audit: 开始删除前必须落库的无正文尝试审计。

        Returns:
            None。
        """
        self._catalog = catalog
        self._content = content
        self._jobs = jobs
        self._plans = plans
        self._documents = documents
        self._docs_path = docs_path
        self._mutation_lock = mutation_lock
        self._audit = audit

    def delete(self, actor_id: UUID, kb_id: str, *, force: bool = False) -> KnowledgeBaseDeleteResult:
        """显式force才清文档，非终态任务永远不是隐式取消对象。

        Args:
            actor_id: 认证管理员身份。
            kb_id: 目标知识库。
            force: 真实授权删除整库文档或重试原批量计划。

        Returns:
            所有计划资源清理完成后的真实结果。
        """
        if kb_id == DEFAULT_KB_ID:
            raise KnowledgeBaseProtectedError("默认知识库不可删除")
        self._check_idle(kb_id)
        with self._mutation_lock:
            self._check_idle(kb_id)
            previous = self._plans.get(kb_id)
            if previous is not None and previous.force and not force:
                raise KnowledgeBaseDeletionInProgressError("请按原force请求重试")
            plan = previous
            if plan is None:
                self._catalog.get(kb_id)
                plan = self._prepare(kb_id, force)
            self._audit.record(
                actor_id, "delete_attempt", "knowledge_base", (), knowledge_base_id=kb_id,
            )
            if previous is None:
                plan = self._plans.begin(plan)
            if plan.state == "deleted":
                return KnowledgeBaseDeleteResult(kb_id=kb_id, documents_deleted=len(plan.items))
            try:
                self._clean(plan)
                self._plans.complete(plan)
            except (
                DocumentDeletePartialError, DocumentStorageCorruptionError,
                VectorStoreError, KnowledgeContentError, KnowledgeBaseStorageError,
                IngestionJobStorageError,
            ) as error:
                logger.error("知识库删除计划未完成", extra={"kb_id": kb_id})
                raise KnowledgeBaseDeletePartialError("知识库删除未完成，请按原请求重试") from error
        return KnowledgeBaseDeleteResult(kb_id=kb_id, documents_deleted=len(plan.items))

    def _check_idle(self, kb_id: str) -> None:
        try:
            jobs = self._jobs.list_knowledge_base_jobs(kb_id)
        except IngestionJobStorageError as error:
            raise KnowledgeBaseStorageError("知识库任务状态暂时不可用") from error
        for job in jobs:
            if job.status in {"queued", "running"}:
                raise KnowledgeBaseIngestionBusyError("知识库仍有可恢复入库任务")

    def _prepare(self, kb_id: str, force: bool) -> KnowledgeDeletionPlan:
        try:
            headers = self._content.list_document_headers(kb_id)
            jobs = self._jobs.list_knowledge_base_jobs(kb_id)
            indexed_jobs = self._jobs.get_document_jobs(tuple(header.doc_id for header in headers))
        except (VectorStoreError, IngestionJobStorageError) as error:
            raise KnowledgeBaseStorageError("知识库文档身份暂时不可用") from error
        items = {}
        for header in headers:
            job = indexed_jobs.get(header.doc_id)
            if job is not None:
                if job.kb_id != kb_id or job.filename != header.filename:
                    raise DocumentStorageCorruptionError("索引与历史任务知识库身份不一致")
            items[header.doc_id] = KnowledgeDeletionItem(doc_id=header.doc_id, filename=header.filename)
        for job in jobs:
            if job.doc_id in items and items[job.doc_id].filename != job.filename:
                raise DocumentStorageCorruptionError("任务与索引来源身份不一致")
            path = safe_source_path(self._docs_path, job.doc_id, job.filename)
            status, _, _ = source_details(path)
            if status == DocumentFileStatus.AVAILABLE:
                items[job.doc_id] = KnowledgeDeletionItem(doc_id=job.doc_id, filename=job.filename)
        for item in items.values():
            safe_source_path(self._docs_path, item.doc_id, item.filename)
        if items and not force:
            raise KnowledgeBaseNotEmptyError("知识库仍有文档或已登记源文件")
        ordered = tuple(items[identifier] for identifier in sorted(items, key=str))
        return KnowledgeDeletionPlan(kb_id=kb_id, force=force, items=ordered, started_at=datetime.now(UTC))

    def _clean(self, plan: KnowledgeDeletionPlan) -> None:
        headers = self._content.list_document_headers(plan.kb_id)
        actual = {header.doc_id: header for header in headers}
        planned = {item.doc_id for item in plan.items}
        jobs = self._jobs.get_document_jobs(tuple(item.doc_id for item in plan.items))
        if not set(actual).issubset(planned):
            raise KnowledgeBaseStorageError("删除开始后出现计划外索引")
        # 所有身份先验证，避免损坏的后半份计划导致前半份已经删除。
        for item in plan.items:
            safe_source_path(self._docs_path, item.doc_id, item.filename)
            job = jobs.get(item.doc_id)
            if job is not None:
                if job.kb_id != plan.kb_id or job.filename != item.filename:
                    raise DocumentStorageCorruptionError("计划与历史任务身份不一致")
            header = actual.get(item.doc_id)
            if header is not None and header.filename != item.filename:
                raise DocumentStorageCorruptionError("计划文档来源名已改变")
        for item in plan.items:
            if item.doc_id in actual:
                self._documents.delete_document(item.doc_id)
            else:
                self._documents.delete_planned_source(item.doc_id, item.filename, plan.kb_id)
        if self._content.list_document_headers(plan.kb_id):
            raise KnowledgeBaseStorageError("计划文档索引仍存在")
