# -*- coding: utf-8 -*-
"""安全聚合、定位和删除已经入库的文档。"""

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path, PurePath, PureWindowsPath
from threading import RLock
from uuid import UUID

from app.application.errors import (
    DocumentDeletePartialError,
    DocumentNotFoundError,
    DocumentStorageCorruptionError,
)
from app.application.ports.ingestion import DocumentDeletion
from app.application.ports.repair_log import RepairLogWriter
from app.application.ports.vector_store import VectorStoreReader
from app.application.ports.document_lifecycle import DocumentLifecycle
from app.application.ports.document_listing import DocumentListing
from app.application.ports.document_registry import DocumentRegistry
from app.application.documents.sources import safe_source_path
from app.application.errors import DocumentIngestionInProgressError
from app.application.errors import IngestionJobStorageError
from app.application.errors import KnowledgeContentError
from app.application.ports.ingestion_jobs import DocumentJobReader
from app.domain.documents import (
    BatchDocumentDeleteResult,
    DocumentChunk,
    DocumentDeleteResult,
    DocumentFileStatus,
    DocumentSummary,
)


logger = logging.getLogger(__name__)


class DocumentService:
    """协调向量元数据、共享入库流水线和上传源文件。"""

    def __init__(
        self,
        docs_path: Path,
        vector_store: VectorStoreReader,
        ingestion_pipeline: DocumentDeletion | Callable[[], DocumentDeletion],
        repair_log: RepairLogWriter,
        mutation_lock: AbstractContextManager[object] | None = None,
        deletion_registry: DocumentLifecycle | None = None,
        listing: DocumentListing | None = None,
        jobs: DocumentJobReader | None = None,
        registry: DocumentRegistry | None = None,
    ) -> None:
        """保存受控目录和共享的索引依赖。

        Args:
            docs_path: 只允许访问的上传源文件目录。
            vector_store: 提供全部文本块快照的向量存储。
            ingestion_pipeline: 共享索引删除流水线或组合根惰性提供函数。
            repair_log: 文件清理部分失败的持久化记录器。
            mutation_lock: 与入库共享的可重入文档写锁。
            deletion_registry: 在已确认身份后记录删除，供来源面板区分410。
            listing: 生产管理元数据只读用例，独立旧单元测试可不提供。
            jobs: 真实任务状态，删除前不得破坏仍可恢复的任务源文件。
            registry: documents 表；按 doc_id 删除分块与登记的唯一写入口。

        Returns:
            None。
        """
        self._docs_path = docs_path
        self._vector_store = vector_store
        self._pipeline_source = ingestion_pipeline
        self._repair_log = repair_log
        self._mutation_lock = mutation_lock or RLock()
        self._deletion_registry = deletion_registry
        self._listing = listing
        self._jobs = jobs
        self._registry = registry

    @property
    def _ingestion_pipeline(self) -> DocumentDeletion:
        if callable(self._pipeline_source):
            return self._pipeline_source()
        return self._pipeline_source

    def list_documents(self, kb_id: str | None = None) -> list[DocumentSummary]:
        """聚合当前文本块并返回稳定排序的文档列表。

        Args:
            kb_id: 可选实际知识库过滤。

        Returns:
            created_at 降序、doc_id 升序的不可变文档摘要。

        Raises:
            DocumentStorageCorruptionError: UUID、文件名或路径元数据损坏。
        """
        if self._listing is not None:
            return self._listing.list_documents(kb_id)
        chunks = self._vector_store.list_chunks()
        if kb_id is not None:
            chunks = [chunk for chunk in chunks if chunk.kb_id == kb_id]
        grouped = self._group_chunks(chunks)
        documents = []
        for doc_id, chunks in grouped.items():
            filename = self._single_filename(chunks)
            source_path = self._safe_source_path(doc_id, filename)
            created_at = None
            file_status = DocumentFileStatus.MISSING
            if source_path.exists():
                try:
                    modified_at = source_path.stat().st_mtime
                except OSError as exc:
                    raise DocumentStorageCorruptionError(
                        "文档源文件状态不可读取"
                    ) from exc
                created_at = datetime.fromtimestamp(
                    modified_at,
                    tz=timezone.utc,
                )
                file_status = DocumentFileStatus.AVAILABLE
            documents.append(
                DocumentSummary(
                    doc_id=doc_id,
                    filename=filename,
                    chunks_count=len(chunks),
                    created_at=created_at,
                    file_status=file_status,
                )
            )
        documents.sort(key=lambda document: str(document.doc_id))
        minimum = datetime.min.replace(tzinfo=timezone.utc)
        documents.sort(
            key=lambda document: document.created_at or minimum,
            reverse=True,
        )
        return documents

    def delete_document(self, doc_id: UUID) -> DocumentDeleteResult:
        """按 doc_id 直接删除分块、登记和源文件，入库失败（无分块）的文档同样可删。

        Args:
            doc_id: 已经通过请求层格式校验的文档 UUID。

        Returns:
            删除成功的稳定领域结果。

        Raises:
            DocumentNotFoundError: 登记、索引和任务都没有该文档。
            DocumentStorageCorruptionError: 来源元数据或路径不安全。
            DocumentDeletePartialError: 分块已删除但源文件无法清理。
            DocumentIngestionInProgressError: 文档仍在入库。
        """
        result = self.delete_documents((doc_id,))
        if result.not_found:
            raise DocumentNotFoundError("文档不存在")
        return DocumentDeleteResult(doc_id=doc_id)

    def delete_documents(self, doc_ids: tuple[UUID, ...]) -> BatchDocumentDeleteResult:
        """批量按 doc_id 删除；分块与登记在一个事务提交，之后再删源文件。

        Args:
            doc_ids: 待删除文档，重复身份只处理一次。

        Returns:
            已删除与不存在的文档身份。

        Raises:
            DocumentStorageCorruptionError: 任一来源元数据不安全，整批不删除。
            DocumentDeletePartialError: 分块已删除但部分源文件无法清理。
            DocumentIngestionInProgressError: 任一文档仍在入库，整批不删除。
        """
        if self._registry is None:
            raise RuntimeError("文档删除需要 documents 登记")
        identifiers = tuple(dict.fromkeys(doc_ids))
        # 先检查避免在正在向量化的任务锁后排队；锁内再核对防止状态变化。
        self._ensure_not_ingesting(identifiers)
        with self._mutation_lock:
            self._ensure_not_ingesting(identifiers)
            return self._delete_unlocked(identifiers)

    def _ensure_not_ingesting(self, doc_ids: tuple[UUID, ...]) -> None:
        if self._jobs is None or not doc_ids:
            return
        try:
            jobs = self._jobs.get_document_jobs(doc_ids)
        except IngestionJobStorageError as error:
            raise KnowledgeContentError("文档任务状态暂时不可用") from error
        if any(job.status in {"queued", "running"} for job in jobs.values()):
            raise DocumentIngestionInProgressError("文档仍在入库，请等待任务结束后删除")

    def _resolve(self, doc_ids: tuple[UUID, ...]) -> tuple[dict[UUID, tuple[str, str]], list[UUID]]:
        """逐个 doc_id 解析身份：登记优先，其次该文档自身分块，最后历史任务。"""
        jobs = {}
        if self._jobs is not None and doc_ids:
            try:
                jobs = self._jobs.get_document_jobs(doc_ids)
            except IngestionJobStorageError as error:
                raise KnowledgeContentError("文档任务状态暂时不可用") from error
        targets: dict[UUID, tuple[str, str]] = {}
        missing: list[UUID] = []
        for doc_id in doc_ids:
            record = self._registry.get(doc_id)
            if record is not None:
                identity = (record.kb_id, record.filename)
            else:
                headers = self._chunk_headers(doc_id)
                job = jobs.get(doc_id)
                if headers:
                    identity = (headers[0].kb_id, headers[0].filename)
                elif job is not None and not self._already_deleted(doc_id):
                    # 入库失败、没有分块的旧文档只剩任务登记，也必须可删。
                    identity = (job.kb_id, job.filename)
                else:
                    missing.append(doc_id)
                    continue
            self._single_filename_value(identity[1])
            self._safe_source_path(doc_id, identity[1])
            targets[doc_id] = identity
        return targets, missing

    def _already_deleted(self, doc_id: UUID) -> bool:
        if self._deletion_registry is None:
            return False
        return self._deletion_registry.was_deleted(doc_id)

    def _chunk_headers(self, doc_id: UUID) -> tuple:
        try:
            return self._vector_store.list_chunk_headers(doc_id)
        except ValueError as error:
            raise DocumentStorageCorruptionError("文档分块元数据不一致") from error

    def _delete_unlocked(self, doc_ids: tuple[UUID, ...]) -> BatchDocumentDeleteResult:
        targets, missing = self._resolve(doc_ids)
        # 所有身份都先校验完再删除，避免损坏的后半批导致前半批已删。
        self._registry.delete_documents({doc_id: kb_id for doc_id, (kb_id, _) in targets.items()})
        partial = False
        for doc_id, (_, filename) in targets.items():
            try:
                self._remove_source(self._safe_source_path(doc_id, filename), doc_id, filename)
            except DocumentDeletePartialError:
                partial = True
        if partial:
            raise DocumentDeletePartialError()
        return BatchDocumentDeleteResult(deleted=tuple(targets), not_found=tuple(missing))

    def delete_planned_source(self, doc_id: UUID, filename: str, kb_id: str) -> None:
        """清理持久KB计划中已无索引的源文件，不接受客户端任意路径。

        Args:
            doc_id: 删除计划已经确认的来源UUID。
            filename: 原计划冻结的安全文件名。
            kb_id: 原计划冻结的知识库。

        Returns:
            None，缺失源文件为幂等成功。
        """
        if self._registry is None:
            raise RuntimeError("文档删除需要 documents 登记")
        self._ensure_not_ingesting((doc_id,))
        with self._mutation_lock:
            self._ensure_not_ingesting((doc_id,))
            if self._chunk_headers(doc_id):
                raise DocumentStorageCorruptionError("计划源清理时索引仍存在")
            if self._jobs is not None:
                job = self._jobs.get_document_jobs((doc_id,)).get(doc_id)
                if job is not None:
                    if job.kb_id != kb_id or job.filename != filename:
                        raise DocumentStorageCorruptionError("计划源与历史任务身份不一致")
            path = self._safe_source_path(doc_id, filename)
            self._registry.delete_documents({doc_id: kb_id})
            self._remove_source(path, doc_id, filename)

    def remove_replaced_source(self, doc_id: UUID) -> None:
        """新版本发布后删除被替换旧版本的源文件；分块已在发布事务中删除。

        Args:
            doc_id: 刚发布的新版本。

        Returns:
            None。
        """
        if self._registry is None:
            return
        old = self._registry.replaced_by(doc_id)
        if old is None:
            return
        with self._mutation_lock:
            self._remove_source(self._safe_source_path(old.doc_id, old.filename), old.doc_id, old.filename)

    def _remove_source(self, source_path: Path, doc_id: UUID, filename: str) -> None:
        try:
            source_path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            try:
                self._repair_log.append_document_deletion(
                    doc_id,
                    filename,
                )
            except Exception:
                logger.critical(
                    "文档删除 repair 记录写入失败",
                    extra={"repair_record_failed": True},
                )
            raise DocumentDeletePartialError() from None

    @staticmethod
    def _group_chunks(
        chunks: list[DocumentChunk],
    ) -> dict[UUID, list[DocumentChunk]]:
        grouped: dict[UUID, list[DocumentChunk]] = {}
        for chunk in chunks:
            try:
                doc_id = UUID(chunk.doc_id)
            except (ValueError, AttributeError) as exc:
                raise DocumentStorageCorruptionError(
                    "文档 UUID 元数据无效"
                ) from exc
            grouped.setdefault(doc_id, []).append(chunk)
        return grouped

    @staticmethod
    def _single_filename(chunks: list[DocumentChunk]) -> str:
        filenames = {chunk.filename for chunk in chunks}
        if len(filenames) != 1:
            raise DocumentStorageCorruptionError(
                "同一文档存在多个来源文件名"
            )
        return DocumentService._single_filename_value(next(iter(filenames)))

    @staticmethod
    def _single_filename_value(filename: str) -> str:
        posix_name = PurePath(filename).name
        windows_name = PureWindowsPath(filename).name
        if filename in {"", ".", ".."}:
            raise DocumentStorageCorruptionError("文档来源文件名无效")
        if posix_name != filename or windows_name != filename:
            raise DocumentStorageCorruptionError("文档来源文件名不安全")
        return filename

    def _safe_source_path(self, doc_id: UUID, filename: str) -> Path:
        return safe_source_path(self._docs_path, doc_id, filename)
