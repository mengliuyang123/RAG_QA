# -*- coding: utf-8 -*-
"""组合权威目录与只读索引元数据，不访问模型或文档正文。"""

from contextlib import AbstractContextManager
from threading import RLock

from app.application.errors import KnowledgeBaseStorageError
from app.application.errors import IngestionJobStorageError
from app.application.errors import VectorStoreError
from app.application.ports.knowledge_bases import KnowledgeBaseRepository
from app.application.ports.knowledge_content import KnowledgeContentReader
from app.application.ports.ingestion_jobs import DocumentPublication
from app.domain.knowledge_bases import KnowledgeBase
from app.domain.knowledge_bases import KnowledgeBasePatch
from app.domain.knowledge_views import AdminKnowledgeBaseCard
from app.domain.knowledge_views import KnowledgeBaseCard
from app.domain.knowledge_views import KnowledgeBaseList


class KnowledgeCatalogService:
    """目录与已有索引共享单进程变更边界，防止分页中计数漂移。"""

    def __init__(
        self, catalog: KnowledgeBaseRepository, content: KnowledgeContentReader,
        mutation_lock: AbstractContextManager[object] | None = None,
        *,
        publication: DocumentPublication | None = None,
    ) -> None:
        """注入组合根提供的权威目录和元数据读取能力。

        Args:
            catalog: 持久化知识库元数据。
            content: 不加载正文的内容统计端口。
            mutation_lock: 与入库、删除共用的可重入锁。
            publication: 只读任务发布登记，生产组合根必须装配。

        Returns:
            None。
        """
        self._catalog = catalog
        self._content = content
        self._mutation_lock = mutation_lock or RLock()
        self._publication = publication

    def list(self) -> KnowledgeBaseList:
        """返回全部已注册目录，空库不能因没有chunk而消失。

        Returns:
            仅含公开卡片字段的目录集合。
        """
        with self._mutation_lock:
            knowledge_bases = self._catalog.list()
            counts = self._counts()
            cards = []
            for kb in knowledge_bases:
                cards.append(KnowledgeBaseCard(
                    id=kb.id, name=kb.name, description=kb.description,
                    icon=kb.icon, documents_count=counts.get(kb.id, 0),
                    updated_at=kb.updated_at,
                ))
        return KnowledgeBaseList(knowledge_bases=tuple(cards))

    def create(self, knowledge_base: KnowledgeBase) -> AdminKnowledgeBaseCard:
        """创建元数据后返回真实计数，不修改或重建文档索引。

        Args:
            knowledge_base: 由应用边界校验的新目录。

        Returns:
            管理员投影及保留的创建时间。
        """
        with self._mutation_lock:
            # 先读统计，避免索引读取失败后已创建目录却向客户端报告失败。
            counts = self._counts()
            stored = self._catalog.create(knowledge_base)
            return AdminKnowledgeBaseCard.model_validate({
                **stored.model_dump(), "documents_count": counts.get(stored.id, 0),
            })

    def patch(self, kb_id: str, changes: KnowledgeBasePatch) -> AdminKnowledgeBaseCard:
        """只更新传入字段，下一次问题理解直接读取新描述。

        Args:
            kb_id: 不可被更新的稳定ID。
            changes: 严格部分更新数据。

        Returns:
            本次事务提交后的管理员投影。
        """
        with self._mutation_lock:
            counts = self._counts()
            stored = self._catalog.patch(kb_id, changes)
            return AdminKnowledgeBaseCard.model_validate({
                **stored.model_dump(), "documents_count": counts.get(kb_id, 0),
            })

    def _counts(self) -> dict[str, int]:
        try:
            if self._publication is not None:
                identities = self._publication.unpublished_document_ids()
                if identities:
                    return self._content.count_documents_by_kb(
                        excluded_doc_ids=tuple(str(item) for item in identities),
                    )
            return self._content.count_documents_by_kb()
        except (VectorStoreError, IngestionJobStorageError) as error:
            raise KnowledgeBaseStorageError("目录文档统计暂时不可用") from error
