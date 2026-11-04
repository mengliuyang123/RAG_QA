# -*- coding: utf-8 -*-
"""声明知识库目录持久化端口，不将索引内容误当作知识库注册表。"""

from typing import Protocol

from app.domain.knowledge_bases import KnowledgeBase
from app.domain.knowledge_bases import KnowledgeBasePatch


class KnowledgeBaseRepository(Protocol):
    """定义目录读取与管理员元数据写入能力。"""

    def list(self) -> list[KnowledgeBase]:
        """读取当前可用目录。

        Returns:
            按稳定ID顺序排列的知识库。
        """

    def get(self, kb_id: str) -> KnowledgeBase:
        """读取指定知识库。

        Args:
            kb_id: 稳定ID。

        Returns:
            持久化元数据。
        """

    def create(self, knowledge_base: KnowledgeBase) -> KnowledgeBase:
        """创建空知识库，拒绝同ID覆盖。

        Args:
            knowledge_base: 已校验的目录元数据。

        Returns:
            新建的目录记录。
        """

    def update(self, knowledge_base: KnowledgeBase) -> KnowledgeBase:
        """更新已存在的元数据。

        Args:
            knowledge_base: 管理员修改后的目录记录。

        Returns:
            持久化后的最新记录。
        """

    def patch(self, kb_id: str, changes: KnowledgeBasePatch) -> KnowledgeBase:
        """原子更新指定字段，不能读整条记录后覆盖写回。

        Args:
            kb_id: 既有知识库ID。
            changes: 仅包含实际传入字段的冻结更新。

        Returns:
            本次事务提交的最新元数据。
        """
