# -*- coding: utf-8 -*-
"""持久删除计划与入队冻结的事务端口。"""

from typing import Protocol

from app.domain.knowledge_deletion import KnowledgeDeletionPlan


class KnowledgeDeletionRepository(Protocol):
    """同一目录库事务保存计划，并阻止新任务引用正在删除的知识库。"""

    def get(self, kb_id: str) -> KnowledgeDeletionPlan | None:
        """读取既有计划，不将存储故障当作不存在。

        Args:
            kb_id: 稳定知识库ID。

        Returns:
            原计划，首次请求尚无计划时为None。
        """

    def begin(self, plan: KnowledgeDeletionPlan) -> KnowledgeDeletionPlan:
        """检查非终态任务并原子冻结新入库，先于任何索引删除。

        Args:
            plan: 已验证元数据和源身份的首次计划。

        Returns:
            实际持久化的计划，重试沿用原身份。
        """

    def complete(self, plan: KnowledgeDeletionPlan) -> None:
        """用例确认资源清理完后，原子提交目录与计划完成状态。

        Args:
            plan: 当前正在恢复的同一计划。

        Returns:
            None。
        """
