# -*- coding: utf-8 -*-
"""知识缺口状态的存储端口。"""

from typing import Protocol

from app.domain.knowledge_gaps import KnowledgeGap


class KnowledgeGapRepository(Protocol):
    """只保存管理员处理状态，缺口本身仍来自拒答与反馈洞察。"""

    def list(self, limit: int) -> list[KnowledgeGap]:
        """按最近修改倒序列出。

        Args:
            limit: 上限。

        Returns:
            缺口列表。
        """

    def save(self, gap: KnowledgeGap) -> KnowledgeGap:
        """按 gap_key 插入或整体替换。

        Args:
            gap: 已校验的缺口状态。

        Returns:
            已保存的记录。
        """
