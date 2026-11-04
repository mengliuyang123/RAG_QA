# -*- coding: utf-8 -*-
"""独立后台反馈端口，不绕过普通会话用例的归属检查。"""

from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.domain.feedback import AdminFeedbackItem
from app.domain.feedback import FeedbackTriage
from app.domain.feedback import FeedbackTriageFilter


class FeedbackAdminRepository(Protocol):
    """只允许已授权后台用例读取有限投影或分类。"""

    def list(
        self, limit: int, kb_id: str | None,
        triage: FeedbackTriageFilter | None,
        before: tuple[datetime, UUID] | None,
    ) -> tuple[AdminFeedbackItem, ...]:
        """过滤后读取有限点踩记录。

        Args:
            limit: 上限。
            kb_id: 实际检索范围过滤。
            triage: 分类过滤。
            before: 上页末条排序边界。

        Returns:
            不含用户身份的单轮投影。
        """

    def triage(
        self, feedback_id: UUID, actor_id: UUID, value: FeedbackTriage | None,
        expected_revision: int | None,
    ) -> AdminFeedbackItem:
        """在同一写事务核对当前评分／版本后分类。

        Args:
            feedback_id: 规范反馈资源标识。
            actor_id: 后端验证的管理员。
            value: 分类或清空。
            expected_revision: 可选乐观锁。

        Returns:
            当前分类投影。
        """
