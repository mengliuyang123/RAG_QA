# -*- coding: utf-8 -*-
"""只操作本人正式轮次的反馈，不调用模型或修改已审核的正文。"""

from uuid import UUID

from app.application.conversations.ports import ConversationRepository
from app.domain.feedback import TurnFeedback


class FeedbackService:
    """通过同一业务数据库事务实现归属检查和幂等覆盖。"""

    def __init__(self, repository: ConversationRepository) -> None:
        """保存权威轮次存储端口。

        Args:
            repository: 与会话服务共享的业务存储。

        Returns:
            None。
        """
        self._repository = repository

    def put(self, turn_id: UUID, owner_id: UUID, feedback: TurnFeedback) -> None:
        """覆盖本人的一条反馈，取消时同时清理原因和备注。

        Args:
            turn_id: 正式轮次标识。
            owner_id: 后端已认证身份。
            feedback: 经过校验的当前选择。

        Returns:
            None。
        """
        self._repository.put_feedback(turn_id, owner_id, feedback)

    def get(self, turn_id: UUID, owner_id: UUID) -> TurnFeedback:
        """读取本人的当前反馈，不能以管理员身份越权。

        Args:
            turn_id: 正式轮次标识。
            owner_id: 后端已认证身份。

        Returns:
            当前反馈，无反馈时rating为None。
        """
        return self._repository.get_feedback(turn_id, owner_id)
