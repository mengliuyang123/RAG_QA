# -*- coding: utf-8 -*-
"""声明会话与无状态外壳共用的高层问题执行端口。"""

from typing import Protocol
from typing import Literal
from uuid import UUID

from app.application.conversations.ports import StatusCallback
from app.domain.conversations import AgentResult
from app.domain.conversations import ConversationTurn
from app.domain.retrieval import RetrievalStrategy
from app.domain.routing import KnowledgeScope


class QuestionExecutor(Protocol):
    """高层端口不暴露图种子，种子只在知识执行器内部交接。"""

    def run(
        self, conversation_id: UUID, question: str,
        history: list[ConversationTurn], on_status: StatusCallback | None = None,
        *, kb_scope: KnowledgeScope | None = None,
        requested_lane: str = "auto",
        strategy: RetrievalStrategy | str | None = None,
        top_k: int | None = None, debug: bool = False,
        execution_entry: Literal["query", "conversation", "evaluation"] = "conversation",
        user_id: UUID | None = None,
        lane_policy: Literal["online", "fast_only"] = "online",
    ) -> AgentResult:
        """执行统一的理解、路由与审核链路。

        Args:
            conversation_id: 执行或正式会话标识。
            question: 原始问题。
            history: token窗口选取后的历史。
            on_status: 安全阶段回调。
            kb_scope: 用户范围。
            requested_lane: 管理检索测试的起点。
            strategy: 可选检索策略。
            top_k: 可选结果上限。
            debug: 不影响审核的调试选项。
            execution_entry: 后端确定的入口，供追踪与评估归因。
            user_id: 已认证的后端身份，不能取自用户问题或请求字段。
            lane_policy: 非线上策略仅允许评估入口使用。

        Returns:
            可发布、可持久化的已验证结果。
        """
