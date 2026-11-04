# -*- coding: utf-8 -*-
"""多轮会话应用层依赖的基础设施端口。"""

from collections.abc import Callable
from collections.abc import Iterator
from typing import Protocol
from uuid import UUID

from app.domain.conversations import AgentResult
from app.domain.conversations import Conversation
from app.domain.conversations import ConversationSummary
from app.domain.conversations import ConversationTurn
from app.domain.execution import EvidenceBundle
from app.domain.retrieval import RetrievalStrategy
from app.domain.routing import KnowledgeScope
from app.domain.feedback import TurnFeedback
from app.domain.conversation_pages import ConversationPage
from app.domain.conversation_pages import ConversationDetailPage


StatusCallback = Callable[[str, int | None], None]


class ConversationRepository(Protocol):
    """定义正式会话历史的权威持久化操作。"""

    def put_feedback(
        self, turn_id: UUID, owner_id: UUID, feedback: TurnFeedback,
    ) -> None:
        """在同一事务检查归属并覆盖反馈。

        Args:
            turn_id: 正式轮次。
            owner_id: 后端认证的用户。
            feedback: 当前选择，rating=None时清除。

        Returns:
            None。
        """

    def get_feedback(self, turn_id: UUID, owner_id: UUID) -> TurnFeedback:
        """只返回当前用户的轮次反馈。

        Args:
            turn_id: 正式轮次。
            owner_id: 后端认证的用户。

        Returns:
            当前反馈，越权或不存在抛出轮次不存在异常。
        """

    def create(
        self, kb_scope: KnowledgeScope | None = None, *,
        owner_id: UUID | None = None,
    ) -> Conversation:
        """创建并返回一个空会话。

        Args:
            kb_scope: 会话默认知识库范围，未指定时自动选库。
            owner_id: 由后端验证的创建者；旧离线调用可不指定。

        Returns:
            已保存的空会话。
        """

    def update(
        self,
        conversation_id: UUID,
        *,
        title: str | None = None,
        kb_scope: KnowledgeScope | None = None,
    ) -> ConversationSummary:
        """原子修改会话标题和默认范围。

        Args:
            conversation_id: 会话标识。
            title: 可选新标题。
            kb_scope: 可选新默认范围。

        Returns:
            修改后的摘要。
        """

    def list_active(
        self, *, owner_id: UUID | None = None,
    ) -> list[ConversationSummary]:
        """按归属过滤后返回摘要，不加载历史或引用。

        Args:
            owner_id: 已验证用户；None仅供可信离线调用。

        Returns:
            最近更新在前的active会话。
        """

    def get(self, conversation_id: UUID) -> Conversation:
        """返回会话及其全部正式轮次。"""

    def list_active_page(
        self, owner_id: UUID, limit: int, cursor: str | None,
    ) -> ConversationPage:
        """在同一快照内按归属读取摘要页和总数。

        Args:
            owner_id: 后端认证用户。
            limit: 每页1–100条。
            cursor: 同范围下一页位置。

        Returns:
            本人active会话页及总数。
        """

    def get_page(
        self, conversation_id: UUID, owner_id: UUID,
        limit: int, cursor: str | None,
    ) -> ConversationDetailPage:
        """同一快照验证归属后只加载当前历史页。

        Args:
            conversation_id: 目标会话。
            owner_id: 后端认证用户。
            limit: 每页1–100轮。
            cursor: 更早一页的位置。

        Returns:
            摘要及页内正序的轮次和引用。
        """

    def get_summary(self, conversation_id: UUID) -> ConversationSummary:
        """只读取状态检查需要的摘要。

        Args:
            conversation_id: 待检查会话标识。

        Returns:
            不加载轮次或引用的会话摘要。
        """

    def recent_turns(
        self,
        conversation_id: UUID,
        limit: int,
    ) -> list[ConversationTurn]:
        """按正序返回最近若干轮正式历史。"""

    def save_turn(
        self,
        conversation_id: UUID,
        question: str,
        result: AgentResult,
        latency_ms: float,
    ) -> ConversationTurn:
        """原子保存成功轮次、引用与会话聚合字段。"""

    def iter_turns_reverse(
        self, conversation_id: UUID, batch_size: int = 32,
    ) -> Iterator[ConversationTurn]:
        """分页倒序读取历史，分页大小不是历史窗口轮数上限。

        Args:
            conversation_id: 会话标识。
            batch_size: 每次读取的记录上限。

        Yields:
            最新在前的完整历史轮次。
        """

    def mark_deleting(self, conversation_id: UUID) -> None:
        """把目标会话标记为可幂等重试的 deleting。"""

    def delete(self, conversation_id: UUID) -> None:
        """删除业务会话并级联清理轮次和引用。"""


class ConversationCheckpointStore(Protocol):
    """定义应用删除流程所需的最小 checkpoint 操作。"""

    def delete_thread(self, conversation_id: UUID) -> None:
        """删除会话对应的全部图 checkpoint。"""


class ConversationAgent(Protocol):
    """定义应用服务调用 Agentic RAG 的稳定边界。"""

    def run(
        self,
        conversation_id: UUID,
        question: str,
        history: list[ConversationTurn],
        on_status: StatusCallback | None = None,
        *,
        seed: EvidenceBundle | None = None,
        kb_ids: tuple[str, ...] | None = None,
        strategy: RetrievalStrategy | str | None = None,
        top_k: int | None = None,
        initial_query: str | None = None,
        sub_queries: tuple[str, ...] = (),
    ) -> AgentResult:
        """运行必须使用证据的 Agent，复用升级时的种子与预算。

        Args:
            conversation_id: 本轮所属会话。
            question: 原始问题或已消解指代的问题。
            history: 按上下文预算选中的正式历史。
            on_status: 实时阶段回调。
            seed: Fast 已完成的检索与审核快照。
            kb_ids: 服务端确定且不可由工具改写的范围。
            strategy: 本轮统一使用的检索策略。
            top_k: 本轮统一使用的结果上限。
            initial_query: 理解层已经改写的首次独立查询。
            sub_queries: 最多三个候选子查询，不代表执行命令。

        Returns:
            已验证的回答、来源及累计执行状态。
        """
