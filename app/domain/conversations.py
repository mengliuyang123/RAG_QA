# -*- coding: utf-8 -*-
"""多轮会话、正式轮次与可信引用的领域模型。"""

from datetime import datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from app.domain.execution import EvidenceBundle
from app.domain.answer_metadata import AnswerMetadata
from app.domain.knowledge_bases import DEFAULT_KB_ID
from app.domain.routing import KnowledgeScope

KNOWLEDGE_REFUSAL_ANSWER = "根据现有知识库，我无法可靠回答这个问题"


class AnswerMode(StrEnum):
    """标识回答是否使用了本轮知识库检索。"""

    DIRECT = "direct"
    RETRIEVAL = "retrieval"


class ConversationStatus(StrEnum):
    """会话在正常使用或可重试删除流程中的状态。"""

    ACTIVE = "active"
    DELETING = "deleting"


class TurnCitation(BaseModel):
    """保存一次正式回答中由证据表映射出的可信来源。"""

    model_config = ConfigDict(frozen=True)

    position: int = Field(ge=1)
    doc_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    page: int | None
    chunk_text: str = Field(min_length=1)
    chunk_id: str | None = None
    kb_id: str = DEFAULT_KB_ID
    doc_updated_at: datetime | None = None


class AgentResult(BaseModel):
    """保存 ConversationAgent 已验证、可持久化的最终结果。"""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)
    answer_mode: AnswerMode
    retrieval_calls: int = Field(ge=0, le=3)
    citations: list[TurnCitation]
    execution: EvidenceBundle | None = None
    metadata: AnswerMetadata | None = None
    quality_issues: tuple[str, ...] = ()
    # 证据不足仍作答或审核未通过时为真，只能以部分依据发布。
    partial: bool = False

    @model_validator(mode="after")
    def validate_mode_and_citations(self) -> Self:
        """校验回答模式、工具调用次数与引用顺序的不变量。"""
        if self.answer_mode is AnswerMode.DIRECT and self.retrieval_calls != 0:
            raise ValueError("direct 回答不得包含检索调用")
        if (
            self.answer_mode is AnswerMode.RETRIEVAL
            and self.retrieval_calls < 1
        ):
            raise ValueError("retrieval 回答必须至少执行一次检索")
        positions = [citation.position for citation in self.citations]
        if len(positions) != len(set(positions)):
            raise ValueError("引用位置不得重复")
        if self.metadata is not None:
            if self.answer_mode is AnswerMode.DIRECT:
                if self.metadata.grounding != "ungrounded":
                    raise ValueError("直接回答不能声明知识库依据")
            elif self.metadata.outcome == "answered" and not self.citations:
                raise ValueError("知识回答必须有已校验引用")
            elif self.metadata.outcome == "answered" and self.metadata.grounding != (
                "partial" if self.partial else "grounded"
            ):
                raise ValueError("知识回答的依据类别必须与是否完整通过审核一致")
        return self


class ConversationTurn(BaseModel):
    """保存一轮已完成且通过验证的正式用户/助手消息。"""

    model_config = ConfigDict(frozen=True)

    id: UUID
    conversation_id: UUID
    user_message: str = Field(min_length=1)
    assistant_message: str = Field(min_length=1)
    answer_mode: AnswerMode
    retrieval_calls: int = Field(ge=0, le=3)
    citations: list[TurnCitation]
    latency_ms: float = Field(ge=0)
    created_at: datetime
    metadata: AnswerMetadata | None = None


class ConversationSummary(BaseModel):
    """保存会话列表需要的轻量元数据。"""

    model_config = ConfigDict(frozen=True)

    id: UUID
    title: str = Field(min_length=1)
    status: ConversationStatus
    created_at: datetime
    updated_at: datetime
    turns_count: int = Field(ge=0)
    kb_scope: KnowledgeScope = Field(default_factory=KnowledgeScope)
    owner_id: UUID | None = None


class Conversation(ConversationSummary):
    """保存会话元数据及其全部正式轮次。"""

    turns: list[ConversationTurn]


class ConversationQueryResult(BaseModel):
    """保存一次会话查询向 HTTP 层公开的完整结果。"""

    model_config = ConfigDict(frozen=True)

    conversation_id: UUID
    turn_id: UUID
    answer: str = Field(min_length=1)
    answer_mode: AnswerMode
    retrieval_calls: int = Field(ge=0, le=3)
    citations: list[TurnCitation]
    latency_ms: float = Field(ge=0)
    metadata: AnswerMetadata | None = None
    created_at: datetime | None = None
