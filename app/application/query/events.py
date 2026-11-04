# -*- coding: utf-8 -*-
"""流式知识库查询使用的应用事件模型。"""

from typing import Literal
from datetime import datetime

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.domain.answers import Citation
from app.domain.retrieval import RetrievalStrategy
from app.domain.answer_metadata import AnswerMetadata
from app.domain.conversations import AnswerMode
from app.domain.query_debug import RetrievalDebugItem
from app.application.conversations.events import StatusConversationData


class TokenStreamData(BaseModel):
    """保存单个回答增量。

    Attributes:
        token: 单个非空回答增量。
    """

    model_config = ConfigDict(frozen=True)
    token: str = Field(min_length=1)


class CompleteStreamData(BaseModel):
    """保存流式查询正常结束时的完整结果。

    Attributes:
        answer: 生成的完整回答。
        citations: 回答引用的可信来源。
        latency_ms: 从查询开始到完成的耗时（毫秒）。
        retrieval_strategy: 本次查询采用的检索策略。
    """

    model_config = ConfigDict(frozen=True)
    answer: str
    citations: list[Citation]
    latency_ms: float = Field(ge=0)
    retrieval_strategy: RetrievalStrategy
    answer_mode: AnswerMode | None = None
    retrieval_calls: int = Field(default=0, ge=0, le=3)
    metadata: AnswerMetadata | None = None
    created_at: datetime | None = None
    retrieved: tuple[RetrievalDebugItem, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )


class ErrorStreamData(BaseModel):
    """保存流开始后的稳定错误信息。

    Attributes:
        code: 稳定的错误代码。
        message: 面向调用方的错误消息。
    """

    model_config = ConfigDict(frozen=True)
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class TokenStreamEvent(BaseModel):
    """保存携带一个回答增量的事件。

    Attributes:
        event: 固定为 ``token`` 的事件名称。
        data: 回答增量负载。
    """

    model_config = ConfigDict(frozen=True)
    event: Literal["token"] = "token"
    data: TokenStreamData


class CompleteStreamEvent(BaseModel):
    """保存一次正常流的唯一终止事件。

    Attributes:
        event: 固定为 ``complete`` 的事件名称。
        data: 完整结果负载。
    """

    model_config = ConfigDict(frozen=True)
    event: Literal["complete"] = "complete"
    data: CompleteStreamData


class ErrorStreamEvent(BaseModel):
    """保存一次失败流的唯一终止事件。

    Attributes:
        event: 固定为 ``error`` 的事件名称。
        data: 稳定错误负载。
    """

    model_config = ConfigDict(frozen=True)
    event: Literal["error"] = "error"
    data: ErrorStreamData


class StatusStreamEvent(BaseModel):
    """查询和会话使用相同的安全阶段模型，不发送推理内容。"""

    model_config = ConfigDict(frozen=True)
    event: Literal["status"] = "status"
    data: StatusConversationData


StreamEvent = TokenStreamEvent | CompleteStreamEvent | ErrorStreamEvent | StatusStreamEvent
