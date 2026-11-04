# -*- coding: utf-8 -*-
"""会话 SSE 只允许公开的类型化安全事件。"""

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.domain.conversations import ConversationQueryResult


class StatusConversationData(BaseModel):
    """保存 Agent 当前可公开阶段及可选检索次数。"""

    model_config = ConfigDict(frozen=True)

    phase: Literal[
        "deciding", "retrieving", "validating", "verifying", "generating",
        "expanding", "escalating",
        "understanding",
        "queued",
    ]
    attempt: int | None = Field(default=None, ge=1, le=3)
    kb_ids: tuple[str, ...] = ()
    label: str | None = None
    position: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None,
    )


class TokenConversationData(BaseModel):
    """保存已验证最终回答的一个非空文本分片。"""

    model_config = ConfigDict(frozen=True)

    token: str = Field(min_length=1)


class ErrorConversationData(BaseModel):
    """保存不泄露内部实现的稳定流式错误。"""

    model_config = ConfigDict(frozen=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class StatusConversationEvent(BaseModel):
    """公开 Agent 阶段变化。"""

    model_config = ConfigDict(frozen=True)

    event: Literal["status"] = "status"
    data: StatusConversationData


class TokenConversationEvent(BaseModel):
    """公开已经校验的最终回答分片。"""

    model_config = ConfigDict(frozen=True)

    event: Literal["token"] = "token"
    data: TokenConversationData


class CompleteConversationEvent(BaseModel):
    """公开已持久化轮次的唯一成功终止事件。"""

    model_config = ConfigDict(frozen=True)

    event: Literal["complete"] = "complete"
    data: ConversationQueryResult


class ErrorConversationEvent(BaseModel):
    """公开唯一失败终止事件。"""

    model_config = ConfigDict(frozen=True)

    event: Literal["error"] = "error"
    data: ErrorConversationData


ConversationStreamEvent = (
    StatusConversationEvent
    | TokenConversationEvent
    | CompleteConversationEvent
    | ErrorConversationEvent
)
