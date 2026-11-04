# -*- coding: utf-8 -*-
"""轮次反馈是用户输入，不作为模型证据或追踪正文。"""

from typing import Literal
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import AwareDatetime
from app.domain.conversations import TurnCitation


FeedbackReason = Literal[
    "incorrect", "incomplete", "irrelevant", "outdated", "citation_wrong",
]


class TurnFeedback(BaseModel):
    """保存一轮当前的反馈，取消时删除整个反馈而非累积事件。

    Attributes:
        rating: 赞、踩或取消。
        reasons: 点踩的有限原因集合。
        comment: 可选人工补充，不传给模型或复制到trace。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rating: Literal["up", "down"] | None
    reasons: tuple[FeedbackReason, ...] = Field(default=(), max_length=5)
    comment: str | None = Field(default=None, max_length=500)


FeedbackTriage = Literal[
    "retrieval_error", "generation_error", "missing_content", "not_an_issue",
]
FeedbackTriageFilter = Literal[
    "retrieval_error", "generation_error", "missing_content", "not_an_issue", "untriaged",
]


class AdminFeedbackItem(BaseModel):
    """管理员单条反馈投影，不提供普通会话越权通道或用户身份。

    Attributes:
        feedback_id: 规范反馈资源标识，一轮只有一份。
        turn_id: 原正式轮次。
        question: 原问题。
        answer: 已保存回答，不重新生成。
        citations: 原引用及编号。
        reasons: 原因。
        comment: 人工备注。
        trace_id: 既有执行追踪。
        revision: 用户反馈版本，防止分类套用到已变化内容。
        triage: 管理员分类。
        created_at: 原轮次创建时间。
        updated_at: 已知反馈修改时间，旧记录未知时为None。
        sort_time: 内部排序边界，不在响应返回。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    feedback_id: UUID
    turn_id: UUID
    question: str
    answer: str
    citations: tuple[TurnCitation, ...]
    reasons: tuple[FeedbackReason, ...]
    comment: str | None
    trace_id: str | None
    revision: int = Field(ge=1)
    triage: FeedbackTriage | None
    created_at: datetime
    updated_at: datetime | None
    sort_time: datetime = Field(exclude=True)


class AdminFeedbackPage(BaseModel):
    """分页的点踩记录，非全会话历史。

    Attributes:
        feedback: 本页记录。
        next_cursor: 下一页游标。
    """

    model_config = ConfigDict(frozen=True)

    feedback: tuple[AdminFeedbackItem, ...]
    next_cursor: str | None = None


class FeedbackPosition(BaseModel):
    """绑定筛选条件的后台反馈游标。

    Attributes:
        sort_time: 上页末条排序时间。
        feedback_id: 同时间记录的唯一标识。
        fingerprint: 当前筛选条件的指纹。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sort_time: AwareDatetime
    feedback_id: UUID
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
