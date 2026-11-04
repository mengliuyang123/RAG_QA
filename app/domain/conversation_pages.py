# -*- coding: utf-8 -*-
"""会话前端分页的不可变结果，不承载SQL游标实现。"""

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.domain.conversations import Conversation
from app.domain.conversations import ConversationSummary


class ConversationPage(BaseModel):
    """保存本人会话摘要页及未分页的总数。

    Attributes:
        conversations: 当前页摘要，最近更新在前。
        total: 当前用户全部active会话数量。
        next_cursor: 下一页位置；末页为None。
    """

    model_config = ConfigDict(frozen=True)

    conversations: tuple[ConversationSummary, ...]
    total: int = Field(ge=0)
    next_cursor: str | None = None


class ConversationDetailPage(Conversation):
    """保存有限历史页，页内按时间正序展示。

    Attributes:
        turns_next_cursor: 更早一页的位置；末页为None。
    """

    turns_next_cursor: str | None = None
