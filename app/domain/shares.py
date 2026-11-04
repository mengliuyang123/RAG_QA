# -*- coding: utf-8 -*-
"""登录团队内单轮分享的冻结契约。"""

from uuid import UUID

from pydantic import AwareDatetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.domain.conversations import TurnCitation


class ShareRequest(BaseModel):
    """限定链接有效期，不接受调用者指定身份。

    Attributes:
        expires_in_days: 有限有效天数，默认30天。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    expires_in_days: int = Field(default=30, ge=1, le=365, strict=True)


class ShareLink(BaseModel):
    """只在签发响应返回明文链接。

    Attributes:
        token: 高熵令牌，不写入存储和日志。
        url: 前端展示路由的相对地址。
        expires_at: UTC绝对截止时间。
    """

    model_config = ConfigDict(frozen=True)
    token: str = Field(repr=False)
    url: str = Field(repr=False)
    expires_at: AwareDatetime


class StoredShare(BaseModel):
    """只保存授权投影，不复制问答正文。

    Attributes:
        token_hash: SHA256摘要。
        turn_id: 正式轮次标识。
        owner_id: 签发时的已认证身份。
        shared_by: 签发时的展示名。
        created_at: 分享创建时间。
        expires_at: 链接失效时间。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    token_hash: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)
    turn_id: UUID
    owner_id: UUID
    shared_by: str = Field(min_length=1, max_length=100)
    created_at: AwareDatetime
    expires_at: AwareDatetime


class SharedTurn(BaseModel):
    """只公开被主动分享的一轮及其原始引用。

    Attributes:
        question: 原始问题。
        answer: 已审核的正式答案。
        citations: 原编号与历史来源，不根据当前索引重写。
        shared_by: 分享者展示名。
        created_at: 分享创建时间。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    question: str
    answer: str
    citations: tuple[TurnCitation, ...]
    shared_by: str
    created_at: AwareDatetime
