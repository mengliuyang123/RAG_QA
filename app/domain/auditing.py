# -*- coding: utf-8 -*-
"""后台敏感数据访问审计只保存操作者、动作和资源标识。"""

from typing import Literal
from uuid import UUID
from uuid import uuid4

from pydantic import AwareDatetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.domain.knowledge_bases import KB_ID_PATTERN


class AdminAccessAudit(BaseModel):
    """不复制问题、回答、评论或凭据的访问记录。

    Attributes:
        id: 独立审计标识。
        actor_id: 后端认证的操作者，仅内部保存。
        action: 读取或分类尝试，不把尝试伪称业务提交。
        resource_kind: 访问对象类别。
        resource_ids: 本次访问的资源标识，空页允许为空。
        knowledge_base_id: 知识库删除尝试对应的稳定非UUID标识。
        created_at: UTC访问时间。
        expires_at: 按明确保留策略设置的截止时间。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    actor_id: UUID
    action: Literal["read", "triage_attempt", "delete_attempt"]
    resource_kind: Literal[
        "feedback", "refusals", "overview", "trace", "document_chunks", "knowledge_base",
    ]
    resource_ids: tuple[UUID, ...] = Field(default=(), max_length=100)
    knowledge_base_id: str | None = Field(default=None, pattern=KB_ID_PATTERN)
    created_at: AwareDatetime
    expires_at: AwareDatetime
