# -*- coding: utf-8 -*-
"""知识库访问控制、文档时效与标准答案的领域契约。"""

from dataclasses import dataclass
from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.domain.knowledge_bases import KB_ID_PATTERN


KbMemberRole = Literal["viewer", "editor"]


class KbMember(BaseModel):
    """知识库成员。

    Attributes:
        user_id: 用户标识。
        role: viewer 可查看提问；editor 还可管理该库内容。
        username: 展示用登录名（读取时填充）。
        name: 展示名（读取时填充）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: UUID
    role: KbMemberRole
    username: str | None = None
    name: str | None = None


class KbAccess(BaseModel):
    """一个知识库的访问设置。

    Attributes:
        kb_id: 知识库。
        restricted: True 时只有成员和管理员可见；False 时所有登录用户可见。
        members: 成员列表；公开库也可以有 editor。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kb_id: str = Field(pattern=KB_ID_PATTERN)
    restricted: bool = False
    members: tuple[KbMember, ...] = Field(default=(), max_length=500)


@dataclass(frozen=True)
class KbPermissions:
    """某个用户对知识库的有效权限。

    Attributes:
        unrestricted: 管理员或离线评估，可见可管全部。
        restricted: 受限知识库集合。
        member_of: 用户所在的知识库。
        editable: 用户是 editor 的知识库。
        deny_all: 禁用或不存在的账号，什么都看不到。
    """

    unrestricted: bool = False
    deny_all: bool = False
    restricted: frozenset[str] = frozenset()
    member_of: frozenset[str] = frozenset()
    editable: frozenset[str] = frozenset()

    def can_view(self, kb_id: str) -> bool:
        """是否可查看并提问：公开库，或自己是成员的受限库。"""
        if self.deny_all:
            return False
        return self.unrestricted or kb_id not in self.restricted or kb_id in self.member_of

    def can_edit(self, kb_id: str) -> bool:
        """是否可管理该库的内容。"""
        return not self.deny_all and (self.unrestricted or kb_id in self.editable)

    def allowed(self, kb_ids: tuple[str, ...] | list[str]) -> frozenset[str] | None:
        """在给定目录中可见的部分；不受限时返回None。"""
        if self.unrestricted:
            return None
        return frozenset(kb_id for kb_id in kb_ids if self.can_view(kb_id))


class DocumentReview(BaseModel):
    """文档复审信息。

    Attributes:
        doc_id: 文档。
        kb_id: 所属知识库。
        review_due_on: 下次复审日期。
        outdated: 已过时；已过时文档不参与检索。
        note: 说明。
        updated_at: 修改时间。
        updated_by: 修改者。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(min_length=1, max_length=64)
    kb_id: str = Field(pattern=KB_ID_PATTERN)
    review_due_on: date | None = None
    outdated: bool = False
    note: str | None = Field(default=None, max_length=500)
    updated_at: AwareDatetime
    updated_by: UUID


class FaqEntry(BaseModel):
    """管理员维护的标准答案。

    Attributes:
        id: 条目标识。
        kb_id: 所属知识库。
        question: 标准问题。
        answer: 标准答案。
        created_at: 创建时间。
        updated_at: 修改时间。
        updated_by: 修改者。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    kb_id: str = Field(pattern=KB_ID_PATTERN)
    question: str = Field(min_length=1, max_length=500)
    answer: str = Field(min_length=1, max_length=5000)
    created_at: AwareDatetime
    updated_at: AwareDatetime
    updated_by: UUID
