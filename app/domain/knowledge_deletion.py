# -*- coding: utf-8 -*-
"""知识库删除计划只保存受控资源身份，不复制正文或用户会话。"""

from typing import Literal
from typing import Self
from uuid import UUID

from pydantic import AwareDatetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from app.domain.knowledge_bases import KB_ID_PATTERN


class KnowledgeDeletionItem(BaseModel):
    """重试清理仍须知道已删除索引的精确源文件身份。

    Attributes:
        doc_id: 属于本计划知识库的文档UUID。
        filename: 已通过受控源路径校验的文件名。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    doc_id: UUID
    filename: str = Field(min_length=1, max_length=255)


class KnowledgeDeletionPlan(BaseModel):
    """跨索引、文件和目录的持久恢复事实，不声称跨库原子事务。

    Attributes:
        kb_id: 已冻结接收入库的知识库。
        force: 首次请求是否明确授权批量删除文档。
        state: deleting为未完成，deleted才是完成事实。
        items: 首次确认的完整资源身份，重试不能扩大范围。
        started_at: 首次持久提交计划的UTC时间。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    kb_id: str = Field(pattern=KB_ID_PATTERN)
    force: bool
    state: Literal["deleting", "deleted"] = "deleting"
    items: tuple[KnowledgeDeletionItem, ...] = ()
    started_at: AwareDatetime

    @model_validator(mode="after")
    def validate_identities(self) -> Self:
        """拒绝重复身份或非force计划夹带文档。

        Returns:
            可安全恢复的冻结计划。
        """
        identifiers = {item.doc_id for item in self.items}
        if len(identifiers) != len(self.items):
            raise ValueError("删除计划文档身份重复")
        if self.items and not self.force:
            raise ValueError("文档删除计划必须有force授权")
        return self


class KnowledgeBaseDeleteResult(BaseModel):
    """仅当计划源文件、索引和可用目录均已清理才返回成功。

    Attributes:
        kb_id: 已逻辑删除的知识库。
        deleted: 确认已完成，不用于表示尝试。
        documents_deleted: 原持久计划中已完成清理的文档数。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    kb_id: str = Field(pattern=KB_ID_PATTERN)
    deleted: Literal[True] = True
    documents_deleted: int = Field(ge=0)
