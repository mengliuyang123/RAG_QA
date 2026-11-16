# -*- coding: utf-8 -*-
"""知识缺口工作台的处理状态：拒答聚类或点踩问题 → 补充文档 → 复测验证。"""

from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.domain.knowledge_bases import KB_ID_PATTERN


GapStatus = Literal["open", "document_added", "verified"]
GapSource = Literal["refusal_cluster", "feedback"]
# cluster:<拒答聚类ID> 或 feedback:<反馈ID>，由前端从现有洞察接口得到。
GAP_KEY_PATTERN = r"^(cluster|feedback):[A-Za-z0-9_-]{1,128}$"
GapQuestion = str
MAX_GAP_QUESTIONS = 10


class GapCheckResult(BaseModel):
    """一条样例问题的复测结果。

    Attributes:
        question: 复测的原问题。
        outcome: 问答接口返回的结果类型。
        answered: 是否得到有依据的回答。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(min_length=1, max_length=8000)
    outcome: Literal["answered", "refused", "clarification", "error"]
    answered: bool


class GapCheck(BaseModel):
    """最近一次复测。

    Attributes:
        checked_at: 复测完成时间。
        kb_id: 复测时限定的知识库，None表示自动路由。
        results: 每条样例问题的结果。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    checked_at: AwareDatetime
    kb_id: str | None = Field(default=None, pattern=KB_ID_PATTERN)
    results: tuple[GapCheckResult, ...] = Field(min_length=1, max_length=MAX_GAP_QUESTIONS)

    @property
    def all_answered(self) -> bool:
        """全部样例问题都得到有依据的回答。"""
        return all(result.answered for result in self.results)


class KnowledgeGap(BaseModel):
    """一条被跟踪的知识缺口。

    Attributes:
        gap_key: 来源稳定键。
        source: 拒答聚类或点踩反馈。
        title: 代表问题。
        questions: 用于复测的样例问题。
        kb_id: 应补充文档的知识库。
        status: 处理状态。
        doc_id: 为此缺口补充的文档。
        last_check: 最近一次复测。
        updated_at: 最近修改时间。
        updated_by: 最近修改的管理员。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    gap_key: str = Field(pattern=GAP_KEY_PATTERN)
    source: GapSource
    title: str = Field(min_length=1, max_length=8000)
    questions: tuple[GapQuestion, ...] = Field(min_length=1, max_length=MAX_GAP_QUESTIONS)
    kb_id: str | None = Field(default=None, pattern=KB_ID_PATTERN)
    status: GapStatus
    doc_id: str | None = Field(default=None, max_length=64)
    last_check: GapCheck | None = None
    updated_at: AwareDatetime
    updated_by: UUID
