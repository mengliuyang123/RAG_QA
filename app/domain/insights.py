# -*- coding: utf-8 -*-
"""后台拒答洞察的有限单轮投影，不包含提问者身份。"""

from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.domain.answer_metadata import RefusalInfo
from app.domain.answer_metadata import AnswerMetadata
from app.domain.routing import Lane


class RefusalItem(BaseModel):
    """当前正式历史中具有明确拒答元数据的单轮。

    Attributes:
        turn_id: 正式轮次，不因此获得普通会话读取权限。
        question: 从已有正式轮次读取的原问题。
        created_at: 正式提交时间。
        trace_id: 可选历史追踪，过期后可能不可读取。
        kb_ids: 实际检索范围，不是模型推测的责任归属。
        lane: 最终执行通道。
        grounding: 拒答时的证据状态。
        refusal: 当时保存的缺口，不重新调用模型。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn_id: UUID
    question: str
    created_at: AwareDatetime
    trace_id: UUID | None = None
    kb_ids: tuple[str, ...]
    lane: Lane
    grounding: Literal["grounded", "partial", "ungrounded"]
    refusal: RefusalInfo


class RefusalPage(BaseModel):
    """游标分页的拒答列表，不暗示覆盖没有明确元数据的旧历史。

    Attributes:
        refusals: 本页明确拒答轮次。
        next_cursor: 下一页边界，无下一页时为None。
        coverage: 明确仅覆盖当前active正式会话的已提交拒答。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    refusals: tuple[RefusalItem, ...]
    next_cursor: str | None = None
    coverage: Literal[
        "active_committed_explicit_refusals"
    ] = "active_committed_explicit_refusals"


class RefusalPosition(BaseModel):
    """绑定知识库筛选条件的稳定时间／主键边界。

    Attributes:
        created_at: 上页最后一条的提交时间。
        turn_id: 同时间的唯一排序键。
        fingerprint: 筛选条件指纹，不是授权凭据。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    created_at: AwareDatetime
    turn_id: UUID
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class InsightMetric(BaseModel):
    """仅用于聚合的单轮元数据投影，不含用户标识、问题或答案。

    Attributes:
        owner_known: 是否记录了服务器身份。
        user_count_increment: 同一只读窗口内该身份只出现一次的计数贡献。
        metadata: 已提交的结果元数据；旧记录缺失时为None。
        feedback: 当前评分，不读取评论或原因正文。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    owner_known: bool
    user_count_increment: int = Field(ge=0, le=1, strict=True)
    metadata: AnswerMetadata | None
    feedback: Literal["up", "down"] | None


class InsightTotals(BaseModel):
    """只统计当前active正式历史，不包括测试或未提交尝试。

    Attributes:
        questions: 窗口内已提交轮次，每轮仅计一次。
        active_users: 已知服务器身份的去重数量。
        unknown_owner_questions: 旧无归属轮次，不猜测为一个用户。
        unknown_metadata_questions: 无法判定旧结果与范围的轮次。
        no_retrieval_questions: 元数据明确没有检索参与库的轮次。
        unknown_scope_questions: 有知识结果元数据但未保存范围的旧轮次。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    questions: int = Field(ge=0)
    active_users: int = Field(ge=0)
    unknown_owner_questions: int = Field(ge=0)
    unknown_metadata_questions: int = Field(ge=0)
    no_retrieval_questions: int = Field(ge=0)
    unknown_scope_questions: int = Field(ge=0)


class KnowledgeInsight(BaseModel):
    """知识库参与请求的实际计数及分母明确的比率。

    Attributes:
        kb_id: 实际参与检索的知识库或当前空目录。
        questions: 参与请求数，跨库请求可参与多行。
        refused_questions: 其中已保存拒答结果的请求数。
        thumbs_down_questions: 当前点踩的请求数，取消或修改后会更新。
        known_initial_auto_questions: 首次自动范围明确包含本库的请求数。
        expanded_questions: 上述已知首次请求中实际扩大范围的数量。
        unknown_initial_auto_questions: 参与本库但首次自动范围未知的旧请求。
        refusal_rate: 拒答数／参与请求数，零分母为None。
        thumbs_down_rate: 点踩数／参与请求数，不按已评分请求作分母。
        expanded_rate: 扩查数／已知首次自动请求数，存在未知分母则None。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kb_id: str
    questions: int = Field(ge=0)
    refused_questions: int = Field(ge=0)
    thumbs_down_questions: int = Field(ge=0)
    known_initial_auto_questions: int = Field(ge=0)
    expanded_questions: int = Field(ge=0)
    unknown_initial_auto_questions: int = Field(ge=0)
    refusal_rate: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    thumbs_down_rate: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    expanded_rate: float | None = Field(ge=0, le=1, allow_inf_nan=False)


class InsightOverview(BaseModel):
    """UTC日历窗口中的正式问答概览，不把日志保留期当业务保留期。

    Attributes:
        period_days: 包含今天的日历天数。
        window_started_at: UTC窗口起点，含边界。
        window_ended_at: 本次统计截止，含边界。
        totals: 全局去重计数。
        knowledge_bases: 当前目录与历史参与库的计数。
        rows_additive: 所有行的questions是否可相加等于全局questions。
        coverage: 只统计当前active正式会话的提交事实。
        attribution: 知识库行是参与归属，不是独占责任归属。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    period_days: int = Field(ge=1, le=365)
    window_started_at: AwareDatetime
    window_ended_at: AwareDatetime
    totals: InsightTotals
    knowledge_bases: tuple[KnowledgeInsight, ...]
    rows_additive: bool
    coverage: Literal["active_committed_conversation_turns"] = "active_committed_conversation_turns"
    attribution: Literal["participating_scope"] = "participating_scope"
