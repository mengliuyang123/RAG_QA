# -*- coding: utf-8 -*-
"""追踪只读投影：提交关联来自业务库，未知用量和成本不补零。"""

from typing import Literal
from typing import Self
from uuid import UUID
from decimal import Decimal

from pydantic import AwareDatetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from app.domain.tracing import TraceSnapshot
from app.domain.tracing import CommitStatus
from app.domain.routing import Lane


class TraceFilters(BaseModel):
    """分页前生效的管理员查询条件。

    Attributes:
        kb_id: 实际检索过的知识库。
        lane: 最终执行通道。
        scope_expanded: 是否扩查。
        escalated: 是否升级。
        outcome: 业务结果。
        feedback: 业务库中的当前评分。
        from_at: UTC开始时间，含边界。
        to_at: UTC结束时间，含边界。
        user_id: 服务端已认证的原始用户标识过滤。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kb_id: str | None = Field(default=None, min_length=1, max_length=100)
    lane: Lane | None = None
    scope_expanded: bool | None = None
    escalated: bool | None = None
    outcome: Literal["answered", "refused", "clarification"] | None = None
    feedback: Literal["up", "down"] | None = None
    from_at: AwareDatetime | None = None
    to_at: AwareDatetime | None = None
    user_id: UUID | None = None

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        """反向时间区间属于非法请求而非空结果。

        Returns:
            时间区间一致的条件。
        """
        if self.from_at is not None and self.to_at is not None:
            if self.from_at > self.to_at:
                raise ValueError("开始时间不能晚于结束时间")
        return self


class TracePosition(BaseModel):
    """稳定排序游标，不是认证令牌。

    Attributes:
        created_at: 上一页末条创建时间。
        trace_id: 同时间记录的唯一排序键。
        fingerprint: 绑定查询条件，避免跨过滤条件误用游标。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    created_at: AwareDatetime
    trace_id: UUID
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class TokenTotals(BaseModel):
    """完整总量未知时仍展示明确报告的小计。

    Attributes:
        input: 任意调用未报告输入时为None。
        output: 任意调用未报告输出时为None。
        known_input: 已报告输入的小计，不能冒充完整总量。
        known_output: 已报告输出的小计。
        unknown_input_calls: 未报告输入的调用数。
        unknown_output_calls: 未报告输出的调用数。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: int | None = Field(ge=0)
    output: int | None = Field(ge=0)
    known_input: int = Field(ge=0)
    known_output: int = Field(ge=0)
    unknown_input_calls: int = Field(ge=0)
    unknown_output_calls: int = Field(ge=0)


class CostGroup(BaseModel):
    """同币种、同费率版本的已知估计小计。

    Attributes:
        currency: 币种。
        pricing_version: 显式配置的费率版本。
        amount: 已知调用的估计，不是实际账单。
        calls: 纳入小计的调用数。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    currency: str
    pricing_version: str
    amount: Decimal = Field(ge=0, allow_inf_nan=False)
    calls: int = Field(ge=1)


class CostTotals(BaseModel):
    """不混合币种或丢弃未知计费调用。

    Attributes:
        status: 全部可估计、部分可估计或全部未知。
        estimates: 各币种／费率版本的已知小计。
        unknown_calls: 无法估价的调用数。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["estimated", "partial", "unknown"]
    estimates: tuple[CostGroup, ...]
    unknown_calls: int = Field(ge=0)


class TraceRecord(BaseModel):
    """同一只读查询内取得的追踪与正式轮次关联。

    Attributes:
        snapshot: 已通过期限检查的执行快照。
        turn_id: 业务库中实际存在的正式轮次。
        conversation_id: 正式轮次所属会话。
        owner_id: 内部关联核对使用，不直接展示身份。
        feedback: 当前反馈，不复制备注。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot: TraceSnapshot
    turn_id: UUID | None = None
    conversation_id: UUID | None = None
    owner_id: UUID | None = None
    feedback: Literal["up", "down"] | None = None


class TraceDetail(TraceSnapshot):
    """供已鉴权管理员查看的执行详情，不默认泄漏提问者身份。

    Attributes:
        turn_available: 正式轮次目前是否仍存在。
        feedback: 当前赞或踩。
        model_calls: 全部供应商调用数，包含向量与重排。
        llm_calls: 排除向量与重排后的语言模型调用数。
        tokens: 全部调用的实际用量与未知数。
        cost: 分币种／版本的估计，不是实际账单。
    """

    user_id: UUID | None = Field(default=None, exclude=True)
    turn_available: bool = False
    feedback: Literal["up", "down"] | None = None
    model_calls: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    tokens: TokenTotals
    cost: CostTotals


class TraceSummary(BaseModel):
    """后台列表只提供有限概要，细节由单条接口读取。

    Attributes:
        trace_id: 追踪标识。
        created_at: UTC开始时间。
        entry: 执行入口。
        lane_policy: 线上执行或固定评估策略，不与最终通道混算。
        question: 仅显式开启正文保存时存在。
        question_sha256: 问题指纹，不代表匿名化保证。
        lane: 最终通道。
        outcome: 回答、拒答或澄清。
        scope_expanded: 是否扩查。
        escalated: 是否升级。
        status: 模型执行状态。
        commit_status: 正式轮次提交状态。
        turn_id: 已确认提交的轮次。
        turn_available: 轮次当前是否仍存在。
        feedback: 当前评分。
        latency_ms: 执行耗时，不伪造传输耗时。
        admission_status: 实际准入结果或旧记录的not_recorded。
        queue_wait_ms: 单独计量的队列等待，未记录时为None。
        execution_duration_ms: 出队后实际用例耗时，未记录时为None。
        llm_calls: 语言模型调用数。
        model_calls: 全部供应商调用数。
        tokens: 实际计数汇总。
        cost: 明确版本费率的估计汇总。
    """

    model_config = ConfigDict(frozen=True)

    trace_id: UUID
    created_at: AwareDatetime
    entry: Literal["query", "conversation", "evaluation"]
    lane_policy: Literal["online", "fast_only"] = "online"
    question: str | None
    question_sha256: str
    lane: Lane | None
    outcome: Literal["answered", "refused", "clarification"] | None
    scope_expanded: bool
    escalated: bool
    status: Literal["running", "completed", "error", "cancelled"]
    commit_status: CommitStatus
    turn_id: UUID | None
    turn_available: bool
    feedback: Literal["up", "down"] | None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    admission_status: str = "not_recorded"
    queue_wait_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    execution_duration_ms: float | None = Field(
        default=None, ge=0, allow_inf_nan=False,
    )
    llm_calls: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    tokens: TokenTotals
    cost: CostTotals


class TracePage(BaseModel):
    """管理员分页列表，不返回用户身份或完整步骤。

    Attributes:
        traces: 本页追踪概要。
        next_cursor: 还有下一页时的不透明游标。
    """

    model_config = ConfigDict(frozen=True)

    traces: tuple[TraceSummary, ...]
    next_cursor: str | None = None
