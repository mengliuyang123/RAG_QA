# -*- coding: utf-8 -*-
"""用量报告的窗口、真实计数与非独占知识库参与归属。"""

from typing import Literal

from pydantic import AwareDatetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.domain.trace_queries import CostTotals
from app.domain.trace_queries import TokenTotals


UsageGroup = Literal["user", "kb", "day"]


class UsageTotals(BaseModel):
    """只汇总当前保留的追踪，不声称是供应商完整账单。

    Attributes:
        questions: 纳入窗口的执行请求数，包含失败／取消。
        running_requests: 仍在执行、用量可能继续变化的请求数。
        model_calls: 已记录的全部模型逻辑调用数。
        llm_calls: 排除Embedding与重排后的逻辑调用数。
        tokens: 实际已报告计数与缺失次数。
        cost: 明确价格版本的估计及未知调用数。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    questions: int = Field(ge=0)
    running_requests: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    tokens: TokenTotals
    cost: CostTotals


class UsageRow(UsageTotals):
    """按用户化名、UTC日期或知识库参与范围分组的用量。

    Attributes:
        key: 稳定分组标识，不默认暴露真实用户ID。
        label: 展示标签，不默认展示姓名。
    """

    key: str
    label: str


class UsageReport(BaseModel):
    """分组行不能代替独立去重的总体统计。

    Attributes:
        group_by: 分组维度。
        days: 含今天的UTC日历天数。
        window_started_at: 实际请求窗口开始。
        window_ended_at: 当前统计截止时间。
        retention_days: 当前保留策略。
        retention_limited: 请求窗口超出保留策略时为真。
        coverage: 明确仅覆盖保留的已记录追踪。
        attribution: 精确请求分组或非独占知识库参与归属。
        rows_additive: 是否允许将所有行相加得到总体。
        totals: 每条追踪只累计一次的总体。
        rows: 各分组小计。
        configurations: 各启动配置版本的独立请求小计，可相加。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_by: UsageGroup
    days: int = Field(ge=1, le=365)
    window_started_at: AwareDatetime
    window_ended_at: AwareDatetime
    retention_days: int = Field(ge=1)
    retention_limited: bool
    coverage: Literal["retained_recorded_traces_only"] = "retained_recorded_traces_only"
    attribution: Literal["request", "participating_scope"]
    rows_additive: bool
    totals: UsageTotals
    rows: tuple[UsageRow, ...]
    configurations: tuple[UsageRow, ...]
