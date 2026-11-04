# -*- coding: utf-8 -*-
"""真实模型用量、版本价格与可追踪费用估计的领域契约。"""

from decimal import Decimal
from typing import Literal
from typing import Self
from uuid import UUID
from uuid import uuid4
from urllib.parse import urlsplit

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from pydantic import JsonValue
from pydantic import field_validator
from pydantic import AwareDatetime

from app.domain.routing import Lane


CommitStatus = Literal[
    "not_applicable", "unknown", "pending", "committed", "failed", "cancelled",
]


class TokenUsage(BaseModel):
    """只接受供应商报告的Token数，None表示未报告，不是零。

    Attributes:
        input: 实际输入Token数。
        output: 实际输出Token数；无生成的Embedding为零。
        cached_input: 输入中供应商明确报告的缓存Token数。
        reasoning_output: 输出中供应商明确报告的思考Token数。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: int | None = Field(default=None, ge=0, strict=True)
    output: int | None = Field(default=None, ge=0, strict=True)
    cached_input: int | None = Field(default=None, ge=0, strict=True)
    reasoning_output: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def validate_subtotals(self) -> Self:
        """缓存及思考不能超过各自总量。

        Returns:
            计数一致的用量。
        """
        if self.input is not None and self.cached_input is not None:
            if self.cached_input > self.input:
                raise ValueError("缓存计数不能超过输入总量")
        if self.output is not None and self.reasoning_output is not None:
            if self.reasoning_output > self.output:
                raise ValueError("思考计数不能超过输出总量")
        return self


class ModelPrice(BaseModel):
    """显式配置的价格版本，不代表自动查询到的最新官方报价。

    Attributes:
        origin: 价格适用的服务来源，不能跨服务复用同名模型价格。
        model: 供应商模型标识。
        version: 可审计的价格版本。
        currency: 三字母币种，同一金额不混合币种。
        input_per_million: 每百万非缓存输入Token费率。
        output_per_million: 每百万输出Token费率。
        cached_input_per_million: 可选独立缓存输入费率。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    origin: str = Field(min_length=1)
    model: str = Field(min_length=1)
    version: str = Field(min_length=1)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    input_per_million: Decimal = Field(ge=0, allow_inf_nan=False)
    output_per_million: Decimal = Field(ge=0, allow_inf_nan=False)
    cached_input_per_million: Decimal | None = Field(
        default=None, ge=0, allow_inf_nan=False,
    )

    @field_validator("origin")
    @classmethod
    def validate_origin(cls, value: str) -> str:
        """价格按服务来源匹配，不接受凭据或请求路径。

        Args:
            value: 显式配置的服务来源。

        Returns:
            不带尾斜杠的来源。
        """
        parsed = urlsplit(value)
        # urllib读取port时才会拒绝非法端口，不能把它落成unknown来源费率。
        parsed.port
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("价格来源必须是HTTP服务来源")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("价格来源不能包含凭据或查询参数")
        if parsed.path not in {"", "/"}:
            raise ValueError("价格来源不能包含API路径")
        return value.rstrip("/")


class CostEstimate(BaseModel):
    """区分按价格表计算的估计与无法确定的费用。

    Attributes:
        status: estimated不是实际账单；unknown不能被当成免费。
        amount: 可确定时的估计金额。
        currency: 估计币种。
        pricing_version: 实际使用的价格版本。
        reason: 未知原因。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["estimated", "unknown"]
    amount: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    currency: str | None = None
    pricing_version: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_estimate(self) -> Self:
        """未知金额不能伪造为零，已知估计必须可追溯。

        Returns:
            语义明确的费用。
        """
        if self.status == "estimated":
            if self.amount is None or not self.currency or not self.pricing_version:
                raise ValueError("费用估计必须包含金额、币种和价格版本")
            if self.reason is not None:
                raise ValueError("费用估计不应包含未知原因")
        elif self.amount is not None or not self.reason:
            raise ValueError("未知费用应说明原因且不含金额")
        return self


def estimate_cost(usage: TokenUsage, price: ModelPrice | None) -> CostEstimate:
    """使用真实计数和明确费率计算，不推测未知缓存或缺失usage。

    Args:
        usage: 供应商返回的计数。
        price: 精确匹配来源和模型后的价格，缺失时为None。

    Returns:
        可审计的金额估计或明确未知状态。
    """
    if usage.input is None or usage.output is None:
        return CostEstimate(
            status="unknown", reason="usage_unavailable",
            pricing_version=price.version if price is not None else None,
            currency=price.currency if price is not None else None,
        )
    if price is None:
        return CostEstimate(status="unknown", reason="price_unavailable")
    cached = 0
    cache_rate = price.input_per_million
    if price.cached_input_per_million is not None:
        if usage.cached_input is None:
            return CostEstimate(
                status="unknown", reason="cache_usage_unavailable",
                pricing_version=price.version, currency=price.currency,
            )
        cached = usage.cached_input
        cache_rate = price.cached_input_per_million
    amount = (
        (usage.input - cached) * price.input_per_million
        + cached * cache_rate
        + usage.output * price.output_per_million
    ) / Decimal(1_000_000)
    return CostEstimate(
        status="estimated", amount=amount, currency=price.currency,
        pricing_version=price.version,
    )


class TraceStep(BaseModel):
    """保存单次实际执行步骤，而非从最终结果倒推时间线。

    Attributes:
        id: 单步标识。
        order: 开始顺序，嵌套步骤结束顺序可能不同。
        type: 应用提供的步骤类型。
        started_at: UTC开始时间。
        duration_ms: 单调时钟测量的耗时。
        status: 运行、完成、失败或取消。
        input: 按隐私策略过滤后的输入元数据。
        output: 来源编号、分数和判断等过滤后结果。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    order: int = Field(ge=1)
    type: str = Field(min_length=1)
    started_at: AwareDatetime
    duration_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    status: Literal["running", "completed", "error", "cancelled"] = "running"
    input: dict[str, JsonValue] = Field(default_factory=dict)
    output: dict[str, JsonValue] = Field(default_factory=dict)


class ModelCallTrace(BaseModel):
    """保存一次真实上游调用，不将提示词估算当作供应商usage。

    Attributes:
        id: 调用标识。
        purpose: 调用用途，例如理解、证据裁判或Embedding。
        origin: 去除路径、凭据和查询参数后的服务来源。
        model: 请求的模型标识。
        response_model: 供应商明确返回的模型版本。
        started_at: UTC开始时间。
        duration_ms: 单调时钟测量的请求耗时。
        status: 调用完成、失败或取消。
        usage: 供应商实际报告的计数。
        cost: 对应配置价格版本的估计或未知状态。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    purpose: str = Field(min_length=1)
    origin: str
    model: str
    response_model: str | None = None
    started_at: AwareDatetime
    duration_ms: float = Field(ge=0, allow_inf_nan=False)
    status: Literal["completed", "error", "cancelled"]
    usage: TokenUsage
    cost: CostEstimate


class TraceSnapshot(BaseModel):
    """保存请求的独立追踪快照，不复制完整历史或检索正文。

    Attributes:
        trace_id: 回答对象关联的追踪标识。
        request_id: 本次执行标识，不与业务会话标识混用。
        entry: 无状态、会话或评估入口。
        conversation_id: 可选正式会话标识。
        turn_id: 可选已提交的正式轮次标识。
        user_id: 后续鉴权接入时由后端填写的归属。
        question: 显式启用内容保存时的原问题。
        question_sha256: 查询指纹；不是不可逆匿名化保证。
        created_at: 开始时间。
        expires_at: 按配置得出的保留截止时间。
        status: 执行状态。
        commit_status: 正式轮次提交状态，与模型执行状态分开。
        lane_policy: 线上或评估固定快速通道策略。
        duration_ms: 本轮执行耗时。
        admission_status: 线上准入结果；旧记录或内部执行为not_recorded。
        queue_wait_ms: 实际等待时长，未记录时为None而非伪造零。
        execution_duration_ms: 出队后的用例耗时，不包括等待和网络传输。
        steps: 按开始顺序排列的实际步骤。
        calls: 实际上游调用与计数。
        lane: 最终通道。
        outcome: 最终业务结果。
        scope_expanded: 是否扩查。
        escalated: 是否升级。
        configuration_version: 不含凭据的执行配置版本。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: UUID
    request_id: UUID
    entry: Literal["query", "conversation", "evaluation"]
    conversation_id: UUID | None = None
    turn_id: UUID | None = None
    user_id: UUID | None = None
    question: str | None = None
    question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: AwareDatetime
    expires_at: AwareDatetime
    status: Literal["running", "completed", "error", "cancelled"] = "running"
    commit_status: CommitStatus = "not_applicable"
    lane_policy: Literal["online", "fast_only"] = "online"
    duration_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    admission_status: Literal[
        "not_recorded", "checking", "queued", "admitted", "rate_limited",
        "queue_full", "queue_timeout", "cancelled", "error",
    ] = "not_recorded"
    queue_wait_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    execution_duration_ms: float | None = Field(
        default=None, ge=0, allow_inf_nan=False,
    )
    steps: tuple[TraceStep, ...] = ()
    calls: tuple[ModelCallTrace, ...] = ()
    lane: Lane | None = None
    outcome: Literal["answered", "refused", "clarification"] | None = None
    scope_expanded: bool = False
    escalated: bool = False
    configuration_version: str = "layered-routing-v1"

    @model_validator(mode="before")
    @classmethod
    def restore_legacy_commit_status(cls, value: object) -> object:
        """旧会话追踪没有提交证据时只能标记未知。

        Args:
            value: 待恢复的快照。

        Returns:
            增补旧记录未知状态的快照数据。
        """
        if isinstance(value, dict) and value.get("entry") == "conversation":
            if "commit_status" not in value:
                return {**value, "commit_status": "unknown"}
        return value

    @model_validator(mode="after")
    def validate_commit_status(self) -> Self:
        """仅已提交的会话执行能声称关联了正式轮次。

        Returns:
            状态一致的冻结快照。
        """
        if self.lane_policy != "online" and self.entry != "evaluation":
            raise ValueError("固定通道策略仅允许评估追踪")
        if self.entry != "conversation":
            if self.commit_status != "not_applicable" or self.turn_id is not None:
                raise ValueError("无状态入口不得声明正式轮次提交")
        elif self.commit_status == "not_applicable":
            raise ValueError("会话执行必须区分提交状态")
        if (self.commit_status == "committed") != (self.turn_id is not None):
            raise ValueError("只有已提交状态可关联正式轮次")
        return self
