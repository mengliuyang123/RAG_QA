# -*- coding: utf-8 -*-
"""定义独立于HTTP外壳的执行通道、依据与拒答元数据。"""

from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from app.domain.knowledge_bases import validate_kb_ids
from app.domain.routing import Lane


class ScopeInfo(BaseModel):
    """区分用户请求范围和实际检索范围。

    Attributes:
        mode: 自动选库或显式选择。
        requested: 用户选择的ID，自动模式为空。
        searched: 实际检索过的ID并集。
        initial: 首次实际检索范围，旧数据未知为None，明确未检索为空。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    mode: Literal["auto", "explicit"] = "auto"
    requested: tuple[str, ...] = ()
    searched: tuple[str, ...] = ()
    initial: tuple[str, ...] | None = None

    @field_validator("requested", "searched", "initial", mode="before")
    @classmethod
    def restore_arrays(cls, value: object) -> object:
        """仅恢复不可变容器，不改变ID类型。

        Args:
            value: JSON范围数组。

        Returns:
            tuple或待拒绝的非法值。
        """
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def enforce_scope(self) -> Self:
        """结果不能声称显式选择之外的库已经被检索。

        Returns:
            语义一致的范围摘要。
        """
        validate_kb_ids(self.requested)
        validate_kb_ids(self.searched)
        if self.initial is not None:
            validate_kb_ids(self.initial)
            if self.searched and not self.initial:
                raise ValueError("已检索范围不为空时不能声称首次未检索")
            if not set(self.initial).issubset(self.searched):
                raise ValueError("首次实际范围必须包含于实际检索并集")
        if self.mode == "auto" and self.requested:
            raise ValueError("自动模式不应含用户指定ID")
        if self.mode == "explicit":
            if not self.requested or not set(self.searched).issubset(self.requested):
                raise ValueError("显式范围结果越界")
        return self


class EvidenceStats(BaseModel):
    """保存检索、相关、引用和排除四种口径的证据数量。

    相关但不足以回答的块既不计入 used 也不计入 excluded，因此只看
    这两个旧字段会出现"采用0、排除0"的误读；retrieved 与 relevant
    补全口径，旧记录中两者为 None 表示未记录。

    Attributes:
        used: 正文实际引用的块数。
        excluded: 裁判判为不相关的去重块数。
        retrieved: 本轮检索到的去重块数；旧记录为 None。
        relevant: 裁判判为相关的去重块数；旧记录为 None。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    used: int = Field(default=0, ge=0)
    excluded: int = Field(default=0, ge=0)
    retrieved: int | None = Field(default=None, ge=0)
    relevant: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_totals(self) -> Self:
        """新口径必须成对出现，并满足检索数等于相关数加排除数。

        Returns:
            口径一致的统计。
        """
        if (self.retrieved is None) != (self.relevant is None):
            raise ValueError("retrieved 与 relevant 必须同时提供")
        if self.retrieved is not None and self.relevant is not None:
            if self.retrieved != self.relevant + self.excluded:
                raise ValueError("检索数必须等于相关数加排除数")
        return self


class RefusalInfo(BaseModel):
    """展示已找到的资料和缺口，不把缺口推测成事实。

    Attributes:
        reason: 没有相关证据或证据不足。
        found_summary: 不调用额外模型的事实性统计说明。
        missing: 明确的信息或验证缺口。
        suggested_kb_id: 仅有可靠元数据依据时提示其他库。
        escalation_available: 是否配置负责人，未配置时为假。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    reason: Literal["no_evidence", "insufficient_evidence"]
    found_summary: str
    missing: tuple[str, ...] = Field(min_length=1)
    suggested_kb_id: str | None = None
    escalation_available: bool = False


class ClarificationInfo(BaseModel):
    """返回可由用户下一轮回答的具体澄清问题。

    Attributes:
        question: 具体需要用户补充的内容。
        options: 可选项，未可靠识别时为空。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    question: str = Field(min_length=1)
    options: tuple[str, ...] = ()


class AnswerMetadata(BaseModel):
    """两入口和正式历史共用的新增字段，不替换旧答案外壳。

    Attributes:
        lane: 最终通道。
        outcome: 回答、拒答或澄清。
        grounding: 证据依据类别，不是理解置信度。
        kb_scope: 请求与实际范围。
        scope_expanded: 是否发生自动扩查。
        escalated: 是否从Fast进入Agent。
        evidence_stats: 实际使用与排除统计。
        refusal: 拒答时的缺口。
        clarification: 澄清内容。
        trace_id: 后续追踪持久化提供；不伪造尚不存在的追踪。
        feedback: 后续反馈接口提供。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lane: Lane
    outcome: Literal["answered", "refused", "clarification"]
    grounding: Literal["grounded", "partial", "ungrounded"]
    kb_scope: ScopeInfo = Field(default_factory=ScopeInfo)
    scope_expanded: bool = False
    escalated: bool = False
    evidence_stats: EvidenceStats = Field(default_factory=EvidenceStats)
    refusal: RefusalInfo | None = None
    clarification: ClarificationInfo | None = None
    trace_id: str | None = None
    feedback: Literal["up", "down"] | None = None

    @model_validator(mode="after")
    def enforce_result(self) -> Self:
        """拒绝前端无法解释或会越权误导的组合。

        Returns:
            通道和结果一致的元数据。
        """
        if self.escalated and self.lane != "agent":
            raise ValueError("只有Agent结果可标记由Fast升级")
        if self.scope_expanded and self.kb_scope.mode != "auto":
            raise ValueError("显式范围不能标记扩大")
        if (self.outcome == "refused") != (self.refusal is not None):
            raise ValueError("拒答字段与结果不一致")
        if (self.outcome == "clarification") != (self.clarification is not None):
            raise ValueError("澄清字段与结果不一致")
        if self.lane in {"direct", "clarify"} and self.grounding != "ungrounded":
            raise ValueError("无检索通道不能标记有知识库依据")
        return self
