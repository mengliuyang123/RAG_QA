# -*- coding: utf-8 -*-
"""定义问题理解、知识库范围与确定性执行通道。"""

from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


Intent = Literal["chitchat", "meta", "knowledge", "ambiguous", "out_of_scope"]
Complexity = Literal["single_hop", "multi_hop", "comparison", "aggregation"]
Lane = Literal["direct", "clarify", "fast", "agent"]


class QueryUnderstanding(BaseModel):
    """保存可校验的问题理解，不允许模型直接指定执行通道。

    Attributes:
        intent: 问题意图。
        complexity: 检索任务复杂度。
        standalone_query: 用于检索的独立问题，不替代原始用户消息。
        sub_queries: 最多三个候选子查询，不代表必须全部执行。
        clarification: 意图明确为模糊问题时的具体澄清问题。
        confidence: 分类把握，仅用于路由及内部追踪。
        preferred_kb_ids: 自动选库的候选 ID，目录和显式范围仍由程序校验。
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False,
    )

    intent: Intent = "knowledge"
    complexity: Complexity = "single_hop"
    standalone_query: str = Field(min_length=1)
    sub_queries: tuple[str, ...] = Field(default=(), max_length=3)
    clarification: str | None = None
    confidence: float = Field(ge=0, le=1)
    preferred_kb_ids: tuple[str, ...] = Field(default=(), max_length=3)

    @field_validator("sub_queries", "preferred_kb_ids", mode="before")
    @classmethod
    def accept_json_array(cls, value: object) -> object:
        """只转换 JSON 数组容器，不把非法元素强制转为字符串。

        Args:
            value: 原始数组或 Python tuple。

        Returns:
            不可变容器或供 Pydantic 拒绝的原始输入。
        """
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """拒绝空白查询及与意图矛盾的澄清字段。

        Returns:
            通过业务校验的理解结果。
        """
        if not self.standalone_query.strip():
            raise ValueError("独立查询不能为空白")
        for query in self.sub_queries:
            if not query.strip():
                raise ValueError("子查询不能为空白")
        if self.intent == "ambiguous":
            if self.clarification is None or not self.clarification.strip():
                raise ValueError("模糊问题必须提供具体澄清问题")
        elif self.clarification is not None:
            raise ValueError("非模糊意图不得夹带澄清问题")
        return self

    @classmethod
    def fallback(cls, question: str) -> "QueryUnderstanding":
        """低置信度或非法输出时回退为必须检索的简单问题。

        Args:
            question: 原问题或可安全保留的独立查询。

        Returns:
            knowledge/single_hop 结果，不保留未经信任的子查询。
        """
        return cls(standalone_query=question, confidence=0.0)


class KnowledgeScope(BaseModel):
    """区分自动选库和用户明确指定的查询范围。

    Attributes:
        mode: auto 或 explicit。
        kb_ids: explicit 指定的非空知识库集合；auto 必须为空。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    mode: Literal["auto", "explicit"] = "auto"
    kb_ids: tuple[str, ...] = ()

    @field_validator("kb_ids", mode="before")
    @classmethod
    def accept_json_array(cls, value: object) -> object:
        """接受 HTTP JSON 数组并保持内部不可变。

        Args:
            value: 知识库 ID 数组。

        Returns:
            不转换元素类型的 tuple。
        """
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        """范围是否合法存在由应用校验，这里保证其含义不含糊。

        Returns:
            语义明确的范围。
        """
        if self.mode == "auto" and self.kb_ids:
            raise ValueError("auto 模式不能指定知识库 ID")
        if self.mode == "explicit" and not self.kb_ids:
            raise ValueError("explicit 模式必须指定知识库")
        if len(set(self.kb_ids)) != len(self.kb_ids):
            raise ValueError("知识库 ID 不得重复")
        for kb_id in self.kb_ids:
            if not kb_id.strip() or kb_id != kb_id.strip():
                raise ValueError("知识库 ID 不得为空或包含首尾空白")
        return self


class RoutingDecision(BaseModel):
    """保存路由的执行起点，而不是检索策略或最终通道。

    Attributes:
        understanding: 路由实际采用的理解结果，可能已经安全回退。
        lane: direct、clarify、fast 或 agent。
        reason: 可公开的简短路由原因，不是模型思维链。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    understanding: QueryUnderstanding
    lane: Lane
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_knowledge_retrieval(self) -> Self:
        """知识意图不得手工构造为不检索的决策。

        Returns:
            满足强制检索规则的决策。
        """
        if self.understanding.intent == "knowledge":
            if self.lane not in {"fast", "agent"}:
                raise ValueError("knowledge 必须走检索通道")
        if self.lane == "clarify" and not self.understanding.clarification:
            raise ValueError("clarify 通道必须有具体澄清问题")
        return self
