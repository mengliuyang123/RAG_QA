# -*- coding: utf-8 -*-
"""定义分通道评估的标签和真实执行观察，不把策略混进通道。"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_serializer

from app.domain.conversations import AgentResult
from app.domain.retrieval import RetrievalStrategy
from app.domain.routing import Lane
from app.domain.tracing import TraceSnapshot


LanePolicy = Literal["auto", "fast_only", "agent"]


class LaneTurn(BaseModel):
    """保存一轮人工标签，reference只能用于评分，不能注入执行历史。

    Attributes:
        question: 固定语料上的问题。
        reference: 人工核对的参考答案。
        expected_lane: 自动路由的期望起点，不是升级后的最终通道。
        expected_outcome: 完整线上流程的期望结果。
        expected_sources: 预期来源文件，不能代替语义支持判断。
        requires_retrieval: 是否必须检索。
        rewrite_terms: 实际独立查询应包含的指代对象关键词。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    expected_lane: Lane
    expected_outcome: Literal["answered", "refused", "clarification"] = "answered"
    expected_sources: tuple[str, ...] = ()
    requires_retrieval: bool = True
    rewrite_terms: tuple[str, ...] = ()


class LaneEpisode(BaseModel):
    """同一策略内顺序运行的独立会话片段。

    Attributes:
        id: 稳定案例标识。
        tags: 多轮、对比、补查等能力标签。
        turns: 顺序运行的用户问题。
        top_k: 案例固定的召回上限，None沿用生产配置。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    tags: tuple[str, ...] = Field(min_length=1)
    turns: tuple[LaneTurn, ...] = Field(min_length=1, max_length=10)
    top_k: int | None = Field(default=None, ge=1, le=20)


class LaneObservation(BaseModel):
    """保留实际输出、真实追踪及规则检查，独立裁判另行评分。

    Attributes:
        episode_id: 固定案例标识。
        turn_index: 零起始轮次。
        execution_id: 当前策略片段自己分配的临时图线程。
        policy: 评估策略，与最终lane分开。
        strategy: 原有三种检索策略之一。
        result: 已审核的真实结果，序列化不保存内部补查正文。
        trace: 生产追踪，遵循同一脱敏与用量口径。
        latency_ms: 本轮执行耗时，不包含后续独立评分。
        query_hashes: 实际检索查询指纹，不默认导出查询正文。
        retrieval_compliant: 必须检索及禁止检索的规则是否满足。
        routing_correct: 仅自动策略检查路由起点，强制策略为None。
        rewrite_matches: 有指代标签时检查真实查询，否则为None。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    turn_index: int = Field(ge=0)
    execution_id: UUID
    policy: LanePolicy
    strategy: RetrievalStrategy
    result: AgentResult
    trace: TraceSnapshot
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    query_hashes: tuple[str, ...]
    retrieval_compliant: bool
    routing_correct: bool | None
    rewrite_matches: bool | None

    @field_serializer("result")
    def serialize_result(self, value: AgentResult) -> dict[str, object]:
        """内部证据种子不复制到报告，防止旁路保存补查查询正文。

        Args:
            value: 当前真实已审核结果。

        Returns:
            保留可恢复历史及引用的安全结果。
        """
        return value.model_dump(mode="json", exclude={"execution"})


class LaneJudgement(BaseModel):
    """独立裁判核对语义、引用支持和完整性，不把自审当外部评分。

    Attributes:
        correctness: 与人工参考语义一致的程度，不是模型理解置信度。
        supported: 知识回答是否逐项受引用正文支持；无知识回答时为None。
        complete: 是否覆盖问题要求，拒答不能冒充完整知识答案。
        reason: 可审阅的简短依据，不请求模型思维链。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correctness: float = Field(ge=0, le=1, allow_inf_nan=False, strict=True)
    supported: bool | None = Field(strict=True)
    complete: bool = Field(strict=True)
    reason: str = Field(min_length=1, max_length=500)


class LaneScoreRecord(BaseModel):
    """评分和答案生成分别保存，续跑评分不重新生成答案。

    Attributes:
        episode_id: 被评分片段。
        turn_index: 被评分轮次。
        policy: 被评分的评估策略。
        strategy: 被评分检索策略。
        source_trace_id: 绑定实际被评分的答案执行，避免串评。
        judgement: 独立模型的严格结构化结论。
        trace: 本次评分的真实调用、耗时和用量，不混入生成用量。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    turn_index: int = Field(ge=0)
    policy: LanePolicy
    strategy: RetrievalStrategy
    source_trace_id: UUID
    judgement: LaneJudgement
    trace: TraceSnapshot
