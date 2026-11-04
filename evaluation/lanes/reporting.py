# -*- coding: utf-8 -*-
"""分通道统计只聚合成功证据，缺失分母与评分费用分别展示。"""

import hashlib
import statistics
from datetime import UTC
from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.application.tracing.totals import cost_totals
from app.application.tracing.totals import token_totals
from app.domain.retrieval import RetrievalStrategy
from app.domain.routing import Lane
from app.domain.trace_queries import CostTotals
from app.domain.trace_queries import TokenTotals
from app.domain.tracing import ModelCallTrace
from evaluation.lanes.checkpoints import LaneRunManifest
from evaluation.lanes.models import LaneEpisode
from evaluation.lanes.models import LaneObservation
from evaluation.lanes.models import LanePolicy
from evaluation.lanes.models import LaneScoreRecord
from evaluation.lanes.models import LaneTurn
from evaluation.metrics import LatencySummary
from evaluation.metrics import summarize_latency


TurnKey = tuple[str, LanePolicy, int]


class Metric(BaseModel):
    """指标和实际分母必须一起解释；不适用或缺失时不补零。

    Attributes:
        value: 比率或均值，分母为空时为None。
        n: 实际被评价的轮次数。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    n: int = Field(ge=0)


class ResourceTotals(BaseModel):
    """只汇总本类别实际成功记录，费用不是账单。

    Attributes:
        model_calls: 全部供应商调用次数，包含向量和重排。
        llm_calls: 排除向量与重排的语言模型调用次数。
        tokens: 已知小计和未知调用数。
        cost: 分币种／价格版本估计，未知不填零。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_calls: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    tokens: TokenTotals
    cost: CostTotals


class PolicySummary(BaseModel):
    """策略或标签子集的统计，不把不同分母的分数混在一起。

    Attributes:
        expected: 本次选择的计划轮次数。
        generated: 已成功生成并审核的轮次数。
        scored: 已成功独立评分的轮次数。
        final_lanes: 实际最终通道分布，与评估策略不同。
        metrics: 每项指标及自己的实际分母。
        latency_ms: 成功生成执行耗时，不包含独立评分。
        generation: 被评估链路的实际调用汇总。
        judging: 外部裁判的实际调用汇总。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected: int = Field(ge=0)
    generated: int = Field(ge=0)
    scored: int = Field(ge=0)
    final_lanes: dict[Lane, int]
    metrics: dict[str, Metric]
    latency_ms: LatencySummary | None
    generation: ResourceTotals
    judging: ResourceTotals


class LaneReport(BaseModel):
    """小样本报告不覆盖旧latest，也不复制完整追踪、提示词或证据正文。

    Attributes:
        schema_version: 独立报告协议版本。
        manifest: 精确运行指纹与选择。
        created_at: 汇总时间。
        purpose: 连通烟雾或能力评估，不相互冒充。
        status: 缺失生成或评分时为incomplete。
        expected_turns: 计划轮次数。
        generated_turns: 成功生成数。
        scored_turns: 成功独立评分数。
        strategies: 仍然只有三种检索策略，不加入fast或agent。
        lanes: 评估策略分别统计，内含最终通道分布。
        knowledge_lanes: 只含知识案例，供相同问题集的策略比较。
        by_tag: 各能力标签统计，重叠标签行不可相加。
        cost_coverage: 仅汇总成功结果；失败尝试需查隔离trace库。
        issues: 当前执行中遇到的安全错误分类，不复制上游异常正文。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["lanes-v1"] = "lanes-v1"
    manifest: LaneRunManifest
    created_at: datetime
    purpose: Literal["connectivity_smoke", "capability_evaluation"]
    status: Literal["complete", "incomplete"]
    expected_turns: int
    generated_turns: int
    scored_turns: int
    strategies: tuple[RetrievalStrategy, ...]
    lanes: dict[LanePolicy, PolicySummary]
    knowledge_lanes: dict[LanePolicy, PolicySummary]
    by_tag: dict[str, dict[LanePolicy, PolicySummary]]
    cost_coverage: Literal["successful_results_and_scores"] = "successful_results_and_scores"
    issues: tuple["LaneRunIssue", ...] = ()


class LaneRunIssue(BaseModel):
    """未完成步骤的稳定分类，不制造默认零分。

    Attributes:
        episode_id: 出现问题的固定案例。
        policy: 对应评估策略。
        turn_index: 当前失败轮次，生成阶段可能还没有结果。
        stage: 生成或独立评分阶段。
        error_category: 无异常正文的稳定分类。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    policy: LanePolicy
    turn_index: int = Field(ge=0)
    stage: Literal["generation", "scoring"]
    error_category: str


def build_lane_report(
    manifest: LaneRunManifest, episodes: tuple[LaneEpisode, ...],
    observations: tuple[LaneObservation, ...], scores: tuple[LaneScoreRecord, ...],
    *, smoke: bool, issues: tuple[LaneRunIssue, ...] = (),
) -> LaneReport:
    """按计划分母汇总，不把失败／未运行样本悄悄排除出覆盖率。

    Args:
        manifest: 已绑定运行配置的选择。
        episodes: 固定人工标签。
        observations: 成功生成记录，允许不完整。
        scores: 成功独立评分记录，允许不完整。
        smoke: 是否仅为连通性烟雾。
        issues: 本次执行遇到的稳定错误，不能用完整样本数掩盖失败。

    Returns:
        缺失状态明确、可交叉比较且不复制正文的冻结报告。
    """
    catalog = {episode.id: episode for episode in episodes}
    planned: dict[TurnKey, LaneTurn] = {}
    tags: dict[str, set[TurnKey]] = {}
    for episode_id, policy in manifest.selections:
        if episode_id not in catalog:
            raise ValueError("运行案例不属于当前标签集")
        episode = catalog[episode_id]
        for index, turn in enumerate(episode.turns):
            key = (episode_id, policy, index)
            if key in planned:
                raise ValueError("计划选择重复")
            planned[key] = turn
            for tag in episode.tags:
                tags.setdefault(tag, set()).add(key)
    observed = {}
    for observation in observations:
        key = (observation.episode_id, observation.policy, observation.turn_index)
        if key in observed:
            raise ValueError("生成记录重复")
        if key not in planned or observation.strategy != manifest.strategy:
            raise ValueError("生成记录不属于本次案例或策略")
        expected_hash = hashlib.sha256(planned[key].question.encode()).hexdigest()
        if observation.trace.question_sha256 != expected_hash:
            raise ValueError("生成问题与人工标签不对应")
        observed[key] = observation
    judged = {}
    for score in scores:
        key = (score.episode_id, score.policy, score.turn_index)
        if key in judged:
            raise ValueError("独立评分重复")
        if (
            key not in observed or score.strategy != manifest.strategy
            or observed[key].trace.trace_id != score.source_trace_id
        ):
            raise ValueError("评分与实际答案不对应")
        judged[key] = score
    lanes = _summaries(planned, observed, judged)
    knowledge = {}
    for key, turn in planned.items():
        if turn.requires_retrieval:
            knowledge[key] = turn
    by_tag = {}
    for tag in sorted(tags):
        subset = {}
        for key in tags[tag]:
            subset[key] = planned[key]
        by_tag[tag] = _summaries(subset, observed, judged)
    complete = len(observed) == len(planned) == len(judged) and not issues
    return LaneReport(
        manifest=manifest, created_at=datetime.now(UTC),
        purpose="connectivity_smoke" if smoke else "capability_evaluation",
        status="complete" if complete else "incomplete",
        expected_turns=len(planned), generated_turns=len(observed), scored_turns=len(judged),
        strategies=(manifest.strategy,), lanes=lanes,
        knowledge_lanes=_summaries(knowledge, observed, judged), by_tag=by_tag,
        issues=issues,
    )


def _summaries(
    planned: dict[TurnKey, LaneTurn], observed: dict[TurnKey, LaneObservation],
    judged: dict[TurnKey, LaneScoreRecord],
) -> dict[LanePolicy, PolicySummary]:
    results = {}
    for policy in ("auto", "fast_only", "agent"):
        keys = []
        for key in planned:
            if key[1] == policy:
                keys.append(key)
        if not keys:
            continue
        numbers: dict[str, list[float]] = {
            name: [] for name in (
                "correctness", "supported", "complete", "routing_accuracy",
                "retrieval_compliance", "outcome_accuracy", "citation_source_recall",
                "refusal_precision", "refusal_recall", "clarification_accuracy",
                "rewrite_matches", "retry_rate",
            )
        }
        final_lanes = {}
        latencies = []
        generation_calls = []
        judging_calls = []
        generated = 0
        scored = 0
        for key in keys:
            if key not in observed:
                continue
            observation = observed[key]
            turn = planned[key]
            metadata = observation.result.metadata
            if metadata is None:
                raise ValueError("生成结果缺少通道元数据")
            generated += 1
            final_lanes[metadata.lane] = final_lanes.get(metadata.lane, 0) + 1
            latencies.append(observation.latency_ms)
            generation_calls.extend(observation.trace.calls)
            numbers["retrieval_compliance"].append(float(observation.retrieval_compliant))
            numbers["outcome_accuracy"].append(float(metadata.outcome == turn.expected_outcome))
            if observation.routing_correct is not None:
                numbers["routing_accuracy"].append(float(observation.routing_correct))
            if observation.rewrite_matches is not None:
                numbers["rewrite_matches"].append(float(observation.rewrite_matches))
            if turn.requires_retrieval:
                numbers["retry_rate"].append(float(observation.result.retrieval_calls > 1))
            if turn.expected_sources:
                sources = {citation.filename for citation in observation.result.citations}
                expected = set(turn.expected_sources)
                numbers["citation_source_recall"].append(len(sources & expected) / len(expected))
            if metadata.outcome == "refused":
                numbers["refusal_precision"].append(float(turn.expected_outcome == "refused"))
            if turn.expected_outcome == "refused":
                numbers["refusal_recall"].append(float(metadata.outcome == "refused"))
            if turn.expected_outcome == "clarification":
                numbers["clarification_accuracy"].append(
                    float(metadata.outcome == "clarification"),
                )
            if key in judged:
                score = judged[key]
                scored += 1
                judging_calls.extend(score.trace.calls)
                numbers["correctness"].append(score.judgement.correctness)
                numbers["complete"].append(float(score.judgement.complete))
                if score.judgement.supported is not None:
                    numbers["supported"].append(float(score.judgement.supported))
        metrics = {}
        for name, values in numbers.items():
            metrics[name] = Metric(
                value=statistics.fmean(values) if values else None, n=len(values),
            )
        results[policy] = PolicySummary(
            expected=len(keys), generated=generated, scored=scored,
            final_lanes=final_lanes, metrics=metrics,
            latency_ms=summarize_latency(latencies) if latencies else None,
            generation=_resources(tuple(generation_calls)),
            judging=_resources(tuple(judging_calls)),
        )
    return results


def _resources(calls: tuple[ModelCallTrace, ...]) -> ResourceTotals:
    return ResourceTotals(
        model_calls=len(calls),
        llm_calls=sum(call.purpose not in {"embedding", "reranking"} for call in calls),
        tokens=token_totals(calls), cost=cost_totals(calls),
    )
