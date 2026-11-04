# -*- coding: utf-8 -*-
"""与外部服务无关的本地评估指标。"""

import math
import statistics

from pydantic import BaseModel, ConfigDict, Field

from app.domain.answers import Citation, RetrievedChunkTrace
from evaluation.models import ExpectedSource


class CaseTrace(BaseModel):
    """计算检索和引用指标所需的一道题追踪。

    Attributes:
        expected_sources: 人工标注的一个或多个期望来源。
        retrieved: 实际交给生成器的有序检索文本块。
        citations: 最终回答实际使用的可信引用。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_sources: list[ExpectedSource] = Field(min_length=1)
    retrieved: list[RetrievedChunkTrace]
    citations: list[Citation]


class RetrievalMetrics(BaseModel):
    """一组 case 的 Hit Rate 与 MRR。

    Attributes:
        hit_rate: 前 k 条至少命中一个期望来源的 case 比例。
        mrr: 首个命中来源倒数排名的 case 均值。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    hit_rate: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)


class CitationMetrics(BaseModel):
    """一组 case 的宏平均引用精确率与召回率。

    Attributes:
        precision: 每题有效引用比例的平均值。
        recall: 每题被引用期望来源比例的平均值。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)


class LatencySummary(BaseModel):
    """一组毫秒延迟的均值与线性插值分位数。

    Attributes:
        mean: 算术平均延迟。
        p50: 第 50 百分位延迟。
        p95: 第 95 百分位延迟。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: float = Field(ge=0.0)
    p50: float = Field(ge=0.0)
    p95: float = Field(ge=0.0)


def compute_retrieval_metrics(
    cases: list[CaseTrace],
    k: int,
) -> RetrievalMetrics:
    """计算固定截断深度的 Hit Rate 与 MRR。

    Args:
        cases: 带人工期望来源的查询追踪。
        k: 参与命中判断的最大检索名次。

    Returns:
        所有 case 的 Hit Rate 和 MRR。

    Raises:
        ValueError: case 为空或 k 不是正整数。
    """
    _validate_cases(cases)
    if k < 1:
        raise ValueError("k 必须为正整数")

    hits = 0
    reciprocal_ranks: list[float] = []
    for case in cases:
        first_match_rank = 0
        for rank, retrieved in enumerate(case.retrieved[:k], start=1):
            if _matches_any_expected(retrieved, case.expected_sources):
                first_match_rank = rank
                break
        if first_match_rank > 0:
            hits += 1
            reciprocal_ranks.append(1.0 / first_match_rank)
        else:
            reciprocal_ranks.append(0.0)
    return RetrievalMetrics(
        hit_rate=hits / len(cases),
        mrr=statistics.fmean(reciprocal_ranks),
    )


def compute_citation_metrics(cases: list[CaseTrace]) -> CitationMetrics:
    """计算逐题引用精确率和召回率的宏平均。

    Args:
        cases: 带人工期望来源和实际引用的查询追踪。

    Returns:
        宏平均 Citation Precision 与 Citation Recall。

    Raises:
        ValueError: case 列表为空。
    """
    _validate_cases(cases)
    precisions: list[float] = []
    recalls: list[float] = []
    for case in cases:
        matching_citations = 0
        recalled_sources: set[tuple[str, int | None]] = set()
        for citation in case.citations:
            matching_sources = _matching_expected_sources(
                citation,
                case.expected_sources,
            )
            if matching_sources:
                matching_citations += 1
                for source in matching_sources:
                    recalled_sources.add((source.filename, source.page))

        precision = 0.0
        if case.citations:
            precision = matching_citations / len(case.citations)
        expected_identities = {
            (source.filename, source.page)
            for source in case.expected_sources
        }
        recall = len(recalled_sources) / len(expected_identities)
        precisions.append(precision)
        recalls.append(recall)
    return CitationMetrics(
        precision=statistics.fmean(precisions),
        recall=statistics.fmean(recalls),
    )


def summarize_latency(values: list[float]) -> LatencySummary:
    """汇总有限非负毫秒值的均值、p50 和 p95。

    Args:
        values: 一项或多项毫秒延迟。

    Returns:
        使用线性插值计算的延迟汇总。

    Raises:
        ValueError: 列表为空或包含非有限、负数值。
    """
    if not values:
        raise ValueError("延迟至少包含一个值")
    for value in values:
        if not math.isfinite(value) or value < 0:
            raise ValueError("延迟必须是有限非负数")

    ordered = sorted(values)
    return LatencySummary(
        mean=statistics.fmean(ordered),
        p50=_linear_percentile(ordered, 0.50),
        p95=_linear_percentile(ordered, 0.95),
    )


def _validate_cases(cases: list[CaseTrace]) -> None:
    if not cases:
        raise ValueError("至少包含一个 case")


def _matches_any_expected(
    item: RetrievedChunkTrace | Citation,
    expected_sources: list[ExpectedSource],
) -> bool:
    return bool(_matching_expected_sources(item, expected_sources))


def _matching_expected_sources(
    item: RetrievedChunkTrace | Citation,
    expected_sources: list[ExpectedSource],
) -> list[ExpectedSource]:
    matches: list[ExpectedSource] = []
    for source in expected_sources:
        if source.filename != item.filename:
            continue
        if source.page is not None and source.page != item.page:
            continue
        matches.append(source)
    return matches


def _linear_percentile(ordered: list[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + (upper_value - lower_value) * fraction
