# -*- coding: utf-8 -*-
"""评估纯指标的精确算术测试。"""

import math

import pytest

from app.domain.answers import Citation, RetrievedChunkTrace
from evaluation.metrics import (
    CaseTrace,
    compute_citation_metrics,
    compute_retrieval_metrics,
    summarize_latency,
)
from evaluation.models import ExpectedSource


def _retrieved(
    rank: int,
    filename: str,
    page: int | None = None,
) -> RetrievedChunkTrace:
    return RetrievedChunkTrace(
        rank=rank,
        chunk_id=f"chunk-{rank}",
        doc_id=f"doc-{rank}",
        text="测试上下文",
        text_sha256="a" * 64,
        filename=filename,
        page=page,
        chunk_index=rank - 1,
        score=1.0 / rank,
    )


def _citation(filename: str, page: int | None = None) -> Citation:
    return Citation(
        doc_id=filename,
        filename=filename,
        page=page,
        chunk_text="引用上下文",
    )


def _case(
    expected: list[ExpectedSource],
    retrieved: list[RetrievedChunkTrace],
    citations: list[Citation] | None = None,
) -> CaseTrace:
    return CaseTrace(
        expected_sources=expected,
        retrieved=retrieved,
        citations=citations or [],
    )


def test_retrieval_metrics_use_first_matching_rank() -> None:
    """Hit Rate 与 MRR 应按首个匹配来源的实际名次计算。"""
    traces = [
        _case(
            [ExpectedSource(filename="a.md")],
            [_retrieved(1, "x.md"), _retrieved(2, "a.md")],
        ),
        _case(
            [ExpectedSource(filename="b.md")],
            [_retrieved(1, "x.md"), _retrieved(2, "y.md")],
        ),
    ]

    metrics = compute_retrieval_metrics(traces, k=5)

    assert metrics.hit_rate == 0.5
    assert metrics.mrr == 0.25


def test_retrieval_metrics_require_expected_page_when_present() -> None:
    """期望页码存在时，同文件不同页不得算命中。"""
    traces = [
        _case(
            [ExpectedSource(filename="guide.pdf", page=3)],
            [
                _retrieved(1, "guide.pdf", page=2),
                _retrieved(2, "guide.pdf", page=3),
            ],
        )
    ]

    metrics = compute_retrieval_metrics(traces, k=1)

    assert metrics.hit_rate == 0.0
    assert metrics.mrr == 0.0


def test_retrieval_metrics_reject_empty_input() -> None:
    """空 case 集没有可解释的平均指标。"""
    with pytest.raises(ValueError, match="至少包含一个 case"):
        compute_retrieval_metrics([], k=5)


def test_citation_metrics_count_duplicates_only_in_precision() -> None:
    """重复引用影响精确率分母，但召回只计算不同期望来源。"""
    traces = [
        _case(
            [
                ExpectedSource(filename="a.md"),
                ExpectedSource(filename="b.md"),
            ],
            [],
            [
                _citation("a.md"),
                _citation("a.md"),
                _citation("wrong.md"),
            ],
        )
    ]

    metrics = compute_citation_metrics(traces)

    assert math.isclose(metrics.precision, 2.0 / 3.0)
    assert metrics.recall == 0.5


def test_citation_metrics_use_zero_for_no_citations() -> None:
    """没有任何引用时精确率与召回率都应为零。"""
    traces = [
        _case([ExpectedSource(filename="a.md")], [], citations=[])
    ]

    metrics = compute_citation_metrics(traces)

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


def test_latency_summary_uses_linear_interpolation() -> None:
    """分位数应使用端点之间的线性插值并保留均值。"""
    summary = summarize_latency([0.0, 10.0, 20.0, 30.0])

    assert summary.mean == 15.0
    assert summary.p50 == 15.0
    assert summary.p95 == pytest.approx(28.5)


def test_latency_summary_rejects_non_finite_or_negative_values() -> None:
    """延迟不得包含负数、NaN 或无穷值。"""
    for values in ([-1.0], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError, match="有限非负数"):
            summarize_latency(values)
