# -*- coding: utf-8 -*-
"""分通道报告保留分母、缺失评分和未知成本，不把强制策略混成自动路由。"""

import hashlib
import json
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from app.domain.answer_metadata import AnswerMetadata
from app.domain.answer_metadata import RefusalInfo
from app.domain.conversations import AgentResult
from app.domain.conversations import AnswerMode
from app.domain.conversations import TurnCitation
from app.domain.retrieval import RetrievalStrategy
from app.domain.tracing import CostEstimate
from app.domain.tracing import ModelCallTrace
from app.domain.tracing import TokenUsage
from app.domain.tracing import TraceSnapshot
from evaluation.lanes.checkpoints import LaneRunManifest
from evaluation.lanes.models import LaneEpisode
from evaluation.lanes.models import LaneJudgement
from evaluation.lanes.models import LaneObservation
from evaluation.lanes.models import LaneScoreRecord
from evaluation.lanes.models import LaneTurn


def _trace(question: str, calls: bool = False) -> TraceSnapshot:
    now = datetime.now(UTC)
    items = ()
    if calls:
        items = (ModelCallTrace(
            purpose="generation", origin="https://model.example", model="offline",
            started_at=now, duration_ms=10, status="completed", usage=TokenUsage(),
            cost=CostEstimate(status="unknown", reason="usage_unavailable"),
        ),)
    return TraceSnapshot(
        trace_id=uuid4(), request_id=uuid4(), entry="evaluation",
        question_sha256=hashlib.sha256(question.encode()).hexdigest(),
        created_at=now, expires_at=now + timedelta(days=30), status="completed",
        calls=items,
    )


def _observation(episode_id: str, *, refused: bool = False, direct: bool = False,
                 policy: str = "auto") -> LaneObservation:
    lane = "fast"
    outcome = "answered"
    grounding = "grounded"
    refusal = None
    citations = [TurnCitation(position=1, doc_id="d", filename="guide.md", page=None,
                              chunk_text="正文", chunk_id="d:0")]
    count = 1
    if direct:
        lane, grounding, citations, count = "direct", "ungrounded", [], 0
    elif refused:
        lane, outcome, grounding, citations, count = "agent", "refused", "partial", [], 3
        refusal = RefusalInfo(reason="insufficient_evidence", found_summary="不足",
                              missing=("缺少第二来源",))
    result = AgentResult(
        answer="回答", answer_mode=AnswerMode.DIRECT if direct else AnswerMode.RETRIEVAL,
        retrieval_calls=count, citations=citations,
        metadata=AnswerMetadata(lane=lane, outcome=outcome, grounding=grounding, refusal=refusal),
    )
    return LaneObservation(
        episode_id=episode_id, turn_index=0, execution_id=uuid4(), policy=policy,
        strategy=RetrievalStrategy.HYBRID, result=result,
        trace=_trace(episode_id, calls=not direct), latency_ms=50 if direct else 100,
        query_hashes=(), retrieval_compliant=True,
        routing_correct=True if policy == "auto" else None, rewrite_matches=None,
    )


def test_report_keeps_incomplete_denominators_and_separate_policy_cohorts(
    tmp_path: Path,
) -> None:
    """漏掉缺失分母、强制策略路由得分或给未知成本补零都应失败。

    Args:
        tmp_path: 不触碰正式latest的报告目录。
    """
    from evaluation.lanes.reporting import build_lane_report
    from evaluation.lanes.artifacts import write_lane_report

    knowledge = LaneTurn(question="answered", reference="正文", expected_lane="fast",
                         expected_sources=("guide.md",))
    retry = LaneTurn(question="retry", reference="两个来源", expected_lane="agent",
                     expected_sources=("guide.md", "other.md"))
    chat = LaneTurn(question="chat", reference="问候", expected_lane="direct",
                    requires_retrieval=False)
    missing = LaneTurn(question="missing", reference="正文", expected_lane="fast")
    episodes = (
        LaneEpisode(id="answered", tags=("single_hop",), turns=(knowledge,)),
        LaneEpisode(id="retry", tags=("retry",), turns=(retry,)),
        LaneEpisode(id="chat", tags=("chitchat",), turns=(chat,)),
        LaneEpisode(id="missing", tags=("single_hop",), turns=(missing,)),
    )
    selections = (("answered", "auto"), ("retry", "auto"), ("chat", "auto"),
                  ("missing", "auto"), ("answered", "fast_only"))
    manifest = LaneRunManifest(
        dataset_sha256="1" * 64, index_fingerprint="2" * 64,
        configuration_version="config", code_sha256="3" * 64,
        judge_configuration_sha256="4" * 64,
        selections=selections, strategy=RetrievalStrategy.HYBRID,
    )
    observations = (
        _observation("answered"), _observation("retry", refused=True),
        _observation("chat", direct=True), _observation("answered", policy="fast_only"),
    )
    scores = []
    for observation, correct, supported, complete in (
        (observations[0], 0.4, False, False),
        (observations[2], 1.0, None, True),
    ):
        scores.append(LaneScoreRecord(
            episode_id=observation.episode_id, turn_index=0, policy=observation.policy,
            strategy=RetrievalStrategy.HYBRID,
            source_trace_id=observation.trace.trace_id,
            judgement=LaneJudgement(correctness=correct, supported=supported,
                                    complete=complete, reason="手工评分"),
            trace=_trace(observation.episode_id, calls=True),
        ))
    report = build_lane_report(manifest, episodes, observations, tuple(scores), smoke=True)
    assert report.status == "incomplete"
    assert report.expected_turns == 5
    assert report.generated_turns == 4
    assert report.scored_turns == 2
    assert report.strategies == (RetrievalStrategy.HYBRID,)
    auto = report.lanes["auto"]
    assert auto.expected == 4
    assert auto.generated == 3
    assert auto.scored == 2
    assert auto.metrics["correctness"].value == pytest.approx(0.7)
    assert auto.metrics["correctness"].n == 2
    assert auto.metrics["supported"].value == 0.0
    assert auto.metrics["supported"].n == 1
    assert auto.metrics["complete"].value == 0.5
    assert auto.metrics["outcome_accuracy"].value == pytest.approx(2 / 3)
    assert auto.metrics["citation_source_recall"].value == 0.5
    assert auto.metrics["retry_rate"].value == 0.5
    assert auto.metrics["refusal_precision"].value == 0.0
    assert auto.metrics["refusal_recall"].value is None
    assert auto.metrics["refusal_recall"].n == 0
    assert auto.generation.tokens.input is None
    assert auto.generation.cost.status == "unknown"
    assert auto.generation.cost.unknown_calls == 2
    assert auto.judging.cost.unknown_calls == 2
    assert report.lanes["fast_only"].metrics["routing_accuracy"].value is None
    assert report.knowledge_lanes["auto"].expected == 3
    assert report.knowledge_lanes["auto"].generated == 2
    assert report.by_tag["retry"]["auto"].metrics["retry_rate"].value == 1.0
    assert report.lanes["auto"].final_lanes == {"fast": 1, "agent": 1, "direct": 1}
    encoded = report.model_dump_json()
    assert "chunk_text" not in encoded
    assert "question_sha256" not in encoded
    assert "steps" not in encoded
    with pytest.raises(ValueError, match="重复"):
        build_lane_report(manifest, episodes, observations + observations[:1], tuple(scores), smoke=True)
    invalid = scores[0].model_copy(update={"source_trace_id": uuid4()})
    with pytest.raises(ValueError, match="对应"):
        build_lane_report(manifest, episodes, observations, (invalid,), smoke=True)
    latest = tmp_path / "latest.json"
    latest.write_text("保留已发布旧报告", encoding="utf-8")
    destination = tmp_path / "run"
    write_lane_report(report, destination)
    payload = json.loads((destination / "report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "incomplete"
    assert payload["purpose"] == "connectivity_smoke"
    assert payload["lanes"]["auto"]["metrics"]["supported"]["n"] == 1
    schema = json.loads((destination / "schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["lanes"]
    markdown = (destination / "report.md").read_text(encoding="utf-8")
    assert "incomplete" in markdown
    assert "fast_only" in markdown
    assert "分母" in markdown
    assert latest.read_text(encoding="utf-8") == "保留已发布旧报告"
