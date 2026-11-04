# -*- coding: utf-8 -*-
"""评估fast_only必须走生产编排器，且不能被线上请求借用绕过通道策略。"""

from pathlib import Path
from uuid import uuid4

import pytest
import httpx

from app.domain.agent_quality import EvidenceAssessment
from tests.test_unified_query_api import _app


def test_fast_only_uses_real_judge_and_refuses_without_agent_upgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """若只把结果改名为fast、实际仍升级Agent，此测试应失败。

    Args:
        tmp_path: 隔离数据。
        monkeypatch: 只控制证据裁判边界和禁止Agent进入的探针。
    """
    app, _, quality = _app(tmp_path)
    root = app.state.container
    try:
        def insufficient(*args: object, **kwargs: object) -> EvidenceAssessment:
            return EvidenceAssessment(sufficient=False, relevant_numbers=[], missing_information=["缺少补充信息"], next_query=None)

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("fast_only不得调用Agent")

        monkeypatch.setattr(quality, "assess_evidence", insufficient)
        monkeypatch.setattr(root._get_conversation_agent(), "run", forbidden)
        result = root.query_orchestrator.run(
            uuid4(), "needle", [], requested_lane="fast",
            execution_entry="evaluation", lane_policy="fast_only",
        )
        assert result.metadata.lane == "fast"
        assert result.metadata.outcome == "refused"
        assert result.retrieval_calls == 1
        assert result.metadata.escalated is False
    finally:
        root.close()


def test_fast_only_is_rejected_outside_evaluation(tmp_path: Path) -> None:
    """后端内部误传也不能把评估策略作为线上执行行为。

    Args:
        tmp_path: 隔离数据。
    """
    app, _, quality = _app(tmp_path)
    try:
        with pytest.raises(ValueError, match="评估"):
            app.state.container.query_orchestrator.run(
                uuid4(), "needle", [], lane_policy="fast_only",
            )
        assert quality.evidence_calls == 0
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_trace_list_retains_evaluation_policy(tmp_path: Path) -> None:
    """列表也要区分固定快速基线，不能只靠最终fast标签混算。

    Args:
        tmp_path: 隔离数据库。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    try:
        with root.trace_service.start(
            "固定案例", request_id=uuid4(), entry="evaluation",
            lane_policy="fast_only",
        ) as trace:
            trace.set_result(lane="fast", outcome="refused", scope_expanded=False, escalated=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get("/api/admin/traces")
        assert response.status_code == 200
        assert response.json()["traces"][0]["lane_policy"] == "fast_only"
    finally:
        root.close()


def test_trace_domain_rejects_fast_only_on_online_entries(tmp_path: Path) -> None:
    """追踪模型不能接受把线上请求冒充固定评估策略的数据。

    Args:
        tmp_path: 隔离数据库。
    """
    app, _, _ = _app(tmp_path)
    try:
        with pytest.raises(ValueError, match="评估"):
            with app.state.container.trace_service.start(
                "question", request_id=uuid4(), entry="query",
                lane_policy="fast_only",
            ):
                pass
    finally:
        app.state.container.close()
