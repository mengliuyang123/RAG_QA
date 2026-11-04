# -*- coding: utf-8 -*-
"""首次实际范围必须可持久化，不能用扩查并集或模型建议替代。"""

from pathlib import Path
from uuid import uuid4

import pytest
import httpx
from pydantic import ValidationError

from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem
from app.domain.answer_metadata import ScopeInfo
from app.domain.routing import QueryUnderstanding
from tests.test_query_orchestrator import _Understanding
from tests.test_query_orchestrator import _build
from tests.test_scoped_agent import _Quality
from tests.test_unified_query_api import _app


@pytest.mark.parametrize("lane", ["auto", "agent"])
def test_initial_scope_comes_from_actual_first_attempt(tmp_path: Path, lane: str) -> None:
    """Agent起点查全库，不能把理解层建议dev误存为首次实际范围。

    Args:
        tmp_path: 隔离真实检索及图。
        lane: 两种实际起点。

    Returns:
        None。
    """
    orchestrator, _ = _build(tmp_path, _Understanding(QueryUnderstanding(
        standalone_query="needle", confidence=0.95, preferred_kb_ids=("dev",),
    )))
    result = orchestrator.run(uuid4(), "原问题", [], requested_lane=lane)
    expected = ("dev",) if lane == "auto" else ("default", "dev", "ops")
    assert result.metadata.kb_scope.initial == expected
    assert result.metadata.kb_scope.initial == result.execution.budget.attempts[0].kb_ids


def test_expansion_preserves_initial_scope_in_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """扩查后searched增加，但首次范围仍为dev，持久字段不得被覆盖。

    Args:
        tmp_path: 真实检索隔离目录。
        monkeypatch: 仅替代远端裁判的结果。

    Returns:
        None。
    """
    count = 0

    def assess(
        self: _Quality, question: str, history: list[dict[str, str]],
        evidence: list[EvidenceItem],
    ) -> EvidenceAssessment:
        """首次模拟不足，扩查后批准实际取回的证据。

        Args:
            self: 离线裁判实例。
            question: 实际问题。
            history: 已预算历史。
            evidence: 实际取回的证据。

        Returns:
            脚本化结构结果。
        """
        nonlocal count
        count += 1
        return EvidenceAssessment(
            sufficient=count > 1, relevant_numbers=[item.number for item in evidence],
            missing_information=["缺少另一份配置"] if count == 1 else [],
            next_query=None,
        )

    monkeypatch.setattr(_Quality, "assess_evidence", assess)
    orchestrator, _ = _build(tmp_path, _Understanding(QueryUnderstanding(
        standalone_query="needle", confidence=0.95, preferred_kb_ids=("dev",),
    )))
    result = orchestrator.run(uuid4(), "原问题", [])
    assert result.metadata.scope_expanded is True
    assert result.metadata.kb_scope.searched == ("default", "dev", "ops")
    assert result.metadata.kb_scope.initial == ("dev",)
    restored = ScopeInfo.model_validate_json(result.metadata.kb_scope.model_dump_json())
    assert restored.initial == ("dev",)


def test_known_no_retrieval_differs_from_old_unmeasured_scope(tmp_path: Path) -> None:
    """新直答已知没有检索；旧数据缺少首次范围只能为未知而不是空元组。

    Args:
        tmp_path: 隔离真实编排器。

    Returns:
        None。
    """
    orchestrator, _ = _build(tmp_path)
    direct = orchestrator.run(uuid4(), "你好", [])
    assert direct.metadata.kb_scope.initial == ()
    old = ScopeInfo.model_validate({"mode": "auto", "searched": ["ops"]})
    assert old.initial is None
    with pytest.raises(ValidationError):
        ScopeInfo.model_validate({"searched": ["ops"], "initial": ["dev"]})
    with pytest.raises(ValidationError):
        ScopeInfo.model_validate({"searched": ["ops"], "initial": []})


@pytest.mark.anyio
async def test_initial_scope_is_retained_in_formal_history(tmp_path: Path) -> None:
    """不仅内存执行结果，正式SQLite历史也必须保留新的首次范围。

    Args:
        tmp_path: 全部真实存储隔离。

    Returns:
        None。
    """
    app, _, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            created = (await client.post("/api/conversations")).json()
            response = await client.post(f"/api/conversations/{created['id']}/query", json={"question": "needle"})
            assert response.status_code == 200
            assert response.json()["kb_scope"]["initial"] == ["default"]
            detail = (await client.get(f"/api/conversations/{created['id']}")).json()
            assert detail["turns"][0]["kb_scope"]["initial"] == ["default"]
    finally:
        app.state.container.close()
