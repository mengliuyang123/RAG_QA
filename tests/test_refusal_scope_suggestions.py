# -*- coding: utf-8 -*-
"""拒答切库建议不构成越过用户显式范围检索的授权。"""

import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem
from app.domain.knowledge_bases import KnowledgeBase
from app.domain.routing import KnowledgeScope
from app.domain.routing import QueryUnderstanding
from tests.test_query_orchestrator import _Understanding
from tests.test_query_orchestrator import _build
from tests.test_query_understanding import _payload
from tests.test_scoped_agent import _Quality
from tests.test_structured_output import _provider
from tests.test_unified_query_api import _app
from tests.test_unified_query_api import _events


def _insufficient(
    self: _Quality, question: str, history: list[dict[str, str]],
    evidence: list[EvidenceItem],
) -> EvidenceAssessment:
    # 没有任何相关证据才会拒答；有相关证据时改为按部分依据作答。
    return EvidenceAssessment(
        sufficient=False, relevant_numbers=[],
        missing_information=["Redis作为checkpoint的具体配置"], next_query=None,
    )


@pytest.mark.parametrize("lane", ["auto", "agent"])
@pytest.mark.parametrize("preferred,confidence,mode,expected", [
    (("dev",), 0.95, "explicit", "dev"),
    (("ops", "dev"), 0.95, "explicit", "dev"),
    (("unknown",), 0.95, "explicit", None),
    (("unknown", "dev"), 0.95, "explicit", None),
    (("ops",), 0.95, "explicit", None),
    ((), 0.95, "explicit", None),
    (("dev",), 0.2, "explicit", None),
    (("dev",), 0.95, "auto", None),
])
def test_suggestion_uses_valid_catalog_hint_without_scope_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str,
    preferred: tuple[str, ...], confidence: float, mode: str,
    expected: str | None,
) -> None:
    """丢失建议、接受未知候选或借推荐跨库检索都应失败。

    Args:
        tmp_path: 真实Chroma和目录的隔离路径。
        monkeypatch: 仅替代外部裁判判断。
        lane: 正常路由或直接Agent起点。
        preferred: 理解层候选。
        confidence: 理解层置信度。
        mode: 用户范围模式。
        expected: 独立给定的建议结果。

    Returns:
        None。
    """
    monkeypatch.setattr(_Quality, "assess_evidence", _insufficient)
    orchestrator, retriever = _build(tmp_path, _Understanding(QueryUnderstanding(
        standalone_query="needle", confidence=confidence,
        preferred_kb_ids=preferred,
    )))
    scope = KnowledgeScope()
    if mode == "explicit":
        scope = KnowledgeScope(mode="explicit", kb_ids=("ops",))
    result = orchestrator.run(
        uuid4(), "needle", [], kb_scope=scope, requested_lane=lane,
    )
    assert result.metadata.outcome == "refused"
    assert result.metadata.refusal.suggested_kb_id == expected
    assert result.metadata.refusal.missing == ("Redis作为checkpoint的具体配置",)
    assert result.metadata.refusal.escalation_available is False
    assert result.citations == []
    if mode == "explicit":
        assert retriever.calls == [("needle", ("ops",))]
        assert result.retrieval_calls == 1
        assert result.metadata.kb_scope.searched == ("ops",)
        assert result.metadata.scope_expanded is False


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_scope_suggestion_survives_http_and_formal_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    conversation: bool, stream: bool,
) -> None:
    """真实结构化理解输出的建议须贯通双入口和历史，仍仅检索用户指定库。

    Args:
        tmp_path: 真实业务存储全部隔离。
        monkeypatch: 只替代外部证据裁判。
        conversation: 是否使用正式会话入口。
        stream: 是否使用SSE外壳。

    Returns:
        None。
    """
    monkeypatch.setattr(_Quality, "assess_evidence", _insufficient)
    provider, transport, requests = _provider(json.dumps(_payload(
        standalone_query="needle", preferred_kb_ids=["dev"],
    )))
    app, _, _ = _app(
        tmp_path, understanding_factory=lambda *args: provider,
    )
    root = app.state.container
    root.knowledge_base_repository.create(KnowledgeBase(
        id="dev", name="研发", description="Redis checkpoint配置",
    ))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            path = "/api/query"
            detail_path = None
            if conversation:
                created = (await client.post("/api/conversations")).json()
                detail_path = "/api/conversations/" + created["id"]
                path = detail_path + "/query"
            if stream:
                path += "/stream"
            response = await client.post(path, json={
                "question": "Redis如何保存checkpoint？",
                "kb_scope": {"mode": "explicit", "kb_ids": ["default"]},
            })
            assert response.status_code == 200
            if stream:
                terminal = [
                    item for item in _events(response.text)
                    if item[0] in {"complete", "error"}
                ]
                assert [name for name, _ in terminal] == ["complete"]
                result = terminal[0][1]
            else:
                result = response.json()
            assert result["outcome"] == "refused"
            assert result["refusal"]["suggested_kb_id"] == "dev"
            assert result["kb_scope"]["searched"] == ["default"]
            assert result["scope_expanded"] is False
            assert result["retrieval_calls"] == 1
            assert len(requests) == 1
            if detail_path is not None:
                saved = (await client.get(detail_path)).json()["turns"][0]
                assert saved["refusal"] == result["refusal"]
                assert saved["kb_scope"] == result["kb_scope"]
    finally:
        root.close()
        transport.close()
