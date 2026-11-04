# -*- coding: utf-8 -*-
"""管理员检索诊断必须反映真实召回，不能凭最终引用反推。"""

from pathlib import Path
from contextlib import AsyncExitStack
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.application.query.fast_lane import FastLane
from app.application.query.generator import AnswerGenerator
from app.application.query.prompting import PromptBuilder
from app.application.query.debug import retrieval_debug
from app.domain.execution import EvidenceBundle
from app.domain.execution import ExecutionBudget
from app.infrastructure.conversations.langgraph_agent import LangGraphConversationAgent
from tests.test_fast_lane import _Provider
from tests.test_fast_lane import _Quality
from tests.test_fast_lane import _Retriever
from tests.test_scoped_agent import _Model
from tests.test_unified_query_api import _app
from tests.test_unified_query_api import _events
from tests.test_shares_api import _client


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lane", ["fast", "agent"])
async def test_admin_debug_contains_real_retrieval_and_judge_result(
    tmp_path: Path, stream: bool, lane: str,
) -> None:
    """只接受debug参数而不输出召回明细，或丢掉SSE字段时失败。

    Args:
        tmp_path: 临时真实索引和会话数据库。
        stream: JSON或SSE。
        lane: 实际Fast或Agent通道。

    Returns:
        None。
    """
    app, chunk, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            path = "/api/query/stream" if stream else "/api/query"
            body = {
                "question": "needle", "strategy": "hybrid", "lane": lane,
                "kb_scope": {"mode": "explicit", "kb_ids": ["default"]},
            }
            response = await client.post(path, json={**body, "debug": True})
            assert response.status_code == 200
            if stream:
                payload = [data for name, data in _events(response.text) if name == "complete"][0]
            else:
                payload = response.json()
            rows = payload["retrieved"]
            assert len(rows) == 1
            row = rows[0]
            assert row["attempt"] == row["rank"] == row["position"] == 1
            assert row["chunk_id"] == chunk.id
            assert row["chunk_text"] == chunk.text
            assert row["kb_id"] == "default"
            assert row["score"] == pytest.approx(2 / 61)
            assert row["approved"] is True
            assert row["cited"] is True
            assert payload["retrieval_calls"] == 1
            normal = await client.post("/api/query", json=body)
            assert "retrieved" not in normal.json()
            direct = await client.post("/api/query", json={
                "question": "你好", "debug": True,
            })
            assert direct.json()["retrieved"] == []
    finally:
        app.state.container.close()


def test_fast_to_agent_debug_keeps_original_ranks_and_excluded_chunks() -> None:
    """种子交接不能丢初查噪声或把最终证据下标伪造成召回排名。"""
    retriever = _Retriever()
    quality = _Quality(sufficient=False)
    seed = FastLane(
        retriever, quality, AnswerGenerator(PromptBuilder(), _Provider()),
        clock=lambda: 1.0,
    ).run(
        "比较A和B", [], query="A 存储", bundle=EvidenceBundle(
            budget=ExecutionBudget(deadline=100.0),
        ),
    ).bundle
    result = LangGraphConversationAgent(
        _Model([AIMessage(content="A使用SQLite[2]，B使用PostgreSQL[3]")]),
        retriever, InMemorySaver(), top_k=5,
        quality_evaluator=quality, clock=lambda: 2.0,
    ).run(uuid4(), "比较A和B", [], seed=seed)
    observations = result.execution.retrieved
    assert [item.model_dump() for item in observations] == [
        {"attempt": 1, "rank": 1, "position": 1, "chunk_id": "noise", "score": 0.9},
        {"attempt": 1, "rank": 2, "position": 2, "chunk_id": "a", "score": 0.8},
        {"attempt": 2, "rank": 1, "position": 3, "chunk_id": "b", "score": 0.8},
    ]
    assert result.execution.approved_numbers == (2, 3)
    assert retriever.calls == [("A 存储", "hybrid_rerank", 5), ("B 存储方式", "hybrid_rerank", 5)]
    rows = retrieval_debug(result)
    assert [item.approved for item in rows] == [False, True, True]
    assert [item.cited for item in rows] == [False, True, True]


def test_repeated_chunk_keeps_each_attempt_rank_without_renumbering() -> None:
    """扩查重复命中同一块时仍应保留两次观察，不能只剩最终去重结果。"""
    lane = FastLane(
        _Retriever(), _Quality(),
        AnswerGenerator(PromptBuilder(), _Provider()), clock=lambda: 1.0,
    )
    first = lane.run("问题", [], query="first", bundle=EvidenceBundle(
        budget=ExecutionBudget(deadline=100.0),
    ))
    second = lane.run("问题", [], query="second", bundle=first.bundle)
    rows = retrieval_debug(second.result)
    assert [(item.attempt, item.rank, item.position) for item in rows] == [
        (1, 1, 1), (1, 2, 2), (2, 1, 1), (2, 2, 2),
    ]
    assert [item.approved for item in rows] == [False, True, False, True]
    assert len(second.bundle.evidence) == 2
    assert second.bundle.budget.retrieval_calls == 2


@pytest.mark.anyio
async def test_debug_remains_admin_only(tmp_path: Path) -> None:
    """不能因为开启调试而向普通成员返回候选正文。

    Args:
        tmp_path: 临时认证和索引。

    Returns:
        None。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        async with AsyncExitStack() as stack:
            member = await _client(stack, app, "member")
            for path in ("/api/query", "/api/query/stream"):
                response = await member.post(path, json={
                    "question": "needle", "debug": True,
                })
                assert response.status_code == 403
                assert "retrieved" not in response.json()
    finally:
        app.state.container.close()
