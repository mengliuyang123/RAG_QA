# -*- coding: utf-8 -*-
"""概览分母必须来自正式事实，不能把参与库误认为首次路由库。"""

import json
import sqlite3
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from uuid import UUID
from uuid import uuid4

import httpx
import pytest

from app.application.errors import TraceStorageError
from app.domain.answer_metadata import AnswerMetadata
from app.domain.answer_metadata import RefusalInfo
from app.domain.answer_metadata import ScopeInfo
from app.domain.conversations import AgentResult
from app.domain.conversations import ConversationTurn
from app.domain.conversations import TurnCitation
from app.domain.feedback import TurnFeedback
from app.domain.knowledge_bases import KnowledgeBase
from app.infrastructure.conversations.sqlite_repository import SQLiteConversationRepository
from tests.test_auth_service import PASSWORD
from tests.test_unified_query_api import _app


ENDPOINT = "/api/admin/insights/overview"


def _turn(
    repository: SQLiteConversationRepository, owner_id: UUID | None,
    scope: ScopeInfo, *, refused: bool = False, expanded: bool = False,
) -> ConversationTurn:
    metadata = AnswerMetadata(
        lane="agent" if refused else "fast",
        outcome="refused" if refused else "answered",
        grounding="ungrounded" if refused else "grounded", kb_scope=scope,
        scope_expanded=expanded,
        refusal=RefusalInfo(reason="no_evidence", found_summary="没有资料", missing=("缺少步骤",))
        if refused else None,
    )
    citations = []
    if not refused:
        citations.append(TurnCitation(
            position=1, doc_id=str(uuid4()), filename="guide.md", page=None,
            chunk_text="敏感引用正文", kb_id=scope.searched[0],
        ))
    return repository.save_turn(repository.create(owner_id=owner_id).id, "私密问题正文", AgentResult(
        answer="无法回答" if refused else "敏感答案[1]", answer_mode="retrieval",
        retrieval_calls=1, citations=citations, metadata=metadata,
    ), 1.0)


@pytest.mark.anyio
async def test_overview_counts_formal_turns_and_exposes_unknown_denominators(tmp_path: Path) -> None:
    """跨库重复、旧首次范围补零、按已评分分母或把测试trace混入都会失败。

    Args:
        tmp_path: 全部真实存储隔离。

    Returns:
        None。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    root.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    root.knowledge_base_repository.create(KnowledgeBase(id="dev", name="研发"))
    repository = SQLiteConversationRepository(root.settings.conversation_db_path)
    owner1 = uuid4()
    owner2 = uuid4()
    first = _turn(repository, owner1, ScopeInfo(searched=("ops",), initial=("ops",)))
    repository.put_feedback(first.id, owner1, TurnFeedback(rating="down"))
    _turn(repository, owner2, ScopeInfo(searched=("ops", "dev"), initial=("ops",)), refused=True, expanded=True)
    _turn(repository, owner2, ScopeInfo(mode="explicit", requested=("dev",), searched=("dev",), initial=("dev",)), refused=True)
    old = _turn(repository, None, ScopeInfo(searched=("ops",)), refused=True, expanded=True)
    repository.save_turn(repository.create(owner_id=owner1).id, "你好", AgentResult(
        answer="你好", answer_mode="direct", retrieval_calls=0, citations=[],
        metadata=AnswerMetadata(lane="direct", outcome="answered", grounding="ungrounded", kb_scope=ScopeInfo(initial=())),
    ), 1.0)
    repository.save_turn(repository.create().id, "旧问题", AgentResult(
        answer="旧答案", answer_mode="direct", retrieval_calls=0, citations=[],
    ), 1.0)
    past = _turn(repository, uuid4(), ScopeInfo(searched=("ops",), initial=("ops",)))
    removed = _turn(repository, uuid4(), ScopeInfo(searched=("ops",), initial=("ops",)))
    repository.mark_deleting(removed.conversation_id)
    with sqlite3.connect(root.settings.conversation_db_path) as connection:
        connection.execute("UPDATE conversation_turns SET created_at=? WHERE id=?", (
            (datetime.now(UTC) - timedelta(days=8)).isoformat(), str(past.id),
        ))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            # 管理检索测试确实执行过，但不是已提交用户问题，不能污染概览。
            assert (await client.post("/api/query", json={"question": "你好"})).status_code == 200
            response = await client.get(ENDPOINT)
            assert response.status_code == 200
            assert response.headers["Cache-Control"] == "no-store"
            body = response.json()
            assert body["period_days"] == 7
            assert body["totals"] == {
                "questions": 6, "active_users": 2, "unknown_owner_questions": 2,
                "unknown_metadata_questions": 1, "no_retrieval_questions": 1,
                "unknown_scope_questions": 0,
            }
            assert body["rows_additive"] is False
            rows = {row["kb_id"]: row for row in body["knowledge_bases"]}
            ops = rows["ops"]
            assert ops["questions"] == 3
            assert ops["refusal_rate"] == pytest.approx(2 / 3)
            assert ops["thumbs_down_rate"] == pytest.approx(1 / 3)
            assert ops["known_initial_auto_questions"] == 2
            assert ops["expanded_questions"] == 1
            assert ops["unknown_initial_auto_questions"] == 1
            assert ops["expanded_rate"] is None
            assert rows["dev"]["questions"] == 2
            assert rows["dev"]["refusal_rate"] == 1
            assert rows["dev"]["known_initial_auto_questions"] == 0
            assert rows["dev"]["expanded_rate"] is None
            assert rows["default"]["questions"] == 0
            assert rows["default"]["refusal_rate"] is None
            assert "私密问题" not in response.text
            assert "敏感答案" not in response.text
            assert str(owner1) not in response.text
            assert str(owner2) not in response.text
            repository.delete(old.conversation_id)
            repository.put_feedback(first.id, owner1, TurnFeedback(rating="up"))
            newer = (await client.get(ENDPOINT)).json()
            ops = next(row for row in newer["knowledge_bases"] if row["kb_id"] == "ops")
            assert ops["questions"] == 2
            assert ops["expanded_rate"] == 0.5
            assert ops["refusal_rate"] == 0.5
            assert ops["thumbs_down_rate"] == 0
            for invalid in ({"days": 0}, {"days": 366}):
                assert (await client.get(ENDPOINT, params=invalid)).status_code == 422
    finally:
        root.close()


@pytest.mark.anyio
async def test_overview_requires_admin_and_audit_even_for_empty_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """匿名／成员不能获取管理员聚合，审计故障也不能悄悄返回统计。

    Args:
        tmp_path: 全部真实存储隔离。
        monkeypatch: 只注入审计故障。

    Returns:
        None。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    root = app.state.container
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://test",
        ) as client:
            assert (await client.get(ENDPOINT)).status_code == 401
            for role in ("member", "admin"):
                root.auth_service.create_user(role, PASSWORD, name=role, role=role)
                await client.post("/api/auth/login", json={"username": role, "password": PASSWORD})
                result = await client.get(ENDPOINT)
                assert result.status_code == (200 if role == "admin" else 403)
            assert result.json()["totals"]["questions"] == 0
            assert result.json()["totals"]["active_users"] == 0
            with sqlite3.connect(root.settings.trace_db_path) as connection:
                rows = connection.execute("SELECT event_json FROM admin_access_audit").fetchall()
            assert len(rows) == 1
            event = json.loads(rows[0][0])
            assert event["resource_kind"] == "overview"
            assert event["resource_ids"] == []

            def fail(event: object) -> None:
                raise TraceStorageError("私密路径")

            monkeypatch.setattr(root.trace_repository, "record_admin_access", fail)
            denied = await client.get(ENDPOINT)
            assert denied.status_code == 503
            assert "私密" not in denied.text
    finally:
        root.close()


@pytest.mark.anyio
async def test_old_missing_scope_is_not_reported_as_no_retrieval(tmp_path: Path) -> None:
    """旧知识回答没有范围时应计未知，不得宣称没有调用检索。

    Args:
        tmp_path: 隔离业务库。

    Returns:
        None。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    repository = SQLiteConversationRepository(root.settings.conversation_db_path)
    repository.save_turn(repository.create().id, "旧问题", AgentResult(
        answer="旧拒答", answer_mode="retrieval", retrieval_calls=1, citations=[],
        metadata=AnswerMetadata(
            lane="agent", outcome="refused", grounding="ungrounded",
            refusal=RefusalInfo(reason="no_evidence", found_summary="未知", missing=("未知",)),
        ),
    ), 1.0)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get(ENDPOINT)
            assert response.status_code == 200
            assert response.json()["totals"]["no_retrieval_questions"] == 0
            assert response.json()["totals"]["unknown_scope_questions"] == 1
    finally:
        root.close()


@pytest.mark.anyio
async def test_overview_storage_failure_is_safe_503(tmp_path: Path) -> None:
    """存储损坏不能变成全0的成功统计，也不能暴露问题或SQL内部信息。

    Args:
        tmp_path: 真实临时业务库。

    Returns:
        None。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    repository = SQLiteConversationRepository(root.settings.conversation_db_path)
    _turn(repository, uuid4(), ScopeInfo(searched=("default",), initial=("default",)))
    try:
        with sqlite3.connect(root.settings.conversation_db_path) as connection:
            connection.execute("UPDATE conversation_turns SET metadata_json='{}'")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get(ENDPOINT)
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "insight_unavailable"
            assert "私密" not in response.text
            assert "conversation_turns" not in response.text
    finally:
        root.close()
