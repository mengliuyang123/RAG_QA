# -*- coding: utf-8 -*-
"""拒答后台必须读取已提交事实，分页前过滤且敏感读取有审计。"""

import json
import sqlite3
from contextlib import AsyncExitStack
from pathlib import Path
from uuid import UUID

import httpx
import pytest

from app.application.errors import TraceStorageError
from app.bootstrap.container import RuntimeOptions
from app.bootstrap.container import build_application_container
from app.config import Settings
from app.domain.answer_metadata import AnswerMetadata
from app.domain.answer_metadata import RefusalInfo
from app.domain.answer_metadata import ScopeInfo
from app.domain.conversations import AgentResult
from app.domain.conversations import ConversationTurn
from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem
from app.domain.knowledge_bases import KnowledgeBase
from app.infrastructure.conversations.sqlite_repository import SQLiteConversationRepository
from tests.http_auth import TEST_USER_ID
from tests.test_auth_service import PASSWORD
from tests.test_unified_query_api import _app


ENDPOINT = "/api/admin/insights/refusals"


def test_refusal_insights_use_runtime_paths_without_models(tmp_path: Path) -> None:
    """后台只读用例不得因无模型凭据而失败，也不能误开生产业务路径。

    Args:
        tmp_path: 候选生产路径与实际隔离路径均在临时目录。

    Returns:
        None。
    """
    settings = Settings(
        _env_file=None, conversation_db_path=tmp_path / "unused-business.sqlite3",
        trace_db_path=tmp_path / "unused-trace.sqlite3",
        chroma_path=tmp_path / "unused-chroma",
        knowledge_base_db_path=tmp_path / "unused-catalog.sqlite3",
    )
    root = build_application_container(settings, RuntimeOptions(
        conversation_db_path=tmp_path / "isolated-business.sqlite3",
        trace_db_path=tmp_path / "isolated-trace.sqlite3",
        knowledge_base_db_path=tmp_path / "isolated-catalog.sqlite3",
    ))
    try:
        page = root.refusal_insight_service.list(TEST_USER_ID)
        assert page.refusals == ()
        assert (tmp_path / "isolated-business.sqlite3").exists()
        assert (tmp_path / "isolated-trace.sqlite3").exists()
        assert not settings.conversation_db_path.exists()
        assert not settings.trace_db_path.exists()
        assert not settings.chroma_path.exists()
        overview = root.insight_overview_service.get(TEST_USER_ID)
        assert overview.totals.questions == 0
        assert overview.totals.active_users == 0
        assert not settings.knowledge_base_db_path.exists()
        assert (tmp_path / "isolated-catalog.sqlite3").exists()
    finally:
        root.close()


def _refusal(
    repository: SQLiteConversationRepository, *, question: str = "私密拒答问题",
    kb_ids: tuple[str, ...] = ("default",), owner_id: UUID = TEST_USER_ID,
) -> ConversationTurn:
    conversation = repository.create(owner_id=owner_id)
    return repository.save_turn(conversation.id, question, AgentResult(
        answer="无法根据现有资料回答", answer_mode="retrieval", retrieval_calls=1,
        citations=[], metadata=AnswerMetadata(
            lane="agent", outcome="refused", grounding="ungrounded",
            kb_scope=ScopeInfo(searched=kb_ids),
            refusal=RefusalInfo(
                reason="no_evidence", found_summary="未找到相关证据",
                missing=("缺少配置步骤",),
            ),
        ),
    ), 15.0)


@pytest.mark.anyio
async def test_refusal_list_authorizes_audits_without_granting_conversation_access(
    tmp_path: Path,
) -> None:
    """越权、原文另存trace或未落审计就返回原文都违背此接口契约。

    Args:
        tmp_path: 全部存储隔离目录。

    Returns:
        None。
    """
    app, _, quality = _app(tmp_path, legacy_identity=False)
    root = app.state.container
    try:
        async with AsyncExitStack() as stack:
            clients = []
            users = []
            for role in ("admin", "member"):
                user = root.auth_service.create_user(
                    role, PASSWORD, name="不应显示的姓名", role=role,
                )
                users.append(user)
                client = await stack.enter_async_context(httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="https://test",
                ))
                assert (await client.post("/api/auth/login", json={
                    "username": role, "password": PASSWORD,
                })).status_code == 200
                clients.append(client)
            admin, member = clients
            repository = SQLiteConversationRepository(root.settings.conversation_db_path)
            refused = _refusal(repository, owner_id=users[1].id)
            assert (await member.get(ENDPOINT)).status_code == 403
            assert (await admin.get(f"/api/conversations/{refused.conversation_id}")).status_code == 404
            response = await admin.get(ENDPOINT)
            assert response.status_code == 200
            assert response.headers["Cache-Control"] == "no-store"
            items = response.json()["refusals"]
            assert len(items) == 1
            assert items[0]["turn_id"] == str(refused.id)
            assert items[0]["question"] == "私密拒答问题"
            assert items[0]["kb_ids"] == ["default"]
            assert items[0]["refusal"]["missing"] == ["缺少配置步骤"]
            assert items[0]["trace_id"] is None
            for field in ("user_id", "owner_id", "user_name", "conversation_id", "answer"):
                assert field not in items[0]
            assert quality.evidence_calls == 0
            with sqlite3.connect(root.settings.trace_db_path) as connection:
                rows = connection.execute("SELECT event_json FROM admin_access_audit").fetchall()
                traces = connection.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
            assert traces == 0
            events = [json.loads(row[0]) for row in rows]
            assert len(events) == 1
            assert events[0]["actor_id"] == str(users[0].id)
            assert events[0]["resource_kind"] == "refusals"
            assert events[0]["resource_ids"] == [str(refused.id)]
            assert "私密拒答问题" not in json.dumps(events, ensure_ascii=False)
            anonymous = await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://test",
            ))
            assert (await anonymous.get(ENDPOINT)).status_code == 401
    finally:
        root.close()


@pytest.mark.anyio
@pytest.mark.parametrize("fault", ["metadata", "missing_table"])
async def test_refusal_storage_failure_is_503_not_empty_or_private_error(
    tmp_path: Path, fault: str,
) -> None:
    """损坏的持久化事实不得伪装成没有拒答，也不能泄露正文或SQL。

    Args:
        tmp_path: 隔离的临时业务库。
        fault: 两类真实SQLite故障。

    Returns:
        None。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    repository = SQLiteConversationRepository(root.settings.conversation_db_path)
    turn = _refusal(repository)
    try:
        # 先完成生产组合根初始化，故障之后读取适配器不能建表“修复”为空。
        root.refusal_insight_service
        with sqlite3.connect(root.settings.conversation_db_path) as connection:
            if fault == "missing_table":
                connection.execute("ALTER TABLE conversation_turns RENAME TO private_missing")
            else:
                connection.execute(
                    "UPDATE conversation_turns SET metadata_json = ? WHERE id = ?",
                    ('{"outcome":"refused","refusal":"私密内容"}', str(turn.id)),
                )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get(ENDPOINT)
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "insight_unavailable"
            assert response.headers["Cache-Control"] == "no-store"
            assert "私密" not in response.text
            assert "private_missing" not in response.text
    finally:
        root.close()


@pytest.mark.anyio
async def test_real_refusal_workflow_lists_only_committed_user_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """统一编排器的真实拒答必须接入洞察，检索测试结果不能混入用户列表。

    Args:
        tmp_path: 隔离所有生产适配器路径。
        monkeypatch: 只替代外部质量模型的确定性响应。

    Returns:
        None。
    """
    app, _, quality = _app(tmp_path)
    root = app.state.container
    root.knowledge_base_repository.create(KnowledgeBase(id="empty", name="空库"))

    def insufficient(
        question: str, history: list[dict[str, str]],
        evidence: list[EvidenceItem],
    ) -> EvidenceAssessment:
        """为外部质量模型返回固定的不充分结果。

        Args:
            question: 实际问题。
            history: 实际预算历史。
            evidence: 实际检索证据。

        Returns:
            明确缺口的结构化判断。
        """
        return EvidenceAssessment(
            sufficient=False, relevant_numbers=[],
            missing_information=["缺少配置步骤"], next_query=None,
        )

    monkeypatch.setattr(quality, "assess_evidence", insufficient)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            body = {
                "question": "needle",
                "kb_scope": {"mode": "explicit", "kb_ids": ["empty"]},
            }
            tested = await client.post("/api/query", json=body)
            assert tested.status_code == 200
            assert tested.json()["outcome"] == "refused"
            assert (await client.get(ENDPOINT)).json()["refusals"] == []
            created = (await client.post("/api/conversations")).json()
            response = await client.post(f"/api/conversations/{created['id']}/query", json=body)
            assert response.status_code == 200
            assert response.json()["outcome"] == "refused"
            page = (await client.get(ENDPOINT, params={"kb_id": "empty"})).json()
            assert len(page["refusals"]) == 1
            item = page["refusals"][0]
            assert item["trace_id"] == response.json()["trace_id"]
            assert item["turn_id"] == response.json()["turn_id"]
            assert item["refusal"] == response.json()["refusal"]
            # 删除正式历史后，尚未过期的执行trace不得重新出现在拒答列表。
            assert (await client.delete(f"/api/conversations/{created['id']}")).status_code == 200
            assert (await client.get(ENDPOINT)).json()["refusals"] == []
    finally:
        root.close()


@pytest.mark.anyio
async def test_refusal_filters_precede_paging_and_cursor_binds_kb(tmp_path: Path) -> None:
    """并列时间、其他库穿插、deleting和旧未知轮次不能造成漏页或假拒答。

    Args:
        tmp_path: 全部存储隔离目录。

    Returns:
        None。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    repository = SQLiteConversationRepository(root.settings.conversation_db_path)
    expected = set()
    for index in range(6):
        kb_ids = ("default", "ops") if index % 2 == 0 else ("other",)
        turn = _refusal(repository, question=f"问题{index}", kb_ids=kb_ids)
        if index % 2 == 0:
            expected.add(str(turn.id))
    removed = _refusal(repository)
    repository.mark_deleting(removed.conversation_id)
    unknown = _refusal(repository)
    repository.save_turn(repository.create(owner_id=TEST_USER_ID).id, "旧拒答文案", AgentResult(
        answer="无法根据现有资料回答", answer_mode="retrieval", retrieval_calls=1, citations=[],
    ), 1.0)
    with sqlite3.connect(root.settings.conversation_db_path) as connection:
        connection.execute("UPDATE conversation_turns SET created_at = '2026-10-20T00:00:00+00:00'")
        connection.execute("UPDATE conversation_turns SET metadata_json = NULL WHERE id = ?", (str(unknown.id),))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            found = []
            params = {"kb_id": "default", "limit": 1}
            first_cursor = None
            for _ in range(4):
                response = await client.get(ENDPOINT, params=params)
                assert response.status_code == 200
                page = response.json()
                found.extend(item["turn_id"] for item in page["refusals"])
                cursor = page["next_cursor"]
                if first_cursor is None:
                    first_cursor = cursor
                if cursor is None:
                    break
                params["cursor"] = cursor
            assert found == sorted(expected, reverse=True)
            assert len(found) == len(set(found)) == 3
            assert (await client.get(ENDPOINT, params={"kb_id": "other", "cursor": first_cursor})).status_code == 422
            assert (await client.get(ENDPOINT, params={"kb_id": "missing"})).json()["refusals"] == []
            for invalid in ({"limit": 101}, {"group": "unknown"}, {"cursor": "bad"}, {"kb_id": "bad/id"}):
                assert (await client.get(ENDPOINT, params=invalid)).status_code == 422
    finally:
        root.close()


@pytest.mark.anyio
async def test_refusal_audit_failure_does_not_return_sensitive_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """审计故障时不能仍把正文交给管理员浏览器。

    Args:
        tmp_path: 全部存储隔离目录。
        monkeypatch: 仅注入审计写入故障。

    Returns:
        None。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    repository = SQLiteConversationRepository(root.settings.conversation_db_path)
    _refusal(repository)

    def fail(event: object) -> None:
        raise TraceStorageError("私密审计路径")

    monkeypatch.setattr(root.trace_repository, "record_admin_access", fail)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get(ENDPOINT)
            assert response.status_code == 503
            assert "私密" not in response.text
    finally:
        root.close()
