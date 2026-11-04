# -*- coding: utf-8 -*-
"""反馈后台必须有独立管理员授权、审计与用户修改时的处理失效保护。"""

from contextlib import AsyncExitStack
from pathlib import Path
import json
import sqlite3

import httpx
import pytest

from tests.test_auth_service import PASSWORD
from tests.test_unified_query_api import _app
from app.application.errors import TraceStorageError


@pytest.mark.anyio
async def test_admin_feedback_triage_preserves_idempotency_and_detects_changes(tmp_path: Path) -> None:
    """不能通过普通会话越权，也不能把旧反馈的分类应用到已变化的内容。

    Args:
        tmp_path: 隔离所有数据。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        async with AsyncExitStack() as stack:
            clients = []
            administrator_id = None
            for username, role in (("alice", "member"), ("admin", "admin")):
                user = app.state.container.auth_service.create_user(username, PASSWORD, name=username, role=role)
                if role == "admin":
                    administrator_id = str(user.id)
                client = await stack.enter_async_context(httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="https://test",
                ))
                login = await client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
                assert login.status_code == 200
                client.headers["X-CSRF-Token"] = client.cookies.get("rag_csrf")
                clients.append(client)
            member, admin = clients
            created = (await member.post("/api/conversations")).json()
            turn = (await member.post(f"/api/conversations/{created['id']}/query", json={"question": "needle"})).json()
            path = f"/api/turns/{turn['turn_id']}/feedback"
            down = {"rating": "down", "reasons": ["incorrect"], "comment": "private note"}
            assert (await member.put(path, json=down)).status_code == 204
            endpoint = "/api/admin/insights/feedback"
            assert (await member.get(endpoint)).status_code == 403
            # 管理洞察是单独授权的投影，不给管理员绕过普通会话归属的例外。
            assert (await admin.get(f"/api/conversations/{created['id']}")).status_code == 404
            response = await admin.get(endpoint)
            assert response.status_code == 200
            item = response.json()["feedback"][0]
            assert item["question"] == "needle"
            assert item["answer"] == turn["answer"]
            assert item["citations"] == turn["citations"]
            assert item["comment"] == "private note"
            assert item["trace_id"] == turn["trace_id"]
            assert "user_id" not in item and "user_name" not in item
            detail_path = endpoint + "/" + item["feedback_id"]
            change = {"triage": "missing_content", "expected_revision": item["revision"]}
            csrf = admin.headers.pop("X-CSRF-Token")
            assert (await admin.patch(detail_path, json=change)).status_code == 403
            admin.headers["X-CSRF-Token"] = csrf
            classified = await admin.patch(detail_path, json=change)
            assert classified.status_code == 200
            assert classified.json()["triage"] == "missing_content"
            assert (await member.put(path, json=down)).status_code == 204
            unchanged = (await admin.get(endpoint)).json()["feedback"][0]
            assert unchanged["revision"] == item["revision"]
            assert unchanged["triage"] == "missing_content"
            assert (await member.put(path, json={**down, "comment": "changed note"})).status_code == 204
            changed = (await admin.get(endpoint)).json()["feedback"][0]
            assert changed["revision"] > item["revision"]
            assert changed["triage"] is None
            assert (await admin.patch(detail_path, json=change)).status_code == 409
            assert (await member.put(path, json={"rating": None})).status_code == 204
            assert (await admin.get(endpoint)).json()["feedback"] == []
            assert (await admin.patch(detail_path, json=change)).status_code == 404
            anonymous = await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://test",
            ))
            assert (await anonymous.get(endpoint)).status_code == 401
            with sqlite3.connect(app.state.container.settings.trace_db_path) as connection:
                rows = connection.execute("SELECT event_json FROM admin_access_audit").fetchall()
            events = [json.loads(row[0]) for row in rows]
            assert any(event["action"] == "read" for event in events)
            assert any(event["action"] == "triage_attempt" for event in events)
            assert all(event["actor_id"] == administrator_id for event in events)
            assert "private note" not in json.dumps(events)
            assert "changed note" not in json.dumps(events)
            assert turn["answer"] not in json.dumps(events, ensure_ascii=False)
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_audit_failure_blocks_sensitive_read_and_triage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """审计不能是吞掉异常的装饰；失败时不得返回正文或执行分类。

    Args:
        tmp_path: 隔离所有数据。
        monkeypatch: 只注入审计写入故障。
    """
    app, _, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            created = (await client.post("/api/conversations")).json()
            turn = (await client.post(f"/api/conversations/{created['id']}/query", json={"question": "needle"})).json()
            assert (await client.put(f"/api/turns/{turn['turn_id']}/feedback", json={"rating": "down", "comment": "do not leak"})).status_code == 204
            endpoint = "/api/admin/insights/feedback"
            item = (await client.get(endpoint)).json()["feedback"][0]
            repository = app.state.container.trace_repository
            original = repository.record_admin_access

            def fail(audit: object) -> None:
                raise TraceStorageError("private audit location")

            monkeypatch.setattr(repository, "record_admin_access", fail)
            denied = await client.get(endpoint)
            assert denied.status_code == 503
            assert "do not leak" not in denied.text
            change = await client.patch(endpoint + "/" + item["feedback_id"], json={"triage": "not_an_issue"})
            assert change.status_code == 503
            monkeypatch.setattr(repository, "record_admin_access", original)
            current = (await client.get(endpoint)).json()["feedback"][0]
            assert current["triage"] is None
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_feedback_admin_filters_before_pagination_and_binds_cursor(tmp_path: Path) -> None:
    """先分页再筛选、漏页、重复或混用不同过滤游标都会使测试失败。

    Args:
        tmp_path: 隔离数据。
    """
    app, _, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            created = (await client.post("/api/conversations")).json()
            expected = set()
            for index in range(4):
                turn = (await client.post(f"/api/conversations/{created['id']}/query", json={"question": "needle"})).json()
                rating = "down" if index < 3 else "up"
                assert (await client.put(f"/api/turns/{turn['turn_id']}/feedback", json={"rating": rating})).status_code == 204
                if rating == "down":
                    expected.add(turn["turn_id"])
            endpoint = "/api/admin/insights/feedback"
            found = []
            cursor = None
            first_cursor = None
            params = {"limit": 1, "kb_id": "default", "triage": "untriaged"}
            for _ in range(4):
                if cursor is not None:
                    params["cursor"] = cursor
                response = await client.get(endpoint, params=params)
                assert response.status_code == 200
                body = response.json()
                found.extend(row["turn_id"] for row in body["feedback"])
                cursor = body["next_cursor"]
                if first_cursor is None:
                    first_cursor = cursor
                if cursor is None:
                    break
            assert set(found) == expected
            assert len(found) == len(set(found)) == 3
            assert first_cursor is not None
            invalid = await client.get(endpoint, params={"cursor": first_cursor, "triage": "missing_content"})
            assert invalid.status_code == 422
            assert (await client.get(endpoint, params={"kb_id": "not_here"})).json()["feedback"] == []
    finally:
        app.state.container.close()
