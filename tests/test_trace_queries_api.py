# -*- coding: utf-8 -*-
"""后台追踪真实权限、脱敏、范围过滤及游标分页契约。"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from uuid import UUID
from uuid import uuid4

import httpx
import pytest

from app.application.tracing import TraceService
from app.application.tracing import traced_step
from tests.test_auth_service import PASSWORD
from tests.test_unified_query_api import _app
from tests.http_auth import TEST_USER_ID


@pytest.mark.anyio
async def test_admin_trace_reads_are_protected_and_redacted(tmp_path: Path) -> None:
    """普通成员即使知道trace_id也不可绕过管理员权限读取。

    Args:
        tmp_path: 隔离所有数据。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        with app.state.container.trace_service.start("private question", request_id=uuid4(), entry="query") as trace:
            pass
        path = f"/api/admin/traces/{trace.id}"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
            assert (await client.get(path)).status_code == 401
            assert (await client.get("/api/admin/traces")).status_code == 401
            assert (await client.get("/api/admin/usage")).status_code == 401
            for role in ("member", "admin"):
                app.state.container.auth_service.create_user(role, PASSWORD, name=role, role=role)
                await client.post("/api/auth/login", json={"username": role, "password": PASSWORD})
                response = await client.get(path)
                if role == "member":
                    assert response.status_code == 403
                    assert (await client.get("/api/admin/traces")).status_code == 403
                    assert (await client.get("/api/admin/usage")).status_code == 403
                else:
                    assert response.status_code == 200
                    body = response.json()
                    assert body["question"] is None
                    assert "user_id" not in body
                    assert "private question" not in response.text
                    assert response.headers["Cache-Control"] == "no-store"
                    assert (await client.get(f"/api/admin/traces/{uuid4()}")).status_code == 404
                    assert (await client.get("/api/admin/usage")).status_code == 200
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_trace_list_filters_before_cursor_pagination(tmp_path: Path) -> None:
    """过滤必须在分页前应用，同时间记录也不能漏页或重复。

    Args:
        tmp_path: 隔离所有数据。
    """
    app, _, _ = _app(tmp_path)
    try:
        now = datetime.now(UTC)
        service = TraceService(app.state.container.trace_repository, wall_clock=lambda: now)
        ids = []
        for index in range(6):
            with service.start(f"question-{index}", request_id=uuid4(), entry="query") as trace:
                with traced_step("retrieval", {"query": "private", "kb_ids": ["ops" if index % 2 == 0 else "dev"]}):
                    pass
                trace.set_result(lane="fast", outcome="answered", scope_expanded=False, escalated=False)
                if index % 2 == 0:
                    ids.append(str(trace.id))
        expired = TraceService(app.state.container.trace_repository, wall_clock=lambda: now - timedelta(days=31))
        with expired.start("expired", request_id=uuid4(), entry="query") as expired_trace:
            pass
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            params = {"kb_id": "ops", "lane": "fast", "limit": 1}
            found = []
            cursor = None
            for _ in range(4):
                if cursor is not None:
                    params["cursor"] = cursor
                response = await client.get("/api/admin/traces", params=params)
                assert response.status_code == 200
                body = response.json()
                found.extend(item["trace_id"] for item in body["traces"])
                cursor = body["next_cursor"]
                if cursor is None:
                    break
            assert found == sorted(ids, reverse=True)
            assert len(found) == len(set(found)) == 3
            bad = await client.get("/api/admin/traces", params={"cursor": "not-json"})
            assert bad.status_code == 422
            assert (await client.get(f"/api/admin/traces/{expired_trace.id}")).status_code == 404
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_trace_feedback_and_user_filters_use_current_business_state(tmp_path: Path) -> None:
    """反馈过滤不能使用追踪写入时的旧评分，也不能先分页再筛选。

    Args:
        tmp_path: 隔离所有数据。
    """
    app, _, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            created = (await client.post("/api/conversations")).json()
            turns = []
            for _ in range(3):
                response = await client.post(f"/api/conversations/{created['id']}/query", json={"question": "needle"})
                assert response.status_code == 200
                turns.append(response.json())
            chosen = turns[0]
            path = f"/api/turns/{chosen['turn_id']}/feedback"
            assert (await client.put(path, json={"rating": "down"})).status_code == 204
            params = {"feedback": "down", "limit": 1, "user_id": str(TEST_USER_ID)}
            response = await client.get("/api/admin/traces", params=params)
            assert response.status_code == 200
            body = response.json()
            assert [item["trace_id"] for item in body["traces"]] == [chosen["trace_id"]]
            assert body["next_cursor"] is None
            assert body["traces"][0]["feedback"] == "down"
            assert "user_id" not in body["traces"][0]
            assert (await client.put(path, json={"rating": "up"})).status_code == 204
            assert (await client.get("/api/admin/traces", params=params)).json()["traces"] == []
            other = await client.get("/api/admin/traces", params={"user_id": str(uuid4())})
            assert other.json()["traces"] == []
    finally:
        app.state.container.close()


@pytest.mark.anyio
@pytest.mark.parametrize("params", [
    {"limit": 0}, {"limit": 101}, {"lane": "hybrid"},
    {"from": "2026-10-24T00:00:00Z", "to": "2026-10-23T00:00:00Z"},
    {"from": "2026-10-23T00:00:00"},
])
async def test_invalid_trace_queries_are_422(tmp_path: Path, params: dict[str, object]) -> None:
    """非法范围和混淆策略／通道不能落成500或无条件查询。

    Args:
        tmp_path: 隔离存储。
        params: 非法查询。
    """
    app, _, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/admin/traces", params=params)
            assert response.status_code == 422
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_trace_read_storage_failure_is_typed_503(tmp_path: Path) -> None:
    """关联库不可读不能变成空结果、读取其他运行的库或泄漏路径。

    Args:
        tmp_path: 隔离存储。
    """
    app, _, _ = _app(tmp_path)
    try:
        app.state.container.settings.conversation_db_path.mkdir()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/admin/traces")
            assert response.status_code == 503
            assert str(tmp_path) not in response.text
    finally:
        app.state.container.close()
