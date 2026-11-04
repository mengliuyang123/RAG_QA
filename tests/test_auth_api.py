# -*- coding: utf-8 -*-
"""真实Cookie登录、CSRF、角色与服务端注销的ASGI契约。"""

from pathlib import Path
from contextlib import AsyncExitStack
from uuid import UUID
from datetime import UTC
from datetime import datetime

import httpx
import pytest

from tests.test_auth_service import PASSWORD
from tests.test_unified_query_api import _app
from tests.test_unified_query_api import _events


@pytest.mark.anyio
async def test_conversations_are_private_even_from_admin(tmp_path: Path) -> None:
    """缺少归属过滤会泄漏列表、正文和写权限，包括管理员越权。

    Args:
        tmp_path: 隔离所有数据。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        auth = app.state.container.auth_service
        async with AsyncExitStack() as stack:
            clients = []
            for username, role in (("alice", "member"), ("bob", "member"), ("admin", "admin")):
                auth.create_user(username, PASSWORD, name=username, role=role)
                client = await stack.enter_async_context(httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="https://test",
                ))
                login = await client.post("/api/auth/login", json={
                    "username": username, "password": PASSWORD,
                })
                assert login.status_code == 200
                client.headers["X-CSRF-Token"] = client.cookies.get("rag_csrf")
                clients.append(client)
            alice, bob, admin = clients
            created = await alice.post("/api/conversations")
            assert created.status_code == 201
            identifier = created.json()["id"]
            path = f"/api/conversations/{identifier}"
            assert (await alice.get(path)).status_code == 200
            for other in (bob, admin):
                assert (await other.get("/api/conversations")).json()["total"] == 0
                for method, suffix, body in (
                    ("GET", "", None), ("PATCH", "", {"title": "stolen"}),
                    ("DELETE", "", None),
                    ("POST", "/query", {"question": "needle"}),
                    ("POST", "/query/stream", {"question": "needle"}),
                ):
                    response = await other.request(method, path + suffix, json=body)
                    assert response.status_code == 404
            assert (await alice.get(path)).json()["title"] == "新对话"
            with app.state.container.conversation_service._acquire(UUID(identifier)):
                busy_other = await bob.post(path + "/query", json={"question": "needle"})
                assert busy_other.status_code == 404
                busy_owner = await alice.post(path + "/query", json={"question": "needle"})
                assert busy_owner.status_code == 409
            forged = await bob.post("/api/conversations", json={"owner_id": identifier})
            assert forged.status_code == 422
            # 无归属的旧数据不应被首位登录者或管理员自动接管。
            legacy = app.state.container.conversation_service.create()
            assert (await admin.get(f"/api/conversations/{legacy.id}")).status_code == 404
            assert (await alice.get("/api/conversations")).json()["total"] == 1
    finally:
        app.state.container.close()


@pytest.mark.anyio
@pytest.mark.parametrize("method,path,body", [
    ("GET", "/api/me", None),
    ("GET", "/api/documents", None),
    ("GET", "/api/conversations", None),
    ("POST", "/api/query", {"question": "needle"}),
    ("POST", "/api/query/stream", {"question": "needle"}),
])
async def test_protected_api_rejects_anonymous_before_model_access(
    tmp_path: Path, method: str, path: str, body: object,
) -> None:
    """不能只隐藏管理按钮或只保护新接口。

    Args:
        tmp_path: 隔离所有数据。
        method: HTTP方法。
        path: 受保护的真实路径。
        body: 可选请求体。
    """
    app, _, quality = _app(tmp_path, legacy_identity=False)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://test",
        ) as client:
            response = await client.request(method, path, json=body)
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "unauthenticated"
            assert (await client.get("/api/health")).status_code == 200
            assert quality.evidence_calls == 0
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_login_cookie_csrf_role_and_logout_are_server_enforced(tmp_path: Path) -> None:
    """Cookie不能包含用户角色，写操作必须校验CSRF，成员不能访问管理检索。

    Args:
        tmp_path: 隔离所有数据。
    """
    app, _, quality = _app(tmp_path, legacy_identity=False)
    try:
        user = app.state.container.auth_service.create_user(
            "alice", PASSWORD, name="小刘", role="member",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://test",
        ) as client:
            login = await client.post("/api/auth/login", json={
                "username": "alice", "password": PASSWORD,
            })
            assert login.status_code == 200
            assert login.json()["user"]["id"] == str(user.id)
            session_cookie = [
                value for value in login.headers.get_list("set-cookie")
                if value.startswith("rag_session=")
            ][0].casefold()
            assert "httponly" in session_cookie
            assert "secure" in session_cookie
            assert "samesite=lax" in session_cookie
            token = client.cookies.get("rag_session")
            assert token not in login.text
            assert (await client.get("/api/me")).json()["role"] == "member"
            no_csrf = await client.post("/api/conversations")
            assert no_csrf.status_code == 403
            csrf = client.cookies.get("rag_csrf")
            client.headers["X-CSRF-Token"] = csrf
            assert (await client.post("/api/conversations")).status_code == 201
            forbidden = await client.post("/api/query", json={"question": "needle"})
            assert forbidden.status_code == 403
            assert quality.evidence_calls == 0
            assert (await client.post("/api/auth/logout")).status_code == 204
            assert (await client.get("/api/me")).status_code == 401
            client.cookies.set("rag_session", token)
            assert (await client.get("/api/me")).status_code == 401
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_foreign_origin_is_rejected_even_with_valid_csrf(tmp_path: Path) -> None:
    """不能只依赖SameSite；同站点不同来源也需要Origin约束。

    Args:
        tmp_path: 隔离所有数据。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        app.state.container.auth_service.create_user("alice", PASSWORD, name="Alice")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://test",
        ) as client:
            rejected = await client.post("/api/auth/login", json={
                "username": "alice", "password": PASSWORD,
            }, headers={"Origin": "https://evil.example"})
            assert rejected.status_code == 403
            assert client.cookies.get("rag_session") is None
            await client.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
            response = await client.post("/api/conversations", headers={
                "X-CSRF-Token": client.cookies.get("rag_csrf"),
                "Origin": "https://evil.example",
            })
            assert response.status_code == 403
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_non_ascii_csrf_is_rejected_without_internal_error(tmp_path: Path) -> None:
    """任意字节的请求头不能使恒定时间比较抛TypeError而返回500。

    Args:
        tmp_path: 隔离所有数据。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        app.state.container.auth_service.create_user("alice", PASSWORD, name="Alice")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://test",
        ) as client:
            await client.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
            response = await client.post(
                "/api/conversations", headers={b"X-CSRF-Token": b"\xff"},
            )
            assert response.status_code == 403
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_cookie_cors_allowlist_and_host_rejection(tmp_path: Path) -> None:
    """凭据跨域只对明确来源开放，恶意Host不能用于伪造同源。

    Args:
        tmp_path: 隔离所有数据。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://test",
        ) as client:
            allowed = await client.options("/api/conversations", headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-CSRF-Token,Content-Type",
            })
            assert allowed.status_code == 200
            assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
            assert allowed.headers["access-control-allow-credentials"] == "true"
            foreign = await client.options("/api/conversations", headers={
                "Origin": "https://evil.example", "Access-Control-Request-Method": "POST",
            })
            assert "access-control-allow-origin" not in foreign.headers
            evil_host = await client.get("/api/health", headers={"Host": "evil.example"})
            assert evil_host.status_code == 400
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_real_login_admin_query_and_member_stream_keep_server_identity(tmp_path: Path) -> None:
    """不能只测试拒绝路径；有效Cookie也必须走统一编排和真实追踪。

    Args:
        tmp_path: 隔离所有数据。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        for role in ("member", "admin"):
            user = app.state.container.auth_service.create_user(role, PASSWORD, name=role, role=role)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://test",
            ) as client:
                await client.post("/api/auth/login", json={"username": role, "password": PASSWORD})
                client.headers["X-CSRF-Token"] = client.cookies.get("rag_csrf")
                created = (await client.post("/api/conversations")).json()
                response = await client.post(
                    f"/api/conversations/{created['id']}/query/stream", json={"question": "needle"},
                )
                assert response.status_code == 200
                events = _events(response.text)
                assert events[-1][0] == "complete"
                trace_id = UUID(events[-1][1]["trace_id"])
                assert app.state.container.trace_repository.get(trace_id).user_id == user.id
                if role == "admin":
                    for suffix in ("", "/stream"):
                        query = await client.post("/api/query" + suffix, json={"question": "needle"})
                        assert query.status_code == 200
                        body = _events(query.text)[-1][1] if suffix else query.json()
                        assert app.state.container.trace_repository.get(UUID(body["trace_id"])).user_id == user.id
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_login_failures_rate_limit_untrusted_forwarded_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """轮换账号和伪造X-Forwarded-For不能绕过服务器观察来源额度。

    Args:
        tmp_path: 隔离认证存储。
        monkeypatch: 固定窗口，避免跨分钟导致测试偶发失效。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        auth = app.state.container.auth_service
        now = datetime.now(UTC)
        monkeypatch.setattr(auth, "_clock", lambda: now)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://test",
        ) as client:
            for attempt in range(10):
                response = await client.post("/api/auth/login", json={
                    "username": f"unknown-{attempt}", "password": PASSWORD,
                }, headers={"X-Forwarded-For": f"192.0.2.{attempt}"})
                assert response.status_code == 401
                assert response.json()["error"]["code"] == "invalid_credentials"
            blocked = await client.post("/api/auth/login", json={
                "username": "another", "password": PASSWORD,
            }, headers={"X-Forwarded-For": "192.0.2.200"})
            assert blocked.status_code == 429
            assert int(blocked.headers["Retry-After"]) > 0
            assert blocked.json()["error"]["code"] == "rate_limited"
            assert PASSWORD not in blocked.text
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_auth_storage_failure_is_503_not_anonymous_fallback(tmp_path: Path) -> None:
    """真实SQLite路径不可用时，不回退匿名，也不泄漏文件路径。

    Args:
        tmp_path: 隔离认证库。
    """
    app, _, quality = _app(tmp_path, legacy_identity=False)
    try:
        # 在惰性认证初始化之前使数据库路径不可用，不替换认证或错误映射。
        app.state.container.settings.auth_db_path.mkdir()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://test",
        ) as client:
            response = await client.post("/api/auth/login", json={
                "username": "alice", "password": PASSWORD,
            })
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "auth_unavailable"
            assert str(tmp_path) not in response.text
            assert quality.evidence_calls == 0
    finally:
        app.state.container.close()
