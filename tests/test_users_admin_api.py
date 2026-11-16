# -*- coding: utf-8 -*-
"""管理员账号管理：权限、口令不回显、自我锁定、最后管理员和会话撤销规则。"""

import logging
from contextlib import AsyncExitStack
from pathlib import Path
from uuid import UUID
from uuid import uuid4

import httpx
import pytest

from app.application.errors import LastAdminError
from app.application.errors import SelfLockoutError
from app.application.errors import UserNotFoundError
from tests.test_auth_service import PASSWORD
from tests.test_auth_service import _service
from tests.test_unified_query_api import _app


NEW_PASSWORD = "replace-me-offline-passphrase"


async def _login(stack: AsyncExitStack, app, username: str, password: str = PASSWORD) -> httpx.AsyncClient:
    client = await stack.enter_async_context(httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://test",
    ))
    login = await client.post("/api/auth/login", json={"username": username, "password": password})
    assert login.status_code == 200, login.text
    client.headers["X-CSRF-Token"] = client.cookies.get("rag_csrf")
    return client


@pytest.fixture
def secured_app(tmp_path: Path):
    app, _, _ = _app(tmp_path, legacy_identity=False)
    auth = app.state.container.auth_service
    auth.create_user("root", PASSWORD, name="管理员", role="admin")
    auth.create_user("alice", PASSWORD, name="小刘", role="member")
    try:
        yield app
    finally:
        app.state.container.close()


def _id_of(users: list[dict], username: str) -> str:
    return next(user["id"] for user in users if user["username"] == username)


@pytest.mark.anyio
async def test_user_endpoints_require_admin(secured_app) -> None:
    """匿名返回401，普通成员返回403，不能只靠前端隐藏菜单。"""
    target = str(uuid4())
    calls = [
        ("GET", "/api/admin/users", None),
        ("POST", "/api/admin/users", {"username": "eve", "name": "e", "password": PASSWORD}),
        ("PATCH", f"/api/admin/users/{target}", {"role": "admin"}),
        ("POST", f"/api/admin/users/{target}/password", {"password": NEW_PASSWORD}),
    ]
    async with AsyncExitStack() as stack:
        anonymous = await stack.enter_async_context(httpx.AsyncClient(
            transport=httpx.ASGITransport(app=secured_app), base_url="https://test",
        ))
        member = await _login(stack, secured_app, "alice")
        for method, path, body in calls:
            assert (await anonymous.request(method, path, json=body)).status_code == 401
            forbidden = await member.request(method, path, json=body)
            assert forbidden.status_code == 403
            assert forbidden.json()["error"]["code"] == "forbidden"
        # 成员不能借此把自己提升为管理员。
        assert (await member.get("/api/me")).json()["role"] == "member"


@pytest.mark.anyio
async def test_admin_lists_and_creates_users_without_exposing_passwords(secured_app, caplog) -> None:
    """响应和日志都不包含口令或哈希；重复登录名返回409。"""
    caplog.set_level(logging.DEBUG)
    async with AsyncExitStack() as stack:
        admin = await _login(stack, secured_app, "root")
        listed = await admin.get("/api/admin/users")
        assert listed.status_code == 200
        assert {user["username"] for user in listed.json()["users"]} == {"root", "alice"}
        assert "password" not in listed.text and "argon2" not in listed.text

        created = await admin.post("/api/admin/users", json={
            "username": " Bob ", "name": " 小王 ", "password": NEW_PASSWORD, "role": "admin",
        })
        assert created.status_code == 201
        body = created.json()
        assert body["username"] == "bob" and body["name"] == "小王"
        assert body["role"] == "admin" and body["disabled"] is False
        assert set(body) == {"id", "username", "name", "role", "disabled", "created_at"}
        assert NEW_PASSWORD not in created.text

        duplicate = await admin.post("/api/admin/users", json={
            "username": "BOB", "name": "x", "password": NEW_PASSWORD,
        })
        assert duplicate.status_code == 409
        assert duplicate.json()["error"]["code"] == "user_exists"

        short = "too-short-pw"
        rejected = await admin.post("/api/admin/users", json={
            "username": "carol", "name": "c", "password": short,
        })
        assert rejected.status_code == 422
        assert short not in rejected.text

        # 新建的管理员可以直接登录并访问后台。
        bob = await _login(stack, secured_app, "bob", NEW_PASSWORD)
        assert (await bob.get("/api/admin/users")).status_code == 200
    assert NEW_PASSWORD not in caplog.text
    assert short not in caplog.text


@pytest.mark.anyio
async def test_role_change_applies_on_next_request_and_keeps_session(secured_app) -> None:
    """角色每次请求从数据库读取：提升与降级都立即生效，会话不被撤销。"""
    async with AsyncExitStack() as stack:
        admin = await _login(stack, secured_app, "root")
        alice = await _login(stack, secured_app, "alice")
        alice_id = _id_of((await admin.get("/api/admin/users")).json()["users"], "alice")
        assert (await alice.get("/api/admin/users")).status_code == 403

        promoted = await admin.patch(f"/api/admin/users/{alice_id}", json={"role": "admin"})
        assert promoted.status_code == 200 and promoted.json()["role"] == "admin"
        assert (await alice.get("/api/admin/users")).status_code == 200

        demoted = await admin.patch(f"/api/admin/users/{alice_id}", json={"role": "member"})
        assert demoted.json()["role"] == "member"
        assert (await alice.get("/api/admin/users")).status_code == 403
        assert (await alice.get("/api/me")).status_code == 200

        empty = await admin.patch(f"/api/admin/users/{alice_id}", json={})
        assert empty.status_code == 422
        missing = await admin.patch(f"/api/admin/users/{uuid4()}", json={"role": "admin"})
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "user_not_found"


@pytest.mark.anyio
async def test_disable_revokes_sessions_and_blocks_login_until_enabled(secured_app) -> None:
    """禁用立即撤销全部会话；重新启用不恢复旧会话，需要重新登录。"""
    async with AsyncExitStack() as stack:
        admin = await _login(stack, secured_app, "root")
        alice = await _login(stack, secured_app, "alice")
        alice_id = _id_of((await admin.get("/api/admin/users")).json()["users"], "alice")

        disabled = await admin.patch(f"/api/admin/users/{alice_id}", json={"disabled": True})
        assert disabled.status_code == 200 and disabled.json()["disabled"] is True
        assert (await alice.get("/api/me")).status_code == 401
        refused = await alice.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
        assert refused.status_code == 401

        enabled = await admin.patch(f"/api/admin/users/{alice_id}", json={"disabled": False})
        assert enabled.json()["disabled"] is False
        assert (await alice.get("/api/me")).status_code == 401
        again = await _login(stack, secured_app, "alice")
        assert (await again.get("/api/me")).status_code == 200

        with secured_app.state.container.database.connection() as connection:
            remaining = connection.execute(
                "SELECT count(*) FROM auth_sessions WHERE user_id = %s", (alice_id,),
            ).fetchone()[0]
        assert remaining == 1


@pytest.mark.anyio
async def test_admin_cannot_demote_or_disable_self(secured_app) -> None:
    """自我降级或禁用返回409，账号保持原状。"""
    async with AsyncExitStack() as stack:
        admin = await _login(stack, secured_app, "root")
        root_id = _id_of((await admin.get("/api/admin/users")).json()["users"], "root")
        for change in ({"role": "member"}, {"disabled": True}):
            response = await admin.patch(f"/api/admin/users/{root_id}", json=change)
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "self_lockout"
        # 不改变权限的自我修改是允许的。
        assert (await admin.patch(f"/api/admin/users/{root_id}", json={"role": "admin"})).status_code == 200
        assert (await admin.get("/api/admin/users")).status_code == 200


@pytest.mark.anyio
async def test_admin_can_demote_another_admin_but_not_the_last_one(secured_app) -> None:
    """有其他启用的管理员时允许降级；最后一名启用管理员受保护。"""
    auth = secured_app.state.container.auth_service
    async with AsyncExitStack() as stack:
        admin = await _login(stack, secured_app, "root")
        created = await admin.post("/api/admin/users", json={
            "username": "bob", "name": "b", "password": NEW_PASSWORD, "role": "admin",
        })
        bob_id = created.json()["id"]
        root_id = _id_of((await admin.get("/api/admin/users")).json()["users"], "root")
        assert (await admin.patch(f"/api/admin/users/{bob_id}", json={"disabled": True})).status_code == 200
    # HTTP 调用者本身就是启用的管理员，最后管理员规则在仓储事务中兜底（例如另一管理员已并发降级）。
    with pytest.raises(LastAdminError):
        auth.update_user(uuid4(), UUID(root_id), role="member")
    with pytest.raises(LastAdminError):
        auth.update_user(uuid4(), UUID(root_id), disabled=True)
    assert {user.username: user.role for user in auth.list_users()}["root"] == "admin"


@pytest.mark.anyio
async def test_reset_password_revokes_all_sessions_of_target(secured_app, caplog) -> None:
    """重置口令后旧口令失效，该账号全部浏览器需重新登录，其他人的会话不受影响。"""
    caplog.set_level(logging.DEBUG)
    async with AsyncExitStack() as stack:
        admin = await _login(stack, secured_app, "root")
        alice = await _login(stack, secured_app, "alice")
        alice_id = _id_of((await admin.get("/api/admin/users")).json()["users"], "alice")

        reset = await admin.post(f"/api/admin/users/{alice_id}/password", json={"password": NEW_PASSWORD})
        assert reset.status_code == 204 and reset.content == b""
        assert (await alice.get("/api/me")).status_code == 401
        assert (await admin.get("/api/me")).status_code == 200
        old = await alice.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
        assert old.status_code == 401
        await _login(stack, secured_app, "alice", NEW_PASSWORD)

        short = await admin.post(f"/api/admin/users/{alice_id}/password", json={"password": "short-pass"})
        assert short.status_code == 422 and "short-pass" not in short.text
        missing = await admin.post(f"/api/admin/users/{uuid4()}/password", json={"password": NEW_PASSWORD})
        assert missing.status_code == 404
    assert NEW_PASSWORD not in caplog.text


def test_service_rules_without_http(tmp_path: Path) -> None:
    """服务层同样执行自我锁定与最后管理员规则，可供命令行等其他入口复用。"""
    service = _service(tmp_path)
    only = service.create_user("root", PASSWORD, name="r", role="admin")
    member = service.admin_create_user("alice", PASSWORD, name="a", role="member")
    with pytest.raises(SelfLockoutError):
        service.update_user(only.id, only.id, disabled=True)
    with pytest.raises(LastAdminError):
        service.update_user(member.id, only.id, role="member")
    with pytest.raises(UserNotFoundError):
        service.reset_password(uuid4(), NEW_PASSWORD)
    with pytest.raises(ValueError):
        service.reset_password(member.id, "short")
    promoted = service.update_user(only.id, member.id, role="admin")
    assert promoted.role == "admin"
    assert service.update_user(member.id, only.id, role="member").role == "member"
    assert all("password" not in user.model_dump() for user in service.list_users())
