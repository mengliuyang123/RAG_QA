# -*- coding: utf-8 -*-
"""知识缺口状态接口：管理员权限、状态流转与"已验证必须复测通过"规则。"""

from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import pytest

from tests.test_auth_service import PASSWORD
from tests.test_unified_query_api import _app


CHECK_OK = {
    "checked_at": "2026-11-17T08:00:00Z", "kb_id": "default",
    "results": [
        {"question": "Redis 缓存怎么配置", "outcome": "answered", "answered": True},
        {"question": "Redis 过期时间", "outcome": "answered", "answered": True},
    ],
}


def _gap(**overrides: object) -> dict[str, object]:
    return {
        "source": "refusal_cluster", "title": "Redis 缓存怎么配置",
        "questions": ["Redis 缓存怎么配置", "Redis 过期时间"],
        "kb_id": "default", "status": "open", **overrides,
    }


@pytest.mark.anyio
async def test_gap_status_flow_and_verification_rule(tmp_path: Path) -> None:
    """open → document_added → verified；未通过复测不能标记已验证。"""
    app, _, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            assert (await client.get("/api/admin/gaps")).json() == {"gaps": []}
            path = "/api/admin/gaps/cluster:cl_0123456789abcdef01234567"
            opened = await client.put(path, json=_gap())
            assert opened.status_code == 200
            assert opened.json()["status"] == "open"

            added = await client.put(path, json=_gap(status="document_added", doc_id="doc-1"))
            assert added.json()["status"] == "document_added"
            assert added.json()["doc_id"] == "doc-1"

            for check in (None, {**CHECK_OK, "results": [
                CHECK_OK["results"][0],
                {"question": "Redis 过期时间", "outcome": "refused", "answered": False},
            ]}):
                rejected = await client.put(path, json=_gap(status="verified", last_check=check))
                assert rejected.status_code == 422
                assert rejected.json()["error"]["code"] == "gap_not_verified"

            # 复测未通过时可以只记录结果，状态保持 document_added。
            partial = await client.put(path, json=_gap(
                status="document_added", doc_id="doc-1",
                last_check={**CHECK_OK, "results": [
                    {"question": "Redis 过期时间", "outcome": "refused", "answered": False},
                ]},
            ))
            assert partial.status_code == 200
            assert partial.json()["last_check"]["results"][0]["answered"] is False

            verified = await client.put(path, json=_gap(
                status="verified", doc_id="doc-1", last_check=CHECK_OK,
            ))
            assert verified.status_code == 200
            assert verified.json()["status"] == "verified"

            await client.put("/api/admin/gaps/feedback:7d7c3f1e-1111-4222-8333-944455556666", json=_gap(
                source="feedback", title="灰度发布流程", questions=["灰度发布流程"], kb_id=None,
            ))
            listed = (await client.get("/api/admin/gaps")).json()["gaps"]
            assert [gap["gap_key"] for gap in listed] == [
                "feedback:7d7c3f1e-1111-4222-8333-944455556666",
                "cluster:cl_0123456789abcdef01234567",
            ]
            assert listed[1]["last_check"]["results"] == CHECK_OK["results"]
            assert listed[1]["questions"] == ["Redis 缓存怎么配置", "Redis 过期时间"]

            for bad_path, body in (
                ("/api/admin/gaps/other:x", _gap()),
                ("/api/admin/gaps/cluster:a%20b", _gap()),
                (path, _gap(questions=[])),
                (path, _gap(questions=[f"q{index}" for index in range(11)])),
                (path, _gap(status="closed")),
                (path, _gap(kb_id="../etc")),
            ):
                assert (await client.put(bad_path, json=body)).status_code == 422
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_gap_endpoints_require_admin(tmp_path: Path) -> None:
    """普通成员不能读取或修改缺口状态。"""
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        app.state.container.auth_service.create_user("alice", PASSWORD, name="a", role="member")
        async with AsyncExitStack() as stack:
            anonymous = await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://test",
            ))
            assert (await anonymous.get("/api/admin/gaps")).status_code == 401
            member = await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://test",
            ))
            await member.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
            member.headers["X-CSRF-Token"] = member.cookies.get("rag_csrf")
            assert (await member.get("/api/admin/gaps")).status_code == 403
            assert (await member.put("/api/admin/gaps/cluster:cl_x", json=_gap())).status_code == 403
    finally:
        app.state.container.close()


def test_migration_records_version_7(tmp_path: Path) -> None:
    """缺口表作为第7版迁移创建。"""
    app, _, _ = _app(tmp_path)
    try:
        with app.state.container.database.connection() as connection:
            assert connection.execute(
                "SELECT version FROM schema_migrations WHERE version=7",
            ).fetchone() is not None
            assert connection.execute("SELECT count(*) FROM knowledge_gaps").fetchone()[0] == 0
    finally:
        app.state.container.close()
