# -*- coding: utf-8 -*-
"""反馈必须幂等持久化、只允许本人操作，且不改写已审核答案。"""

from contextlib import AsyncExitStack
from pathlib import Path
from uuid import uuid4
from uuid import UUID

import httpx
import pytest

from tests.test_auth_service import PASSWORD
from tests.test_unified_query_api import _app


@pytest.mark.anyio
async def test_feedback_is_private_idempotent_and_visible_in_history(tmp_path: Path) -> None:
    """省略归属或只更新内存都会导致此真实ASGI与SQLite测试失败。

    Args:
        tmp_path: 隔离全部数据库。
    """
    app, _, quality = _app(tmp_path, legacy_identity=False)
    try:
        async with AsyncExitStack() as stack:
            clients = []
            for username, role in (("alice", "member"), ("bob", "member"), ("admin", "admin")):
                app.state.container.auth_service.create_user(username, PASSWORD, name=username, role=role)
                client = await stack.enter_async_context(httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="https://test",
                ))
                login = await client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
                assert login.status_code == 200
                client.headers["X-CSRF-Token"] = client.cookies.get("rag_csrf")
                clients.append(client)
            alice, bob, admin = clients
            created = (await alice.post("/api/conversations")).json()
            path = f"/api/conversations/{created['id']}"
            answer = (await alice.post(path + "/query", json={"question": "needle"})).json()
            turn_id = answer["turn_id"]
            feedback_path = f"/api/turns/{turn_id}/feedback"
            assert (await alice.get(feedback_path)).json() == {"rating": None, "reasons": [], "comment": None}
            calls = quality.evidence_calls
            for other in (bob, admin):
                assert (await other.put(feedback_path, json={"rating": "down"})).status_code == 404
                assert (await other.get(feedback_path)).status_code == 404
            for _ in range(2):
                assert (await alice.put(feedback_path, json={"rating": "up"})).status_code == 204
            assert (await alice.get(path)).json()["turns"][0]["feedback"] == "up"
            down = {"rating": "down", "reasons": ["incorrect", "citation_wrong"], "comment": "第二步错误"}
            assert (await alice.put(feedback_path, json=down)).status_code == 204
            assert (await alice.get(feedback_path)).json() == down
            after = (await alice.get(path)).json()["turns"][0]
            assert after["feedback"] == "down"
            assert after["assistant_message"] == answer["answer"]
            assert after["citations"] == answer["citations"]
            assert after["trace_id"] == answer["trace_id"]
            trace = app.state.container.trace_repository.get(UUID(answer["trace_id"]))
            assert "第二步错误" not in trace.model_dump_json()
            assert quality.evidence_calls == calls
            csrf = alice.headers.pop("X-CSRF-Token")
            assert (await alice.put(feedback_path, json={"rating": "up"})).status_code == 403
            alice.headers["X-CSRF-Token"] = csrf
            assert (await alice.put(feedback_path, json={"rating": None})).status_code == 204
            assert (await alice.get(feedback_path)).json() == {"rating": None, "reasons": [], "comment": None}
            assert (await alice.get(path)).json()["turns"][0]["feedback"] is None
            absent = await alice.put(f"/api/turns/{uuid4()}/feedback", json={"rating": "up"})
            assert absent.status_code == 404
            assert (await alice.delete(path)).status_code == 200
            assert (await alice.get(feedback_path)).status_code == 404
    finally:
        app.state.container.close()


@pytest.mark.anyio
@pytest.mark.parametrize("body", [
    {"rating": "bad"}, {"rating": "down", "reasons": ["arbitrary"]},
    {"rating": "down", "comment": "字" * 501},
    {"rating": "up", "user_id": str(uuid4())},
])
async def test_feedback_rejects_invalid_body_and_owner_forgery(tmp_path: Path, body: dict[str, object]) -> None:
    """不能接受伪造归属、未定义原因或超长正文。

    Args:
        tmp_path: 隔离全部数据库。
        body: 非法输入。
    """
    app, _, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.put(f"/api/turns/{uuid4()}/feedback", json=body)
            assert response.status_code == 422
    finally:
        app.state.container.close()
