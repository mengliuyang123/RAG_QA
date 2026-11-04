# -*- coding: utf-8 -*-
"""两个HTTP入口共享个人额度，排队不提前调用模型且错误外壳兼容。"""

from pathlib import Path
from uuid import uuid4
import json

import anyio
import httpx
import pytest

from app.application.errors import ConversationBusyError
from app.application.errors import QuestionQueueFullError
from tests.test_auth_service import PASSWORD
from tests.test_unified_query_api import _app
from tests.test_unified_query_api import _events
from tests.http_auth import TEST_USER_ID
from tests.test_live_query_streams import _scope


@pytest.mark.anyio
async def test_query_and_conversation_share_real_quota_and_429_headers(tmp_path: Path) -> None:
    """如果两个入口各算一次窗口，跨入口第十一次不会被正确拒绝。

    Args:
        tmp_path: 全部真实存储隔离的应用。
    """
    app, _, quality = _app(tmp_path, legacy_identity=False)
    root = app.state.container
    user = root.auth_service.create_user("admin", PASSWORD, name="管理员", role="admin")
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
            login = await client.post("/api/auth/login", json={"username": "admin", "password": PASSWORD})
            assert login.status_code == 200
            client.headers["X-CSRF-Token"] = client.cookies["rag_csrf"]
            conversation = (await client.post("/api/conversations")).json()["id"]
            for _ in range(9):
                with root.question_admission.acquire(user.id):
                    pass
            before = await client.get("/api/me")
            assert before.json()["quota"]["remaining"] == 1
            response = await client.post("/api/query", json={"question": "你好"})
            assert response.status_code == 200
            blocked = await client.post(f"/api/conversations/{conversation}/query", json={"question": "needle"})
            assert blocked.status_code == 429
            assert int(blocked.headers["Retry-After"]) >= 1
            assert blocked.json()["error"]["code"] == "rate_limited"
            assert blocked.json()["error"]["details"]["retry_after"] == int(blocked.headers["Retry-After"])
            streamed = await client.post("/api/query/stream", json={"question": "needle"})
            assert streamed.status_code == 429
            assert "text/event-stream" not in streamed.headers.get("Content-Type", "")
            forged = await client.post("/api/query", json={
                "question": "你好", "user_id": str(uuid4()),
            })
            assert forged.status_code == 429
            detail = await client.get(f"/api/conversations/{conversation}")
            assert detail.json()["turns"] == []
            assert quality.evidence_calls == 0
            assert (await client.get("/api/me")).json()["quota"]["remaining"] == 0
    finally:
        root.close()


def test_streams_report_queued_position_before_any_model_call(tmp_path: Path) -> None:
    """排队状态必须先于理解／模型，不把等待当作已经开始理解。

    Args:
        tmp_path: 隔离实际应用与存储。
    """
    app, _, quality = _app(tmp_path, legacy_identity=False)
    root = app.state.container
    user = root.auth_service.create_user("admin", PASSWORD, name="管理员", role="admin")
    conversation = root.conversation_service.create(owner_id=user.id)
    active = []
    stream = None
    try:
        for _ in range(root.settings.question_max_active):
            slot = root.question_admission.acquire(user.id)
            slot.__enter__()
            active.append(slot)
        stream = root.conversation_service.stream_query(conversation.id, "needle", owner_id=user.id)
        queued = next(stream)
        assert queued.event == "status"
        assert queued.data.phase == "queued"
        assert queued.data.position == 0
        assert quality.evidence_calls == 0
        stream.close()
    finally:
        if stream is not None:
            stream.close()
        for slot in active:
            slot.__exit__(None, None, None)
        root.close()


@pytest.mark.anyio
async def test_ownership_and_invalid_scope_precede_rate_limit(tmp_path: Path) -> None:
    """用户没有额度时也不能用429／404差异探测他人的会话。

    Args:
        tmp_path: 隔离真实身份与会话。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    root = app.state.container
    owner = root.auth_service.create_user("owner", PASSWORD, name="归属者", role="member")
    visitor = root.auth_service.create_user("visitor", PASSWORD, name="访问者", role="admin")
    conversation = root.conversation_service.create(owner_id=owner.id)
    try:
        for _ in range(10):
            with root.question_admission.acquire(visitor.id):
                pass
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
            await client.post("/api/auth/login", json={"username": "visitor", "password": PASSWORD})
            client.headers["X-CSRF-Token"] = client.cookies["rag_csrf"]
            for suffix in ("query", "query/stream"):
                response = await client.post(f"/api/conversations/{conversation.id}/{suffix}", json={"question": "needle"})
                assert response.status_code == 404
            invalid = await client.post("/api/query", json={"question": "needle", "kb_scope": {"mode": "explicit", "kb_ids": ["missing"]}})
            assert invalid.status_code == 422
    finally:
        root.close()


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_queue_timeout_has_one_error_and_no_model_or_formal_turn(
    tmp_path: Path, conversation: bool, streaming: bool,
) -> None:
    """只测纯控制器不够，实际SSE等待截止后不能发complete或保存轮次。

    Args:
        tmp_path: 隔离实际应用。
        conversation: 两个实际入口。
        streaming: 同时检查HTTP503和流内唯一error。
    """
    app, _, quality = _app(tmp_path, legacy_identity=False, settings_overrides={
        "question_max_active": 1, "question_queue_timeout_seconds": 0.05,
    })
    root = app.state.container
    user = root.auth_service.create_user("admin", PASSWORD, name="管理员", role="admin")
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
            await client.post("/api/auth/login", json={"username": "admin", "password": PASSWORD})
            client.headers["X-CSRF-Token"] = client.cookies["rag_csrf"]
            created = (await client.post("/api/conversations")).json()["id"]
            path = "/api/query"
            if conversation:
                path = f"/api/conversations/{created}/query"
            if streaming:
                path += "/stream"
            with root.question_admission.acquire(user.id):
                with anyio.fail_after(1):
                    response = await client.post(path, json={"question": "needle"})
            if streaming:
                assert response.status_code == 200
                events = _events(response.text)
                assert [name for name, _ in events] == ["status", "error"]
                assert events[0][1]["phase"] == "queued"
                assert events[0][1]["position"] == 0
                assert events[-1][1]["code"] == "queue_timeout"
            else:
                assert response.status_code == 503
                assert response.json()["error"]["code"] == "queue_timeout"
            assert quality.evidence_calls == 0
            assert (await client.get(f"/api/conversations/{created}")).json()["turns"] == []
    finally:
        root.close()


@pytest.mark.anyio
async def test_full_queue_rejects_all_four_routes_before_sse_headers_and_does_not_charge(tmp_path: Path) -> None:
    """满队列必须对四个路径一致，且不能发deciding再伪装成流内错误。

    Args:
        tmp_path: 隔离实际应用。
    """
    app, _, quality = _app(tmp_path, legacy_identity=False, settings_overrides={
        "question_max_active": 1, "question_max_waiting": 1,
    })
    root = app.state.container
    user = root.auth_service.create_user("admin", PASSWORD, name="管理员", role="admin")
    stream = None
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
            await client.post("/api/auth/login", json={"username": "admin", "password": PASSWORD})
            client.headers["X-CSRF-Token"] = client.cookies["rag_csrf"]
            created = (await client.post("/api/conversations")).json()["id"]
            with root.question_admission.acquire(user.id):
                stream = root.query_service.stream_query("needle", user_id=user.id)
                queued = await anyio.to_thread.run_sync(next, stream)
                assert queued.data.phase == "queued"
                before = root.question_admission.quota(user.id).remaining
                for path in (
                    "/api/query", "/api/query/stream",
                    f"/api/conversations/{created}/query",
                    f"/api/conversations/{created}/query/stream",
                ):
                    response = await client.post(path, json={"question": "needle"})
                    assert response.status_code == 503
                    assert response.json()["error"]["code"] == "queue_full"
                    assert "text/event-stream" not in response.headers.get("Content-Type", "")
                assert root.question_admission.quota(user.id).remaining == before
                assert quality.evidence_calls == 0
                assert (await client.get("/api/health")).status_code == 200
    finally:
        if stream is not None:
            stream.close()
        root.close()


@pytest.mark.anyio
async def test_same_conversation_stays_busy_while_queued_and_cancellation_releases_it(tmp_path: Path) -> None:
    """排队也属于同会话未完成请求，不能积累同一历史上的多个待执行问题。

    Args:
        tmp_path: 隔离实际应用。
    """
    app, _, quality = _app(tmp_path, legacy_identity=False, settings_overrides={
        "question_max_active": 1,
    })
    root = app.state.container
    user = root.auth_service.create_user("admin", PASSWORD, name="管理员", role="admin")
    conversation = root.conversation_service.create(owner_id=user.id)
    stream = None
    try:
        with root.question_admission.acquire(user.id):
            stream = root.conversation_service.stream_query(conversation.id, "needle", owner_id=user.id)
            queued = await anyio.to_thread.run_sync(next, stream)
            assert queued.data.phase == "queued"
            with pytest.raises(ConversationBusyError):
                root.conversation_service.query(conversation.id, "second", owner_id=user.id)
            stream.close()
            with anyio.fail_after(1):
                while True:
                    try:
                        changed = root.conversation_service.update(conversation.id, title="released", owner_id=user.id)
                        break
                    except ConversationBusyError:
                        await anyio.sleep(0.01)
            assert changed.title == "released"
            assert quality.evidence_calls == 0
    finally:
        if stream is not None:
            stream.close()
        root.close()


@pytest.mark.anyio
async def test_default_full_waiting_queue_keeps_health_responsive(tmp_path: Path) -> None:
    """真实非流式等待会占线程槽，默认满队列也必须保留其他HTTP路由余量。

    Args:
        tmp_path: 隔离应用与所有临时正式记录。
    """
    app, _, quality = _app(tmp_path, settings_overrides={
        "question_rate_limit_per_minute": 100,
    })
    root = app.state.container
    active = []
    responses = []
    try:
        for _ in range(3):
            slot = root.question_admission.acquire(uuid4())
            slot.__enter__()
            active.append(slot)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            async def ask() -> None:
                responses.append(await client.post("/api/query", json={"question": "你好"}))

            async with anyio.create_task_group() as tasks:
                for _ in range(20):
                    tasks.start_soon(ask)
                try:
                    with anyio.fail_after(2):
                        while root.question_admission.quota(TEST_USER_ID).remaining != 80:
                            await anyio.sleep(0.01)
                    with anyio.fail_after(0.8):
                        assert (await client.get("/api/health")).status_code == 200
                    assert responses == []
                    assert quality.evidence_calls == 0
                finally:
                    for slot in reversed(active):
                        slot.__exit__(None, None, None)
                    active.clear()
            assert len(responses) == 20
            assert all(response.status_code == 200 for response in responses)
    finally:
        for slot in reversed(active):
            slot.__exit__(None, None, None)
        root.close()


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
async def test_asgi_disconnect_while_queued_removes_waiter_without_model_calls(
    tmp_path: Path, conversation: bool,
) -> None:
    """真实ASGI断连应关闭新队列ticket，而非仅调用测试迭代器.close。

    Args:
        tmp_path: 隔离实际应用。
        conversation: 两个真实SSE入口。
    """
    app, _, quality = _app(tmp_path, settings_overrides={
        "question_max_active": 1, "question_max_waiting": 1,
    })
    root = app.state.container
    body = json.dumps({"question": "needle"}).encode()
    queued = anyio.Event()
    frames = []
    received = False
    path = "/api/query/stream"
    if conversation:
        created = root.conversation_service.create(owner_id=TEST_USER_ID)
        path = f"/api/conversations/{created.id}/query/stream"

    async def receive() -> dict[str, object]:
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await queued.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        if message["type"] == "http.response.body":
            frame = message.get("body", b"")
            frames.append(frame)
            if b'"phase":"queued"' in frame:
                queued.set()

    class AcceptedProbe(Exception):
        """用实际容量接受证明旧ticket已移除。"""

    def stop_probe(position: int) -> None:
        raise AcceptedProbe()

    try:
        with root.question_admission.acquire(uuid4()):
            with anyio.fail_after(1):
                await app(_scope(path, body), receive, send)
            with anyio.fail_after(1):
                while True:
                    try:
                        with root.question_admission.acquire(uuid4(), on_queued=stop_probe):
                            pytest.fail("执行槽仍占用，不得开始执行")
                    except AcceptedProbe:
                        break
                    except QuestionQueueFullError:
                        await anyio.sleep(0.01)
            assert quality.evidence_calls == 0
            assert [name for name, _ in _events(b"".join(frames).decode())] == ["status"]
    finally:
        root.close()
