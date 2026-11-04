# -*- coding: utf-8 -*-
"""普通HTTP请求断连同样应取消等待和后续调用，不仅保护SSE。"""

import json
from pathlib import Path
from uuid import uuid4

import anyio
import pytest

from app.application.errors import QuestionQueueFullError
from tests.http_auth import TEST_USER_ID
from tests.test_live_query_streams import _BlockedUnderstanding
from tests.test_live_query_streams import _scope
from tests.test_unified_query_api import _app


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
async def test_disconnect_while_nonstream_queued_does_not_start_question(
    tmp_path: Path, conversation: bool,
) -> None:
    """普通客户端停止等待后不能在腾出名额时仍执行模型。

    Args:
        tmp_path: 所有持久化隔离的应用。
        conversation: 两种非流式入口。
    """
    app, _, quality = _app(tmp_path, settings_overrides={"question_max_active": 1})
    root = app.state.container
    path = "/api/query"
    if conversation:
        created = root.conversation_service.create(owner_id=TEST_USER_ID)
        path = f"/api/conversations/{created.id}/query"
    received = False
    body = json.dumps({"question": "needle"}).encode()
    done = anyio.Event()
    frames = []

    async def receive() -> dict[str, object]:
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        while root.question_admission.quota(TEST_USER_ID).remaining == 10:
            await anyio.sleep(0.005)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        frames.append(message)

    async def request() -> None:
        try:
            await app(_scope(path, body), receive, send)
        finally:
            done.set()

    slot = root.question_admission.acquire(uuid4())
    slot.__enter__()
    try:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(request)
            try:
                with anyio.move_on_after(0.8):
                    await done.wait()
                finished_before_release = done.is_set()
            finally:
                slot.__exit__(None, None, None)
        assert finished_before_release is True
        assert quality.evidence_calls == 0
        status = [item["status"] for item in frames if item["type"] == "http.response.start"]
        assert status == [499]
        if conversation:
            assert root.conversation_service.get(created.id, owner_id=TEST_USER_ID).turns == []
    finally:
        root.close()


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
async def test_disconnect_during_nonstream_model_keeps_permit_until_call_returns(
    tmp_path: Path, conversation: bool,
) -> None:
    """同步SDK不能强杀，但不能提前释放名额或在其返回后继续检索／提交。

    Args:
        tmp_path: 隔离实际应用。
        conversation: 两种非流式入口。
    """
    provider = _BlockedUnderstanding()
    app, _, quality = _app(tmp_path, understanding_factory=lambda *args: provider, settings_overrides={
        "question_max_active": 1, "question_max_waiting": 0,
    })
    root = app.state.container
    path = "/api/query"
    if conversation:
        created = root.conversation_service.create(owner_id=TEST_USER_ID)
        path = f"/api/conversations/{created.id}/query"
    body = json.dumps({"question": "needle"}).encode()
    received = False
    disconnected = anyio.Event()
    frames = []

    async def receive() -> dict[str, object]:
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        while not provider.entered.is_set():
            await anyio.sleep(0.005)
        disconnected.set()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        frames.append(message)

    try:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(app, _scope(path, body), receive, send)
            try:
                with anyio.move_on_after(0.8):
                    await disconnected.wait()
                observed_disconnect = disconnected.is_set()
                with pytest.raises(QuestionQueueFullError):
                    with root.question_admission.acquire(uuid4()):
                        pytest.fail("上游未返回前不能提前释放执行名额")
            finally:
                provider.release.set()
        assert observed_disconnect is True
        assert quality.evidence_calls == 0
        status = [item["status"] for item in frames if item["type"] == "http.response.start"]
        assert status == [499]
        with root.question_admission.acquire(uuid4()):
            pass
        if conversation:
            assert root.conversation_service.get(created.id, owner_id=TEST_USER_ID).turns == []
    finally:
        provider.release.set()
        root.close()
