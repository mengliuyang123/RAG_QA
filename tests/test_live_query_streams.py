# -*- coding: utf-8 -*-
"""真实ASGI传输应实时发送阶段；断连不能继续计费或提交会话。"""

import json
from pathlib import Path
from threading import Event

import anyio
import httpx
import pytest
from starlette.types import Message
from starlette.types import Scope

from app.application.ports.structured_output import ResponseModel
from app.domain.agent_quality import AnswerAssessment
from app.domain.agent_quality import EvidenceItem
from app.domain.conversations import KNOWLEDGE_REFUSAL_ANSWER
from tests.test_query_understanding import _payload
from tests.test_unified_query_api import _app
from tests.test_unified_query_api import _events


class _BlockedUnderstanding:
    """仅模拟不能被Python强制打断的一次同步上游请求。"""

    def __init__(self) -> None:
        """建立测试控制的请求生命周期信号。"""
        self.entered = Event()
        self.release = Event()
        self.exited = Event()

    def generate(
        self, messages: list[dict[str, object]],
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        """等待上游放行后返回理解结果。

        Args:
            messages: 实际理解请求。
            response_model: 严格领域模型。

        Returns:
            有效独立查询。
        """
        self.entered.set()
        try:
            assert self.release.wait(timeout=3)
            return response_model.model_validate(_payload(
                standalone_query="needle", preferred_kb_ids=[],
            ))
        finally:
            self.exited.set()


def _scope(path: str, body: bytes) -> Scope:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "state": {},
        "headers": [
            (b"host", b"test"), (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 12345), "server": ("test", 80),
    }


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
async def test_stage_is_sent_while_understanding_model_is_still_running(
    tmp_path: Path, conversation: bool,
) -> None:
    """若阶段仍缓冲，客户端在模型放行前就收不到understanding。

    Args:
        tmp_path: 全部持久化的隔离路径。
        conversation: 分别覆盖无状态与正式会话入口。
    """
    provider = _BlockedUnderstanding()
    app, _, _ = _app(tmp_path, understanding_factory=lambda *args: provider)
    body = json.dumps({"question": "needle"}).encode()
    stage_sent = anyio.Event()
    frames = []
    path = "/api/query/stream"
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            if conversation:
                created = (await client.post("/api/conversations")).json()
                path = f"/api/conversations/{created['id']}/query/stream"
            request_received = False

            async def receive() -> Message:
                nonlocal request_received
                if not request_received:
                    request_received = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await anyio.sleep_forever()

            async def send(message: Message) -> None:
                if message["type"] == "http.response.body":
                    frames.append(message.get("body", b""))
                    if b'"phase":"understanding"' in frames[-1]:
                        stage_sent.set()

            async with anyio.create_task_group() as group:
                group.start_soon(app, _scope(path, body), receive, send)
                try:
                    with anyio.fail_after(0.8):
                        await stage_sent.wait()
                    assert not provider.exited.is_set()
                    with anyio.fail_after(0.8):
                        assert (await client.get("/api/health")).status_code == 200
                finally:
                    provider.release.set()
        events = _events(b"".join(frames).decode())
        assert sum(name == "complete" for name, _ in events) == 1
        assert not any(name == "error" for name, _ in events)
        tokens = "".join(data["token"] for name, data in events if name == "token")
        assert tokens == "资料[1]"
    finally:
        provider.release.set()
        app.state.container.close()


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
async def test_disconnect_returns_promptly_and_never_continues_after_model(
    tmp_path: Path, conversation: bool,
) -> None:
    """同步请求尚未返回时先取消消费；不能提前释放会话锁或继续模型流水线。

    Args:
        tmp_path: 隔离数据目录。
        conversation: 两个HTTP入口分别验证。
    """
    provider = _BlockedUnderstanding()
    app, _, quality = _app(tmp_path, understanding_factory=lambda *args: provider)
    body = json.dumps({"question": "needle"}).encode()
    stage_sent = anyio.Event()
    request_done = anyio.Event()
    frames = []
    path = "/api/query/stream"
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            if conversation:
                created = (await client.post("/api/conversations")).json()
                path = f"/api/conversations/{created['id']}/query/stream"
            request_received = False

            async def receive() -> Message:
                nonlocal request_received
                if not request_received:
                    request_received = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await stage_sent.wait()
                while not provider.entered.is_set():
                    await anyio.sleep(0.005)
                return {"type": "http.disconnect"}

            async def send(message: Message) -> None:
                if message["type"] == "http.response.body":
                    frames.append(message.get("body", b""))
                    if b'"phase":"understanding"' in frames[-1]:
                        stage_sent.set()

            async def run_request() -> None:
                try:
                    await app(_scope(path, body), receive, send)
                finally:
                    request_done.set()

            async with anyio.create_task_group() as group:
                group.start_soon(run_request)
                try:
                    with anyio.fail_after(0.8):
                        await request_done.wait()
                    assert not provider.exited.is_set()
                    if conversation:
                        busy = await client.post(
                            f"/api/conversations/{created['id']}/query",
                            json={"question": "你好"},
                        )
                        assert busy.status_code == 409
                finally:
                    provider.release.set()
            with anyio.fail_after(1):
                while not provider.exited.is_set():
                    await anyio.sleep(0.005)
                if conversation:
                    service = app.state.container.conversation_service
                    while service._conversation_locks:
                        await anyio.sleep(0.005)
                    detail = (await client.get(
                        f"/api/conversations/{created['id']}",
                    )).json()
                    assert detail["turns_count"] == 0
                    assert detail["turns"] == []
            assert quality.evidence_calls == 0
            assert not any(
                name in {"token", "complete", "error"}
                for name, _ in _events(b"".join(frames).decode())
            )
    finally:
        provider.release.set()
        app.state.container.close()


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
async def test_unsupported_agent_draft_is_sent_as_partial_answer(
    tmp_path: Path, conversation: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """编号合法但审核不通过时，修复后仍不通过则按部分依据发布，不再整体拒答。

    Args:
        tmp_path: 隔离真实索引和正式历史。
        conversation: 两种流式入口。
        monkeypatch: 仅替代外部裁判的支持性输出。
    """
    app, _, quality = _app(tmp_path)

    def reject(
        question: str, history: list[dict[str, str]], answer: str,
        evidence: list[EvidenceItem],
    ) -> AnswerAssessment:
        return AnswerAssessment(
            supported=False, complete=True, issues=["存储结论未获支持"],
        )

    monkeypatch.setattr(quality, "assess_answer", reject)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            if conversation:
                created = (await client.post("/api/conversations")).json()
                path = f"/api/conversations/{created['id']}/query/stream"
                request = {"question": "比较needle存储"}
            else:
                path = "/api/query/stream"
                request = {"question": "needle", "lane": "agent"}
            response = await client.post(path, json=request)
            events = _events(response.text)
            completed = [data for name, data in events if name == "complete"]
            assert len(completed) == 1
            body = completed[0]
            assert body["outcome"] == "answered"
            assert body["grounding"] == "partial"
            assert body["refusal"] is None
            assert body["citations"]
            tokens = "".join(data["token"] for name, data in events if name == "token")
            assert tokens != KNOWLEDGE_REFUSAL_ANSWER
            assert tokens == body["answer"]
            if conversation:
                saved = (await client.get(
                    f"/api/conversations/{created['id']}",
                )).json()["turns"][0]
                assert saved["outcome"] == "answered"
                assert saved["grounding"] == "partial"
                assert saved["citations"] == body["citations"]
    finally:
        app.state.container.close()


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("agent", [False, True])
async def test_retrieving_status_has_actual_scope_and_safe_label(
    tmp_path: Path, conversation: bool, agent: bool,
) -> None:
    """前端应按实际范围展示阶段，不能把用户请求范围猜成已执行范围。

    Args:
        tmp_path: 隔离数据目录。
        conversation: 两种流式入口。
        agent: 分别覆盖Fast和真实图工具阶段。
    """
    app, _, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            path = "/api/query/stream"
            if conversation:
                created = (await client.post("/api/conversations")).json()
                path = f"/api/conversations/{created['id']}/query/stream"
            request = {
                "question": "比较needle存储" if agent else "needle",
                "kb_scope": {"mode": "explicit", "kb_ids": ["default"]},
            }
            if agent and not conversation:
                request["lane"] = "agent"
            response = await client.post(path, json=request)
            statuses = [
                data for name, data in _events(response.text)
                if name == "status" and data["phase"] == "retrieving"
            ]
            assert statuses
            assert statuses[0]["attempt"] == 1
            assert statuses[0]["kb_ids"] == ["default"]
            assert statuses[0]["label"] == "正在查找资料"
    finally:
        app.state.container.close()
