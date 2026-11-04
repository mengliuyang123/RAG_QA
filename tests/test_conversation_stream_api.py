# -*- coding: utf-8 -*-
"""多轮会话安全 SSE 编码与断连关闭测试。"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

import anyio
import httpx
import pytest
from fastapi import FastAPI

from tests.http_auth import allow_test_identity
from starlette.types import Message
from starlette.types import Scope

from app.application.conversations.events import CompleteConversationEvent
from app.application.conversations.events import StatusConversationData
from app.application.conversations.events import StatusConversationEvent
from app.application.conversations.events import TokenConversationData
from app.application.conversations.events import TokenConversationEvent
from app.domain.conversations import AnswerMode
from app.domain.conversations import ConversationQueryResult
from app.interfaces.http.dependencies import get_conversation_service
from app.interfaces.http.router import router


class _ClosableEvents:
    def __init__(self, events: list[object], endless: bool = False) -> None:
        self._events = iter(events)
        self.endless = endless
        self.closed = False

    def __iter__(self) -> "_ClosableEvents":
        return self

    def __next__(self) -> object:
        try:
            return next(self._events)
        except StopIteration:
            if self.endless:
                return TokenConversationEvent(
                    data=TokenConversationData(token="继续")
                )
            raise

    def close(self) -> None:
        self.closed = True


class _StreamingService:
    def __init__(self, iterator: _ClosableEvents) -> None:
        self.iterator = iterator
        self.calls: list[tuple[UUID, str]] = []

    def stream_query(
        self,
        conversation_id: UUID,
        question: str,
        *, owner_id: UUID | None = None,
    ) -> Iterator[object]:
        self.calls.append((conversation_id, question))
        return self.iterator


def _app(service: _StreamingService) -> FastAPI:
    application = FastAPI()
    allow_test_identity(application)
    application.include_router(router)
    application.dependency_overrides[get_conversation_service] = (
        lambda: service
    )
    return application


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_conversation_sse_encodes_only_safe_validated_events() -> None:
    conversation_id = uuid4()
    turn_id = uuid4()
    iterator = _ClosableEvents(
        [
            StatusConversationEvent(
                data=StatusConversationData(phase="deciding")
            ),
            StatusConversationEvent(
                data=StatusConversationData(phase="retrieving", attempt=1)
            ),
            StatusConversationEvent(
                data=StatusConversationData(phase="validating")
            ),
            TokenConversationEvent(
                data=TokenConversationData(token="最终回答")
            ),
            CompleteConversationEvent(
                data=ConversationQueryResult(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    answer="最终回答",
                    answer_mode=AnswerMode.DIRECT,
                    retrieval_calls=0,
                    citations=[],
                    latency_ms=8.5,
                )
            ),
        ]
    )
    service = _StreamingService(iterator)
    transport = httpx.ASGITransport(app=_app(service))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/api/conversations/{conversation_id}/query/stream",
            json={"question": "  你好  "},
        )

    frames = response.text.split("\n\n")[:-1]
    assert response.status_code == 200
    assert [frame.split("\n", 1)[0] for frame in frames] == [
        "event: status",
        "event: status",
        "event: status",
        "event: token",
        "event: complete",
    ]
    assert "scratchpad" not in response.text
    assert "reasoning" not in response.text
    assert service.calls == [(conversation_id, "你好")]
    assert iterator.closed is True


@pytest.mark.anyio
async def test_conversation_sse_disconnect_closes_application_iterator(
) -> None:
    conversation_id = uuid4()
    iterator = _ClosableEvents(
        [
            StatusConversationEvent(
                data=StatusConversationData(phase="deciding")
            )
        ],
        endless=True,
    )
    service = _StreamingService(iterator)
    application = _app(service)
    request_body = json.dumps({"question": "问题"}).encode()
    first_body_sent = anyio.Event()
    received = False

    async def receive() -> Message:
        nonlocal received
        if not received:
            received = True
            return {
                "type": "http.request",
                "body": request_body,
                "more_body": False,
            }
        await first_body_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            first_body_sent.set()

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": f"/api/conversations/{conversation_id}/query/stream",
        "raw_path": (
            f"/api/conversations/{conversation_id}/query/stream".encode()
        ),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(request_body)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
        "state": {},
    }

    await application(scope, receive, send)

    assert first_body_sent.is_set()
    assert iterator.closed is True
