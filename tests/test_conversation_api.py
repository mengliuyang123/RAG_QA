# -*- coding: utf-8 -*-
"""多轮会话 CRUD 与非流式查询 HTTP 契约测试。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from app.application.errors import ConversationAgentError
from app.application.errors import ConversationBusyError
from app.application.errors import ConversationDeleteError
from app.application.errors import ConversationDeletingError
from app.application.errors import ConversationNotFoundError
from app.application.errors import ConversationStorageError
from app.domain.conversations import AnswerMode
from app.domain.conversations import Conversation
from app.domain.conversations import ConversationQueryResult
from app.domain.conversations import ConversationStatus
from app.domain.conversations import ConversationSummary
from app.domain.conversations import ConversationTurn
from app.domain.conversations import TurnCitation
from app.interfaces.http.dependencies import get_conversation_service
from app.interfaces.http.error_handlers import register_error_handlers
from app.interfaces.http.router import router
from tests.api_compat import legacy_answer_view
from tests.http_auth import allow_test_identity


class _FakeConversationService:
    def __init__(self) -> None:
        self.conversation_id = uuid4()
        self.turn_id = uuid4()
        self.now = datetime(2026, 8, 26, tzinfo=UTC)
        self.queries: list[tuple[UUID, str]] = []
        self.error: Exception | None = None

    def _raise(self) -> None:
        if self.error is not None:
            raise self.error

    def create(self, *, owner_id: UUID | None = None) -> Conversation:
        self._raise()
        return self._conversation([])

    def list(self, *, owner_id: UUID | None = None) -> list[ConversationSummary]:
        self._raise()
        return [
            ConversationSummary(
                id=self.conversation_id,
                title="新对话",
                status=ConversationStatus.ACTIVE,
                created_at=self.now,
                updated_at=self.now,
                turns_count=0,
            )
        ]

    def get(self, conversation_id: UUID, *, owner_id: UUID | None = None) -> Conversation:
        self._raise()
        assert conversation_id == self.conversation_id
        turn = ConversationTurn(
            id=self.turn_id,
            conversation_id=conversation_id,
            user_message="问题",
            assistant_message="回答[1]",
            answer_mode=AnswerMode.RETRIEVAL,
            retrieval_calls=1,
            citations=[self._citation()],
            latency_ms=12.5,
            created_at=self.now,
        )
        return self._conversation([turn])

    def delete(self, conversation_id: UUID, *, owner_id: UUID | None = None) -> None:
        self._raise()
        assert conversation_id == self.conversation_id

    def query(
        self,
        conversation_id: UUID,
        question: str,
        *, owner_id: UUID | None = None,
    ) -> ConversationQueryResult:
        self._raise()
        self.queries.append((conversation_id, question))
        return ConversationQueryResult(
            conversation_id=conversation_id,
            turn_id=self.turn_id,
            answer="回答[1]",
            answer_mode=AnswerMode.RETRIEVAL,
            retrieval_calls=1,
            citations=[self._citation()],
            latency_ms=12.5,
        )

    def _conversation(
        self,
        turns: list[ConversationTurn],
    ) -> Conversation:
        return Conversation(
            id=self.conversation_id,
            title="新对话" if not turns else "问题",
            status=ConversationStatus.ACTIVE,
            created_at=self.now,
            updated_at=self.now,
            turns_count=len(turns),
            turns=turns,
        )

    @staticmethod
    def _citation() -> TurnCitation:
        return TurnCitation(
            position=1,
            doc_id="doc-a",
            filename="a.md",
            page=None,
            chunk_text="证据",
        )


def _app(service: _FakeConversationService) -> FastAPI:
    application = FastAPI()
    allow_test_identity(application)
    register_error_handlers(application)
    application.include_router(router)
    application.dependency_overrides[get_conversation_service] = (
        lambda: service
    )
    return application


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_conversation_crud_and_query_have_stable_shapes() -> None:
    service = _FakeConversationService()
    transport = httpx.ASGITransport(app=_app(service))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        created = await client.post("/api/conversations")
        listed = await client.get("/api/conversations")
        detail = await client.get(
            f"/api/conversations/{service.conversation_id}"
        )
        queried = await client.post(
            f"/api/conversations/{service.conversation_id}/query",
            json={"question": "  第三步是什么意思？  "},
        )
        deleted = await client.delete(
            f"/api/conversations/{service.conversation_id}"
        )

    assert created.status_code == 201
    assert created.json()["id"] == str(service.conversation_id)
    assert listed.json()["total"] == 1
    assert detail.json()["turns"][0]["citations"] == [
        {
            "doc_id": "doc-a",
            "filename": "a.md",
            "page": None,
            "chunk_text": "证据",
            "position": 1,
            "chunk_id": None, "kb_id": "default", "doc_updated_at": None,
        }
    ]
    assert legacy_answer_view(queried.json(), (
        "conversation_id", "turn_id", "answer", "answer_mode",
        "retrieval_calls", "citations", "latency_ms",
    )) == {
        "conversation_id": str(service.conversation_id),
        "turn_id": str(service.turn_id),
        "answer": "回答[1]",
        "answer_mode": "retrieval",
        "retrieval_calls": 1,
        "citations": [
            {
                "doc_id": "doc-a",
                "filename": "a.md",
                "page": None,
                "chunk_text": "证据",
                "position": 1,
            }
        ],
        "latency_ms": 12.5,
    }
    assert "strategy" not in queried.json()
    assert deleted.json() == {
        "conversation_id": str(service.conversation_id),
        "deleted": True,
    }
    assert service.queries == [
        (service.conversation_id, "第三步是什么意思？")
    ]


@pytest.mark.anyio
async def test_conversation_query_forbids_strategy_top_k_and_blank_question(
) -> None:
    service = _FakeConversationService()
    transport = httpx.ASGITransport(app=_app(service))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        extra = await client.post(
            f"/api/conversations/{service.conversation_id}/query",
            json={"question": "问题", "strategy": "vector", "top_k": 20},
        )
        blank = await client.post(
            f"/api/conversations/{service.conversation_id}/query",
            json={"question": "   "},
        )

    assert extra.status_code == 422
    assert blank.status_code == 422
    assert service.queries == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (ConversationNotFoundError("private"), 404, "会话不存在"),
        (ConversationBusyError("private"), 409, "该会话已有请求正在运行"),
        (ConversationDeletingError("private"), 409, "会话正在删除"),
        (ConversationAgentError("private"), 502, "会话生成服务暂时不可用"),
        (ConversationStorageError("private"), 500, "会话存储暂时不可用"),
        (ConversationDeleteError("private"), 500, "会话删除未完成"),
    ],
)
async def test_conversation_errors_have_stable_http_mapping(
    error: Exception,
    status_code: int,
    detail: str,
) -> None:
    service = _FakeConversationService()
    service.error = error
    transport = httpx.ASGITransport(app=_app(service))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post("/api/conversations")

    assert response.status_code == status_code
    assert response.json()["detail"] == detail
    assert "private" not in response.text
