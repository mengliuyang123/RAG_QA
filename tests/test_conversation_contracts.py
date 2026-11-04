# -*- coding: utf-8 -*-
"""多轮会话领域模型与应用端口契约测试。"""

from datetime import UTC, datetime
from typing import get_type_hints
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.application.conversations.events import (
    CompleteConversationEvent,
    StatusConversationData,
    StatusConversationEvent,
)
from app.application.conversations.ports import ConversationAgent
from app.application.conversations.ports import ConversationCheckpointStore
from app.application.conversations.ports import ConversationRepository
from app.domain.conversations import AgentResult
from app.domain.conversations import AnswerMode
from app.domain.conversations import Conversation
from app.domain.conversations import ConversationQueryResult
from app.domain.conversations import ConversationStatus
from app.domain.conversations import ConversationTurn
from app.domain.conversations import TurnCitation


def _citation(position: int = 1) -> TurnCitation:
    return TurnCitation(
        position=position,
        doc_id="doc-a",
        filename="a.md",
        page=None,
        chunk_text="证据正文",
    )


def test_direct_agent_result_allows_no_citations() -> None:
    result = AgentResult(
        answer="这是一个直接回答。",
        answer_mode=AnswerMode.DIRECT,
        retrieval_calls=0,
        citations=[],
    )

    assert result.answer_mode is AnswerMode.DIRECT
    assert result.citations == []


def test_agent_result_enforces_mode_and_retrieval_call_invariants() -> None:
    with pytest.raises(ValidationError):
        AgentResult(
            answer="错误的 direct 回答",
            answer_mode=AnswerMode.DIRECT,
            retrieval_calls=1,
            citations=[],
        )

    with pytest.raises(ValidationError):
        AgentResult(
            answer="错误的 retrieval 回答",
            answer_mode=AnswerMode.RETRIEVAL,
            retrieval_calls=0,
            citations=[],
        )

    with pytest.raises(ValidationError):
        AgentResult(
            answer="超过调用上限",
            answer_mode=AnswerMode.RETRIEVAL,
            retrieval_calls=4,
            citations=[],
        )


def test_agent_result_rejects_duplicate_citation_positions() -> None:
    with pytest.raises(ValidationError):
        AgentResult(
            answer="回答[1]",
            answer_mode=AnswerMode.RETRIEVAL,
            retrieval_calls=1,
            citations=[_citation(), _citation()],
        )


def test_conversation_models_preserve_full_official_turn() -> None:
    conversation_id = uuid4()
    turn_id = uuid4()
    created_at = datetime.now(UTC)
    citation = _citation()
    turn = ConversationTurn(
        id=turn_id,
        conversation_id=conversation_id,
        user_message="问题",
        assistant_message="回答[1]",
        answer_mode=AnswerMode.RETRIEVAL,
        retrieval_calls=1,
        citations=[citation],
        latency_ms=12.5,
        created_at=created_at,
    )
    conversation = Conversation(
        id=conversation_id,
        title="问题",
        status=ConversationStatus.ACTIVE,
        created_at=created_at,
        updated_at=created_at,
        turns_count=1,
        turns=[turn],
    )

    assert conversation.turns == [turn]
    assert conversation.turns[0].citations == [citation]


def test_conversation_events_have_stable_safe_shape() -> None:
    conversation_id = uuid4()
    turn_id = uuid4()
    status_event = StatusConversationEvent(
        data=StatusConversationData(
            phase="retrieving", attempt=2,
            kb_ids=("ops",), label="正在查找资料",
        )
    )
    result = ConversationQueryResult(
        conversation_id=conversation_id,
        turn_id=turn_id,
        answer="回答[1]",
        answer_mode=AnswerMode.RETRIEVAL,
        retrieval_calls=2,
        citations=[_citation()],
        latency_ms=25.0,
    )
    complete_event = CompleteConversationEvent(data=result)

    assert status_event.model_dump(mode="json") == {
        "event": "status",
        "data": {
            "phase": "retrieving", "attempt": 2,
            "kb_ids": ["ops"], "label": "正在查找资料",
        },
    }
    assert complete_event.data.turn_id == turn_id
    assert "scratchpad" not in complete_event.model_dump_json()


def test_conversation_ports_expose_only_application_contracts() -> None:
    repository_members = {
        "create",
        "list_active",
        "get",
        "recent_turns",
        "save_turn",
        "mark_deleting",
        "delete",
    }

    assert repository_members.issubset(ConversationRepository.__dict__)
    assert "run" in ConversationAgent.__dict__
    assert "delete_thread" in ConversationCheckpointStore.__dict__
    assert get_type_hints(ConversationAgent.run)["return"] is AgentResult
