# -*- coding: utf-8 -*-
"""SQLite 正式会话历史 repository 测试。"""

import sqlite3
from pathlib import Path

import pytest

from app.application.errors import ConversationNotFoundError
from app.application.errors import ConversationStorageError
from app.domain.conversations import AgentResult
from app.domain.conversations import AnswerMode
from app.domain.conversations import ConversationStatus
from app.domain.conversations import TurnCitation
from app.infrastructure.conversations.sqlite_repository import (
    SQLiteConversationRepository,
)


def _retrieval_result() -> AgentResult:
    return AgentResult(
        answer="LangGraph 使用状态图编排工作流[1]。",
        answer_mode=AnswerMode.RETRIEVAL,
        retrieval_calls=2,
        citations=[
            TurnCitation(
                position=1,
                doc_id="doc-a",
                filename="langgraph.md",
                page=3,
                chunk_text="LangGraph 使用状态图编排工作流。",
            )
        ],
    )


def _direct_result(answer: str = "你好。") -> AgentResult:
    return AgentResult(
        answer=answer,
        answer_mode=AnswerMode.DIRECT,
        retrieval_calls=0,
        citations=[],
    )


def test_repository_persists_crud_title_citations_and_recent_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "conversations.sqlite3"
    repository = SQLiteConversationRepository(path)
    conversation = repository.create()

    assert conversation.title == "新对话"
    assert conversation.status is ConversationStatus.ACTIVE
    assert conversation.turns == []

    first = repository.save_turn(
        conversation.id,
        "  LangGraph 是什么？  ",
        _retrieval_result(),
        12.5,
    )
    second = repository.save_turn(
        conversation.id,
        "它能直接回答吗？",
        _direct_result("可以。"),
        8.0,
    )

    reopened = SQLiteConversationRepository(path)
    loaded = reopened.get(conversation.id)
    assert loaded.title == "LangGraph 是什么？"
    assert loaded.turns_count == 2
    assert loaded.turns == [first, second]
    assert loaded.turns[0].citations[0].chunk_text.startswith("LangGraph")
    assert reopened.recent_turns(conversation.id, 1) == [second]
    assert reopened.list_active()[0].id == conversation.id


def test_repository_lists_active_conversations_by_recent_update(
    tmp_path: Path,
) -> None:
    repository = SQLiteConversationRepository(tmp_path / "conversations.db")
    older = repository.create()
    newer = repository.create()
    repository.save_turn(older.id, "更新旧会话", _direct_result(), 1.0)
    repository.mark_deleting(newer.id)

    summaries = repository.list_active()

    assert [item.id for item in summaries] == [older.id]
    assert repository.get(newer.id).status is ConversationStatus.DELETING


def test_first_title_is_unicode_safe_and_never_replaced(
    tmp_path: Path,
) -> None:
    repository = SQLiteConversationRepository(tmp_path / "conversations.db")
    conversation = repository.create()
    question = "你" * 60

    repository.save_turn(conversation.id, question, _direct_result(), 1.0)
    repository.save_turn(
        conversation.id,
        "第二个问题",
        _direct_result(),
        1.0,
    )

    loaded = repository.get(conversation.id)
    assert loaded.title == "你" * 50
    assert len(loaded.title) == 50


def test_delete_cascades_turns_and_citations(tmp_path: Path) -> None:
    path = tmp_path / "conversations.db"
    repository = SQLiteConversationRepository(path)
    conversation = repository.create()
    repository.save_turn(
        conversation.id,
        "问题",
        _retrieval_result(),
        1.0,
    )

    repository.mark_deleting(conversation.id)
    repository.delete(conversation.id)

    with pytest.raises(ConversationNotFoundError):
        repository.get(conversation.id)
    with sqlite3.connect(path) as connection:
        turns = connection.execute(
            "SELECT COUNT(*) FROM conversation_turns"
        ).fetchone()[0]
        citations = connection.execute(
            "SELECT COUNT(*) FROM turn_citations"
        ).fetchone()[0]
    assert (turns, citations) == (0, 0)


def test_save_turn_rolls_back_all_changes_when_citation_insert_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = SQLiteConversationRepository(tmp_path / "conversations.db")
    conversation = repository.create()

    def fail_insert(*args: object, **kwargs: object) -> None:
        del args
        del kwargs
        raise sqlite3.DatabaseError("private database detail")

    monkeypatch.setattr(repository, "_insert_citations", fail_insert)

    with pytest.raises(ConversationStorageError) as exc_info:
        repository.save_turn(
            conversation.id,
            "不能留下半轮",
            _retrieval_result(),
            1.0,
        )

    loaded = repository.get(conversation.id)
    assert loaded.turns_count == 0
    assert loaded.turns == []
    assert "private database detail" not in str(exc_info.value)


def test_repository_rejects_missing_conversation_and_invalid_limit(
    tmp_path: Path,
) -> None:
    repository = SQLiteConversationRepository(tmp_path / "conversations.db")

    with pytest.raises(ConversationNotFoundError):
        repository.get(__import__("uuid").uuid4())
    with pytest.raises(ValueError, match="limit"):
        repository.recent_turns(__import__("uuid").uuid4(), 0)
