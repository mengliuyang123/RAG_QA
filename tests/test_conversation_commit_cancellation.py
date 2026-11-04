# -*- coding: utf-8 -*-
"""已观察取消时，真实SQLite事务不得提交半途写入的轮次和引用。"""

import sqlite3
from pathlib import Path
from threading import Event
from uuid import UUID

import pytest

from app.application.execution_control import StreamCancelled
from app.application.execution_control import cancellation_scope
from app.domain.conversations import TurnCitation
from app.infrastructure.conversations.sqlite_repository import SQLiteConversationRepository
from tests.test_conversation_repository import _retrieval_result


def test_cancelled_before_commit_rolls_back_turn_citations_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """只在生成前检查不足，取消可能发生在引用插入之后、事务提交之前。

    Args:
        tmp_path: 隔离正式会话库。
        monkeypatch: 在真实引用写入后模拟断连，不替代存储事务。
    """
    path = tmp_path / "conversations.sqlite3"
    repository = SQLiteConversationRepository(path)
    conversation = repository.create()
    cancelled = Event()
    insert = repository._insert_citations

    def insert_then_cancel(
        connection: sqlite3.Connection, turn_id: UUID,
        citations: list[TurnCitation],
    ) -> None:
        insert(connection, turn_id, citations)
        cancelled.set()

    def check() -> None:
        if cancelled.is_set():
            raise StreamCancelled()

    monkeypatch.setattr(repository, "_insert_citations", insert_then_cancel)
    with cancellation_scope(check):
        with pytest.raises(StreamCancelled):
            repository.save_turn(conversation.id, "问题", _retrieval_result(), 1)
    actual = repository.get(conversation.id)
    assert actual.turns == []
    assert actual.title == "新对话"
    assert actual.turns_count == 0
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM turn_citations").fetchone()[0] == 0
