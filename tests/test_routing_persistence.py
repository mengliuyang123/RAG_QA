# -*- coding: utf-8 -*-
"""真实旧版SQLite迁移不能损坏历史，来源顺序和路由元数据必须可恢复。"""

import sqlite3
from datetime import datetime
from datetime import timezone
from pathlib import Path
from uuid import uuid4

import pytest

from app.application.errors import ConversationStorageError
from app.domain.answer_metadata import AnswerMetadata
from app.domain.answer_metadata import ScopeInfo
from app.domain.conversations import AgentResult
from app.domain.conversations import AnswerMode
from app.domain.conversations import TurnCitation
from app.domain.routing import KnowledgeScope
from app.infrastructure.conversations.sqlite_repository import SQLiteConversationRepository


def test_sparse_citations_keep_first_appearance_order_after_reopen(tmp_path: Path) -> None:
    """正文[5]在[2]之前时，存储不能按编号升序重排来源。

    Args:
        tmp_path: 隔离SQLite目录。
    """
    path = tmp_path / "conversation.sqlite3"
    repository = SQLiteConversationRepository(path)
    conversation = repository.create()
    result = AgentResult(
        answer="先[5]，后[2]", answer_mode=AnswerMode.RETRIEVAL,
        retrieval_calls=2,
        citations=[TurnCitation(
            position=number, doc_id="d", filename="a.md", page=None,
            chunk_text=f"资料{number}", chunk_id=f"d:{number}", kb_id="ops",
        ) for number in (5, 2)],
        metadata=AnswerMetadata(
            lane="agent", outcome="answered", grounding="grounded",
            escalated=True, scope_expanded=False,
            kb_scope=ScopeInfo(mode="explicit", requested=("ops",), searched=("ops",)),
        ),
    )
    saved = repository.save_turn(conversation.id, "原问题", result, 1.0)
    restored = SQLiteConversationRepository(path).get(conversation.id).turns[0]
    assert [item.position for item in restored.citations] == [5, 2]
    assert [item.chunk_id for item in restored.citations] == ["d:5", "d:2"]
    assert restored.metadata == result.metadata == saved.metadata


def test_real_legacy_schema_migrates_without_rejudging_content(tmp_path: Path) -> None:
    """旧表没有新增列时可打开，原文与编号不变，不能伪造已充分审核标签。

    Args:
        tmp_path: 旧版数据库目录。
    """
    path = tmp_path / "old.sqlite3"
    conversation_id = uuid4()
    turn_id = uuid4()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY, title TEXT, status TEXT,
                created_at TEXT, updated_at TEXT, turns_count INTEGER
            );
            CREATE TABLE conversation_turns (
                id TEXT PRIMARY KEY, conversation_id TEXT, user_message TEXT,
                assistant_message TEXT, answer_mode TEXT, retrieval_calls INTEGER,
                latency_ms REAL, created_at TEXT
            );
            CREATE TABLE turn_citations (
                turn_id TEXT, position INTEGER, doc_id TEXT, filename TEXT,
                page INTEGER, chunk_text TEXT, PRIMARY KEY(turn_id, position)
            );
        """)
        connection.execute(
            "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?)",
            (str(conversation_id), "旧会话", "active", now, now, 1),
        )
        connection.execute(
            "INSERT INTO conversation_turns VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(turn_id), str(conversation_id), "旧问题", "旧回答[5]", "retrieval", 1, 1.0, now),
        )
        connection.execute(
            "INSERT INTO turn_citations VALUES (?, ?, ?, ?, ?, ?)",
            (str(turn_id), 5, "old", "old.md", None, "旧证据"),
        )
    repository = SQLiteConversationRepository(path)
    turn = repository.get(conversation_id).turns[0]
    assert repository.get_summary(conversation_id).kb_scope.mode == "auto"
    assert turn.assistant_message == "旧回答[5]"
    assert turn.citations[0].position == 5
    assert turn.citations[0].chunk_id is None
    assert turn.metadata.grounding == "partial"
    assert repository.recent_turns(conversation_id, 1)[0] == turn
    assert list(repository.iter_turns_reverse(conversation_id))[0] == turn
    assert SQLiteConversationRepository(path).get(conversation_id).turns[0] == turn


def test_corrupt_routing_metadata_is_a_storage_error(tmp_path: Path) -> None:
    """非法元数据不能被静默当成旧记录或作为可信状态返回。

    Args:
        tmp_path: 隔离数据库目录。
    """
    path = tmp_path / "conversation.sqlite3"
    repository = SQLiteConversationRepository(path)
    conversation = repository.create()
    turn = repository.save_turn(conversation.id, "问题", AgentResult(
        answer="你好", answer_mode=AnswerMode.DIRECT, retrieval_calls=0,
        citations=[],
    ), 1.0)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE conversation_turns SET metadata_json = ? WHERE id = ?",
            ('{"lane":"unsupported"}', str(turn.id)),
        )
    with pytest.raises(ConversationStorageError):
        repository.get(conversation.id)


def test_default_scope_survives_reopen_and_manual_title_survives_first_turn(
    tmp_path: Path,
) -> None:
    """首轮自动标题不能覆盖用户设置，默认范围也不能只保存在内存中。

    Args:
        tmp_path: 隔离数据库目录。
    """
    path = tmp_path / "scoped.sqlite3"
    repository = SQLiteConversationRepository(path)
    conversation = repository.create(
        kb_scope=KnowledgeScope(mode="explicit", kb_ids=("ops",)),
    )
    repository.update(conversation.id, title="手动标题")
    reopened = SQLiteConversationRepository(path)
    reopened.save_turn(conversation.id, "自动标题不能覆盖我", AgentResult(
        answer="你好", answer_mode=AnswerMode.DIRECT,
        retrieval_calls=0, citations=[],
    ), 1.0)
    summary = reopened.get_summary(conversation.id)
    assert summary.title == "手动标题"
    assert summary.kb_scope.mode == "explicit"
    assert summary.kb_scope.kb_ids == ("ops",)
    assert reopened.list_active()[0].kb_scope == summary.kb_scope
