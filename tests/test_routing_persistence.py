from tests.postgres_support import pg_database, pg_connect
# -*- coding: utf-8 -*-
"""真实旧版SQLite迁移不能损坏历史，来源顺序和路由元数据必须可恢复。"""

import psycopg
from app.infrastructure.persistence.database import record_row, Record
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
from app.infrastructure.conversations.postgres_repository import PostgresConversationRepository


def test_sparse_citations_keep_first_appearance_order_after_reopen(tmp_path: Path) -> None:
    """正文[5]在[2]之前时，存储不能按编号升序重排来源。

    Args:
        tmp_path: 隔离SQLite目录。
    """
    path = tmp_path / "conversation.db"
    repository = PostgresConversationRepository(pg_database(path))
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
    restored = PostgresConversationRepository(pg_database(path)).get(conversation.id).turns[0]
    assert [item.position for item in restored.citations] == [5, 2]
    assert [item.chunk_id for item in restored.citations] == ["d:5", "d:2"]
    assert restored.metadata == result.metadata == saved.metadata


def test_postgres_migration_preserves_saved_answers(tmp_path: Path) -> None:
    from app.infrastructure.persistence.migrations import migrate_database
    from app.infrastructure.conversations.postgres_repository import PostgresConversationRepository
    from app.domain.conversations import AgentResult, AnswerMode
    database=pg_database(tmp_path/'business.db')
    repository=PostgresConversationRepository(database)
    conversation=repository.create()
    repository.save_turn(conversation.id,'question',AgentResult(answer='original answer',answer_mode=AnswerMode.DIRECT,retrieval_calls=0,citations=[]),1.0)
    before=repository.get(conversation.id)
    migrate_database(database._dsn,database.schema)
    assert PostgresConversationRepository(database).get(conversation.id)==before


def test_corrupt_routing_metadata_is_a_storage_error(tmp_path: Path) -> None:
    """非法元数据不能被静默当成旧记录或作为可信状态返回。

    Args:
        tmp_path: 隔离数据库目录。
    """
    path = tmp_path / "conversation.db"
    repository = PostgresConversationRepository(pg_database(path))
    conversation = repository.create()
    turn = repository.save_turn(conversation.id, "问题", AgentResult(
        answer="你好", answer_mode=AnswerMode.DIRECT, retrieval_calls=0,
        citations=[],
    ), 1.0)
    with pg_connect(path) as connection:
        connection.execute(
            "UPDATE conversation_turns SET metadata_json = %s WHERE id = %s",
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
    path = tmp_path / "scoped.db"
    repository = PostgresConversationRepository(pg_database(path))
    conversation = repository.create(
        kb_scope=KnowledgeScope(mode="explicit", kb_ids=("ops",)),
    )
    repository.update(conversation.id, title="手动标题")
    reopened = PostgresConversationRepository(pg_database(path))
    reopened.save_turn(conversation.id, "自动标题不能覆盖我", AgentResult(
        answer="你好", answer_mode=AnswerMode.DIRECT,
        retrieval_calls=0, citations=[],
    ), 1.0)
    summary = reopened.get_summary(conversation.id)
    assert summary.title == "手动标题"
    assert summary.kb_scope.mode == "explicit"
    assert summary.kb_scope.kb_ids == ("ops",)
    assert reopened.list_active()[0].kb_scope == summary.kb_scope
