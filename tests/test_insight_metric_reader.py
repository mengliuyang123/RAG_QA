# -*- coding: utf-8 -*-
"""统计投影必须在同一窗口快照去重用户，且不能读取聊天正文。"""

import sqlite3
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from uuid import UUID
from uuid import uuid4

import pytest

from app.application.errors import InsightStorageError
from app.domain.conversations import AgentResult
from app.domain.conversations import ConversationTurn
from app.domain.feedback import TurnFeedback
from app.infrastructure.conversations.sqlite_repository import SQLiteConversationRepository
from app.infrastructure.persistence.insights import SQLiteInsightMetricReader


def _old_turn(
    repository: SQLiteConversationRepository, owner_id: UUID | None,
) -> ConversationTurn:
    return repository.save_turn(repository.create(owner_id=owner_id).id, "私密问题", AgentResult(
        answer="私密答案", answer_mode="direct", retrieval_calls=0, citations=[],
    ), 1.0)


def test_metric_window_includes_boundaries_and_counts_known_users_once(tmp_path: Path) -> None:
    """边界不含、同人多算、未知归属算同一人或泄露评论均应失败。

    Args:
        tmp_path: 真实临时业务库。

    Returns:
        None。
    """
    path = tmp_path / "business.sqlite3"
    repository = SQLiteConversationRepository(path)
    start = datetime(2026, 10, 20, tzinfo=UTC)
    end = start + timedelta(days=1)
    first_user = uuid4()
    second_user = uuid4()
    fixtures = [
        (first_user, start), (first_user, end),
        (second_user, start + timedelta(hours=1)),
        (None, start + timedelta(hours=2)),
        (uuid4(), start - timedelta(microseconds=1)),
        (uuid4(), end + timedelta(microseconds=1)),
    ]
    for owner_id, when in fixtures:
        turn = _old_turn(repository, owner_id)
        if owner_id == first_user:
            repository.put_feedback(turn.id, first_user, TurnFeedback(rating="down", comment="私密评论"))
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE conversation_turns SET created_at=? WHERE id=?", (when.isoformat(), str(turn.id)))
    metrics = list(SQLiteInsightMetricReader(path).iter_metrics(start, end))
    assert len(metrics) == 4
    assert sum(metric.user_count_increment for metric in metrics) == 2
    assert sum(not metric.owner_known for metric in metrics) == 1
    assert sum(metric.feedback == "down" for metric in metrics) == 2
    serialized = "".join(metric.model_dump_json() for metric in metrics)
    for private in ("私密问题", "私密答案", "私密评论", str(first_user), str(second_user)):
        assert private not in serialized


def test_metric_batches_keep_one_snapshot_across_concurrent_feedback_change(tmp_path: Path) -> None:
    """第101条进入下一批时仍须沿用原快照，不能混入另一个时间点的评分。

    Args:
        tmp_path: 真实WAL业务库。

    Returns:
        None。
    """
    path = tmp_path / "business.sqlite3"
    repository = SQLiteConversationRepository(path)
    owner = uuid4()
    for _ in range(102):
        _old_turn(repository, owner)
    now = datetime.now(UTC)
    reader = SQLiteInsightMetricReader(path)
    stream = reader.iter_metrics(now - timedelta(days=1), now)
    first = next(stream)
    with sqlite3.connect(path) as connection:
        # 一次真实并发提交更新所有当前评分，不能影响已经打开的统计快照。
        connection.execute("UPDATE conversation_turns SET feedback_json = ?", ('{"rating":"down"}',))
    previous = [first, *stream]
    assert len(previous) == 102
    assert all(metric.feedback is None for metric in previous)
    assert sum(metric.user_count_increment for metric in previous) == 1
    current = list(reader.iter_metrics(now - timedelta(days=1), now))
    assert len(current) == 102
    assert all(metric.feedback == "down" for metric in current)


def test_metric_sql_does_not_read_question_answer_or_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用真实SQL授权器拒绝正文列，统计必须仍可执行，不靠替身掩盖读取。

    Args:
        tmp_path: 真实临时业务库。
        monkeypatch: 给实际只读连接安装列级授权器。

    Returns:
        None。
    """
    path = tmp_path / "business.sqlite3"
    repository = SQLiteConversationRepository(path)
    _old_turn(repository, uuid4())
    connect = sqlite3.connect

    def _authorize(action, table, column, database, trigger) -> int:
        if action == sqlite3.SQLITE_READ:
            if table == "turn_citations":
                return sqlite3.SQLITE_DENY
            if table == "conversation_turns" and column in {"user_message", "assistant_message"}:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def _connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = connect(*args, **kwargs)
        connection.set_authorizer(_authorize)
        return connection

    monkeypatch.setattr(sqlite3, "connect", _connect)
    now = datetime.now(UTC)
    metrics = list(SQLiteInsightMetricReader(path).iter_metrics(now - timedelta(days=1), now))
    assert len(metrics) == 1


def test_corrupt_metric_metadata_does_not_turn_into_zero_counts(tmp_path: Path) -> None:
    """损坏的元数据不是旧数据缺失，必须明确失败而非返回空统计。

    Args:
        tmp_path: 真实临时业务库。

    Returns:
        None。
    """
    path = tmp_path / "business.sqlite3"
    repository = SQLiteConversationRepository(path)
    _old_turn(repository, uuid4())
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE conversation_turns SET metadata_json='{}'")
    now = datetime.now(UTC)
    with pytest.raises(InsightStorageError):
        list(SQLiteInsightMetricReader(path).iter_metrics(now - timedelta(days=1), now))
