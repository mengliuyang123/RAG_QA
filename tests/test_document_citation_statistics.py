# -*- coding: utf-8 -*-
"""正式引用计数应有明确窗口、归属、快照和失败语义。"""

import sqlite3
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from app.application.documents.citations import CitationStatisticsService
from app.application.errors import KnowledgeContentError
from app.infrastructure.conversations.sqlite_repository import SQLiteConversationRepository
from app.infrastructure.persistence.document_citations import SQLiteDocumentCitationReader
from tests.test_admin_chunk_diagnostics import _cite


def test_calendar_window_and_exact_chunk_attribution(tmp_path: Path) -> None:
    """滚动七天、丢失边界、跨文档同块ID或按问题计数均会给出错误数。

    Args:
        tmp_path: 隔离的真实正式业务库。

    Returns:
        None。
    """
    path = tmp_path / "business.sqlite3"
    repository = SQLiteConversationRepository(path)
    now = datetime(2026, 10, 25, 15, 30, tzinfo=UTC)
    start = datetime(2026, 10, 19, tzinfo=UTC)
    doc_id = uuid4()
    other = uuid4()
    fixtures = [
        (doc_id, ["opaque/a", "opaque/b"], start),
        (doc_id, ["opaque/a"], now),
        (doc_id, [None, ""], now - timedelta(hours=1)),
        (doc_id, ["opaque/a"], start - timedelta(microseconds=1)),
        (doc_id, ["opaque/a"], now + timedelta(microseconds=1)),
        (other, ["opaque/a"], now),
    ]
    for identifier, chunks, when in fixtures:
        turn = _cite(repository, identifier, chunks)
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE conversation_turns SET created_at=? WHERE id=?", (when.isoformat(), str(turn.id)))
    deleted = _cite(repository, doc_id, ["opaque/a"])
    repository.mark_deleting(deleted.conversation_id)
    service = CitationStatisticsService(SQLiteDocumentCitationReader(path), clock=lambda: now)
    page = service.chunks(doc_id, ("opaque/a", "opaque/b", "unused"))
    assert page.window_started_at == start
    assert page.window_ended_at == now
    assert {row.chunk_id: row.count for row in page.counts} == {"opaque/a": 2, "opaque/b": 1, "unused": 0}
    assert page.unattributed == 2
    totals = service.documents((doc_id, other, uuid4()))
    assert totals[doc_id].total == 5
    assert totals[doc_id].unattributed == 2
    assert totals[other].total == 1


def test_statistics_sql_cannot_read_chat_or_citation_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """若统计偷读正文、用户或metadata，真实列级授权器会拒绝查询。

    Args:
        tmp_path: 隔离正式业务库。
        monkeypatch: 仅给真实连接加授权器，保留真实SQL结果。

    Returns:
        None。
    """
    path = tmp_path / "business.sqlite3"
    repository = SQLiteConversationRepository(path)
    doc_id = uuid4()
    _cite(repository, doc_id, ["opaque/a"])
    connect = sqlite3.connect

    def _authorize(action, table, column, database, trigger) -> int:
        if action == sqlite3.SQLITE_READ:
            allowed = {
                "turn_citations": {"turn_id", "doc_id", "chunk_id"},
                "conversation_turns": {"id", "conversation_id", "created_at"},
                "conversations": {"id", "status"},
            }
            if table in allowed and column not in allowed[table]:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def _connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = connect(*args, **kwargs)
        connection.set_authorizer(_authorize)
        return connection

    monkeypatch.setattr(sqlite3, "connect", _connect)
    service = CitationStatisticsService(SQLiteDocumentCitationReader(path))
    assert service.documents((doc_id,))[doc_id].total == 1
    assert service.chunks(doc_id, ("opaque/a",)).counts[0].count == 1


def test_document_batches_use_one_read_only_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """101条统计跨批时若重新打开快照，会混入并发提交的另一批引用。

    Args:
        tmp_path: 真实WAL业务库。
        monkeypatch: 观察实际查询，并在首批完成后提交一条正式引用。

    Returns:
        None。
    """
    path = tmp_path / "business.sqlite3"
    repository = SQLiteConversationRepository(path)
    identifiers = tuple(uuid4() for _ in range(101))
    _cite(repository, identifiers[0], ["a"])
    _cite(repository, identifiers[-1], ["last"])
    now = datetime.now(UTC)
    start = now - timedelta(days=1)
    connect = sqlite3.connect
    statements = []
    injected = False

    class ObservedConnection(sqlite3.Connection):
        """只观察真实SQL边界，不替换任何结果。"""

        def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
            """在第二个实际聚合前提交并发引用。

            Args:
                sql: 真实待执行SQL。
                parameters: 实际查询参数。

            Returns:
                原查询游标。
            """
            nonlocal injected
            if sql.startswith("SELECT citation.doc_id"):
                statements.append(sql)
                if len(statements) == 2 and not injected:
                    injected = True
                    turn = _cite(repository, identifiers[-1], ["new"])
                    with connect(path) as writer:
                        writer.execute("UPDATE conversation_turns SET created_at=? WHERE id=?", (now.isoformat(), str(turn.id)))
            return super().execute(sql, parameters)

    def _connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        if kwargs.get("uri") is True:
            kwargs["factory"] = ObservedConnection
        return connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _connect)
    reader = SQLiteDocumentCitationReader(path)
    totals = reader.documents(identifiers, start, now)
    assert len(statements) == 2
    assert len(totals) == 101
    assert totals[identifiers[0]].total == 1
    assert totals[identifiers[50]].total == 0
    assert totals[identifiers[-1]].total == 1
    assert reader.documents((identifiers[-1],), start, now)[identifiers[-1]].total == 2


@pytest.mark.parametrize("method", ["documents", "chunks"])
@pytest.mark.parametrize("damage", ["missing", "corrupt", "schema"])
def test_unreadable_business_store_is_not_reported_as_zero(
    tmp_path: Path, method: str, damage: str,
) -> None:
    """缺失、坏库或缺少引用列必须报告不可用，而非返回伪造全零。

    Args:
        tmp_path: 仅临时坏库。
        method: 文档或块统计。
        damage: 故障类型。

    Returns:
        None。
    """
    path = tmp_path / "business.sqlite3"
    if damage == "corrupt":
        path.write_bytes(b"not sqlite")
    elif damage == "schema":
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE incomplete (id TEXT)")
    reader = SQLiteDocumentCitationReader(path)
    now = datetime.now(UTC)
    with pytest.raises(KnowledgeContentError):
        if method == "documents":
            reader.documents((uuid4(),), now - timedelta(days=1), now)
        else:
            reader.chunks(uuid4(), ("opaque/a",), now - timedelta(days=1), now)


def test_business_migration_indexes_document_chunk_after_legacy_columns(tmp_path: Path) -> None:
    """旧库没有chunk_id时迁移仍应完成，并得到实际可用的统计索引。

    Args:
        tmp_path: 隔离旧业务库。

    Returns:
        None。
    """
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE turn_citations (turn_id TEXT, position INTEGER, doc_id TEXT, filename TEXT, page INTEGER, chunk_text TEXT, PRIMARY KEY(turn_id, position))")
    SQLiteConversationRepository(path)
    with sqlite3.connect(path) as connection:
        columns = connection.execute("PRAGMA index_info(idx_citations_document_chunk)").fetchall()
        assert [column[2] for column in columns] == ["doc_id", "chunk_id", "turn_id"]
