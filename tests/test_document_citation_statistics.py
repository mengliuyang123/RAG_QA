from tests.postgres_support import pg_database, pg_connect
# -*- coding: utf-8 -*-
"""正式引用计数应有明确窗口、归属、快照和失败语义。"""

import psycopg
from app.infrastructure.persistence.database import record_row, Record
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from app.application.documents.citations import CitationStatisticsService
from app.application.errors import KnowledgeContentError
from app.infrastructure.conversations.postgres_repository import PostgresConversationRepository
from app.infrastructure.persistence.document_citations import PostgresDocumentCitationReader
from tests.test_admin_chunk_diagnostics import _cite


def test_calendar_window_and_exact_chunk_attribution(tmp_path: Path) -> None:
    """滚动七天、丢失边界、跨文档同块ID或按问题计数均会给出错误数。

    Args:
        tmp_path: 隔离的真实正式业务库。

    Returns:
        None。
    """
    path = tmp_path / "business.db"
    repository = PostgresConversationRepository(pg_database(path))
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
        with pg_connect(path) as connection:
            connection.execute("UPDATE conversation_turns SET created_at=%s WHERE id=%s", (when.isoformat(), str(turn.id)))
    deleted = _cite(repository, doc_id, ["opaque/a"])
    repository.mark_deleting(deleted.conversation_id)
    service = CitationStatisticsService(PostgresDocumentCitationReader(pg_database(path)), clock=lambda: now)
    page = service.chunks(doc_id, ("opaque/a", "opaque/b", "unused"))
    assert page.window_started_at == start
    assert page.window_ended_at == now
    assert {row.chunk_id: row.count for row in page.counts} == {"opaque/a": 2, "opaque/b": 1, "unused": 0}
    assert page.unattributed == 2
    totals = service.documents((doc_id, other, uuid4()))
    assert totals[doc_id].total == 5
    assert totals[doc_id].unattributed == 2
    assert totals[other].total == 1


def test_statistics_sql_cannot_read_chat_or_citation_text(tmp_path: Path, monkeypatch) -> None:
    from tests.postgres_support import observe_queries
    database=pg_database(tmp_path/'business.db')
    repository=PostgresConversationRepository(database)
    doc_id=uuid4()
    _cite(repository,doc_id,['opaque/a'])
    def guard(query):
        assert not any(column in query for column in ('user_message','assistant_message','chunk_text','metadata_json','owner_id'))
    with observe_queries(database,guard):
        service=CitationStatisticsService(PostgresDocumentCitationReader(database))
        assert service.documents((doc_id,))[doc_id].total==1
        assert service.chunks(doc_id,('opaque/a',)).counts[0].count==1


def test_document_batches_use_one_read_only_snapshot(tmp_path: Path, monkeypatch) -> None:
    from tests.postgres_support import observe_queries
    database=pg_database(tmp_path/'business.db')
    repository=PostgresConversationRepository(database)
    identifiers=tuple(uuid4() for _ in range(101))
    _cite(repository,identifiers[0],['a'])
    _cite(repository,identifiers[-1],['last'])
    now=datetime.now(UTC)
    start=now-timedelta(days=1)
    statements=[]
    def before(query):
        if query.startswith('SELECT citation.doc_id'):
            statements.append(query)
            if len(statements)==2:
                turn=_cite(repository,identifiers[-1],['new'])
                with database.connection() as connection:
                    connection.execute('UPDATE conversation_turns SET created_at=%s WHERE id=%s',(now.isoformat(),str(turn.id)))
    reader=PostgresDocumentCitationReader(database)
    with observe_queries(database,before):
        totals=reader.documents(identifiers,start,now)
    assert len(statements)==2
    assert len(totals)==101
    assert totals[identifiers[0]].total==1
    assert totals[identifiers[50]].total==0
    assert totals[identifiers[-1]].total==1
    assert reader.documents((identifiers[-1],),start,now)[identifiers[-1]].total==2


@pytest.mark.parametrize("method", ["documents", "chunks"])
@pytest.mark.parametrize("damage", ["missing", "corrupt", "schema"])
def test_unreadable_business_store_is_not_reported_as_zero(tmp_path: Path, method: str, damage: str) -> None:
    database=pg_database(tmp_path/'business.db')
    with database.connection() as connection:
        if damage=='missing': connection.execute('DROP TABLE turn_citations CASCADE')
        elif damage=='corrupt': connection.execute('ALTER TABLE turn_citations ALTER COLUMN doc_id TYPE INTEGER USING NULL')
        else: connection.execute('ALTER TABLE turn_citations DROP COLUMN chunk_id CASCADE')
    reader=PostgresDocumentCitationReader(database)
    now=datetime.now(UTC)
    with pytest.raises(KnowledgeContentError):
        if method=='documents': reader.documents((uuid4(),),now-timedelta(days=1),now)
        else: reader.chunks(uuid4(),('opaque/a',),now-timedelta(days=1),now)


def test_business_migration_indexes_document_chunk_after_legacy_columns(tmp_path: Path) -> None:
    database=pg_database(tmp_path/'business.db')
    from app.infrastructure.persistence.migrations import migrate_database
    migrate_database(database._dsn,database.schema)
    with database.connection(read_only=True) as connection:
        row=connection.execute("SELECT indexdef FROM pg_indexes WHERE schemaname=current_schema() AND indexname='idx_citations_document_chunk'").fetchone()
        assert '(doc_id, chunk_id, turn_id)' in row[0]
