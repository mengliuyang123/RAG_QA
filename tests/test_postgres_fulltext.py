"""真实 PostgreSQL 全文索引：中文、回填与跨进程一致性。"""
import os
import subprocess
import sys
from uuid import uuid4

import pytest

from app.domain.documents import DocumentChunk
from app.infrastructure.vectorstores.postgres import PostgresVectorStore
from tests.postgres_support import isolated_database

pytestmark = pytest.mark.database


def chunk(text, *, kb_id='ops', identity=None):
    identity = identity or str(uuid4())
    return DocumentChunk(id=identity + ':0', doc_id=identity, text=text,
                         filename='test.md', page=None, chunk_index=0, kb_id=kb_id)


def keyword(database, collection='knowledge'):
    from app.infrastructure.indexes.postgres import PostgresKeywordIndex
    return PostgresKeywordIndex(database, collection)


def test_chinese_words_identifiers_and_safe_or_query():
    with isolated_database() as database:
        store = PostgresVectorStore(database, 'knowledge')
        docs = [chunk('数据库连接配置 response_model 返回 HTTP 422'),
                chunk('数据报表与仓库管理'), chunk('无关的天气预报')]
        store.add_chunks(docs, [[1., 0.]] * len(docs))
        index = keyword(database)
        assert [r.chunk_id for r in index.search('数据库', 5)] == [docs[0].id]
        for query in ('连接配置', 'response_model', 'HTTP', '422', "不存在 | '数据库':* & !"):
            assert docs[0].id in [r.chunk_id for r in index.search(query, 5)]
        assert index.search(' !!! & | :* \\ ', 5) == []
        assert index.search('量子纠缠', 5) == []
        with pytest.raises(ValueError, match='top_k'):
            index.search('数据库', 0)


def test_scope_collection_and_exclusions_before_limit():
    with isolated_database() as database:
        store = PostgresVectorStore(database, 'knowledge')
        docs = [chunk('数据库 数据库 数据库', kb_id='private'), chunk('数据库配置')]
        store.add_chunks(docs, [[1., 0.]] * 2)
        PostgresVectorStore(database, 'other').add_chunks([chunk('数据库')], [[1., 0.]])
        index = keyword(database)
        assert [r.chunk_id for r in index.search('数据库', 1, kb_ids=('ops',))] == [docs[1].id]
        assert index.search('数据库', 1, kb_ids=()) == []
        assert index.search('数据库', 1, kb_ids=('ops',), excluded_doc_ids=(docs[1].doc_id,)) == []
        assert len(index.search('数据库', 10)) == 2


def test_migration_backfills_old_chunks_without_changing_vectors():
    from app.infrastructure.persistence.migrations import migrate_database
    with isolated_database() as database:
        store = PostgresVectorStore(database, 'knowledge')
        doc = chunk('数据库连接配置 PostgreSQL')
        store.add_chunks([doc], [[1., 0.]])
        with database.connection() as connection:
            before = connection.execute('SELECT embedding::text,metadata FROM vector_chunks').fetchone()
            columns = connection.execute("SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='vector_chunks'").fetchall()
            assert 'search_vector' in [r[0] for r in columns]
            connection.execute('ALTER TABLE vector_chunks DROP COLUMN search_vector, DROP COLUMN search_text')
            connection.execute('DELETE FROM schema_migrations WHERE version=4')
        migrate_database(database._dsn, database.schema)
        migrate_database(database._dsn, database.schema)
        assert keyword(database).search('数据库', 1)[0].chunk_id == doc.id
        with database.connection() as connection:
            after = connection.execute('SELECT embedding::text,metadata FROM vector_chunks').fetchone()
            assert after == before
            assert connection.execute("SELECT indexdef FROM pg_indexes WHERE schemaname=current_schema() AND indexname='vector_chunks_search'").fetchone()[0].find('USING gin') > 0
            connection.execute('SET LOCAL enable_seqscan=off')
            plan = connection.execute("EXPLAIN SELECT id FROM vector_chunks WHERE search_vector @@ to_tsquery('simple','数据库')").fetchall()
            assert 'vector_chunks_search' in str(plan)


def test_independent_process_commits_and_deletes_are_visible_without_rebuild():
    with isolated_database() as database:
        PostgresVectorStore(database, 'knowledge')
        index = keyword(database)
        assert index.search('数据库', 5) == []
        script = '''
import os
from app.infrastructure.persistence.database import Database
from app.infrastructure.vectorstores.postgres import PostgresVectorStore
from app.domain.documents import DocumentChunk
d=Database(os.environ['FTS_TEST_DSN'],os.environ['FTS_TEST_SCHEMA'])
try:
 s=PostgresVectorStore(d,'knowledge')
 if os.environ['FTS_TEST_ACTION']=='add':
  s.add_chunks([DocumentChunk(id='child:0',doc_id='child',text='数据库连接配置',filename='child.md',page=None,chunk_index=0)],[[1.,0.]])
 else:
  s.delete_document('child')
finally:
 d.close()
'''
        env = {**os.environ, 'FTS_TEST_DSN': database._dsn, 'FTS_TEST_SCHEMA': database.schema}
        for action in ('add', 'delete'):
            subprocess.run([sys.executable, '-c', script], env={**env, 'FTS_TEST_ACTION': action},
                           check=True, capture_output=True, timeout=40)
            results = index.search('数据库', 5)
            assert [r.chunk_id for r in results] == (['child:0'] if action == 'add' else [])


def test_failed_batch_leaves_no_vector_or_keyword_rows():
    from app.application.errors import VectorStoreError
    with isolated_database() as database:
        store = PostgresVectorStore(database, 'knowledge')
        doc = chunk('数据库')
        with pytest.raises(VectorStoreError):
            store.add_chunks([doc, doc], [[1., 0.], [1., 0.]])
        assert store.count() == 0
        assert keyword(database).search('数据库', 5) == []


def test_rebuild_cannot_publish_stale_snapshot_or_resurrect_deleted_chunks():
    with isolated_database() as database:
        store = PostgresVectorStore(database, 'knowledge')
        doc = chunk('数据库')
        store.add_chunks([doc], [[1., 0.]])
        index = keyword(database)
        index.rebuild([])
        assert index.search('数据库', 1)[0].chunk_id == doc.id
        store.delete_document(doc.doc_id)
        index.rebuild([doc])
        assert index.search('数据库', 1) == []


def test_migration_failure_rolls_back_ddl_and_backfill(monkeypatch):
    from app.infrastructure.persistence.migrations import migrate_database
    from app.infrastructure.indexes import tokenization
    with isolated_database() as database:
        store = PostgresVectorStore(database, 'knowledge')
        store.add_chunks([chunk('数据库')], [[1., 0.]])
        with database.connection() as connection:
            connection.execute('ALTER TABLE vector_chunks DROP COLUMN search_vector, DROP COLUMN search_text')
            connection.execute('DELETE FROM schema_migrations WHERE version=4')
        original = tokenization.search_text
        def fail(text):
            raise RuntimeError('backfill failed')
        monkeypatch.setattr(tokenization, 'search_text', fail)
        with pytest.raises(RuntimeError, match='backfill failed'):
            migrate_database(database._dsn, database.schema)
        with database.connection() as connection:
            assert connection.execute('SELECT COUNT(*) FROM schema_migrations WHERE version=4').fetchone()[0] == 0
            assert connection.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='vector_chunks' AND column_name='search_text'").fetchone()[0] == 0
            assert connection.execute('SELECT COUNT(*) FROM vector_chunks').fetchone()[0] == 1
        monkeypatch.setattr(tokenization, 'search_text', original)
        migrate_database(database._dsn, database.schema)
        assert len(keyword(database).search('数据库', 1)) == 1


def test_uncommitted_and_rolled_back_text_changes_are_invisible():
    from app.infrastructure.indexes.tokenization import search_text
    with isolated_database() as database:
        store = PostgresVectorStore(database, 'knowledge')
        store.add_chunks([chunk('数据库')], [[1., 0.]])
        index = keyword(database)
        with pytest.raises(RuntimeError):
            with database.connection() as writer:
                writer.execute('UPDATE vector_chunks SET text=%s, search_text=%s',
                               ('量子纠缠', search_text('量子纠缠')))
                assert index.search('量子纠缠', 1) == []
                assert len(index.search('数据库', 1)) == 1
                raise RuntimeError('rollback')
        assert index.search('量子纠缠', 1) == []
        assert len(index.search('数据库', 1)) == 1


def test_default_factory_and_ingestion_do_not_load_full_corpus(monkeypatch):
    from app.bootstrap.factories import AdapterFactories
    from app.application.ingestion.pipeline import IngestionPipeline
    with isolated_database() as database:
        store = PostgresVectorStore(database, 'knowledge')
        doc = chunk('数据库')
        store.add_chunks([doc], [[1., 0.]])
        def forbidden():
            raise AssertionError('全文检索不应载入全库文本')
        monkeypatch.setattr(store, 'list_chunks', forbidden)
        index = AdapterFactories().keyword_index(store)
        assert index.search('数据库', 1)[0].chunk_id == doc.id
        pipeline = object.__new__(IngestionPipeline)
        pipeline._keyword_index, pipeline._vector_store = index, store
        pipeline._rebuild_keyword_index()
        pipeline._delete_document_unlocked(doc.doc_id)
        assert index.search('数据库', 1) == []
