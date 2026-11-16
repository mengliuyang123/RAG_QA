"""显式、串行且有版本记录的 PostgreSQL 初始化。"""
from pathlib import Path
import psycopg
from psycopg import sql
from app.infrastructure.persistence.database import validate_schema


def migrate_database(dsn: str, schema: str = 'rag_qa') -> None:
    validate_schema(schema)
    with psycopg.connect(dsn, prepare_threshold=None, connect_timeout=5) as connection:
        connection.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))',
                           ('rag_qa:migrate:' + schema,))
        connection.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(schema)))
        connection.execute(sql.SQL('SET LOCAL search_path TO {}, public').format(sql.Identifier(schema)))
        if connection.execute("SELECT 1 FROM pg_extension WHERE extname='vector'").fetchone() is None:
            raise RuntimeError('请由数据库管理员先启用 pgvector 扩展')
        connection.execute('CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)')
        if connection.execute('SELECT 1 FROM schema_migrations WHERE version=1').fetchone() is None:
            connection.execute(Path(__file__).with_name('schema.sql').read_text(encoding='utf-8'))
            connection.execute('INSERT INTO schema_migrations VALUES (1)')
        if connection.execute('SELECT 1 FROM schema_migrations WHERE version=2').fetchone() is None:
            connection.execute(Path(__file__).with_name('schema_v2.sql').read_text(encoding='utf-8'))
            connection.execute('INSERT INTO schema_migrations VALUES (2)')
        if connection.execute('SELECT 1 FROM schema_migrations WHERE version=3').fetchone() is None:
            connection.execute(Path(__file__).with_name('schema_v3.sql').read_text(encoding='utf-8'))
            connection.execute('INSERT INTO schema_migrations VALUES (3)')
        if connection.execute('SELECT 1 FROM schema_migrations WHERE version=4').fetchone() is None:
            from app.infrastructure.indexes.tokenization import search_text
            connection.execute(Path(__file__).with_name('schema_v4.sql').read_text(encoding='utf-8'))
            # ALTER TABLE 的锁持续到提交，回填与旧写入不能交错；流式读取限制内存。
            with connection.cursor(name='fulltext_backfill') as reader:
                reader.execute('SELECT collection,id,text FROM vector_chunks')
                while rows := reader.fetchmany(500):
                    with connection.cursor() as writer:
                        writer.executemany('UPDATE vector_chunks SET search_text=%s WHERE collection=%s AND id=%s',
                                           [(search_text(text), collection, identity)
                                            for collection, identity, text in rows])
            connection.execute('ALTER TABLE vector_chunks ALTER COLUMN search_text SET NOT NULL')
            connection.execute('CREATE INDEX vector_chunks_search ON vector_chunks USING GIN(search_vector)')
            connection.execute('INSERT INTO schema_migrations VALUES (4)')
        if connection.execute('SELECT 1 FROM schema_migrations WHERE version=5').fetchone() is None:
            connection.execute(Path(__file__).with_name('schema_v5.sql').read_text(encoding='utf-8'))
            connection.execute('INSERT INTO schema_migrations VALUES (5)')
        if connection.execute('SELECT 1 FROM schema_migrations WHERE version=6').fetchone() is None:
            connection.execute(Path(__file__).with_name('schema_v6.sql').read_text(encoding='utf-8'))
            connection.execute('INSERT INTO schema_migrations VALUES (6)')
        if connection.execute('SELECT 1 FROM schema_migrations WHERE version=7').fetchone() is None:
            connection.execute(Path(__file__).with_name('schema_v7.sql').read_text(encoding='utf-8'))
            connection.execute('INSERT INTO schema_migrations VALUES (7)')
    # 官方 checkpoint 迁移包含并发索引，要求 autocommit。
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row,
                         options=f'-c search_path={schema},public', prepare_threshold=None) as connection:
        connection.execute('SELECT pg_advisory_lock(hashtextextended(%s, 0))', ('rag_qa:checkpoint:' + schema,))
        try:
            PostgresSaver(connection).setup()
        finally:
            connection.execute('SELECT pg_advisory_unlock(hashtextextended(%s, 0))', ('rag_qa:checkpoint:' + schema,))
