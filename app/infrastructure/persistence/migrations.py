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
