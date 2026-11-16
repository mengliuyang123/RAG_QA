"""PostgreSQL 连接池由容器关闭，向量适配器不关闭其他拥有者。"""
from pathlib import Path
from app.infrastructure.vectorstores.postgres import PostgresVectorStore
from app.infrastructure.persistence.database import Database
from tests.postgres_support import pg_database, test_database_url as database_url
from tests.test_unified_query_api import _app
from tests.test_vectorstore import make_chunks


def test_container_close_releases_owned_postgres_pool(tmp_path: Path):
    app, _, _ = _app(tmp_path)
    database = app.state.container.database
    assert not database._pool.closed
    app.state.container.close()
    assert database._pool.closed


def test_store_close_keeps_other_client_alive_and_reopens_data(tmp_path: Path):
    database = pg_database(tmp_path / 'vectors.db')
    first = PostgresVectorStore(database, 'lifecycle')
    second = PostgresVectorStore(database, 'lifecycle')
    first.add_chunks(make_chunks(), [[1.0,0.0],[0.0,1.0]])
    first.close()
    first.close()
    assert second.list_chunks() == make_chunks()
    reopened_database = Database(database_url(), database.schema)
    try:
        assert PostgresVectorStore(reopened_database, 'lifecycle').list_chunks() == make_chunks()
    finally:
        reopened_database.close()
