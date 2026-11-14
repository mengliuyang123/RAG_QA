"""仅在独立测试数据库的随机 schema 中运行持久化测试。"""
from contextlib import contextmanager
from uuid import uuid4

from dotenv import dotenv_values
import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict


def test_database_url() -> str:
    import os
    values = {**dotenv_values('.env'), **os.environ}
    target = values.get('TEST_DATABASE_URL')
    if not target:
        raise RuntimeError('请配置独立 TEST_DATABASE_URL')
    actual = conninfo_to_dict(target)
    production = conninfo_to_dict(values['DATABASE_URL'])
    if all(actual.get(k) == production.get(k) for k in ('host', 'port', 'dbname')):
        raise ValueError('测试数据库必须与应用数据库分离')
    return target


@contextmanager
def isolated_database():
    from app.infrastructure.persistence.database import Database
    from app.infrastructure.persistence.migrations import migrate_database
    dsn = test_database_url()
    schema = 'test_' + uuid4().hex
    migrate_database(dsn, schema)
    database = Database(dsn, schema)
    try:
        yield database
    finally:
        database.close()
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))


# 每个测试使用独立命名空间；路径仅用作测试案例内的逻辑分组键。
_active_databases = None
_active_root = None
_active_schema = None


def pg_database(identity=None):
    from pathlib import Path
    from hashlib import sha256
    from app.infrastructure.persistence.database import Database
    if isinstance(identity, Database):
        return identity
    if _active_databases is None:
        raise RuntimeError('数据库测试必须通过 pytest 隔离 fixture 运行')
    parent = (Path(identity).parent if Path(identity).suffix in ('.db','.db','.sqlite') else Path(identity)).resolve() if identity is not None else _active_root
    key = str(parent)
    if key not in _active_databases:
        suffix = '' if parent == _active_root else '_' + sha256(key.encode()).hexdigest()[:8]
        _active_databases[key] = Database(test_database_url(), _active_schema + suffix)
    return _active_databases[key]


def pg_connect(identity=None, **kwargs):
    return pg_database(identity).connection()


@contextmanager
def observe_queries(database, before=None):
    """观察真实 PostgreSQL 查询边界，不替换查询或结果。"""
    original = database.connection
    queries = []
    class Observed:
        def __init__(self, connection): self._connection = connection
        def __getattr__(self, name): return getattr(self._connection, name)
        def execute(self, query, *args, **kwargs):
            queries.append(query)
            if before is not None: before(query)
            return self._connection.execute(query, *args, **kwargs)
    @contextmanager
    def observed(**kwargs):
        with original(**kwargs) as connection:
            yield Observed(connection)
    database.connection = observed
    try:
        yield queries
    finally:
        database.connection = original


def pg_schema(identity):
    from pathlib import Path
    from hashlib import sha256
    parent=(Path(identity).parent if Path(identity).suffix in ('.db','.db','.sqlite') else Path(identity)).resolve()
    suffix='' if parent==_active_root else '_'+sha256(str(parent).encode()).hexdigest()[:8]
    return _active_schema+suffix
