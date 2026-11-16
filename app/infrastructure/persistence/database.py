"""应用拥有的 PostgreSQL 连接池和事务边界。"""
from collections.abc import Iterator
from contextlib import contextmanager
import re
from datetime import datetime, UTC

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


class Record(dict):
    """领域查询可以按列名读取，单值聚合也可按位置读取。"""
    def __getitem__(self, key):
        if isinstance(key, int):
            return tuple(self.values())[key]
        return super().__getitem__(key)


def record_row(cursor):
    factory = dict_row(cursor)
    def build(values):
        return Record({k: v.astimezone(UTC).isoformat() if isinstance(v, datetime) else v
                       for k, v in factory(values).items()})
    return build


def validate_schema(schema: str) -> str:
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,62}', schema):
        raise ValueError('PostgreSQL schema 名称无效')
    return schema


class Database:
    def __init__(self, dsn: str, schema: str = 'rag_qa') -> None:
        self.schema = validate_schema(schema)
        self._dsn = dsn
        self._pool = ConnectionPool(
            dsn, min_size=0, max_size=8, open=False, timeout=10,
            kwargs={'row_factory': record_row, 'options': f'-c search_path={schema},public',
                    'connect_timeout': 5, 'prepare_threshold': None},
        )
        self._pool.open()

    @contextmanager
    def connection(self, *, read_only: bool = False) -> Iterator[psycopg.Connection]:
        with self._pool.connection() as connection:
            with connection.transaction():
                if read_only:
                    connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
                yield connection

    def close(self) -> None:
        self._pool.close()

    def acquire(self) -> psycopg.Connection:
        """供需要显式取消检查和提交边界的会话写入借用连接。"""
        return self._pool.getconn()

    def release(self, connection: psycopg.Connection) -> None:
        self._pool.putconn(connection)
