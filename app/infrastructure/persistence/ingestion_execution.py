"""用 PostgreSQL 会话咨询锁排斥同一 schema 的后台执行器。"""
from contextlib import contextmanager
from app.application.errors import IngestionWorkerUnavailableError
from app.infrastructure.persistence.database import Database


class PostgresIngestionExecutionGuard:
    def __init__(self, database: Database) -> None:
        self._database = database

    @contextmanager
    def hold(self):
        with self._database.connection() as connection:
            key = self._database.schema + ':ingestion-worker'
            acquired = connection.execute('SELECT pg_try_advisory_lock(hashtextextended(%s,0))', (key,)).fetchone()[0]
            if not acquired:
                raise IngestionWorkerUnavailableError('入库执行器暂时不可用')
            try:
                yield
            finally:
                connection.execute('SELECT pg_advisory_unlock(hashtextextended(%s,0))', (key,))
