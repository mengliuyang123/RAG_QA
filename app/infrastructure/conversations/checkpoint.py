"""LangGraph 官方 PostgreSQL checkpoint，初始化由迁移命令负责。"""
from uuid import UUID
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from app.application.errors import ConversationDeleteError
from app.infrastructure.persistence.database import Database


class PostgresCheckpointStore:
    def __init__(self, database: Database) -> None:
        self._pool = ConnectionPool(database._dsn, min_size=0, max_size=4,
            kwargs={'autocommit': True, 'prepare_threshold': None, 'row_factory': dict_row,
                    'connect_timeout': 5, 'options': f'-c search_path={database.schema},public'}, open=False)
        self._pool.open()
        self._checkpointer = PostgresSaver(self._pool, serde=JsonPlusSerializer(allowed_msgpack_modules=[]))

    @property
    def checkpointer(self) -> PostgresSaver:
        return self._checkpointer

    def delete_thread(self, conversation_id: UUID) -> None:
        try:
            self._checkpointer.delete_thread(str(conversation_id))
        except psycopg.Error as exc:
            raise ConversationDeleteError('会话 checkpoint 删除失败') from exc

    def close(self) -> None:
        self._pool.close()
