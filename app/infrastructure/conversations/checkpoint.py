# -*- coding: utf-8 -*-
"""独立管理 LangGraph SQLite checkpoint 连接和 thread 清理。"""

import sqlite3
from pathlib import Path
from threading import RLock
from uuid import UUID

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from app.application.errors import ConversationDeleteError
from app.application.errors import ConversationStorageError


class SQLiteCheckpointStore:
    """持有同步 SqliteSaver，并提供幂等资源关闭边界。"""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        serializer = JsonPlusSerializer(allowed_msgpack_modules=[])
        self._checkpointer = SqliteSaver(
            self._connection,
            serde=serializer,
        )
        self._checkpointer.setup()
        self._close_lock = RLock()
        self._closed = False

    @property
    def checkpointer(self) -> SqliteSaver:
        """返回供 StateGraph 编译使用的共享 checkpointer。"""
        return self._checkpointer

    def delete_thread(self, conversation_id: UUID) -> None:
        """删除目标 thread 的全部 checkpoint 和 pending writes。"""
        try:
            self._checkpointer.delete_thread(str(conversation_id))
        except sqlite3.Error as exc:
            raise ConversationDeleteError("会话 checkpoint 删除失败") from exc
        except Exception as exc:
            raise ConversationStorageError("会话 checkpoint 操作失败") from exc

    def close(self) -> None:
        """幂等关闭 SQLite 连接。"""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._connection.close()
