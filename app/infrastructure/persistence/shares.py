# -*- coding: utf-8 -*-
"""用业务库事务约束单轮分享，删除正式轮次同时撤销授权。"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from pydantic import AwareDatetime
from pydantic import TypeAdapter

from app.application.errors import ShareExpiredError
from app.application.errors import ShareNotFoundError
from app.application.errors import ShareStorageError
from app.application.errors import TurnNotFoundError
from app.domain.conversations import TurnCitation
from app.domain.shares import SharedTurn
from app.domain.shares import StoredShare


class SQLiteShareRepository:
    """分享只保存摘要和身份，内容从不可变的正式轮次投影。"""

    def __init__(self, path: Path) -> None:
        """由组合根先初始化正式schema，再增加分享表。

        Args:
            path: 当前运行的正式会话数据库。

        Returns:
            None。
        """
        self._path = path
        with self._connection() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS turn_shares (
                token_hash TEXT PRIMARY KEY,
                turn_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                shared_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(turn_id) REFERENCES conversation_turns(id)
                    ON DELETE CASCADE
            )""")
            connection.execute("""CREATE INDEX IF NOT EXISTS idx_shares_turn
                ON turn_shares(turn_id)""")

    def create(self, share: StoredShare) -> None:
        """将归属检查与授权写入放在同一个写事务内。

        Args:
            share: 不含明文令牌的授权。

        Returns:
            None。
        """
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""SELECT t.id
                FROM conversation_turns t
                JOIN conversations c ON c.id = t.conversation_id
                WHERE t.id = ? AND c.owner_id = ? AND c.status = 'active'
            """, (str(share.turn_id), str(share.owner_id))).fetchone()
            if row is None:
                raise TurnNotFoundError("轮次不存在")
            connection.execute("""INSERT INTO turn_shares (
                token_hash, turn_id, owner_id, shared_by,
                created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?)""", (
                share.token_hash, str(share.turn_id), str(share.owner_id),
                share.shared_by, share.created_at.isoformat(),
                share.expires_at.isoformat(),
            ))

    def read(self, token_hash: str, now: datetime) -> SharedTurn:
        """先检查授权和活动会话，再从同一快照读取该轮正文。

        Args:
            token_hash: 高熵令牌摘要。
            now: 当前UTC时间。

        Returns:
            指定单轮内容，编号保持原顺序。
        """
        with self._connection() as connection:
            connection.execute("BEGIN")
            share = connection.execute("""SELECT s.turn_id, s.shared_by,
                s.created_at, s.expires_at
                FROM turn_shares s
                JOIN conversation_turns t ON t.id = s.turn_id
                JOIN conversations c ON c.id = t.conversation_id
                WHERE s.token_hash = ? AND c.owner_id = s.owner_id
                    AND c.status = 'active'
            """, (token_hash,)).fetchone()
            if share is None:
                raise ShareNotFoundError("分享不存在")
            expires_at = TypeAdapter(AwareDatetime).validate_python(
                share["expires_at"],
            )
            if expires_at <= now:
                raise ShareExpiredError("分享已过期")
            turn = connection.execute("""SELECT user_message,
                assistant_message FROM conversation_turns WHERE id = ?
            """, (share["turn_id"],)).fetchone()
            citations = []
            rows = connection.execute("""SELECT position, doc_id, filename,
                page, chunk_text, chunk_id, kb_id, doc_updated_at
                FROM turn_citations WHERE turn_id = ?
                ORDER BY COALESCE(citation_order, position)
            """, (share["turn_id"],))
            for row in rows:
                values = dict(row)
                values["kb_id"] = values["kb_id"] or "default"
                citations.append(TurnCitation.model_validate(values))
            return SharedTurn(
                question=turn["user_message"],
                answer=turn["assistant_message"], citations=tuple(citations),
                shared_by=share["shared_by"], created_at=share["created_at"],
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        try:
            uri = self._path.resolve().as_uri() + "?mode=rw"
            with closing(sqlite3.connect(uri, uri=True, timeout=5)) as db:
                db.row_factory = sqlite3.Row
                db.execute("PRAGMA foreign_keys = ON")
                with db:
                    yield db
        except (sqlite3.Error, OSError, ValueError, TypeError) as error:
            raise ShareStorageError("分享存储暂时不可用") from error
