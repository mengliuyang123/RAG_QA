# -*- coding: utf-8 -*-
"""认证账号、服务端会话及登录限流的独立SQLite存储。"""

import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime
from pathlib import Path
from uuid import UUID

from app.application.errors import AuthStorageError
from app.application.errors import LoginRateLimitError
from app.application.errors import UserAlreadyExistsError
from app.domain.auth import AuthenticatedSession
from app.domain.auth import StoredSession
from app.domain.auth import User
from app.domain.auth import UserAccount


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('member', 'admin')),
    disabled INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash TEXT PRIMARY KEY,
    csrf_hash TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    last_seen_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS auth_sessions_expiry ON auth_sessions(expires_at);
CREATE TABLE IF NOT EXISTS login_buckets (
    bucket_key TEXT PRIMARY KEY,
    window_start INTEGER NOT NULL,
    attempts INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS login_buckets_window ON login_buckets(window_start);
"""


class SQLiteAuthRepository:
    """短连接、参数化事务存储，不在日志中打印认证数据。"""

    def __init__(self, path: Path) -> None:
        """初始化独立认证表，不创建任何默认账号。

        Args:
            path: 认证数据库路径。

        Returns:
            None。
        """
        self._path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(_SCHEMA)
        except (OSError, sqlite3.Error) as exc:
            raise AuthStorageError("认证存储初始化失败") from exc

    def create_user(self, account: UserAccount) -> None:
        """创建规范化且唯一的登录账号。

        Args:
            account: 已验证和哈希的账号。

        Returns:
            None。
        """
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO users(
                        id, username, name, password_hash, role, disabled, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(account.id), account.username, account.name,
                        account.password_hash, account.role, int(account.disabled),
                        account.created_at.timestamp(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise UserAlreadyExistsError("登录标识已存在") from exc
        except sqlite3.Error as exc:
            raise AuthStorageError("账号创建失败") from exc

    def user_by_name(self, username: str) -> UserAccount | None:
        """读取账号，仅供密码验证和可信管理工具使用。

        Args:
            username: 规范化标识。

        Returns:
            账号或None。
        """
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM users WHERE username = ?", (username,),
                ).fetchone()
            if row is None:
                return None
            return UserAccount(
                id=UUID(row["id"]), username=row["username"], name=row["name"],
                password_hash=row["password_hash"], role=row["role"],
                disabled=bool(row["disabled"]),
                created_at=datetime.fromtimestamp(row["created_at"], UTC),
            )
        except (sqlite3.Error, ValueError) as exc:
            raise AuthStorageError("账号读取失败") from exc

    def save_session(self, session: StoredSession) -> None:
        """保存新令牌摘要，同时清理已绝对过期的认证令牌。

        Args:
            session: 服务器会话记录。

        Returns:
            None。
        """
        try:
            with self._connect() as connection:
                connection.execute("DELETE FROM auth_sessions WHERE expires_at <= ?", (
                    session.created_at.timestamp(),
                ))
                connection.execute(
                    """INSERT INTO auth_sessions(
                        token_hash, csrf_hash, user_id, created_at, expires_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        session.token_hash, session.csrf_hash, str(session.user_id),
                        session.created_at.timestamp(), session.expires_at.timestamp(),
                        session.last_seen_at.timestamp(),
                    ),
                )
        except sqlite3.Error as exc:
            raise AuthStorageError("认证会话保存失败") from exc

    def resolve_session(
        self, token_hash: str, now: datetime, idle_seconds: int,
    ) -> AuthenticatedSession | None:
        """使用数据库当前角色与状态，原子检查并刷新闲置时间。

        Args:
            token_hash: 请求令牌摘要。
            now: UTC当前时间。
            idle_seconds: 闲置时限。

        Returns:
            有效身份或None。
        """
        stamp = now.timestamp()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """SELECT u.id, u.name, u.role, s.csrf_hash, s.expires_at
                    FROM auth_sessions s JOIN users u ON u.id = s.user_id
                    WHERE s.token_hash = ? AND u.disabled = 0
                        AND s.expires_at > ? AND s.last_seen_at > ?""",
                    (token_hash, stamp, stamp - idle_seconds),
                ).fetchone()
                if row is None:
                    return None
                connection.execute(
                    "UPDATE auth_sessions SET last_seen_at = ? WHERE token_hash = ?",
                    (stamp, token_hash),
                )
                return AuthenticatedSession(
                    user=User(id=UUID(row["id"]), name=row["name"], role=row["role"]),
                    csrf_hash=row["csrf_hash"],
                    expires_at=datetime.fromtimestamp(row["expires_at"], UTC),
                )
        except (sqlite3.Error, ValueError) as exc:
            raise AuthStorageError("认证会话读取失败") from exc

    def revoke_session(self, token_hash: str) -> None:
        """幂等删除精确会话，不影响其他浏览器的登录。

        Args:
            token_hash: 待撤销令牌摘要。

        Returns:
            None。
        """
        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,),
                )
        except sqlite3.Error as exc:
            raise AuthStorageError("认证会话撤销失败") from exc

    def consume_login_attempts(
        self, keys: tuple[str, ...], now: datetime,
        max_attempts: int, window_seconds: int,
    ) -> None:
        """以写事务预留全部额度，避免并发穿透或计数表无限累积。

        Args:
            keys: 账号和来源的指纹。
            now: 当前UTC时间。
            max_attempts: 每窗口额度。
            window_seconds: 固定窗口时长。

        Returns:
            None，超限抛出限流异常。
        """
        stamp = now.timestamp()
        window = int(stamp // window_seconds) * window_seconds
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DELETE FROM login_buckets WHERE window_start < ?", (window,))
                for key in keys:
                    row = connection.execute(
                        "SELECT attempts FROM login_buckets WHERE bucket_key = ?",
                        (key,),
                    ).fetchone()
                    if row is not None and row[0] >= max_attempts:
                        raise LoginRateLimitError(math.ceil(window + window_seconds - stamp))
                for key in keys:
                    connection.execute(
                        """INSERT INTO login_buckets(bucket_key, window_start, attempts)
                        VALUES (?, ?, 1) ON CONFLICT(bucket_key) DO UPDATE
                        SET attempts = attempts + 1, window_start = excluded.window_start""",
                        (key, window),
                    )
        except sqlite3.Error as exc:
            raise AuthStorageError("登录限流存储失败") from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()
