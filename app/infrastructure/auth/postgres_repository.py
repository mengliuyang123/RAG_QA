# -*- coding: utf-8 -*-
"""认证账号、服务端会话及登录限流的独立Postgres存储。"""

import math
import psycopg
from app.infrastructure.persistence.database import Database, Record
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime
from pathlib import Path
from uuid import UUID

from app.application.errors import AuthStorageError
from app.application.errors import LastAdminError
from app.application.errors import LoginRateLimitError
from app.application.errors import UserAlreadyExistsError
from app.application.errors import UserNotFoundError
from app.domain.auth import AuthenticatedSession
from app.domain.auth import StoredSession
from app.domain.auth import User
from app.domain.auth import UserAccount
from app.domain.auth import UserRole


_BUSINESS_LOCK = "SELECT pg_advisory_xact_lock(hashtextextended(current_schema() || 'business_write', 0))"


def _account(row: Record) -> UserAccount:
    return UserAccount(
        id=UUID(row["id"]), username=row["username"], name=row["name"],
        password_hash=row["password_hash"], role=row["role"],
        disabled=bool(row["disabled"]),
        created_at=datetime.fromtimestamp(row["created_at"], UTC),
    )




class PostgresAuthRepository:
    """短连接、参数化事务存储，不在日志中打印认证数据。"""

    def __init__(self, database: Database) -> None:
        self._database = database

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
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (
                        str(account.id), account.username, account.name,
                        account.password_hash, account.role, int(account.disabled),
                        account.created_at.timestamp(),
                    ),
                )
        except psycopg.IntegrityError as exc:
            raise UserAlreadyExistsError("登录标识已存在") from exc
        except psycopg.Error as exc:
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
                    "SELECT * FROM users WHERE username = %s", (username,),
                ).fetchone()
            return None if row is None else _account(row)
        except (psycopg.Error, ValueError) as exc:
            raise AuthStorageError("账号读取失败") from exc

    def list_users(self) -> list[UserAccount]:
        """按创建时间列出账号，供管理员页面使用。

        Returns:
            账号列表。
        """
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT * FROM users ORDER BY created_at, username",
                ).fetchall()
            return [_account(row) for row in rows]
        except (psycopg.Error, ValueError) as exc:
            raise AuthStorageError("账号读取失败") from exc

    def user_by_id(self, user_id: UUID) -> UserAccount | None:
        """按标识读取账号。

        Args:
            user_id: 用户标识。

        Returns:
            账号或None。
        """
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM users WHERE id = %s", (str(user_id),),
                ).fetchone()
            return None if row is None else _account(row)
        except (psycopg.Error, ValueError) as exc:
            raise AuthStorageError("账号读取失败") from exc

    def update_user(
        self, user_id: UUID, *, role: UserRole | None, disabled: bool | None,
    ) -> UserAccount:
        """业务写锁内检查最后一名管理员，避免两个管理员并发互相降级。

        Args:
            user_id: 目标账号。
            role: 新角色或None。
            disabled: 新禁用状态或None。

        Returns:
            修改后的账号。
        """
        try:
            with self._connect() as connection:
                connection.execute(_BUSINESS_LOCK)
                row = connection.execute(
                    "SELECT * FROM users WHERE id = %s", (str(user_id),),
                ).fetchone()
                if row is None:
                    raise UserNotFoundError("账号不存在")
                current = _account(row)
                new_role = current.role if role is None else role
                new_disabled = current.disabled if disabled is None else disabled
                was_admin = current.role == "admin" and not current.disabled
                stays_admin = new_role == "admin" and not new_disabled
                if was_admin and not stays_admin:
                    others = connection.execute(
                        "SELECT count(*) FROM users WHERE role = 'admin' AND disabled = 0 AND id <> %s",
                        (str(user_id),),
                    ).fetchone()[0]
                    if others == 0:
                        raise LastAdminError("至少需要保留一名启用的管理员")
                connection.execute(
                    "UPDATE users SET role = %s, disabled = %s WHERE id = %s",
                    (new_role, int(new_disabled), str(user_id)),
                )
                if new_disabled and not current.disabled:
                    connection.execute(
                        "DELETE FROM auth_sessions WHERE user_id = %s", (str(user_id),),
                    )
            return current.model_copy(update={"role": new_role, "disabled": new_disabled})
        except psycopg.Error as exc:
            raise AuthStorageError("账号修改失败") from exc

    def set_password(
        self, user_id: UUID, password_hash: str, *, keep_token_hash: str | None = None,
    ) -> None:
        """替换口令并撤销该账号会话，旧口令登录的浏览器需要重新登录。

        Args:
            user_id: 目标账号。
            password_hash: 新哈希。
            keep_token_hash: 用户自己改口令时保留当前浏览器的会话。

        Returns:
            None。
        """
        try:
            with self._connect() as connection:
                connection.execute(_BUSINESS_LOCK)
                updated = connection.execute(
                    "UPDATE users SET password_hash = %s WHERE id = %s",
                    (password_hash, str(user_id)),
                ).rowcount
                if updated == 0:
                    raise UserNotFoundError("账号不存在")
                connection.execute(
                    "DELETE FROM auth_sessions WHERE user_id = %s AND token_hash <> %s",
                    (str(user_id), keep_token_hash or ""),
                )
        except psycopg.Error as exc:
            raise AuthStorageError("口令重置失败") from exc

    def save_session(self, session: StoredSession) -> None:
        """保存新令牌摘要，同时清理已绝对过期的认证令牌。

        Args:
            session: 服务器会话记录。

        Returns:
            None。
        """
        try:
            with self._connect() as connection:
                connection.execute("DELETE FROM auth_sessions WHERE expires_at <= %s", (
                    session.created_at.timestamp(),
                ))
                connection.execute(
                    """INSERT INTO auth_sessions(
                        token_hash, csrf_hash, user_id, created_at, expires_at, last_seen_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)""",
                    (
                        session.token_hash, session.csrf_hash, str(session.user_id),
                        session.created_at.timestamp(), session.expires_at.timestamp(),
                        session.last_seen_at.timestamp(),
                    ),
                )
        except psycopg.Error as exc:
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
                connection.execute(_BUSINESS_LOCK)
                row = connection.execute(
                    """SELECT u.id, u.name, u.role, s.csrf_hash, s.expires_at
                    FROM auth_sessions s JOIN users u ON u.id = s.user_id
                    WHERE s.token_hash = %s AND u.disabled = 0
                        AND s.expires_at > %s AND s.last_seen_at > %s""",
                    (token_hash, stamp, stamp - idle_seconds),
                ).fetchone()
                if row is None:
                    return None
                connection.execute(
                    "UPDATE auth_sessions SET last_seen_at = %s WHERE token_hash = %s",
                    (stamp, token_hash),
                )
                return AuthenticatedSession(
                    user=User(id=UUID(row["id"]), name=row["name"], role=row["role"]),
                    csrf_hash=row["csrf_hash"],
                    expires_at=datetime.fromtimestamp(row["expires_at"], UTC),
                )
        except (psycopg.Error, ValueError) as exc:
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
                    "DELETE FROM auth_sessions WHERE token_hash = %s", (token_hash,),
                )
        except psycopg.Error as exc:
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
                connection.execute(_BUSINESS_LOCK)
                connection.execute("DELETE FROM login_buckets WHERE window_start < %s", (window,))
                for key in keys:
                    row = connection.execute(
                        "SELECT attempts FROM login_buckets WHERE bucket_key = %s",
                        (key,),
                    ).fetchone()
                    if row is not None and row[0] >= max_attempts:
                        raise LoginRateLimitError(math.ceil(window + window_seconds - stamp))
                for key in keys:
                    connection.execute(
                        """INSERT INTO login_buckets(bucket_key, window_start, attempts)
                        VALUES (%s, %s, 1) ON CONFLICT(bucket_key) DO UPDATE
                        SET attempts = login_buckets.attempts + 1, window_start = excluded.window_start""",
                        (key, window),
                    )
        except psycopg.Error as exc:
            raise AuthStorageError("登录限流存储失败") from exc

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection]:
        try:
            with self._database.connection() as connection:
                yield connection
        except psycopg.IntegrityError:
            # 唯一约束冲突由调用方映射为稳定业务错误（如登录名重复）。
            raise
        except psycopg.Error as exc:
            raise AuthStorageError("PostgreSQL 仓储操作失败") from exc
