# -*- coding: utf-8 -*-
"""以独立SQLite存储追踪，防止业务会话和评估追踪路径混用。"""

import sqlite3
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime
from pathlib import Path
from uuid import UUID

from app.application.errors import TraceNotFoundError
from app.application.errors import TraceStorageError
from app.domain.tracing import TraceSnapshot
from app.domain.tracing import CommitStatus
from app.domain.auditing import AdminAccessAudit


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SQLiteTraceRepository:
    """使用短连接和事务保存追踪，不保留跨请求SQLite连接。"""

    def __init__(
        self, path: Path, clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """创建追踪专属表。

        Args:
            path: 独立数据库路径。
            clock: 用于保留期限检查的UTC时钟。

        Returns:
            None。
        """
        self._path = path
        self._clock = clock
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript("""
                    CREATE TABLE IF NOT EXISTS traces (
                        trace_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        snapshot_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS traces_created
                        ON traces(created_at, trace_id);
                    CREATE INDEX IF NOT EXISTS traces_expiry
                        ON traces(expires_at);
                    CREATE TABLE IF NOT EXISTS admin_access_audit (
                        id TEXT PRIMARY KEY,
                        expires_at TEXT NOT NULL,
                        event_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS admin_access_audit_expiry
                        ON admin_access_audit(expires_at);
                """)
        except (sqlite3.Error, OSError) as exc:
            raise TraceStorageError("追踪存储初始化失败") from exc

    def save(self, snapshot: TraceSnapshot) -> None:
        """事务替换同一追踪，不生成重复记录。

        Args:
            snapshot: 待保存的已校验快照。

        Returns:
            None。
        """
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO traces(
                        trace_id, created_at, expires_at, snapshot_json
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(trace_id) DO UPDATE SET
                        expires_at = excluded.expires_at,
                        snapshot_json = excluded.snapshot_json
                    """,
                    (
                        str(snapshot.trace_id),
                        snapshot.created_at.astimezone(UTC).isoformat(),
                        snapshot.expires_at.astimezone(UTC).isoformat(),
                        snapshot.model_dump_json(),
                    ),
                )
        except (sqlite3.Error, ValueError) as exc:
            raise TraceStorageError("追踪存储写入失败") from exc

    def record_admin_access(self, audit: AdminAccessAudit) -> None:
        """先持久化审计再允许接口返回敏感内容。

        Args:
            audit: 不含内容正文的访问记录。

        Returns:
            None。
        """
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO admin_access_audit(id, expires_at, event_json) "
                    "VALUES (?, ?, ?)",
                    (str(audit.id), audit.expires_at.astimezone(UTC).isoformat(), audit.model_dump_json()),
                )
        except (sqlite3.Error, ValueError) as error:
            raise TraceStorageError("后台访问审计写入失败") from error

    def get(self, trace_id: UUID) -> TraceSnapshot:
        """读取精确ID，过期追踪不对查询接口开放。

        Args:
            trace_id: 待查询的追踪标识。

        Returns:
            完整快照。
        """
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT snapshot_json FROM traces WHERE trace_id = ?",
                    (str(trace_id),),
                ).fetchone()
            if row is None:
                raise TraceNotFoundError("追踪不存在")
            snapshot = TraceSnapshot.model_validate_json(row[0])
            if snapshot.expires_at <= self._clock():
                raise TraceNotFoundError("追踪不存在")
            return snapshot
        except (sqlite3.Error, ValueError) as exc:
            raise TraceStorageError("追踪存储读取失败") from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=5.0)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record_commit(
        self, trace_id: UUID, status: CommitStatus, turn_id: UUID | None,
    ) -> None:
        """独立短事务记录提交，不插入新追踪或复活过期记录。

        Args:
            trace_id: 现有追踪标识。
            status: 本轮正式提交结果。
            turn_id: 成功提交后的真实轮次标识。

        Returns:
            None，已确认提交的状态不可降级。
        """
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT snapshot_json FROM traces WHERE trace_id = ? "
                    "AND expires_at > ?",
                    (str(trace_id), self._clock().astimezone(UTC).isoformat()),
                ).fetchone()
                if row is None:
                    raise TraceNotFoundError("追踪不存在")
                previous = TraceSnapshot.model_validate_json(row[0])
                if previous.commit_status == "committed":
                    return
                updated = TraceSnapshot.model_validate({
                    **previous.model_dump(), "commit_status": status,
                    "turn_id": turn_id,
                })
                connection.execute(
                    "UPDATE traces SET snapshot_json = ? WHERE trace_id = ?",
                    (updated.model_dump_json(), str(trace_id)),
                )
        except (sqlite3.Error, ValueError) as exc:
            raise TraceStorageError("追踪提交状态写入失败") from exc

    def prune_expired(self) -> int:
        """按精确截止时间清理，不扫描或删除其他数据库。

        Returns:
            删除的过期追踪数。
        """
        try:
            with self._connect() as connection:
                deleted = connection.execute(
                    "DELETE FROM traces WHERE expires_at <= ?",
                    (self._clock().astimezone(UTC).isoformat(),),
                )
                connection.execute(
                    "DELETE FROM admin_access_audit WHERE expires_at <= ?",
                    (self._clock().astimezone(UTC).isoformat(),),
                )
                return deleted.rowcount
        except (sqlite3.Error, ValueError, TypeError) as exc:
            raise TraceStorageError("追踪过期清理失败") from exc
