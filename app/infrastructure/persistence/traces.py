# -*- coding: utf-8 -*-
"""以独立Postgres存储追踪，防止业务会话和评估追踪路径混用。"""

import psycopg
from app.infrastructure.persistence.database import Database, Record
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


class PostgresTraceRepository:
    """使用短连接和事务保存追踪，不保留跨请求Postgres连接。"""

    def __init__(self, database: Database, clock: Callable[[], datetime] = _utc_now) -> None:
        self._database = database
        self._clock = clock

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
                    ) VALUES (%s, %s, %s, %s)
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
        except (psycopg.Error, ValueError) as exc:
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
                    "VALUES (%s, %s, %s)",
                    (str(audit.id), audit.expires_at.astimezone(UTC).isoformat(), audit.model_dump_json()),
                )
        except (psycopg.Error, ValueError) as error:
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
                    "SELECT snapshot_json FROM traces WHERE trace_id = %s",
                    (str(trace_id),),
                ).fetchone()
            if row is None:
                raise TraceNotFoundError("追踪不存在")
            snapshot = TraceSnapshot.model_validate_json(row[0])
            if snapshot.expires_at <= self._clock():
                raise TraceNotFoundError("追踪不存在")
            return snapshot
        except (psycopg.Error, ValueError) as exc:
            raise TraceStorageError("追踪存储读取失败") from exc

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection]:
        try:
            with self._database.connection(read_only=False) as connection:
                yield connection
        except psycopg.Error as exc:
            raise TraceStorageError("PostgreSQL 仓储操作失败") from exc

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
                connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(current_schema() || 'business_write', 0))")
                row = connection.execute(
                    "SELECT snapshot_json FROM traces WHERE trace_id = %s "
                    "AND expires_at > %s",
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
                    "UPDATE traces SET snapshot_json = %s WHERE trace_id = %s",
                    (updated.model_dump_json(), str(trace_id)),
                )
        except (psycopg.Error, ValueError) as exc:
            raise TraceStorageError("追踪提交状态写入失败") from exc

    def prune_expired(self) -> int:
        """按精确截止时间清理，不扫描或删除其他数据库。

        Returns:
            删除的过期追踪数。
        """
        try:
            with self._connect() as connection:
                deleted = connection.execute(
                    "DELETE FROM traces WHERE expires_at <= %s",
                    (self._clock().astimezone(UTC).isoformat(),),
                )
                connection.execute(
                    "DELETE FROM admin_access_audit WHERE expires_at <= %s",
                    (self._clock().astimezone(UTC).isoformat(),),
                )
                return deleted.rowcount
        except (psycopg.Error, ValueError, TypeError) as exc:
            raise TraceStorageError("追踪过期清理失败") from exc
