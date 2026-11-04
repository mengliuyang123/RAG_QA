# -*- coding: utf-8 -*-
"""以只读SQLite关联查询恢复真实提交事实，不做跨库假事务。"""

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
from app.domain.feedback import TurnFeedback
from app.domain.trace_queries import TraceRecord
from app.domain.trace_queries import TraceFilters
from app.domain.trace_queries import TracePosition
from app.domain.tracing import TraceSnapshot


def _utc_now() -> datetime:
    return datetime.now(UTC)


_SELECT = """
    SELECT tr.snapshot_json, t.id AS turn_id,
        t.conversation_id, c.owner_id, t.feedback_json,
        (SELECT COUNT(*) FROM business.conversation_turns linked
         WHERE linked.trace_id = tr.trace_id) AS linked_count
    FROM traces tr
    LEFT JOIN business.conversation_turns t ON t.trace_id = tr.trace_id
    LEFT JOIN business.conversations c ON c.id = t.conversation_id
"""


class SQLiteTraceReader:
    """短连接读取双库，不复制问题、回答、引用正文或反馈备注。"""

    def __init__(
        self, trace_path: Path, conversation_path: Path,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """保存由组合根确定的同一运行隔离路径。

        Args:
            trace_path: 已初始化的追踪库。
            conversation_path: 已初始化的业务库。
            clock: UTC时间。

        Returns:
            None。
        """
        self._trace_path = trace_path
        self._conversation_path = conversation_path
        self._clock = clock

    def get(self, trace_id: UUID) -> TraceRecord:
        """单次只读查询获取当前提交及反馈，不加载全部会话历史。

        Args:
            trace_id: 精确追踪标识。

        Returns:
            未过期的追踪及权威正式轮次关联。
        """
        try:
            with self._connect() as connection:
                row = connection.execute(
                    _SELECT + " WHERE tr.trace_id = ? AND tr.expires_at > ?",
                    (str(trace_id), self._clock().astimezone(UTC).isoformat()),
                ).fetchone()
                if row is None:
                    raise TraceNotFoundError("追踪不存在")
                return self._record(row)
        except (sqlite3.Error, OSError, ValueError) as exc:
            raise TraceStorageError("追踪关联查询失败") from exc

    @staticmethod
    def _record(row: sqlite3.Row) -> TraceRecord:
        if row["linked_count"] > 1:
            raise TraceStorageError("追踪关联不唯一")
        feedback = None
        if row["feedback_json"]:
            feedback = TurnFeedback.model_validate_json(row["feedback_json"]).rating
        return TraceRecord(
            snapshot=TraceSnapshot.model_validate_json(row["snapshot_json"]),
            turn_id=UUID(row["turn_id"]) if row["turn_id"] else None,
            conversation_id=UUID(row["conversation_id"]) if row["conversation_id"] else None,
            owner_id=UUID(row["owner_id"]) if row["owner_id"] else None,
            feedback=feedback,
        )

    def find(
        self, filters: TraceFilters, position: TracePosition | None,
        limit: int,
    ) -> tuple[TraceRecord, ...]:
        """在SQL中先关联／过滤再分页，不全量加载或N+1读取历史。

        Args:
            filters: 后端校验的查询条件。
            position: 可选稳定排序边界。
            limit: 读取上限。

        Returns:
            时间和ID倒序的只读投影。
        """
        if not 1 <= limit <= 101:
            raise ValueError("追踪读取上限无效")
        query, parameters = self._selection(filters, position)
        parameters.append(limit)
        try:
            with self._connect() as connection:
                rows = connection.execute(query + " LIMIT ?", parameters).fetchall()
                return tuple(self._record(row) for row in rows)
        except (sqlite3.Error, OSError, ValueError) as exc:
            raise TraceStorageError("追踪列表查询失败") from exc

    def iter_records(self, filters: TraceFilters) -> Iterator[TraceRecord]:
        """在同一只读快照分批读取，不用列表API循环累积全部记录。

        Args:
            filters: 已验证的窗口与筛选条件。

        Yields:
            只读关联投影，每次最多读取100条。
        """
        query, parameters = self._selection(filters, None)
        try:
            with self._connect() as connection:
                cursor = connection.execute(query, parameters)
                while True:
                    rows = cursor.fetchmany(100)
                    if not rows:
                        break
                    for row in rows:
                        yield self._record(row)
        except (sqlite3.Error, OSError, ValueError) as exc:
            raise TraceStorageError("追踪统计读取失败") from exc

    def _selection(
        self, filters: TraceFilters, position: TracePosition | None,
    ) -> tuple[str, list[object]]:
        conditions = ["tr.expires_at > ?"]
        parameters: list[object] = [self._clock().astimezone(UTC).isoformat()]
        for field in ("lane", "outcome", "scope_expanded", "escalated", "user_id"):
            value = getattr(filters, field)
            if value is not None:
                # 字段来自固定白名单；所有请求值均使用参数，不能插入SQL。
                conditions.append(f"json_extract(tr.snapshot_json, '$.{field}') = ?")
                parameters.append(str(value) if field == "user_id" else value)
        if filters.kb_id is not None:
            conditions.append("""EXISTS (
                SELECT 1 FROM json_each(tr.snapshot_json, '$.steps') step,
                    json_each(step.value, '$.input.kb_ids') kb
                WHERE json_extract(step.value, '$.type') = 'retrieval'
                    AND kb.value = ?
            )""")
            parameters.append(filters.kb_id)
        if filters.feedback is not None:
            conditions.append("json_extract(t.feedback_json, '$.rating') = ?")
            parameters.append(filters.feedback)
        for value, operator in ((filters.from_at, ">="), (filters.to_at, "<=")):
            if value is not None:
                conditions.append(f"tr.created_at {operator} ?")
                parameters.append(value.astimezone(UTC).isoformat())
        if position is not None:
            conditions.append("(tr.created_at, tr.trace_id) < (?, ?)")
            parameters.extend((
                position.created_at.astimezone(UTC).isoformat(),
                str(position.trace_id),
            ))
        query = _SELECT + " WHERE " + " AND ".join(conditions)
        query += " ORDER BY tr.created_at DESC, tr.trace_id DESC"
        return query, parameters

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self._trace_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("ATTACH DATABASE ? AS business", (
                self._conversation_path.resolve().as_uri() + "?mode=ro",
            ))
            connection.execute("BEGIN")
            yield connection
        finally:
            connection.close()
