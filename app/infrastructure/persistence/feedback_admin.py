# -*- coding: utf-8 -*-
"""后台单轮反馈投影与分类，独立于普通会话CRUD适配器。"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime
from pathlib import Path
from uuid import UUID

from app.application.errors import FeedbackChangedError
from app.application.errors import FeedbackNotFoundError
from app.application.errors import ConversationStorageError
from app.domain.conversations import TurnCitation
from app.domain.feedback import AdminFeedbackItem
from app.domain.feedback import FeedbackTriage
from app.domain.feedback import FeedbackTriageFilter
from app.domain.feedback import TurnFeedback


_SELECT = """SELECT t.*, COALESCE(t.feedback_updated_at, t.created_at) AS sort_time
    FROM conversation_turns t JOIN conversations c ON c.id = t.conversation_id
    WHERE c.status = 'active' AND t.feedback_json IS NOT NULL
        AND json_extract(t.feedback_json, '$.rating') = 'down'"""


class SQLiteFeedbackAdminRepository:
    """只读有限投影，分类只更新当前反馈的处理字段。"""

    def __init__(self, path: Path) -> None:
        """保存由组合根初始化并隔离的业务路径。

        Args:
            path: 当前运行业务库。

        Returns:
            None。
        """
        self._path = path

    def list(
        self, limit: int, kb_id: str | None,
        triage: FeedbackTriageFilter | None,
        before: tuple[datetime, UUID] | None,
    ) -> tuple[AdminFeedbackItem, ...]:
        """在SQL过滤后分页，并一次批量读取引用而非加载全会话。

        Args:
            limit: 有界读取条数。
            kb_id: 搜索范围或引用来源知识库。
            triage: 分类过滤。
            before: 可选排序边界。

        Returns:
            当前点踩投影。
        """
        if not 1 <= limit <= 101:
            raise ValueError("反馈分页上限无效")
        query = _SELECT
        values: list[object] = []
        if kb_id is not None:
            query += """ AND (EXISTS (
                SELECT 1 FROM json_each(t.metadata_json, '$.kb_scope.searched') kb
                WHERE kb.value = ?
            ) OR EXISTS (SELECT 1 FROM turn_citations citation
                WHERE citation.turn_id = t.id AND citation.kb_id = ?))"""
            values.extend((kb_id, kb_id))
        if triage == "untriaged":
            query += " AND t.feedback_triage IS NULL"
        elif triage is not None:
            query += " AND t.feedback_triage = ?"
            values.append(triage)
        if before is not None:
            query += " AND (COALESCE(t.feedback_updated_at, t.created_at), t.id) < (?, ?)"
            values.extend((before[0].isoformat(), str(before[1])))
        query += " ORDER BY sort_time DESC, t.id DESC LIMIT ?"
        values.append(limit)
        try:
            with self._connect(read_only=True) as connection:
                rows = connection.execute(query, values).fetchall()
                return self._items(connection, rows)
        except (sqlite3.Error, OSError, ValueError) as error:
            raise ConversationStorageError("后台反馈读取失败") from error

    def triage(
        self, feedback_id: UUID, actor_id: UUID, value: FeedbackTriage | None,
        expected_revision: int | None,
    ) -> AdminFeedbackItem:
        """核对版本与当前点踩状态，用户改动后旧分类不能沿用。

        Args:
            feedback_id: 每轮唯一的反馈资源ID。
            actor_id: 已验证管理员。
            value: 新分类或清空。
            expected_revision: 可选预期用户反馈版本。

        Returns:
            已保存的处理状态。
        """
        try:
            with self._connect(read_only=False) as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(_SELECT + " AND t.id = ?", (str(feedback_id),)).fetchone()
                if row is None:
                    raise FeedbackNotFoundError("反馈不存在")
                if expected_revision is not None and row["feedback_revision"] != expected_revision:
                    raise FeedbackChangedError("反馈已变化，请刷新后重试")
                connection.execute(
                    """UPDATE conversation_turns SET feedback_triage = ?,
                        feedback_triaged_by = ?, feedback_triaged_at = ? WHERE id = ?""",
                    (value, str(actor_id), datetime.now(UTC).isoformat(), str(feedback_id)),
                )
                row = connection.execute(_SELECT + " AND t.id = ?", (str(feedback_id),)).fetchone()
                return self._items(connection, [row])[0]
        except FeedbackChangedError:
            raise
        except (sqlite3.Error, OSError, ValueError) as error:
            raise ConversationStorageError("后台反馈分类失败") from error

    @staticmethod
    def _items(
        connection: sqlite3.Connection, rows: list[sqlite3.Row],
    ) -> tuple[AdminFeedbackItem, ...]:
        if not rows:
            return ()
        identifiers = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in identifiers)
        citation_rows = connection.execute(
            "SELECT * FROM turn_citations WHERE turn_id IN (" + placeholders + ") "
            "ORDER BY turn_id, COALESCE(citation_order, position)", identifiers,
        ).fetchall()
        citations: dict[str, list[TurnCitation]] = {}
        for citation in citation_rows:
            item = TurnCitation.model_validate({
                "position": citation["position"], "doc_id": citation["doc_id"],
                "filename": citation["filename"], "page": citation["page"],
                "chunk_text": citation["chunk_text"], "chunk_id": citation["chunk_id"],
                "kb_id": citation["kb_id"] or "default",
                "doc_updated_at": citation["doc_updated_at"],
            })
            citations.setdefault(citation["turn_id"], []).append(item)
        items = []
        for row in rows:
            feedback = TurnFeedback.model_validate_json(row["feedback_json"])
            items.append(AdminFeedbackItem(
                feedback_id=UUID(row["id"]), turn_id=UUID(row["id"]),
                question=row["user_message"], answer=row["assistant_message"],
                citations=tuple(citations.get(row["id"], [])),
                reasons=feedback.reasons, comment=feedback.comment,
                trace_id=row["trace_id"], revision=row["feedback_revision"],
                triage=row["feedback_triage"], created_at=row["created_at"],
                updated_at=row["feedback_updated_at"], sort_time=row["sort_time"],
            ))
        return tuple(items)

    @contextmanager
    def _connect(self, read_only: bool) -> Iterator[sqlite3.Connection]:
        suffix = "?mode=ro" if read_only else "?mode=rw"
        connection = sqlite3.connect(self._path.resolve().as_uri() + suffix, uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                if read_only:
                    connection.execute("BEGIN")
                yield connection
        finally:
            connection.close()
