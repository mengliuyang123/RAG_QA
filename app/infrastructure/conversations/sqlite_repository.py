# -*- coding: utf-8 -*-
"""使用独立 SQLite 数据库持久化正式多轮会话历史。"""

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from app.application.errors import ConversationDeletingError
from app.application.errors import ConversationNotFoundError
from app.application.execution_control import check_cancellation
from app.application.execution_control import StreamCancelled
from app.application.errors import ConversationStorageError
from app.domain.conversations import AgentResult
from app.domain.conversations import AnswerMode
from app.domain.conversations import Conversation
from app.domain.conversations import ConversationStatus
from app.domain.conversations import ConversationSummary
from app.domain.conversations import ConversationTurn
from app.domain.conversations import TurnCitation
from app.domain.answer_metadata import AnswerMetadata
from app.domain.answer_metadata import EvidenceStats
from app.domain.answer_metadata import RefusalInfo
from app.domain.answer_metadata import ScopeInfo
from app.domain.conversations import KNOWLEDGE_REFUSAL_ANSWER
from app.domain.routing import KnowledgeScope
from app.domain.feedback import TurnFeedback
from app.application.errors import TurnNotFoundError
from app.domain.conversation_pages import ConversationPage
from app.domain.conversation_pages import ConversationDetailPage
from app.domain.errors import InvalidConversationCursorError
from app.infrastructure.conversations.pagination import decode_cursor
from app.infrastructure.conversations.pagination import encode_cursor


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active', 'deleting')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    turns_count INTEGER NOT NULL DEFAULT 0 CHECK(turns_count >= 0)
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    answer_mode TEXT NOT NULL CHECK(answer_mode IN ('direct', 'retrieval')),
    retrieval_calls INTEGER NOT NULL CHECK(retrieval_calls BETWEEN 0 AND 3),
    latency_ms REAL NOT NULL CHECK(latency_ms >= 0),
    created_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_conversation
    ON conversation_turns(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS turn_citations (
    turn_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK(position >= 1),
    doc_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    page INTEGER,
    chunk_text TEXT NOT NULL,
    PRIMARY KEY(turn_id, position),
    FOREIGN KEY(turn_id) REFERENCES conversation_turns(id)
        ON DELETE CASCADE
);
"""


class SQLiteConversationRepository:
    """把正式会话历史存入参数化、事务化的 SQLite 表。"""

    def put_feedback(
        self, turn_id: UUID, owner_id: UUID, feedback: TurnFeedback,
    ) -> None:
        """原子检查归属并覆盖，单轮只有一条当前反馈。

        Args:
            turn_id: 正式轮次。
            owner_id: 后端身份，无管理员例外。
            feedback: 当前反馈，取消时不保留备注。

        Returns:
            None。
        """
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = self._owned_feedback_row(connection, turn_id, owner_id)
                normalized = feedback if feedback.rating is not None else TurnFeedback(rating=None)
                if self._feedback_from_json(current["feedback_json"]) == normalized:
                    return
                connection.execute(
                    """UPDATE conversation_turns SET feedback_json = ?,
                        feedback_revision = feedback_revision + 1,
                        feedback_updated_at = ?, feedback_triage = NULL,
                        feedback_triaged_by = NULL, feedback_triaged_at = NULL
                    WHERE id = ?""",
                    (
                        feedback.model_dump_json() if feedback.rating is not None else None,
                        datetime.now(UTC).isoformat(),
                        str(turn_id),
                    ),
                )
        except sqlite3.Error as exc:
            raise ConversationStorageError("轮次反馈保存失败") from exc

    def get_feedback(self, turn_id: UUID, owner_id: UUID) -> TurnFeedback:
        """读取本人当前反馈，不加载会话历史和引用。

        Args:
            turn_id: 正式轮次。
            owner_id: 后端身份。

        Returns:
            当前反馈，无反馈时rating为None。
        """
        try:
            with self._connect() as connection:
                row = self._owned_feedback_row(connection, turn_id, owner_id)
                return self._feedback_from_json(row["feedback_json"])
        except (sqlite3.Error, ValueError) as exc:
            raise ConversationStorageError("轮次反馈读取失败") from exc

    @staticmethod
    def _owned_feedback_row(
        connection: sqlite3.Connection, turn_id: UUID, owner_id: UUID,
    ) -> sqlite3.Row:
        row = connection.execute(
            """SELECT t.feedback_json, t.feedback_revision FROM conversation_turns t
            JOIN conversations c ON c.id = t.conversation_id
            WHERE t.id = ? AND c.owner_id = ? AND c.status = 'active'""",
            (str(turn_id), str(owner_id)),
        ).fetchone()
        if row is None:
            raise TurnNotFoundError("轮次不存在")
        return row

    @staticmethod
    def _feedback_from_json(encoded: str | None) -> TurnFeedback:
        if encoded is None:
            return TurnFeedback(rating=None)
        return TurnFeedback.model_validate_json(encoded)

    def __init__(
        self,
        path: Path,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        if busy_timeout_ms < 1:
            raise ValueError("busy_timeout_ms 必须大于零")
        self._path = path
        self._busy_timeout_ms = busy_timeout_ms
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create(
        self, kb_scope: KnowledgeScope | None = None, *,
        owner_id: UUID | None = None,
    ) -> Conversation:
        """保存空会话及默认检索范围。

        Args:
            kb_scope: 未指定时使用自动选库。
            owner_id: 后端确定的归属，旧记录保持无归属不自动认领。

        Returns:
            持久化后的空会话。
        """
        scope = kb_scope or KnowledgeScope()
        conversation_id = uuid4()
        now = datetime.now(UTC)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO conversations(
                        id, title, status, created_at, updated_at,
                        turns_count, kb_scope_json, owner_id
                    ) VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        str(conversation_id),
                        "新对话",
                        ConversationStatus.ACTIVE.value,
                        now.isoformat(),
                        now.isoformat(),
                        scope.model_dump_json(),
                        str(owner_id) if owner_id is not None else None,
                    ),
                )
        except sqlite3.Error as exc:
            raise ConversationStorageError("会话存储操作失败") from exc
        return Conversation(
            id=conversation_id,
            title="新对话",
            status=ConversationStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            turns_count=0,
            turns=[],
            kb_scope=scope,
            owner_id=owner_id,
        )

    def update(
        self,
        conversation_id: UUID,
        *,
        title: str | None = None,
        kb_scope: KnowledgeScope | None = None,
    ) -> ConversationSummary:
        """事务性修改摘要，不改变正式轮次或引用。

        Args:
            conversation_id: 待修改会话。
            title: 非空标题；None 表示保持不变。
            kb_scope: 默认范围；None 表示保持不变。

        Returns:
            修改后的会话摘要。
        """
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = self._conversation_row(connection, conversation_id)
                if row["status"] == ConversationStatus.DELETING.value:
                    raise ConversationDeletingError("会话正在删除")
                connection.execute(
                    """
                    UPDATE conversations
                    SET title = COALESCE(?, title),
                        kb_scope_json = COALESCE(?, kb_scope_json),
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        title,
                        kb_scope.model_dump_json() if kb_scope else None,
                        datetime.now(UTC).isoformat(),
                        str(conversation_id),
                    ),
                )
                return self._summary_from_row(
                    self._conversation_row(connection, conversation_id),
                )
        except (sqlite3.Error, ValueError) as exc:
            raise ConversationStorageError("会话摘要修改失败") from exc

    def list_active(
        self, *, owner_id: UUID | None = None,
    ) -> list[ConversationSummary]:
        """在SQL层筛选当前用户，避免读取其他人的历史。

        Args:
            owner_id: 已验证用户；None仅供可信内部调用。

        Returns:
            属于指定用户的active摘要。
        """
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT id, title, status, created_at, updated_at,
                           turns_count, kb_scope_json, owner_id
                    FROM conversations
                    WHERE status = ? AND (? IS NULL OR owner_id = ?)
                    ORDER BY updated_at DESC, rowid DESC
                    """,
                    (
                        ConversationStatus.ACTIVE.value,
                        str(owner_id) if owner_id is not None else None,
                        str(owner_id) if owner_id is not None else None,
                    ),
                ).fetchall()
        except sqlite3.Error as exc:
            raise ConversationStorageError("会话存储操作失败") from exc
        return [self._summary_from_row(row) for row in rows]

    def get(self, conversation_id: UUID) -> Conversation:
        try:
            with self._connect() as connection:
                row = self._conversation_row(connection, conversation_id)
                turn_rows = connection.execute(
                    """
                    SELECT rowid AS turn_rowid, id, conversation_id,
                           user_message, assistant_message, answer_mode,
                           retrieval_calls, latency_ms, created_at, metadata_json,
                           feedback_json
                    FROM conversation_turns
                    WHERE conversation_id = ?
                    ORDER BY created_at ASC, turn_rowid ASC
                    """,
                    (str(conversation_id),),
                ).fetchall()
                turns = [
                    self._turn_from_row(connection, turn_row)
                    for turn_row in turn_rows
                ]
        except ConversationNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise ConversationStorageError("会话存储操作失败") from exc
        summary = self._summary_from_row(row)
        return Conversation(**summary.model_dump(), turns=turns)

    def list_active_page(
        self, owner_id: UUID, limit: int, cursor: str | None,
    ) -> ConversationPage:
        """归属过滤和分页都在SQL内，总数与当前页共用快照。

        Args:
            owner_id: 后端认证用户。
            limit: 当前页大小。
            cursor: 同用户列表的下一页位置。

        Returns:
            本人active摘要页与精确总数。
        """
        scope = f"conversations:{owner_id}"
        position = decode_cursor(cursor, scope)
        clause = ""
        parameters: list[object] = [str(owner_id)]
        if position is not None:
            clause = " AND (updated_at, rowid) < (?, ?)"
            parameters.extend([position.timestamp, position.rowid])
        parameters.append(limit + 1)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN")
                total = connection.execute(
                    "SELECT COUNT(*) FROM conversations "
                    "WHERE owner_id = ? AND status = 'active'",
                    (str(owner_id),),
                ).fetchone()[0]
                rows = connection.execute(
                    "SELECT rowid AS page_rowid, * FROM conversations "
                    "WHERE owner_id = ? AND status = 'active'" + clause
                    + " ORDER BY updated_at DESC, rowid DESC LIMIT ?",
                    parameters,
                ).fetchall()
                next_cursor = None
                if len(rows) > limit:
                    last = rows[limit - 1]
                    next_cursor = encode_cursor(
                        last["updated_at"], last["page_rowid"], scope,
                    )
                items = tuple(self._summary_from_row(row) for row in rows[:limit])
                return ConversationPage(
                    conversations=items, total=total, next_cursor=next_cursor,
                )
        except (sqlite3.Error, ValueError) as error:
            raise ConversationStorageError("会话分页读取失败") from error

    def get_page(
        self, conversation_id: UUID, owner_id: UUID,
        limit: int, cursor: str | None,
    ) -> ConversationDetailPage:
        """同快照先检查归属，再读取选中轮次与批量引用。

        Args:
            conversation_id: 目标会话。
            owner_id: 已认证用户。
            limit: 当前页大小。
            cursor: 同用户同会话的更早页位置。

        Returns:
            页内正序的有限历史及下一页游标。
        """
        scope = f"turns:{owner_id}:{conversation_id}"
        try:
            with self._connect() as connection:
                connection.execute("BEGIN")
                row = self._conversation_row(connection, conversation_id)
                if row["owner_id"] != str(owner_id):
                    raise ConversationNotFoundError("会话不存在")
                position = decode_cursor(cursor, scope)
                clause = ""
                parameters: list[object] = [str(conversation_id)]
                if position is not None:
                    clause = " AND (created_at, rowid) < (?, ?)"
                    parameters.extend([position.timestamp, position.rowid])
                parameters.append(limit + 1)
                # 多取一条仅限ID和排序键，不读取不在本页的正文。
                keys = connection.execute(
                    "SELECT rowid AS page_rowid, id, created_at "
                    "FROM conversation_turns WHERE conversation_id = ?"
                    + clause + " ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    parameters,
                ).fetchall()
                next_cursor = None
                if len(keys) > limit:
                    last = keys[limit - 1]
                    next_cursor = encode_cursor(
                        last["created_at"], last["page_rowid"], scope,
                    )
                turns = self._page_turns(connection, keys[:limit])
                summary = self._summary_from_row(row)
                return ConversationDetailPage(
                    **summary.model_dump(), turns=turns,
                    turns_next_cursor=next_cursor,
                )
        except InvalidConversationCursorError:
            raise
        except (sqlite3.Error, ValueError) as error:
            raise ConversationStorageError("会话历史分页读取失败") from error

    def _page_turns(
        self, connection: sqlite3.Connection, keys: list[sqlite3.Row],
    ) -> list[ConversationTurn]:
        if not keys:
            return []
        ids = [row["id"] for row in keys]
        placeholders = ",".join("?" for _ in ids)
        rows = connection.execute(
            "SELECT * FROM conversation_turns "
            f"WHERE id IN ({placeholders}) ORDER BY created_at ASC, rowid ASC",
            ids,
        ).fetchall()
        citation_rows = connection.execute(
            f"SELECT * FROM turn_citations WHERE turn_id IN ({placeholders}) "
            "ORDER BY COALESCE(citation_order, position) ASC",
            ids,
        ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for citation in citation_rows:
            grouped.setdefault(citation["turn_id"], []).append(citation)
        turns = []
        for row in rows:
            turns.append(self._turn_from_row(
                connection, row, citation_rows=grouped.get(row["id"], []),
            ))
        return turns

    def get_summary(self, conversation_id: UUID) -> ConversationSummary:
        """用单次查询读取会话状态与摘要。

        Args:
            conversation_id: 要检查的会话标识。

        Returns:
            不包含历史及引用的会话摘要。
        """
        try:
            with self._connect() as connection:
                row = self._conversation_row(connection, conversation_id)
                return self._summary_from_row(row)
        except ConversationNotFoundError:
            raise
        except (sqlite3.Error, ValueError) as exc:
            raise ConversationStorageError("会话摘要读取失败") from exc

    def recent_turns(
        self,
        conversation_id: UUID,
        limit: int,
    ) -> list[ConversationTurn]:
        if limit < 1:
            raise ValueError("limit 必须大于零")
        try:
            with self._connect() as connection:
                self._conversation_row(connection, conversation_id)
                rows = connection.execute(
                    """
                    SELECT * FROM (
                        SELECT rowid AS turn_rowid, id, conversation_id,
                               user_message, assistant_message, answer_mode,
                               retrieval_calls, latency_ms, created_at, metadata_json,
                               feedback_json
                        FROM conversation_turns
                        WHERE conversation_id = ?
                        ORDER BY created_at DESC, turn_rowid DESC
                        LIMIT ?
                    )
                    ORDER BY created_at ASC, turn_rowid ASC
                    """,
                    (str(conversation_id), limit),
                ).fetchall()
                return [
                    self._turn_from_row(connection, row) for row in rows
                ]
        except ConversationNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise ConversationStorageError("会话存储操作失败") from exc

    def iter_turns_reverse(
        self, conversation_id: UUID, batch_size: int = 32,
    ) -> Iterator[ConversationTurn]:
        """以时间和 rowid 游标分页，连接在向外 yield 前已关闭。

        Args:
            conversation_id: 正式会话标识。
            batch_size: 技术分页大小，不决定保留多少轮历史。

        Yields:
            按最新到最旧排列的完整历史轮次。
        """
        if batch_size < 1:
            raise ValueError("batch_size 必须大于零")
        cursor: tuple[str, int] | None = None
        while True:
            condition = ""
            parameters: list[object] = [str(conversation_id)]
            if cursor is not None:
                condition = " AND (created_at, rowid) < (?, ?)"
                parameters.extend(cursor)
            parameters.append(batch_size)
            try:
                with self._connect() as connection:
                    self._conversation_row(connection, conversation_id)
                    # condition 仅来自固定字符串；所有用户值均参数化。
                    rows = connection.execute(
                        "SELECT rowid AS turn_rowid, * "
                        "FROM conversation_turns WHERE conversation_id = ?"
                        + condition
                        + " ORDER BY created_at DESC, rowid DESC LIMIT ?",
                        parameters,
                    ).fetchall()
                    page = [self._turn_from_row(connection, row) for row in rows]
            except ConversationNotFoundError:
                raise
            except (sqlite3.Error, ValueError) as exc:
                raise ConversationStorageError("会话历史读取失败") from exc
            if not rows:
                return
            yield from page
            if len(rows) < batch_size:
                return
            last = rows[-1]
            cursor = (last["created_at"], last["turn_rowid"])

    def save_turn(
        self,
        conversation_id: UUID,
        question: str,
        result: AgentResult,
        latency_ms: float,
    ) -> ConversationTurn:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("问题不能为空")
        if latency_ms < 0:
            raise ValueError("latency_ms 不得小于零")
        turn_id = uuid4()
        created_at = datetime.now(UTC)
        connection: sqlite3.Connection | None = None
        try:
            connection = self._new_connection()
            check_cancellation()
            connection.execute("BEGIN IMMEDIATE")
            check_cancellation()
            conversation_row = self._conversation_row(
                connection,
                conversation_id,
            )
            if conversation_row["status"] == ConversationStatus.DELETING.value:
                raise ConversationDeletingError("会话正在删除")
            connection.execute(
                """
                INSERT INTO conversation_turns(
                    id, conversation_id, user_message, assistant_message,
                    answer_mode, retrieval_calls, latency_ms, created_at,
                    metadata_json, trace_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(turn_id),
                    str(conversation_id),
                    normalized_question,
                    result.answer,
                    result.answer_mode.value,
                    result.retrieval_calls,
                    latency_ms,
                    created_at.isoformat(),
                    result.metadata.model_dump_json() if result.metadata else None,
                    result.metadata.trace_id if result.metadata else None,
                ),
            )
            self._insert_citations(connection, turn_id, result.citations)
            connection.execute(
                """
                UPDATE conversations
                SET title = CASE
                        WHEN turns_count = 0 AND title = '新对话' THEN ?
                        ELSE title END,
                    updated_at = ?,
                    turns_count = turns_count + 1
                WHERE id = ?
                """,
                (
                    normalized_question[:50],
                    created_at.isoformat(),
                    str(conversation_id),
                ),
            )
            check_cancellation()
            connection.commit()
        except (ConversationNotFoundError, ConversationDeletingError, StreamCancelled):
            if connection is not None:
                connection.rollback()
            raise
        except (sqlite3.Error, ValueError) as exc:
            if connection is not None:
                connection.rollback()
            raise ConversationStorageError("会话存储操作失败") from exc
        finally:
            if connection is not None:
                connection.close()
        return ConversationTurn(
            id=turn_id,
            conversation_id=conversation_id,
            user_message=normalized_question,
            assistant_message=result.answer,
            answer_mode=result.answer_mode,
            retrieval_calls=result.retrieval_calls,
            citations=result.citations,
            latency_ms=latency_ms,
            created_at=created_at,
            metadata=result.metadata or self._legacy_metadata(
                result.answer_mode, result.answer, result.citations,
            ),
        )

    def mark_deleting(self, conversation_id: UUID) -> None:
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE conversations SET status = ? WHERE id = ?",
                    (
                        ConversationStatus.DELETING.value,
                        str(conversation_id),
                    ),
                )
                if cursor.rowcount == 0:
                    raise ConversationNotFoundError("会话不存在")
        except ConversationNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise ConversationStorageError("会话存储操作失败") from exc

    def delete(self, conversation_id: UUID) -> None:
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "DELETE FROM conversations WHERE id = ?",
                    (str(conversation_id),),
                )
                if cursor.rowcount == 0:
                    raise ConversationNotFoundError("会话不存在")
        except ConversationNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise ConversationStorageError("会话存储操作失败") from exc

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(_SCHEMA)
                connection.execute("BEGIN IMMEDIATE")
                additions = {
                    "conversations": {
                        "kb_scope_json": "TEXT", "owner_id": "TEXT",
                    },
                    "conversation_turns": {
                        "metadata_json": "TEXT", "feedback_json": "TEXT",
                        "trace_id": "TEXT",
                        "feedback_revision": "INTEGER NOT NULL DEFAULT 0",
                        "feedback_updated_at": "TEXT",
                        "feedback_triage": "TEXT",
                        "feedback_triaged_by": "TEXT",
                        "feedback_triaged_at": "TEXT",
                    },
                    "turn_citations": {
                        "chunk_id": "TEXT", "kb_id": "TEXT DEFAULT 'default'",
                        "doc_updated_at": "TEXT", "citation_order": "INTEGER",
                    },
                }
                backfill_trace_ids = False
                for table, columns in additions.items():
                    existing = {
                        row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
                    }
                    for name, definition in columns.items():
                        if name not in existing:
                            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
                            if table == "conversation_turns" and name == "trace_id":
                                backfill_trace_ids = True
                if backfill_trace_ids:
                    # 旧历史已有真实trace关联，只提取索引列，不重跑模型或改写正文。
                    connection.execute("""
                        UPDATE conversation_turns SET trace_id =
                            CASE WHEN json_valid(metadata_json) THEN
                                CASE WHEN json_type(metadata_json, '$.trace_id') = 'text'
                                THEN json_extract(metadata_json, '$.trace_id') END
                            END
                        WHERE metadata_json IS NOT NULL
                    """)
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_turns_trace "
                    "ON conversation_turns(trace_id)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_citations_document_chunk "
                    "ON turn_citations(doc_id, chunk_id, turn_id)"
                )
                connection.execute(
                    "UPDATE conversation_turns SET feedback_revision = 1 "
                    "WHERE feedback_json IS NOT NULL AND feedback_revision = 0"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_turns_feedback_time "
                    "ON conversation_turns(COALESCE(feedback_updated_at, created_at), id) "
                    "WHERE feedback_json IS NOT NULL"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conversations_owner "
                    "ON conversations(owner_id, status, updated_at)"
                )
        except sqlite3.Error as exc:
            raise ConversationStorageError("会话存储初始化失败") from exc

    def _new_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=self._busy_timeout_ms / 1_000,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        return connection

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = self._new_connection()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _conversation_row(
        connection: sqlite3.Connection,
        conversation_id: UUID,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT id, title, status, created_at, updated_at, turns_count,
                   kb_scope_json, owner_id
            FROM conversations
            WHERE id = ?
            """,
            (str(conversation_id),),
        ).fetchone()
        if row is None:
            raise ConversationNotFoundError("会话不存在")
        return row

    @staticmethod
    def _summary_from_row(row: sqlite3.Row) -> ConversationSummary:
        return ConversationSummary(
            id=UUID(row["id"]),
            title=row["title"],
            status=ConversationStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            turns_count=row["turns_count"],
            owner_id=UUID(row["owner_id"]) if row["owner_id"] else None,
            kb_scope=(
                KnowledgeScope.model_validate_json(row["kb_scope_json"])
                if row["kb_scope_json"] else KnowledgeScope()
            ),
        )

    def _turn_from_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *, citation_rows: Sequence[sqlite3.Row] | None = None,
    ) -> ConversationTurn:
        if citation_rows is None:
            citation_rows = connection.execute(
                """
                SELECT position, doc_id, filename, page, chunk_text,
                       chunk_id, kb_id, doc_updated_at
                FROM turn_citations
                WHERE turn_id = ?
                ORDER BY COALESCE(citation_order, position) ASC
                """,
                (row["id"],),
            ).fetchall()
        citations = [
            TurnCitation(
                position=citation_row["position"],
                doc_id=citation_row["doc_id"],
                filename=citation_row["filename"],
                page=citation_row["page"],
                chunk_text=citation_row["chunk_text"],
                chunk_id=citation_row["chunk_id"], kb_id=citation_row["kb_id"] or "default",
                doc_updated_at=datetime.fromisoformat(citation_row["doc_updated_at"])
                if citation_row["doc_updated_at"] else None,
            )
            for citation_row in citation_rows
        ]
        try:
            metadata = AnswerMetadata.model_validate_json(row["metadata_json"]) if row["metadata_json"] else self._legacy_metadata(
                AnswerMode(row["answer_mode"]), row["assistant_message"], citations,
            )
            metadata = metadata.model_copy(update={
                "feedback": self._feedback_from_json(row["feedback_json"]).rating,
            })
        except (ValueError, TypeError) as exc:
            raise ConversationStorageError("历史轮次的路由元数据无效") from exc
        return ConversationTurn(
            id=UUID(row["id"]),
            conversation_id=UUID(row["conversation_id"]),
            user_message=row["user_message"],
            assistant_message=row["assistant_message"],
            answer_mode=AnswerMode(row["answer_mode"]),
            retrieval_calls=row["retrieval_calls"],
            citations=citations,
            latency_ms=row["latency_ms"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=metadata,
        )

    @staticmethod
    def _legacy_metadata(
        mode: AnswerMode, answer: str, citations: Sequence[TurnCitation],
    ) -> AnswerMetadata:
        # 不为旧回答补做收费审核，也不能仅因有编号就标记为已充分验证。
        refused = answer in {
            KNOWLEDGE_REFUSAL_ANSWER, "根据现有知识库，我无法回答这个问题",
        }
        searched = tuple(sorted({citation.kb_id for citation in citations}))
        if not searched and mode is AnswerMode.RETRIEVAL:
            searched = ("default",)
        return AnswerMetadata(
            lane="direct" if mode is AnswerMode.DIRECT else "agent",
            outcome="refused" if refused else "answered",
            grounding="partial" if citations else "ungrounded",
            kb_scope=ScopeInfo(searched=searched),
            evidence_stats=EvidenceStats(used=len(citations)),
            refusal=RefusalInfo(
                reason="insufficient_evidence", found_summary="历史记录未保留完整审核结论。",
                missing=("足以可靠回答问题的证据",),
            ) if refused else None,
        )

    @staticmethod
    def _insert_citations(
        connection: sqlite3.Connection,
        turn_id: UUID,
        citations: Sequence[TurnCitation],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO turn_citations(
                turn_id, position, doc_id, filename, page, chunk_text,
                chunk_id, kb_id, doc_updated_at, citation_order
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(turn_id),
                    citation.position,
                    citation.doc_id,
                    citation.filename,
                    citation.page,
                    citation.chunk_text,
                    citation.chunk_id, citation.kb_id,
                    citation.doc_updated_at.isoformat() if citation.doc_updated_at else None,
                    ordinal,
                )
                for ordinal, citation in enumerate(citations)
            ],
        )
