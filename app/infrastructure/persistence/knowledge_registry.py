# -*- coding: utf-8 -*-
"""与目录同路径保存无正文删除记录及显式预设，不扫描会话或反馈。"""

import json
import psycopg
from app.infrastructure.persistence.database import Database, Record
from collections.abc import Iterator
from contextlib import closing
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime
from pathlib import Path
from uuid import UUID
from uuid import uuid5

from app.application.errors import KnowledgeBaseNotFoundError
from app.application.errors import KnowledgeContentError
from app.domain.suggestions import CuratedQuestionsUpdate
from app.domain.suggestions import SuggestedQuestion


QUESTION_NAMESPACE = UUID("36832ea3-3ee8-4bc1-b565-aed27eeaa83a")


class PostgresKnowledgeReadRegistry:
    """不持有长连接，预设与来源生命周期不混入模型执行追踪。"""

    def __init__(self, database: Database) -> None:
        self._database = database

    def record_deletion(self, doc_id: UUID, kb_id: str) -> None:
        """只记录已验证的文档身份，不保存文件名、正文或操作者历史。

        Args:
            doc_id: 删除前已确认存在的文档UUID。
            kb_id: 真实来源范围。

        Returns:
            None。
        """
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO document_deletions VALUES (%s, %s, %s) "
                "ON CONFLICT(doc_id) DO UPDATE SET kb_id=excluded.kb_id, recorded_at=excluded.recorded_at",
                (str(doc_id), kb_id, datetime.now(UTC).isoformat()),
            )

    def was_deleted(self, doc_id: UUID) -> bool:
        """只在当前索引已确认无文档时，由用例核对删除事实。

        Args:
            doc_id: 待核对的文档UUID。

        Returns:
            是否存在持久化删除记录，故障不返回假阴性。
        """
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM document_deletions WHERE doc_id = %s", (str(doc_id),),
            ).fetchone()
        return row is not None

    def list_curated(self, kb_id: str | None, limit: int) -> tuple[SuggestedQuestion, ...]:
        """只读取管理员预设表，不从真实历史自动发布问题。

        Args:
            kb_id: 可选知识库范围。
            limit: 1–50条读取上限。

        Returns:
            当前已注册库的稳定有序预设。
        """
        if not 1 <= limit <= 50:
            raise ValueError("建议问题读取上限无效")
        where = " WHERE k.lifecycle = 'active'"
        parameters = []
        if kb_id is not None:
            where += " AND q.kb_id = %s"
            parameters.append(kb_id)
        parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT q.id, q.kb_id, q.text FROM curated_questions q "
                "JOIN knowledge_bases k ON k.id = q.kb_id" + where
                + " ORDER BY q.kb_id, q.sort_order, q.id LIMIT %s", parameters,
            ).fetchall()
        try:
            return tuple(SuggestedQuestion(id=row[0], kb_id=row[1], text=row[2]) for row in rows)
        except ValueError as error:
            raise KnowledgeContentError("预设问题记录无效") from error

    def replace_curated(
        self, kb_id: str, update: CuratedQuestionsUpdate,
    ) -> tuple[SuggestedQuestion, ...]:
        """同一写事务核对知识库并替换，稳定ID使完全相同PUT重试幂等。

        Args:
            kb_id: 既有知识库。
            update: 管理员明确提供的问题集合。

        Returns:
            真实已提交的问题集合。
        """
        validated = CuratedQuestionsUpdate.model_validate(update.model_dump())
        questions = []
        for text in validated.questions:
            identity = json.dumps([kb_id, text], ensure_ascii=False)
            questions.append(SuggestedQuestion(
                id=f"sq_{uuid5(QUESTION_NAMESPACE, identity)}", kb_id=kb_id, text=text,
            ))
        with self._connection() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(current_schema() || 'business_write', 0))")
            if connection.execute("SELECT 1 FROM knowledge_bases WHERE id = %s AND lifecycle = 'active'", (kb_id,)).fetchone() is None:
                raise KnowledgeBaseNotFoundError("知识库不存在")
            connection.execute("DELETE FROM curated_questions WHERE kb_id = %s", (kb_id,))
            for index, question in enumerate(questions):
                connection.execute(
                    "INSERT INTO curated_questions VALUES (%s, %s, %s, %s)",
                    (question.id, kb_id, question.text, index),
                )
        return tuple(questions)

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        try:
            with self._database.connection() as connection:
                yield connection
        except psycopg.Error as exc:
            raise KnowledgeContentError("PostgreSQL 仓储操作失败") from exc
