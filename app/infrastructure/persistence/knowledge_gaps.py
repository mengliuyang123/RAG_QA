# -*- coding: utf-8 -*-
"""知识缺口处理状态的 PostgreSQL 存储。"""

import json
from datetime import datetime
from uuid import UUID

from app.domain.knowledge_gaps import GapCheck
from app.domain.knowledge_gaps import KnowledgeGap
from app.infrastructure.persistence.database import Database
from app.infrastructure.persistence.database import Record


def _gap(row: Record) -> KnowledgeGap:
    check = row["last_check"]
    return KnowledgeGap(
        gap_key=row["gap_key"], source=row["source"], title=row["title"],
        questions=tuple(json.loads(row["questions"])), kb_id=row["kb_id"],
        status=row["status"], doc_id=row["doc_id"],
        last_check=GapCheck.model_validate_json(check) if check else None,
        updated_at=datetime.fromisoformat(row["updated_at"]),
        updated_by=UUID(row["updated_by"]),
    )


class PostgresKnowledgeGapRepository:
    """一张小表，按 gap_key 整体替换。"""

    def __init__(self, database: Database) -> None:
        self._database = database

    def list(self, limit: int) -> list[KnowledgeGap]:
        with self._database.connection(read_only=True) as connection:
            rows = connection.execute(
                "SELECT * FROM knowledge_gaps ORDER BY updated_at DESC, gap_key LIMIT %s",
                (limit,),
            ).fetchall()
        return [_gap(row) for row in rows]

    def save(self, gap: KnowledgeGap) -> KnowledgeGap:
        with self._database.connection() as connection:
            row = connection.execute(
                """INSERT INTO knowledge_gaps(
                    gap_key, source, title, questions, kb_id, status, doc_id,
                    last_check, updated_at, updated_by
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (gap_key) DO UPDATE SET
                    source = excluded.source, title = excluded.title,
                    questions = excluded.questions, kb_id = excluded.kb_id,
                    status = excluded.status, doc_id = excluded.doc_id,
                    last_check = excluded.last_check, updated_at = excluded.updated_at,
                    updated_by = excluded.updated_by
                RETURNING *""",
                (
                    gap.gap_key, gap.source, gap.title,
                    json.dumps(list(gap.questions), ensure_ascii=False), gap.kb_id,
                    gap.status, gap.doc_id,
                    gap.last_check.model_dump_json() if gap.last_check else None,
                    gap.updated_at, str(gap.updated_by),
                ),
            ).fetchone()
        return _gap(row)
