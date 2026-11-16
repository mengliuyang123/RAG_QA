# -*- coding: utf-8 -*-
"""知识库访问控制、文档复审与标准答案的 PostgreSQL 存储。"""

from __future__ import annotations

from datetime import date
from datetime import datetime
from uuid import UUID

from app.domain.knowledge_access import DocumentReview
from app.domain.knowledge_access import FaqEntry
from app.domain.knowledge_access import KbAccess
from app.domain.knowledge_access import KbMember
from app.domain.knowledge_access import KbMemberRole
from app.infrastructure.persistence.database import Database


class PostgresKnowledgeAccessRepository:
    """访问设置；保存时整体替换成员。"""

    def __init__(self, database: Database) -> None:
        self._database = database

    def user_role(self, user_id: UUID) -> str | None:
        with self._database.connection(read_only=True) as connection:
            row = connection.execute(
                "SELECT role, disabled FROM users WHERE id = %s", (str(user_id),),
            ).fetchone()
        if row is None:
            return None
        return "disabled" if row["disabled"] else row["role"]

    def restricted_kb_ids(self) -> frozenset[str]:
        with self._database.connection(read_only=True) as connection:
            rows = connection.execute(
                "SELECT kb_id FROM knowledge_base_access WHERE restricted",
            ).fetchall()
        return frozenset(row["kb_id"] for row in rows)

    def memberships(self, user_id: UUID) -> dict[str, KbMemberRole]:
        with self._database.connection(read_only=True) as connection:
            rows = connection.execute(
                "SELECT kb_id, role FROM knowledge_base_members WHERE user_id = %s",
                (str(user_id),),
            ).fetchall()
        return {row["kb_id"]: row["role"] for row in rows}

    def get(self, kb_id: str) -> KbAccess:
        with self._database.connection(read_only=True) as connection:
            access = connection.execute(
                "SELECT restricted FROM knowledge_base_access WHERE kb_id = %s", (kb_id,),
            ).fetchone()
            rows = connection.execute(
                """SELECT m.user_id, m.role, u.username, u.name
                FROM knowledge_base_members m JOIN users u ON u.id = m.user_id
                WHERE m.kb_id = %s ORDER BY u.username""",
                (kb_id,),
            ).fetchall()
        return KbAccess(
            kb_id=kb_id, restricted=bool(access and access["restricted"]),
            members=tuple(
                KbMember(user_id=UUID(row["user_id"]), role=row["role"],
                         username=row["username"], name=row["name"])
                for row in rows
            ),
        )

    def save(self, access: KbAccess) -> KbAccess:
        with self._database.connection() as connection:
            connection.execute(
                """INSERT INTO knowledge_base_access(kb_id, restricted) VALUES (%s, %s)
                ON CONFLICT (kb_id) DO UPDATE SET restricted = excluded.restricted""",
                (access.kb_id, access.restricted),
            )
            connection.execute("DELETE FROM knowledge_base_members WHERE kb_id = %s", (access.kb_id,))
            for member in access.members:
                connection.execute(
                    "INSERT INTO knowledge_base_members(kb_id, user_id, role) VALUES (%s, %s, %s)",
                    (access.kb_id, str(member.user_id), member.role),
                )
        return self.get(access.kb_id)


class PostgresDocumentReviewRepository:
    """文档复审信息。"""

    def __init__(self, database: Database) -> None:
        self._database = database

    @staticmethod
    def _review(row) -> DocumentReview:
        due = row["review_due_on"]
        return DocumentReview(
            doc_id=row["doc_id"], kb_id=row["kb_id"],
            review_due_on=due if isinstance(due, date) or due is None else date.fromisoformat(str(due)),
            outdated=bool(row["outdated"]), note=row["note"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
            updated_by=UUID(row["updated_by"]),
        )

    def list(self, kb_id: str) -> list[DocumentReview]:
        with self._database.connection(read_only=True) as connection:
            rows = connection.execute(
                "SELECT * FROM document_reviews WHERE kb_id = %s ORDER BY doc_id", (kb_id,),
            ).fetchall()
        return [self._review(row) for row in rows]

    def save(self, review: DocumentReview) -> DocumentReview:
        with self._database.connection() as connection:
            row = connection.execute(
                """INSERT INTO document_reviews(
                    doc_id, kb_id, review_due_on, outdated, note, updated_at, updated_by
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    kb_id = excluded.kb_id, review_due_on = excluded.review_due_on,
                    outdated = excluded.outdated, note = excluded.note,
                    updated_at = excluded.updated_at, updated_by = excluded.updated_by
                RETURNING *""",
                (review.doc_id, review.kb_id, review.review_due_on, review.outdated,
                 review.note, review.updated_at, str(review.updated_by)),
            ).fetchone()
        return self._review(row)

    def outdated_doc_ids(self) -> tuple[str, ...]:
        with self._database.connection(read_only=True) as connection:
            rows = connection.execute(
                "SELECT doc_id FROM document_reviews WHERE outdated ORDER BY doc_id",
            ).fetchall()
        return tuple(row["doc_id"] for row in rows)


class PostgresFaqRepository:
    """标准答案。"""

    def __init__(self, database: Database) -> None:
        self._database = database

    @staticmethod
    def _entry(row) -> FaqEntry:
        return FaqEntry(
            id=row["id"], kb_id=row["kb_id"], question=row["question"], answer=row["answer"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            updated_by=UUID(row["updated_by"]),
        )

    def list(self, kb_id: str) -> list[FaqEntry]:
        with self._database.connection(read_only=True) as connection:
            rows = connection.execute(
                "SELECT * FROM faq_entries WHERE kb_id = %s ORDER BY created_at, id", (kb_id,),
            ).fetchall()
        return [self._entry(row) for row in rows]

    def save(self, entry: FaqEntry) -> FaqEntry:
        with self._database.connection() as connection:
            row = connection.execute(
                """INSERT INTO faq_entries(
                    id, kb_id, question, answer, created_at, updated_at, updated_by
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    question = excluded.question, answer = excluded.answer,
                    updated_at = excluded.updated_at, updated_by = excluded.updated_by
                RETURNING *""",
                (entry.id, entry.kb_id, entry.question, entry.answer,
                 entry.created_at, entry.updated_at, str(entry.updated_by)),
            ).fetchone()
        return self._entry(row)

    def delete(self, kb_id: str, entry_id: str) -> bool:
        with self._database.connection() as connection:
            deleted = connection.execute(
                "DELETE FROM faq_entries WHERE kb_id = %s AND id = %s", (kb_id, entry_id),
            ).rowcount
        return deleted > 0
