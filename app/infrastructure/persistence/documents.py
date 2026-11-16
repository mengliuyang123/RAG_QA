# -*- coding: utf-8 -*-
"""documents 表：上传查重、同名替换和按 doc_id 直接删除分块。"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime
from uuid import UUID

import psycopg

from app.application.errors import DocumentAlreadyExistsError
from app.application.errors import DocumentNameConflictError
from app.application.errors import DocumentReplaceConflictError
from app.application.errors import KnowledgeContentError
from app.domain.documents import ConflictResolution
from app.domain.documents import StoredDocument
from app.infrastructure.persistence.database import Database
from app.infrastructure.persistence.database import Record


_COLUMNS = ("doc_id", "kb_id", "filename", "sha256", "status", "replaces_doc_id", "uploaded_at", "deleted_at")
_SELECT = ",".join(_COLUMNS)
_BUSINESS_LOCK = "SELECT pg_advisory_xact_lock(hashtextextended(current_schema() || 'business_write', 0))"


def stored_document(row: Record) -> StoredDocument:
    """把数据库行转换为领域登记。

    Args:
        row: documents 表一行。

    Returns:
        校验后的登记。
    """
    return StoredDocument(
        doc_id=UUID(row["doc_id"]), kb_id=row["kb_id"], filename=row["filename"],
        sha256=row["sha256"], status=row["status"],
        replaces_doc_id=UUID(row["replaces_doc_id"]) if row["replaces_doc_id"] else None,
        uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
        deleted_at=datetime.fromisoformat(row["deleted_at"]) if row["deleted_at"] else None,
    )


def register_upload(
    connection: psycopg.Connection, *, doc_id: UUID, kb_id: str, filename: str,
    sha256: str, uploaded_at: datetime, conflict: ConflictResolution | None,
    replace_doc_id: UUID | None = None,
) -> None:
    """在入队事务内查重并登记 pending 文档；调用方已持有业务写锁。

    Args:
        connection: 与入队共享的事务连接。
        doc_id: 新文档 UUID。
        kb_id: 目标知识库。
        filename: 已清理文件名。
        sha256: 内容摘要。
        uploaded_at: 登记时间。
        conflict: 同名时用户的选择；None 表示尚未选择。
        replace_doc_id: 替换时明确指定的旧文档。

    Returns:
        None。

    Raises:
        DocumentAlreadyExistsError: 同库已有相同内容。
        DocumentNameConflictError: 同名不同内容且用户尚未选择。
        DocumentReplaceConflictError: 替换目标仍在入库或已被其他任务替换。
    """
    same = connection.execute(
        f"SELECT {_SELECT} FROM documents WHERE kb_id=%s AND sha256=%s AND deleted_at IS NULL",
        (kb_id, sha256),
    ).fetchone()
    if same is not None:
        raise DocumentAlreadyExistsError("知识库中已存在内容相同的文档", stored_document(same))
    replaces = None
    if conflict != "keep_both":
        if conflict == "replace" and replace_doc_id is not None:
            target = connection.execute(
                f"SELECT {_SELECT} FROM documents WHERE doc_id=%s AND kb_id=%s AND deleted_at IS NULL",
                (str(replace_doc_id), kb_id),
            ).fetchone()
        else:
            target = connection.execute(
                f"SELECT {_SELECT} FROM documents WHERE kb_id=%s AND filename=%s AND deleted_at IS NULL "
                "ORDER BY uploaded_at DESC, doc_id LIMIT 1",
                (kb_id, filename),
            ).fetchone()
        if target is not None:
            existing = stored_document(target)
            if conflict is None:
                raise DocumentNameConflictError("知识库中已有同名文档", existing)
            if existing.status == "pending":
                raise DocumentReplaceConflictError("被替换的文档仍在入库，请稍后重试")
            busy = connection.execute(
                "SELECT 1 FROM documents WHERE replaces_doc_id=%s AND status='pending' "
                "AND deleted_at IS NULL LIMIT 1", (str(existing.doc_id),),
            ).fetchone()
            if busy is not None:
                raise DocumentReplaceConflictError("该文档已有替换任务在进行")
            replaces = str(existing.doc_id)
    connection.execute(
        "INSERT INTO documents (doc_id,kb_id,filename,sha256,status,replaces_doc_id,uploaded_at) "
        "VALUES (%s,%s,%s,%s,'pending',%s,%s)",
        (str(doc_id), kb_id, filename, sha256, replaces, uploaded_at),
    )


def publish_document(connection: psycopg.Connection, doc_id: UUID, now: datetime) -> None:
    """与任务成功同一事务发布新版本，并原子删除被替换旧版本的分块。

    任一 SQL 语句快照只会看到旧版本已发布或新版本已发布，没有两者都不可见的提交点。

    Args:
        connection: 任务 succeed 事务连接。
        doc_id: 刚完成入库的文档。
        now: 删除时间。

    Returns:
        None。
    """
    row = connection.execute(
        "UPDATE documents SET status='ready' WHERE doc_id=%s AND deleted_at IS NULL RETURNING replaces_doc_id",
        (str(doc_id),),
    ).fetchone()
    if row is None or row["replaces_doc_id"] is None:
        return
    old = connection.execute(
        "UPDATE documents SET deleted_at=%s WHERE doc_id=%s AND deleted_at IS NULL RETURNING kb_id",
        (now, row["replaces_doc_id"]),
    ).fetchone()
    if old is None:
        return
    connection.execute("DELETE FROM vector_chunks WHERE doc_id=%s", (row["replaces_doc_id"],))
    connection.execute(
        "INSERT INTO document_deletions VALUES (%s,%s,%s) ON CONFLICT (doc_id) DO NOTHING",
        (row["replaces_doc_id"], old["kb_id"], now),
    )


def fail_document(connection: psycopg.Connection, doc_id: UUID) -> None:
    """入库失败只标记新登记，旧版本保持不变。

    Args:
        connection: 任务 fail 事务连接。
        doc_id: 失败任务的文档。

    Returns:
        None。
    """
    connection.execute(
        "UPDATE documents SET status='failed' WHERE doc_id=%s AND deleted_at IS NULL", (str(doc_id),),
    )


class PostgresDocumentRegistry:
    """documents 表读取与按 doc_id 删除；不扫描整个向量库。"""

    def __init__(self, database: Database) -> None:
        self._database = database

    def get(self, doc_id: UUID) -> StoredDocument | None:
        """读取未删除登记。

        Args:
            doc_id: 文档 UUID。

        Returns:
            登记或 None。
        """
        with self._connection(read_only=True) as connection:
            row = connection.execute(
                f"SELECT {_SELECT} FROM documents WHERE doc_id=%s AND deleted_at IS NULL", (str(doc_id),),
            ).fetchone()
        return None if row is None else stored_document(row)

    def list_live(self, kb_id: str | None = None) -> tuple[StoredDocument, ...]:
        """列出未删除登记。

        Args:
            kb_id: 可选知识库过滤。

        Returns:
            按上传时间倒序的登记。
        """
        query = f"SELECT {_SELECT} FROM documents WHERE deleted_at IS NULL"
        values: list[object] = []
        if kb_id is not None:
            query += " AND kb_id=%s"
            values.append(kb_id)
        with self._connection(read_only=True) as connection:
            rows = connection.execute(query + " ORDER BY uploaded_at DESC, doc_id", values).fetchall()
        return tuple(stored_document(row) for row in rows)

    def replaced_by(self, doc_id: UUID) -> StoredDocument | None:
        """读取被新版本替换并已删除的旧登记。

        Args:
            doc_id: 新版本。

        Returns:
            旧登记或 None。
        """
        columns = ",".join("o." + column for column in _COLUMNS)
        with self._connection(read_only=True) as connection:
            row = connection.execute(
                f"SELECT {columns} FROM documents n JOIN documents o ON o.doc_id=n.replaces_doc_id "
                "WHERE n.doc_id=%s AND n.status='ready' AND o.deleted_at IS NOT NULL",
                (str(doc_id),),
            ).fetchone()
        return None if row is None else stored_document(row)

    def delete_documents(self, targets: dict[UUID, str]) -> None:
        """一个事务内按 doc_id 删除分块与登记，并记录删除事实。

        Args:
            targets: doc_id 到知识库的映射。

        Returns:
            None。
        """
        if not targets:
            return
        identifiers = [str(doc_id) for doc_id in targets]
        now = datetime.now(UTC)
        with self._connection() as connection:
            connection.execute(_BUSINESS_LOCK)
            connection.execute("DELETE FROM vector_chunks WHERE doc_id = ANY(%s)", (identifiers,))
            connection.execute(
                "UPDATE documents SET deleted_at=%s WHERE doc_id = ANY(%s) AND deleted_at IS NULL",
                (now, identifiers),
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO document_deletions VALUES (%s,%s,%s) ON CONFLICT (doc_id) DO NOTHING",
                    [(str(doc_id), kb_id, now) for doc_id, kb_id in targets.items()],
                )

    @contextmanager
    def _connection(self, *, read_only: bool = False) -> Iterator[psycopg.Connection]:
        try:
            with self._database.connection(read_only=read_only) as connection:
                yield connection
        except (psycopg.Error, OSError) as error:
            raise KnowledgeContentError("文档登记暂时不可用") from error
