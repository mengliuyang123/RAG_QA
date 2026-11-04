# -*- coding: utf-8 -*-
"""只用资源标识与正式轮次时间统计引用行，不读取聊天或引用正文。"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime
from pathlib import Path
from uuid import UUID

from app.application.errors import KnowledgeContentError
from app.domain.document_diagnostics import ChunkCitationCount
from app.domain.document_diagnostics import ChunkCitationTotals
from app.domain.document_diagnostics import DocumentCitationTotals


_FROM = """ FROM turn_citations citation
    JOIN conversation_turns turn ON turn.id = citation.turn_id
    JOIN conversations conversation ON conversation.id = turn.conversation_id
    WHERE conversation.status = 'active'
        AND turn.created_at >= ? AND turn.created_at <= ? """


class SQLiteDocumentCitationReader:
    """业务库只读计数，读取失败不能补造空统计。"""

    def __init__(self, path: Path) -> None:
        """保存组合根已初始化的当前业务路径。

        Args:
            path: 当前运行的正式业务库。

        Returns:
            None。
        """
        self._path = path

    def documents(
        self, doc_ids: tuple[UUID, ...], started_at: datetime, ended_at: datetime,
    ) -> dict[UUID, DocumentCitationTotals]:
        """每100个文档分组计数，整个请求使用同一事务快照。

        Args:
            doc_ids: 当前索引文档身份。
            started_at: 统计起点。
            ended_at: 统计截止。

        Returns:
            文档总引用数与未知块归属数。
        """
        if not doc_ids:
            return {}
        identifiers = tuple(dict.fromkeys(doc_ids))
        totals = {}
        with self._snapshot() as connection:
            for identifier in identifiers:
                totals[identifier] = DocumentCitationTotals(total=0, unattributed=0)
            for offset in range(0, len(identifiers), 100):
                batch = identifiers[offset:offset + 100]
                marks = ",".join("?" for _ in batch)
                query = """SELECT citation.doc_id, COUNT(*) AS total,
                    SUM(CASE WHEN citation.chunk_id IS NULL OR citation.chunk_id = ''
                        THEN 1 ELSE 0 END) AS unattributed """
                query += _FROM + " AND citation.doc_id IN (" + marks + ") GROUP BY citation.doc_id"
                values = [started_at.astimezone(UTC).isoformat(), ended_at.astimezone(UTC).isoformat()]
                values.extend(str(identifier) for identifier in batch)
                for row in connection.execute(query, values):
                    totals[UUID(row["doc_id"])] = DocumentCitationTotals(
                        total=row["total"], unattributed=row["unattributed"],
                    )
        return totals

    def chunks(
        self, doc_id: UUID, chunk_ids: tuple[str, ...],
        started_at: datetime, ended_at: datetime,
    ) -> ChunkCitationTotals:
        """只对精确文档与所选块计数，不用正文匹配旧引用。

        Args:
            doc_id: 限定来源。
            chunk_ids: 至多100个当前页块ID。
            started_at: UTC起点。
            ended_at: UTC截止。

        Returns:
            只读块计数与旧未知归属。
        """
        if len(chunk_ids) > 100:
            raise ValueError("引用统计块页过大")
        counts = dict.fromkeys(chunk_ids, 0)
        unattributed = 0
        unknown = "citation.chunk_id IS NULL OR citation.chunk_id = ''"
        values: list[object] = [started_at.astimezone(UTC).isoformat(), ended_at.astimezone(UTC).isoformat(), str(doc_id)]
        condition = unknown
        if chunk_ids:
            condition += " OR citation.chunk_id IN (" + ",".join("?" for _ in chunk_ids) + ")"
            values.extend(chunk_ids)
        query = "SELECT citation.chunk_id, COUNT(*) AS total" + _FROM
        query += " AND citation.doc_id = ? AND (" + condition + ") GROUP BY citation.chunk_id"
        with self._snapshot() as connection:
            for row in connection.execute(query, values):
                identifier = row["chunk_id"]
                if identifier is None or identifier == "":
                    unattributed += row["total"]
                else:
                    counts[identifier] = row["total"]
        return ChunkCitationTotals(
            counts=tuple(ChunkCitationCount(chunk_id=identifier, count=count) for identifier, count in counts.items()),
            unattributed=unattributed, window_started_at=started_at,
            window_ended_at=ended_at,
        )

    @contextmanager
    def _snapshot(self) -> Iterator[sqlite3.Connection]:
        connection = None
        try:
            connection = sqlite3.connect(self._path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN")
            yield connection
        except (sqlite3.Error, OSError, ValueError) as error:
            raise KnowledgeContentError("正式引用统计暂时不可用") from error
        finally:
            if connection is not None:
                connection.close()
