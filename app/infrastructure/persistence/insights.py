# -*- coding: utf-8 -*-
"""以只读SQL筛选正式拒答，不扫描全部会话或复制完整聊天历史。"""

import sqlite3
from collections.abc import Iterator
from datetime import UTC
from datetime import datetime
from pathlib import Path
from uuid import UUID

from app.application.errors import InsightStorageError
from app.domain.answer_metadata import AnswerMetadata
from app.domain.insights import RefusalItem
from app.domain.insights import InsightMetric
from app.domain.refusal_clusters import RefusalClusterSample


_REFUSALS = """
    SELECT t.id, t.user_message, t.created_at, t.metadata_json, t.trace_id
    FROM conversation_turns t
    JOIN conversations c ON c.id = t.conversation_id
    WHERE c.status = 'active'
        AND json_extract(t.metadata_json, '$.outcome') = 'refused'
"""


class SQLiteInsightMetricReader:
    """使用窗口函数去重用户，在应用层仅保留固定大小的各库计数器。"""

    def __init__(self, path: Path) -> None:
        """保存已初始化的业务路径，不读取追踪库或模型。

        Args:
            path: 当前运行的正式会话数据库。

        Returns:
            None。
        """
        self._path = path

    def iter_metrics(
        self, started_at: datetime, ended_at: datetime,
    ) -> Iterator[InsightMetric]:
        """在同一只读事务中按批次读取，不加载问题、答案或反馈评论。

        Args:
            started_at: UTC窗口开始。
            ended_at: UTC窗口截止。

        Yields:
            不向应用层泄漏身份的计数投影。
        """
        query = """SELECT t.metadata_json, c.owner_id,
            json_extract(t.feedback_json, '$.rating') AS feedback,
            CASE WHEN c.owner_id IS NOT NULL AND
                ROW_NUMBER() OVER (PARTITION BY c.owner_id ORDER BY t.id) = 1
                THEN 1 ELSE 0 END AS user_increment
            FROM conversation_turns t
            JOIN conversations c ON c.id = t.conversation_id
            WHERE c.status = 'active' AND t.created_at >= ? AND t.created_at <= ?
        """
        connection = None
        try:
            connection = sqlite3.connect(
                self._path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN")
            cursor = connection.execute(query, (
                started_at.astimezone(UTC).isoformat(),
                ended_at.astimezone(UTC).isoformat(),
            ))
            while True:
                rows = cursor.fetchmany(100)
                if not rows:
                    break
                for row in rows:
                    if row["owner_id"] is not None:
                        UUID(row["owner_id"])
                    metadata = None
                    if row["metadata_json"] is not None:
                        metadata = AnswerMetadata.model_validate_json(row["metadata_json"])
                    yield InsightMetric(
                        owner_known=row["owner_id"] is not None,
                        user_count_increment=row["user_increment"],
                        metadata=metadata, feedback=row["feedback"],
                    )
        except (sqlite3.Error, OSError, ValueError) as error:
            raise InsightStorageError("概览统计读取失败") from error
        finally:
            if connection is not None:
                connection.close()


class SQLiteRefusalReader:
    """使用组合根初始化的业务库，只读有界拒答投影。"""

    def __init__(self, path: Path) -> None:
        """保存隔离业务路径，不迁移、不创建文件。

        Args:
            path: 已初始化的正式会话数据库。

        Returns:
            None。
        """
        self._path = path

    def cluster_sample(
        self, started_at: datetime, ended_at: datetime, kb_id: str | None,
    ) -> RefusalClusterSample:
        """用只读事务固定计数与最新500条，时间/实际KB过滤先于LIMIT。

        Args:
            started_at: UTC起点，含边界。
            ended_at: UTC截止，含边界。
            kb_id: 实际检索范围，可不限定。

        Returns:
            不含用户身份的聚类样本。
        """
        query = _REFUSALS + " AND t.created_at >= ? AND t.created_at <= ?"
        parameters = [
            started_at.astimezone(UTC).isoformat(),
            ended_at.astimezone(UTC).isoformat(),
        ]
        if kb_id is not None:
            query += """ AND EXISTS (
                SELECT 1 FROM json_each(t.metadata_json, '$.kb_scope.searched') kb
                WHERE kb.value = ?
            )"""
            parameters.append(kb_id)
        connection = None
        try:
            connection = sqlite3.connect(
                self._path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN")
            total = connection.execute(
                "SELECT COUNT(*) FROM (" + query + ")", parameters,
            ).fetchone()[0]
            rows = connection.execute(
                query + " ORDER BY t.created_at DESC, t.id DESC LIMIT 500",
                parameters,
            ).fetchall()
            return RefusalClusterSample(
                items=tuple(self._item(row) for row in rows), total=total,
            )
        except (sqlite3.Error, OSError, ValueError) as error:
            raise InsightStorageError("拒答聚类快照读取失败") from error
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _item(row: sqlite3.Row) -> RefusalItem:
        metadata = AnswerMetadata.model_validate_json(row["metadata_json"])
        if metadata.refusal is None:
            raise ValueError("拒答元数据缺少缺口")
        trace_id = UUID(row["trace_id"]) if row["trace_id"] else None
        if metadata.trace_id != (str(trace_id) if trace_id is not None else None):
            raise ValueError("拒答追踪标识不一致")
        return RefusalItem(
            turn_id=UUID(row["id"]), question=row["user_message"],
            created_at=row["created_at"], trace_id=trace_id,
            kb_ids=metadata.kb_scope.searched, lane=metadata.lane,
            grounding=metadata.grounding, refusal=metadata.refusal,
        )

    def list(
        self, limit: int, kb_id: str | None,
        before: tuple[datetime, UUID] | None,
    ) -> tuple[RefusalItem, ...]:
        """SQL先过滤再分页，一条查询只读取有限单轮字段。

        Args:
            limit: 包含续页探测时至多101条。
            kb_id: 实际检索知识库过滤。
            before: UTC时间与ID排序边界。

        Returns:
            不含提问者身份或回答正文的投影。
        """
        if not 1 <= limit <= 101:
            raise ValueError("拒答读取上限无效")
        query = _REFUSALS
        parameters: list[object] = []
        if kb_id is not None:
            query += """ AND EXISTS (
                SELECT 1 FROM json_each(t.metadata_json, '$.kb_scope.searched') kb
                WHERE kb.value = ?
            )"""
            parameters.append(kb_id)
        if before is not None:
            query += " AND (t.created_at, t.id) < (?, ?)"
            parameters.extend((before[0].astimezone(UTC).isoformat(), str(before[1])))
        query += " ORDER BY t.created_at DESC, t.id DESC LIMIT ?"
        parameters.append(limit)
        connection = None
        try:
            connection = sqlite3.connect(
                self._path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5,
            )
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, parameters).fetchall()
            items = []
            for row in rows:
                metadata = AnswerMetadata.model_validate_json(row["metadata_json"])
                if metadata.refusal is None:
                    raise ValueError("拒答元数据缺少缺口")
                trace_id = UUID(row["trace_id"]) if row["trace_id"] else None
                if metadata.trace_id != (str(trace_id) if trace_id is not None else None):
                    raise ValueError("拒答追踪标识不一致")
                items.append(RefusalItem(
                    turn_id=UUID(row["id"]), question=row["user_message"],
                    created_at=row["created_at"], trace_id=trace_id,
                    kb_ids=metadata.kb_scope.searched, lane=metadata.lane,
                    grounding=metadata.grounding, refusal=metadata.refusal,
                ))
            return tuple(items)
        except (sqlite3.Error, OSError, ValueError) as error:
            raise InsightStorageError("拒答洞察读取失败") from error
        finally:
            if connection is not None:
                connection.close()
