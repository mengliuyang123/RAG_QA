# -*- coding: utf-8 -*-
"""只读正式引用计数，不读取问题、答案、引用正文或trace。"""

from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.domain.document_diagnostics import ChunkCitationTotals
from app.domain.document_diagnostics import DocumentCitationTotals


class DocumentCitationReader(Protocol):
    """以active正式引用行为事实，不把检索候选数当作引用次数。"""

    def documents(
        self, doc_ids: tuple[UUID, ...], started_at: datetime, ended_at: datetime,
    ) -> dict[UUID, DocumentCitationTotals]:
        """同一只读快照批量返回文档计数。

        Args:
            doc_ids: 当前管理视图中的文档。
            started_at: UTC起点，含边界。
            ended_at: UTC截止，含边界。

        Returns:
            每个请求身份的实际总量，成功查询空记录才为0。
        """

    def chunks(
        self, doc_id: UUID, chunk_ids: tuple[str, ...],
        started_at: datetime, ended_at: datetime,
    ) -> ChunkCitationTotals:
        """只统计所选页与该文档旧未知块归属。

        Args:
            doc_id: 限定文档。
            chunk_ids: 当前有界页，不超过100块。
            started_at: UTC起点。
            ended_at: UTC截止。

        Returns:
            精确块计数和不可分配的旧引用数。
        """
