# -*- coding: utf-8 -*-
"""统一文档与块的七日UTC日历统计窗口。"""

from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from uuid import UUID

from app.application.ports.document_citations import DocumentCitationReader
from app.domain.document_diagnostics import ChunkCitationTotals
from app.domain.document_diagnostics import DocumentCitationTotals


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CitationStatisticsService:
    """统计引用行，而非提问次数或检索命中次数。"""

    def __init__(
        self, reader: DocumentCitationReader, clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """保存只读端口及可验证的UTC时钟。

        Args:
            reader: 当前业务库中的正式引用读取器。
            clock: 统计截止时刻。

        Returns:
            None。
        """
        self._reader = reader
        self._clock = clock

    def documents(self, doc_ids: tuple[UUID, ...]) -> dict[UUID, DocumentCitationTotals]:
        """批量读取同一窗口的文档计数。

        Args:
            doc_ids: 实际索引中的来源。

        Returns:
            所有来源的真实引用总量。
        """
        started, ended = self._window()
        return self._reader.documents(doc_ids, started, ended)

    def chunks(self, doc_id: UUID, chunk_ids: tuple[str, ...]) -> ChunkCitationTotals:
        """读取当前页的精确块引用，不补造旧未知归属。

        Args:
            doc_id: 实际文档。
            chunk_ids: 有界当前页块身份。

        Returns:
            计数及实际统计起止时间。
        """
        started, ended = self._window()
        return self._reader.chunks(doc_id, chunk_ids, started, ended)

    def _window(self) -> tuple[datetime, datetime]:
        ended = self._clock().astimezone(UTC)
        started = ended.replace(hour=0, minute=0, second=0, microsecond=0)
        return started - timedelta(days=6), ended
