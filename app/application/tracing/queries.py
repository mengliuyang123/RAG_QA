# -*- coding: utf-8 -*-
"""在返回后台追踪前核对正式提交，不把日志陈旧状态当业务事实。"""

from uuid import UUID
import base64
import binascii
import hashlib

from app.application.errors import TraceStorageError
from app.application.ports.trace_queries import TraceReader
from app.domain.trace_queries import TraceDetail
from app.domain.trace_queries import TraceRecord
from app.domain.trace_queries import TraceFilters
from app.domain.trace_queries import TracePosition
from app.domain.trace_queries import TracePage
from app.domain.trace_queries import TraceSummary
from app.application.tracing.totals import token_totals
from app.application.tracing.totals import cost_totals


def validate_trace_link(record: TraceRecord) -> None:
    """拒绝不一致的正式关联，避免查询与用量统计各自放宽归属。

    Args:
        record: 同一只读查询获取的追踪与正式轮次。

    Returns:
        None，归属不一致时抛出存储异常。
    """
    snapshot = record.snapshot
    if record.turn_id is not None:
        if (
            snapshot.entry != "conversation"
            or snapshot.conversation_id != record.conversation_id
            or (
                snapshot.user_id is not None
                and snapshot.user_id != record.owner_id
            )
        ):
            raise TraceStorageError("追踪与正式轮次归属不一致")


class TraceQueryService:
    """提供受保护接口所需的只读投影，不再次调用上游模型。"""

    def __init__(self, reader: TraceReader) -> None:
        """保存只读端口。

        Args:
            reader: 同一运行的追踪与业务关联读取器。

        Returns:
            None。
        """
        self._reader = reader

    def get(self, trace_id: UUID) -> TraceDetail:
        """读取未过期详情并以业务库核对提交关联。

        Args:
            trace_id: 追踪标识。

        Returns:
            不默认展示提问者身份的追踪详情。
        """
        return self._detail(self._reader.get(trace_id))

    def list(
        self, filters: TraceFilters, *, limit: int = 20,
        cursor: str | None = None,
    ) -> TracePage:
        """以过滤条件绑定的游标分页，不以全量内存截取代替数据库分页。

        Args:
            filters: 已验证条件。
            limit: 每页1–100条。
            cursor: 上次返回的游标。

        Returns:
            脱敏概要及下一页游标。
        """
        if not 1 <= limit <= 100:
            raise ValueError("每页数量无效")
        fingerprint = hashlib.sha256(filters.model_dump_json().encode()).hexdigest()
        position = None
        if cursor is not None:
            try:
                data = base64.b64decode(cursor, altchars=b"-_", validate=True)
                position = TracePosition.model_validate_json(data)
                if position.fingerprint != fingerprint:
                    raise ValueError("游标的查询条件不一致")
            except (ValueError, binascii.Error) as error:
                raise ValueError("追踪游标无效") from error
        records = self._reader.find(filters, position, limit + 1)
        items = []
        for record in records[:limit]:
            detail = self._detail(record)
            payload = detail.model_dump(exclude={"steps", "calls", "duration_ms"})
            payload["latency_ms"] = detail.duration_ms
            items.append(TraceSummary.model_validate(payload))
        next_cursor = None
        if len(records) > limit:
            last = records[limit - 1].snapshot
            value = TracePosition(
                created_at=last.created_at, trace_id=last.trace_id,
                fingerprint=fingerprint,
            ).model_dump_json().encode()
            next_cursor = base64.urlsafe_b64encode(value).decode()
        return TracePage(traces=tuple(items), next_cursor=next_cursor)

    @staticmethod
    def _detail(record: TraceRecord) -> TraceDetail:
        validate_trace_link(record)
        snapshot = record.snapshot
        payload = snapshot.model_dump()
        if record.turn_id is not None:
            payload.update(commit_status="committed", turn_id=record.turn_id)
        return TraceDetail.model_validate({
            **payload, "turn_available": record.turn_id is not None,
            "feedback": record.feedback,
            "model_calls": len(snapshot.calls),
            "llm_calls": sum(
                call.purpose not in {"embedding", "reranking"}
                for call in snapshot.calls
            ),
            "tokens": token_totals(snapshot.calls),
            "cost": cost_totals(snapshot.calls),
        })
