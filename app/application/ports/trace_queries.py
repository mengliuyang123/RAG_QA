# -*- coding: utf-8 -*-
"""声明执行快照与权威业务提交关联的只读端口。"""

from typing import Protocol
from collections.abc import Iterator
from uuid import UUID

from app.domain.trace_queries import TraceRecord
from app.domain.trace_queries import TraceFilters
from app.domain.trace_queries import TracePosition


class TraceReader(Protocol):
    """查询投影不调用模型、不创建业务轮次。"""

    def iter_records(self, filters: TraceFilters) -> Iterator[TraceRecord]:
        """流式读取过滤后的追踪，不在内存保留整个时间窗口。

        Args:
            filters: 已验证的统计窗口和其他条件。

        Yields:
            未过期的只读关联记录。
        """

    def find(
        self, filters: TraceFilters, position: TracePosition | None,
        limit: int,
    ) -> tuple[TraceRecord, ...]:
        """过滤后按时间与ID倒序读取有限条数。

        Args:
            filters: 所有条件，包含当前反馈。
            position: 排序边界，不含该条记录。
            limit: 最多读取数量。

        Returns:
            未过期的只读关联投影。
        """

    def get(self, trace_id: UUID) -> TraceRecord:
        """返回未过期追踪及当前业务关联。

        Args:
            trace_id: 精确追踪标识。

        Returns:
            当前追踪和只包含标识／评分的正式关联。
        """
