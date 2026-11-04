# -*- coding: utf-8 -*-
"""声明不绕过普通会话归属的后台拒答专用读取端口。"""

from datetime import datetime
from collections.abc import Iterator
from typing import Protocol
from uuid import UUID

from app.domain.insights import RefusalItem
from app.domain.insights import InsightMetric
from app.domain.refusal_clusters import RefusalClusterSample


class InsightMetricReader(Protocol):
    """以同一只读快照提供去重身份贡献，不传递问题、答案或用户标识。"""

    def iter_metrics(
        self, started_at: datetime, ended_at: datetime,
    ) -> Iterator[InsightMetric]:
        """在数据库筛选窗口内分批读取元数据。

        Args:
            started_at: UTC起点，含边界。
            ended_at: UTC截止，含边界。

        Yields:
            同一身份计数贡献合计仅为1的正式轮次元数据。
        """


class RefusalReader(Protocol):
    """只暴露明确拒答的有限单轮投影，不提供全会话读取。"""

    def cluster_sample(
        self, started_at: datetime, ended_at: datetime, kb_id: str | None,
    ) -> RefusalClusterSample:
        """在同一快照先过滤计数，再取最新至多500条。

        Args:
            started_at: UTC包含起点。
            ended_at: UTC包含截止。
            kb_id: 实际检索范围过滤。

        Returns:
            样本及完整窗口计数。
        """
        ...

    def list(
        self, limit: int, kb_id: str | None,
        before: tuple[datetime, UUID] | None,
    ) -> tuple[RefusalItem, ...]:
        """先按事实范围过滤，再返回时间／ID倒序有限条数。

        Args:
            limit: 最大读取数量，含应用层的一条续页探测。
            kb_id: 实际检索知识库过滤。
            before: 不包含边界本身的上一页末条。

        Returns:
            当前active正式会话的已提交拒答投影。
        """
