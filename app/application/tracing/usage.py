# -*- coding: utf-8 -*-
"""流式汇总保留期内用量，用户化名默认不暴露身份。"""

import hashlib
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from datetime import timedelta

from app.application.ports.knowledge_bases import KnowledgeBaseRepository
from app.application.ports.trace_queries import TraceReader
from app.application.tracing.queries import validate_trace_link
from app.application.tracing.totals import CostAccumulator
from app.application.tracing.totals import TokenAccumulator
from app.domain.trace_queries import TraceFilters
from app.domain.tracing import TraceSnapshot
from app.domain.usage import UsageGroup
from app.domain.usage import UsageReport
from app.domain.usage import UsageRow
from app.domain.usage import UsageTotals


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _UsageAccumulator:
    def __init__(self) -> None:
        """初始化空请求集合的明确零计数。

        Returns:
            None。
        """
        self.questions = 0
        self.running = 0
        self.model_calls = 0
        self.llm_calls = 0
        self.tokens = TokenAccumulator()
        self.cost = CostAccumulator()

    def add(self, snapshot: TraceSnapshot) -> None:
        """累加一条追踪的请求及已记录调用。

        Args:
            snapshot: 当前执行快照。

        Returns:
            None。
        """
        self.questions += 1
        if snapshot.status == "running":
            self.running += 1
        for call in snapshot.calls:
            self.model_calls += 1
            if call.purpose not in {"embedding", "reranking"}:
                self.llm_calls += 1
            self.tokens.add(call.usage)
            self.cost.add(call.cost)

    def result(self) -> UsageTotals:
        """返回不再累积变化的报告小计。

        Returns:
            已报告用量和未知计数。
        """
        return UsageTotals(
            questions=self.questions, running_requests=self.running,
            model_calls=self.model_calls, llm_calls=self.llm_calls,
            tokens=self.tokens.result(), cost=self.cost.result(),
        )


class UsageQueryService:
    """分组报告只覆盖已记录追踪，不作为独占知识库账单。"""

    def __init__(
        self, reader: TraceReader, *, retention_days: int = 30,
        catalog: KnowledgeBaseRepository | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """保存只读端口及明确保留策略。

        Args:
            reader: 当前运行隔离的追踪读取器。
            retention_days: 当前追踪保留天数。
            catalog: 用于知识库标签，不影响历史分组ID。
            clock: UTC统计时间。

        Returns:
            None。
        """
        if retention_days < 1:
            raise ValueError("保留天数必须大于零")
        self._reader = reader
        self._retention_days = retention_days
        self._catalog = catalog
        self._clock = clock

    def get(self, days: int = 7, group_by: UsageGroup = "day") -> UsageReport:
        """统计含今天的UTC日历窗口，跨库总量只计算一次。

        Args:
            days: 1–365天。
            group_by: 用户化名、实际检索知识库或UTC日。

        Returns:
            去重总量、分组行及可加性／保留范围声明。
        """
        if not 1 <= days <= 365 or group_by not in {"user", "kb", "day"}:
            raise ValueError("用量窗口或分组无效")
        now = self._clock().astimezone(UTC)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start -= timedelta(days=days - 1)
        groups: dict[str, _UsageAccumulator] = {}
        configurations: dict[str, _UsageAccumulator] = {}
        total = _UsageAccumulator()
        rows_additive = True
        for record in self._reader.iter_records(TraceFilters(from_at=start, to_at=now)):
            validate_trace_link(record)
            snapshot = record.snapshot
            total.add(snapshot)
            version = snapshot.configuration_version
            if version not in configurations:
                configurations[version] = _UsageAccumulator()
            configurations[version].add(snapshot)
            keys = self._keys(snapshot, group_by)
            if len(keys) > 1:
                rows_additive = False
            for key in keys:
                if key not in groups:
                    groups[key] = _UsageAccumulator()
                groups[key].add(snapshot)
        labels = {}
        if group_by == "kb" and self._catalog is not None:
            labels = {item.id: item.name for item in self._catalog.list()}
        rows = []
        for key in sorted(groups):
            rows.append(UsageRow(
                key=key, label=self._label(key, group_by, labels),
                **groups[key].result().model_dump(),
            ))
        configuration_rows = []
        for version in sorted(configurations):
            configuration_rows.append(UsageRow(
                key=version, label=version,
                **configurations[version].result().model_dump(),
            ))
        return UsageReport(
            group_by=group_by, days=days, window_started_at=start,
            window_ended_at=now, retention_days=self._retention_days,
            retention_limited=start < now - timedelta(days=self._retention_days),
            attribution="participating_scope" if group_by == "kb" else "request",
            rows_additive=rows_additive, totals=total.result(), rows=tuple(rows),
            configurations=tuple(configuration_rows),
        )

    @staticmethod
    def _keys(snapshot: TraceSnapshot, group_by: UsageGroup) -> tuple[str, ...]:
        if group_by == "day":
            return (snapshot.created_at.astimezone(UTC).date().isoformat(),)
        if group_by == "user":
            if snapshot.user_id is None:
                return ("unknown-user:",)
            digest = hashlib.sha256(str(snapshot.user_id).encode()).hexdigest()
            return (f"user:{digest}",)
        kb_ids = set()
        has_retrieval = False
        for step in snapshot.steps:
            if step.type != "retrieval":
                continue
            has_retrieval = True
            values = step.input.get("kb_ids")
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str) and value:
                        kb_ids.add(value)
        if kb_ids:
            return tuple(sorted(kb_ids))
        return ("unknown-scope:" if has_retrieval else "no-retrieval:",)

    @staticmethod
    def _label(key: str, group_by: UsageGroup, labels: dict[str, str]) -> str:
        if group_by == "user":
            if key == "unknown-user:":
                return "未关联用户"
            return f"用户化名 {key[5:17]}"
        if key == "no-retrieval:":
            return "未执行检索"
        if key == "unknown-scope:":
            return "未保留检索范围"
        return labels.get(key, key)
