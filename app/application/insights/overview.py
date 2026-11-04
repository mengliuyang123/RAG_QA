# -*- coding: utf-8 -*-
"""流式汇总正式问答元数据，明确首次路由分母和未知覆盖。"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from uuid import UUID

from app.application.ports.insights import InsightMetricReader
from app.application.ports.knowledge_bases import KnowledgeBaseRepository
from app.application.tracing.auditing import AdminAuditService
from app.domain.insights import InsightMetric
from app.domain.insights import InsightOverview
from app.domain.insights import InsightTotals
from app.domain.insights import KnowledgeInsight


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _rate(count: int, total: int) -> float | None:
    if total == 0:
        return None
    return count / total


@dataclass
class _KnowledgeCounts:
    """每个实际参与库只保存计数器，不累积用户标识或轮次正文。"""

    questions: int = 0
    refused: int = 0
    down: int = 0
    initial_auto: int = 0
    expanded: int = 0
    unknown_initial: int = 0

    def add(self, metric: InsightMetric, kb_id: str) -> None:
        """累加实际参与请求，扩查分母只看明确首次自动范围。

        Args:
            metric: 单轮元数据。
            kb_id: 本计数器所属参与库。

        Returns:
            None。
        """
        metadata = metric.metadata
        if metadata is None:
            return
        self.questions += 1
        if metadata.outcome == "refused":
            self.refused += 1
        if metric.feedback == "down":
            self.down += 1
        scope = metadata.kb_scope
        if scope.mode == "auto":
            if scope.initial is None:
                self.unknown_initial += 1
            elif kb_id in scope.initial:
                self.initial_auto += 1
                if metadata.scope_expanded:
                    self.expanded += 1

    def result(self, kb_id: str) -> KnowledgeInsight:
        """产生分母明确的只读结果，未知旧范围不混入已知比例。

        Args:
            kb_id: 当前统计知识库。

        Returns:
            不可变计数及比率。
        """
        expanded_rate = None
        if self.unknown_initial == 0:
            expanded_rate = _rate(self.expanded, self.initial_auto)
        return KnowledgeInsight(
            kb_id=kb_id, questions=self.questions,
            refused_questions=self.refused, thumbs_down_questions=self.down,
            known_initial_auto_questions=self.initial_auto,
            expanded_questions=self.expanded,
            unknown_initial_auto_questions=self.unknown_initial,
            refusal_rate=_rate(self.refused, self.questions),
            thumbs_down_rate=_rate(self.down, self.questions),
            expanded_rate=expanded_rate,
        )


class InsightOverviewService:
    """只统计正式提交事实；日志过期不影响当前业务元数据。"""

    def __init__(
        self, reader: InsightMetricReader, catalog: KnowledgeBaseRepository,
        audit: AdminAuditService, clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """保存只读端口与UTC窗口策略。

        Args:
            reader: 同一快照的有界元数据迭代器。
            catalog: 当前目录，仅补齐空库行。
            audit: 返回前的无正文访问审计。
            clock: 真实统计截止时刻。

        Returns:
            None。
        """
        self._reader = reader
        self._catalog = catalog
        self._audit = audit
        self._clock = clock

    def get(self, actor_id: UUID, days: int = 7) -> InsightOverview:
        """统计含今天的UTC日历窗口，不将参与库小计相加为全局总量。

        Args:
            actor_id: 认证后的管理员身份，不返回给前端。
            days: 1–365天，包含今天。

        Returns:
            全局去重总量与分母可解释的各库比率。
        """
        if not 1 <= days <= 365:
            raise ValueError("概览统计窗口无效")
        ended = self._clock().astimezone(UTC)
        started = ended.replace(hour=0, minute=0, second=0, microsecond=0)
        started -= timedelta(days=days - 1)
        groups = {}
        for knowledge_base in self._catalog.list():
            groups[knowledge_base.id] = _KnowledgeCounts()
        questions = 0
        active_users = 0
        unknown_owner = 0
        unknown_metadata = 0
        no_retrieval = 0
        unknown_scope = 0
        rows_additive = True
        for metric in self._reader.iter_metrics(started, ended):
            questions += 1
            active_users += metric.user_count_increment
            if not metric.owner_known:
                unknown_owner += 1
            metadata = metric.metadata
            if metadata is None:
                unknown_metadata += 1
                rows_additive = False
                continue
            kb_ids = metadata.kb_scope.searched
            if not kb_ids:
                if metadata.lane in {"direct", "clarify"}:
                    no_retrieval += 1
                else:
                    unknown_scope += 1
                rows_additive = False
            elif len(kb_ids) > 1:
                rows_additive = False
            for kb_id in kb_ids:
                if kb_id not in groups:
                    groups[kb_id] = _KnowledgeCounts()
                groups[kb_id].add(metric, kb_id)
        report = InsightOverview(
            period_days=days, window_started_at=started, window_ended_at=ended,
            totals=InsightTotals(
                questions=questions, active_users=active_users,
                unknown_owner_questions=unknown_owner,
                unknown_metadata_questions=unknown_metadata,
                no_retrieval_questions=no_retrieval,
                unknown_scope_questions=unknown_scope,
            ),
            knowledge_bases=tuple(groups[kb_id].result(kb_id) for kb_id in sorted(groups)),
            rows_additive=rows_additive,
        )
        self._audit.record(actor_id, "read", "overview", ())
        return report
