# -*- coding: utf-8 -*-
"""使用生产端口执行多轮案例，不持久化为真实用户会话。"""

import hashlib
from collections.abc import Callable
from time import perf_counter
from uuid import UUID
from uuid import uuid4

from app.application.conversations.history_window import HistoryWindowBuilder
from app.application.ports.query_execution import QuestionExecutor
from app.application.ports.tracing import TraceRepository
from app.domain.conversations import ConversationTurn
from app.domain.retrieval import RetrievalStrategy
from evaluation.lanes.models import LaneEpisode
from evaluation.lanes.models import LaneObservation
from evaluation.lanes.models import LanePolicy
from evaluation.lanes.models import LaneTurn


class LaneEpisodeRunner:
    """每次策略运行建立独立历史，共用线上预算与审核机制。"""

    def __init__(
        self, executor: QuestionExecutor, history_window: HistoryWindowBuilder,
        traces: TraceRepository, cleanup: Callable[[UUID], None],
    ) -> None:
        """注入由组合根创建的生产组件。

        Args:
            executor: 两个HTTP入口共用的问题编排器。
            history_window: 同一配置的历史Token窗口。
            traces: 隔离评估路径下的追踪存储。
            cleanup: 只清理本次新建临时图线程的回调。

        Returns:
            None。
        """
        self._executor = executor
        self._history_window = history_window
        self._traces = traces
        self._cleanup = cleanup

    def run(
        self, episode: LaneEpisode, policy: LanePolicy,
        strategy: RetrievalStrategy,
        *, cached: tuple[LaneObservation, ...] = (),
        on_turn: Callable[[LaneObservation], None] | None = None,
    ) -> tuple[LaneObservation, ...]:
        """执行完整片段，后续问题只能读取该策略的真实前轮答案。

        Args:
            episode: 已标注片段。
            policy: auto、fast_only或agent。
            strategy: 与策略独立的检索方式。
            cached: 同一严格运行指纹下已成功生成的连续前缀。
            on_turn: 每轮完成后立即持久化；失败中断而非偷偷重新生成。

        Returns:
            顺序排列的实际观察；执行失败抛出，不能伪装为通过。
        """
        if policy not in {"auto", "fast_only", "agent"}:
            raise ValueError("评估策略无效")
        if policy != "auto":
            if any(not turn.requires_retrieval for turn in episode.turns):
                raise ValueError("强制知识基线不接受非知识案例")
        _validate_cached(episode, policy, strategy, cached)
        execution_id = cached[0].execution_id if cached else uuid4()
        history = []
        for record in cached:
            history.append(_as_history(record, episode.turns[record.turn_index].question))
        observations = list(cached)
        try:
            for index, turn in enumerate(episode.turns):
                if index < len(cached):
                    continue
                started = perf_counter()
                selected = self._history_window.build(
                    turn.question, reversed(history),
                )
                result = self._executor.run(
                    execution_id, turn.question, selected,
                    requested_lane="fast" if policy == "fast_only" else policy,
                    strategy=strategy, top_k=episode.top_k,
                    execution_entry="evaluation",
                    lane_policy="fast_only" if policy == "fast_only" else "online",
                )
                latency_ms = (perf_counter() - started) * 1000
                if result.metadata is None or result.metadata.trace_id is None:
                    raise ValueError("评估编排结果缺少真实追踪")
                trace = self._traces.get(UUID(result.metadata.trace_id))
                queries = []
                if result.execution is not None:
                    queries = [item.query for item in result.execution.budget.attempts]
                routing_correct = None
                if policy == "auto":
                    routing_correct = False
                    for step in trace.steps:
                        if step.type == "routing":
                            routing_correct = step.output.get("lane") == turn.expected_lane
                observation = LaneObservation(
                    episode_id=episode.id, turn_index=index,
                    execution_id=execution_id, policy=policy, strategy=strategy,
                    result=result, trace=trace, latency_ms=latency_ms,
                    query_hashes=tuple(
                        hashlib.sha256(query.encode("utf-8")).hexdigest()
                        for query in queries
                    ),
                    retrieval_compliant=(
                        (result.retrieval_calls > 0) == turn.requires_retrieval
                    ),
                    routing_correct=routing_correct,
                    rewrite_matches=_rewrite_matches(turn, queries),
                )
                observations.append(observation)
                if on_turn is not None:
                    on_turn(observation)
                history.append(_as_history(observation, turn.question))
        finally:
            self._cleanup(execution_id)
        return tuple(observations)


def _validate_cached(
    episode: LaneEpisode, policy: LanePolicy, strategy: RetrievalStrategy,
    cached: tuple[LaneObservation, ...],
) -> None:
    if len(cached) > len(episode.turns):
        raise ValueError("缓存轮次超过案例长度")
    for index, record in enumerate(cached):
        question_hash = hashlib.sha256(
            episode.turns[index].question.encode("utf-8"),
        ).hexdigest()
        if (
            record.episode_id != episode.id or record.turn_index != index
            or record.policy != policy or record.strategy != strategy
            or record.execution_id != cached[0].execution_id
            or record.trace.question_sha256 != question_hash
        ):
            raise ValueError("评估缓存不是当前案例策略的连续前缀")


def _as_history(record: LaneObservation, question: str) -> ConversationTurn:
    result = record.result
    return ConversationTurn(
        id=uuid4(), conversation_id=record.execution_id,
        user_message=question, assistant_message=result.answer,
        answer_mode=result.answer_mode, retrieval_calls=result.retrieval_calls,
        citations=result.citations, latency_ms=record.latency_ms,
        created_at=record.trace.created_at, metadata=result.metadata,
    )


def _rewrite_matches(turn: LaneTurn, queries: list[str]) -> bool | None:
    if not turn.rewrite_terms:
        return None
    if not queries:
        return False
    first_query = queries[0].casefold()
    return all(term.casefold() in first_query for term in turn.rewrite_terms)
