# -*- coding: utf-8 -*-
"""创建独立请求追踪与配置明确的内容保留策略。"""

import hashlib
import logging
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from time import perf_counter
from typing import Literal
from uuid import UUID
from uuid import uuid4

from app.application.errors import TraceStorageError
from app.application.errors import TraceNotFoundError
from app.application.execution_control import StreamCancelled
from app.application.ports.tracing import TraceRepository
from app.application.tracing.context import bind_trace
from app.application.tracing.context import current_trace
from app.application.tracing.recorder import TraceRecorder
from app.application.tracing.recorder import normalize_origin
from app.domain.tracing import ModelPrice
from app.domain.tracing import TraceSnapshot
from app.domain.tracing import CommitStatus


logger = logging.getLogger(__name__)


@dataclass
class _PreparedExecution:
    owner: object
    execution_id: UUID
    recorder: TraceRecorder
    claimed: bool = False


_PREPARED_EXECUTION: ContextVar[_PreparedExecution | None] = ContextVar(
    "rag_prepared_execution", default=None,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class TraceService:
    """按请求创建记录器，不以常驻字典积累已结束请求。"""

    def __init__(
        self, repository: TraceRepository, *, capture_queries: bool = False,
        retention_days: int = 30, prices: tuple[ModelPrice, ...] = (),
        configuration_version: str = "layered-routing-v1",
        clock: Callable[[], float] = perf_counter,
        wall_clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """保存已由组合根校验的追踪策略。

        Args:
            repository: 独立追踪存储端口。
            capture_queries: 是否显式保留查询正文，默认关闭。
            retention_days: 可查询保留天数。
            prices: 明确版本的费用估计费率，不自动查询价格。
            configuration_version: 不含凭据的执行配置标识。
            clock: 单调时钟。
            wall_clock: UTC时间。

        Returns:
            None。
        """
        if retention_days < 1:
            raise ValueError("追踪保留天数必须大于零")
        identities = set()
        for price in prices:
            key = (normalize_origin(price.origin), price.model)
            if key in identities:
                raise ValueError("同一来源和模型不能配置多个价格版本")
            identities.add(key)
        self._repository = repository
        self._capture_queries = capture_queries
        self._retention_days = retention_days
        self._prices = prices
        self._configuration_version = configuration_version
        self._clock = clock
        self._wall_clock = wall_clock

    @contextmanager
    def start(
        self, question: str, *, request_id: UUID,
        entry: Literal["query", "conversation", "evaluation"],
        conversation_id: UUID | None = None,
        user_id: UUID | None = None,
        lane_policy: Literal["online", "fast_only"] = "online",
    ) -> Iterator[TraceRecorder]:
        """创建、绑定并终止追踪，失败和取消不能保存成成功。

        Args:
            question: 原始用户问题。
            request_id: 本次执行标识。
            entry: 实际入口类型。
            conversation_id: 可选正式会话归属。
            user_id: 可选后端认证用户，不接受模型推断。
            lane_policy: 不与执行通道或检索策略混淆的策略。

        Yields:
            可由各层同步共享的本请求记录器。
        """
        now = self._wall_clock()
        recorder = TraceRecorder(
            self._repository,
            TraceSnapshot(
                trace_id=uuid4(), request_id=request_id, entry=entry,
                conversation_id=conversation_id, user_id=user_id,
                lane_policy=lane_policy,
                commit_status="pending" if entry == "conversation" else "not_applicable",
                question=question if self._capture_queries else None,
                question_sha256=hashlib.sha256(question.encode()).hexdigest(),
                created_at=now,
                expires_at=now + timedelta(days=self._retention_days),
                configuration_version=self._configuration_version,
            ),
            capture_queries=self._capture_queries, prices=self._prices,
            clock=self._clock, wall_clock=self._wall_clock,
        )
        with bind_trace(recorder):
            try:
                yield recorder
            except BaseException as error:
                try:
                    recorder.finish(
                        "cancelled" if isinstance(error, StreamCancelled) else "error",
                    )
                except TraceStorageError:
                    # 记录失败不能覆盖真正的业务错误或取消，也不能打印其正文。
                    logger.warning("追踪失败状态无法写入")
                if entry == "conversation":
                    self.record_commit(
                        recorder.id,
                        "cancelled" if isinstance(error, StreamCancelled) else "failed",
                    )
                raise
            else:
                try:
                    recorder.finish("completed")
                except TraceStorageError:
                    if entry == "conversation":
                        self.record_commit(recorder.id, "failed")
                    raise

    @contextmanager
    def start_execution(
        self, question: str, *, execution_id: UUID,
        entry: Literal["query", "conversation"], user_id: UUID,
    ) -> Iterator[TraceRecorder]:
        """从准入开始追踪，并允许匹配的编排调用认领一次。

        Args:
            question: 本次问题，遵守现有正文保存策略。
            execution_id: 无状态临时执行或正式会话标识。
            entry: 已由服务端确定的入口。
            user_id: 认证后的当前身份。

        Yields:
            准入与编排共用的唯一记录器。
        """
        with self.start(
            question, request_id=execution_id if entry == "query" else uuid4(),
            entry=entry, user_id=user_id,
            conversation_id=execution_id if entry == "conversation" else None,
        ) as recorder:
            prepared = _PreparedExecution(self, execution_id, recorder)
            token = _PREPARED_EXECUTION.set(prepared)
            try:
                yield recorder
            finally:
                _PREPARED_EXECUTION.reset(token)

    def claim_execution_trace(
        self, question: str, *, execution_id: UUID,
        entry: Literal["query", "conversation", "evaluation"],
        user_id: UUID | None, lane_policy: str,
    ) -> TraceRecorder | None:
        """仅复用同服务、身份、入口和问题的单次预备追踪。

        Args:
            question: 实际编排问题。
            execution_id: 实际编排执行标识。
            entry: 服务端入口。
            user_id: 认证身份。
            lane_policy: 线上策略才可认领准入追踪。

        Returns:
            匹配且未认领的记录器，否则None并沿用独立追踪逻辑。
        """
        prepared = _PREPARED_EXECUTION.get()
        if prepared is None or prepared.owner is not self or prepared.claimed:
            return None
        snapshot = prepared.recorder.snapshot
        if (
            current_trace() is not prepared.recorder
            or prepared.execution_id != execution_id
            or snapshot.entry != entry
            or snapshot.user_id != user_id
            or snapshot.lane_policy != lane_policy
            or snapshot.question_sha256 != hashlib.sha256(question.encode()).hexdigest()
        ):
            return None
        prepared.claimed = True
        return prepared.recorder

    def record_commit(
        self, trace_id: UUID, status: CommitStatus, turn_id: UUID | None = None,
    ) -> None:
        """尽力记录提交结果，追踪故障不能回滚已经提交的正式历史。

        Args:
            trace_id: 真实执行追踪标识。
            status: 真实业务提交状态。
            turn_id: 提交成功后的正式轮次标识。

        Returns:
            None；查询层还会以业务存储的trace关联核对提交事实。
        """
        try:
            self._repository.record_commit(trace_id, status, turn_id)
        except (TraceStorageError, TraceNotFoundError):
            logger.warning("追踪提交状态暂时无法写入，正式业务状态保持不变")
