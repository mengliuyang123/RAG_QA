# -*- coding: utf-8 -*-
"""记录请求级步骤与供应商调用，线程共享但不共享请求内容。"""

import hashlib
import json
import logging
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from threading import RLock
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import JsonValue

from app.application.execution_control import StreamCancelled
from app.application.errors import TraceStorageError
from app.application.ports.tracing import TraceRepository
from app.domain.tracing import ModelCallTrace
from app.domain.tracing import ModelPrice
from app.domain.tracing import TokenUsage
from app.domain.tracing import TraceSnapshot
from app.domain.tracing import TraceStep
from app.domain.tracing import estimate_cost
from app.domain.admission import AdmissionDecision


_PRIVATE_KEYS = frozenset({
    "authorization", "password", "secret", "access_token", "refresh_token",
    "messages", "history", "text", "chunk_text", "content", "headers",
    "prompt", "system_prompt", "answer",
})
_QUERY_KEYS = frozenset({
    "query", "question", "standalone_query", "sub_queries", "clarification",
    "missing_information", "issues", "next_query",
})
logger = logging.getLogger(__name__)


def normalize_origin(url: str) -> str:
    """去除服务地址中的凭据、路径和查询参数。

    Args:
        url: 原始兼容API地址。

    Returns:
        来源标识；无法识别时为unknown。
    """
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if not host or parsed.scheme not in {"http", "https"}:
            return "unknown"
        if ":" in host:
            host = f"[{host}]"
        port = parsed.port
        if port is not None:
            default_port = 443 if parsed.scheme == "https" else 80
            if port != default_port:
                host = f"{host}:{port}"
        return f"{parsed.scheme}://{host}"
    except ValueError:
        return "unknown"


@dataclass
class StepCapture:
    """保存上下文区间内的结果，结束时由记录器复制并过滤。

    Attributes:
        output: 尚未持久化的结构化结果。
    """

    output: dict[str, JsonValue] = field(default_factory=dict)

    def set_output(self, output: dict[str, JsonValue]) -> None:
        """保存本步骤的结构化输出。

        Args:
            output: 只应包含业务追踪所需的元数据。

        Returns:
            None。
        """
        self.output = output


@dataclass
class ProviderCallCapture:
    """只接收经过适配器提取的usage，不保存原始响应对象。

    Attributes:
        usage: 实际供应商计数，默认未知。
        response_model: 供应商返回的实际模型版本。
    """

    usage: TokenUsage = field(default_factory=TokenUsage)
    response_model: str | None = None


class TraceRecorder:
    """同一请求中的图节点共享记录器，以锁保护有序快照写入。"""

    def __init__(
        self, repository: TraceRepository, snapshot: TraceSnapshot,
        *, capture_queries: bool, prices: tuple[ModelPrice, ...],
        clock: Callable[[], float], wall_clock: Callable[[], datetime],
    ) -> None:
        """保存端口和本请求不可变配置。

        Args:
            repository: 追踪权威存储。
            snapshot: 当前请求初始快照。
            capture_queries: 是否允许保存查询正文。
            prices: 明确的价格版本列表。
            clock: 单调耗时时钟。
            wall_clock: UTC记录时间。

        Returns:
            None。
        """
        self._repository = repository
        self._snapshot = snapshot
        self._capture_queries = capture_queries
        self._prices = prices
        self._clock = clock
        self._wall_clock = wall_clock
        self._started = clock()
        self._lock = RLock()
        repository.save(snapshot)

    @property
    def id(self) -> UUID:
        """返回当前追踪标识。

        Returns:
            追踪UUID。
        """
        return self._snapshot.trace_id

    @property
    def snapshot(self) -> TraceSnapshot:
        """读取当前冻结快照。

        Returns:
            已持久化快照。
        """
        with self._lock:
            return self._snapshot

    @contextmanager
    def step(
        self, kind: str, input: dict[str, JsonValue] | None = None,
    ) -> Iterator[StepCapture]:
        """立即记录开始，结束时保存结果和实际耗时。

        Args:
            kind: 步骤类型。
            input: 待过滤的输入元数据。

        Yields:
            本步骤结果收集器。
        """
        started = self._clock()
        with self._lock:
            step = TraceStep(
                order=len(self._snapshot.steps) + 1, type=kind,
                started_at=self._wall_clock(), input=self._sanitize(input or {}),
            )
            self._save(steps=(*self._snapshot.steps, step))
        capture = StepCapture()
        status = "completed"
        try:
            yield capture
        except StreamCancelled:
            status = "cancelled"
            raise
        except BaseException:
            status = "error"
            raise
        finally:
            final = TraceStep.model_validate({
                **step.model_dump(), "status": status,
                "duration_ms": max(0.0, (self._clock() - started) * 1000),
                "output": self._sanitize(capture.output),
            })
            with self._lock:
                steps = []
                for previous in self._snapshot.steps:
                    steps.append(final if previous.id == step.id else previous)
                self._save_preserving_error(status != "completed", steps=tuple(steps))

    @contextmanager
    def provider_call(
        self, purpose: str, origin: str, model: str,
    ) -> Iterator[ProviderCallCapture]:
        """记录一次供应商调用，失败也不伪称零用量。

        Args:
            purpose: 明确调用用途。
            origin: API地址，仅持久化来源。
            model: 请求模型。

        Yields:
            实际计数与版本收集器。
        """
        started = self._clock()
        started_at = self._wall_clock()
        capture = ProviderCallCapture()
        status = "completed"
        try:
            yield capture
        except StreamCancelled:
            status = "cancelled"
            raise
        except BaseException:
            status = "error"
            raise
        finally:
            normalized = normalize_origin(origin)
            price = None
            for candidate in self._prices:
                if normalize_origin(candidate.origin) == normalized:
                    if candidate.model == model:
                        price = candidate
                        break
            call = ModelCallTrace(
                purpose=purpose, origin=normalized, model=model,
                response_model=capture.response_model,
                started_at=started_at,
                duration_ms=max(0.0, (self._clock() - started) * 1000),
                status=status, usage=capture.usage,
                cost=estimate_cost(capture.usage, price),
            )
            with self._lock:
                self._save_preserving_error(
                    status != "completed", calls=(*self._snapshot.calls, call),
                )

    def set_result(
        self, *, lane: str, outcome: str, scope_expanded: bool,
        escalated: bool,
    ) -> None:
        """记录实际最终通道，不把起点或检索策略误当通道。

        Args:
            lane: 最终执行通道。
            outcome: 最终业务结果。
            scope_expanded: 是否扩查。
            escalated: 是否升级。

        Returns:
            None。
        """
        with self._lock:
            self._save(
                lane=lane, outcome=outcome, scope_expanded=scope_expanded,
                escalated=escalated,
            )

    def set_admission(self, decision: AdmissionDecision) -> None:
        """保存控制器报告的真实准入结果。

        Args:
            decision: 不含正文的实际准入结果。

        Returns:
            None。
        """
        with self._lock:
            self._save(
                admission_status=decision.status,
                queue_wait_ms=decision.queue_wait_ms,
            )

    def mark_queued(self) -> None:
        """记录正在等待，尚不知道最终等待耗时。

        Returns:
            None。
        """
        with self._lock:
            self._save(admission_status="queued")

    def finish(
        self, status: Literal["completed", "error", "cancelled"],
    ) -> None:
        """保存执行终止状态，不包含异常原始文案。

        Args:
            status: 实际终止类型。

        Returns:
            None。
        """
        with self._lock:
            execution_duration = None
            for step in self._snapshot.steps:
                if step.type == "execution" and step.status != "running":
                    execution_duration = step.duration_ms
            if self._snapshot.admission_status in {
                "rate_limited", "queue_full", "queue_timeout", "cancelled",
                "error",
            }:
                execution_duration = 0.0
            self._save(
                status=status,
                duration_ms=max(0.0, (self._clock() - self._started) * 1000),
                execution_duration_ms=execution_duration,
            )

    def _save(self, **updates: object) -> None:
        snapshot = TraceSnapshot.model_validate({
            **self._snapshot.model_dump(), **updates,
        })
        self._repository.save(snapshot)
        self._snapshot = snapshot

    def _save_preserving_error(self, error_in_progress: bool, **updates: object) -> None:
        try:
            self._save(**updates)
        except TraceStorageError:
            if not error_in_progress:
                raise
            logger.warning("追踪步骤状态无法写入")

    def _sanitize(self, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
        clean: dict[str, JsonValue] = {}
        for key, value in payload.items():
            normalized = key.casefold()
            if normalized in _PRIVATE_KEYS or normalized.endswith("api_key"):
                clean[key] = None
                continue
            if normalized in _QUERY_KEYS and value is not None:
                serialized = value if isinstance(value, str) else json.dumps(
                    value, ensure_ascii=False, sort_keys=True,
                )
                clean[f"{key}_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
                if not self._capture_queries:
                    clean[key] = None
                    continue
            clean[key] = self._sanitize_value(value)
        return clean

    def _sanitize_value(self, value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            return self._sanitize(value)
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        return value
