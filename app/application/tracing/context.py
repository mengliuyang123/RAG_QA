# -*- coding: utf-8 -*-
"""使适配器和图节点共享请求追踪，不向内层导入具体存储。"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import JsonValue

from app.application.tracing.recorder import ProviderCallCapture
from app.application.tracing.recorder import StepCapture
from app.application.tracing.recorder import TraceRecorder


_CURRENT_TRACE: ContextVar[TraceRecorder | None] = ContextVar(
    "rag_trace_recorder", default=None,
)


@contextmanager
def bind_trace(recorder: TraceRecorder) -> Iterator[None]:
    """为本次执行绑定上下文，并保证退出后恢复。

    Args:
        recorder: 本请求记录器。

    Yields:
        追踪有效的执行区间。
    """
    token = _CURRENT_TRACE.set(recorder)
    try:
        yield
    finally:
        _CURRENT_TRACE.reset(token)


def current_trace() -> TraceRecorder | None:
    """读取当前请求记录器，无追踪运行时不创建任何存储。

    Returns:
        当前记录器或None。
    """
    return _CURRENT_TRACE.get()


@contextmanager
def traced_step(
    kind: str, input: dict[str, JsonValue] | None = None,
) -> Iterator[StepCapture]:
    """记录实际执行区间，没有启用追踪时保持用例行为不变。

    Args:
        kind: 步骤类型。
        input: 待过滤的输入元数据。

    Yields:
        当前步骤的结果收集器。
    """
    recorder = current_trace()
    if recorder is None:
        yield StepCapture()
    else:
        with recorder.step(kind, input) as capture:
            yield capture


@contextmanager
def provider_call(
    purpose: str, origin: str, model: str,
) -> Iterator[ProviderCallCapture]:
    """记录一次真实上游调用，不主动发请求、不估算Token。

    Args:
        purpose: 调用用途。
        origin: 实际API地址。
        model: 请求模型。

    Yields:
        用量和响应模型的收集器。
    """
    recorder = current_trace()
    if recorder is None:
        yield ProviderCallCapture()
    else:
        with recorder.provider_call(purpose, origin, model) as capture:
            yield capture
