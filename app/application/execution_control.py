# -*- coding: utf-8 -*-
"""在同步调用和图节点中传播请求级协作取消，不引入HTTP依赖。"""

from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from app.application.errors import ConversationTimeoutError


class StreamCancelled(BaseException):
    """表示断连取消，不得被上游Exception兜底转换成补查或重试。"""


_CANCELLATION_CHECK: ContextVar[Callable[[], None] | None] = ContextVar(
    "rag_cancellation_check", default=None,
)
_DEADLINES: ContextVar[tuple[tuple[float, Callable[[], float]], ...]] = (
    ContextVar("rag_execution_deadlines", default=())
)


@contextmanager
def deadline_scope(
    deadline: float, clock: Callable[[], float],
) -> Iterator[None]:
    """让嵌套外部操作沿用整轮截止时间，不覆盖已有取消检查。

    Args:
        deadline: 原预算中的绝对单调截止时间。
        clock: 与该预算相同的单调时钟。

    Yields:
        所有边界检查均遵守该截止时间的执行区间。
    """
    token = _DEADLINES.set((*_DEADLINES.get(), (deadline, clock)))
    try:
        yield
    finally:
        _DEADLINES.reset(token)


@contextmanager
def cancellation_scope(check: Callable[[], None]) -> Iterator[None]:
    """绑定当前请求的检查函数，不影响并行请求或后续线程复用。

    Args:
        check: 请求拥有的取消检查。

    Yields:
        继承取消上下文的同步执行区间。
    """
    token = _CANCELLATION_CHECK.set(check)
    try:
        yield
    finally:
        _CANCELLATION_CHECK.reset(token)


def check_cancellation() -> None:
    """在操作边界检查断连和继承的截止时间，无上下文时不受影响。

    Returns:
        None。
    """
    check = _CANCELLATION_CHECK.get()
    if check is not None:
        check()
    for deadline, clock in _DEADLINES.get():
        if clock() >= deadline:
            raise ConversationTimeoutError("整轮时间预算已耗尽")
