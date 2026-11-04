# -*- coding: utf-8 -*-
"""单应用实例共享的个人滑窗及有界FIFO问答准入，不保存问题正文。"""

from collections import deque
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from contextlib import nullcontext
from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import ceil
from math import isfinite
from threading import Condition
from time import monotonic
from uuid import UUID
from uuid import uuid4

from app.application.errors import QuestionQueueFullError
from app.application.errors import QuestionQueueTimeoutError
from app.application.errors import QuestionRateLimitError
from app.application.execution_control import check_cancellation
from app.application.execution_control import StreamCancelled
from app.domain.admission import AdmissionDecision
from app.domain.admission import AdmissionStatus
from app.domain.admission import QuestionQuota


@dataclass
class _Ticket:
    identity: UUID
    deadline: float
    queued_at: float
    expired: bool = False


class QuestionAdmissionController:
    """共享active容量与等待队列；配额按接收计数，不按成功答案计数。"""

    def __init__(
        self, *, per_user_limit: int = 10, window_seconds: int = 60,
        max_active: int = 3, max_waiting: int = 20,
        queue_timeout_seconds: float = 60,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """保存有界配置，构造时不启动线程或打开存储。

        Args:
            per_user_limit: 每个用户滑动窗口内的接收上限。
            window_seconds: 滑动窗口长度。
            max_active: 同时执行问答链路的上限。
            max_waiting: 等待名额数，0表示不排队。
            queue_timeout_seconds: 独立的最大等待秒数。
            clock: 不受系统时间调整影响的单调时钟。

        Returns:
            None。
        """
        for value in (per_user_limit, window_seconds, max_active):
            if type(value) is not int or value <= 0:
                raise ValueError("问答额度、窗口和执行容量必须为正整数")
        if type(max_waiting) is not int or max_waiting < 0:
            raise ValueError("等待容量必须为非负整数")
        if not isfinite(queue_timeout_seconds) or queue_timeout_seconds <= 0:
            raise ValueError("排队超时必须为有限正数")
        self._limit = per_user_limit
        self._window = window_seconds
        self._max_active = max_active
        self._max_waiting = max_waiting
        self._queue_timeout = queue_timeout_seconds
        self._clock = clock
        self._condition = Condition()
        self._active = 0
        self._waiting: deque[_Ticket] = deque()
        self._accepted: deque[tuple[float, UUID]] = deque()
        self._users: dict[UUID, deque[float]] = {}
        self._closed = False

    @contextmanager
    def acquire(
        self, user_id: UUID, *,
        on_queued: Callable[[int], None] | None = None,
        on_decision: Callable[[AdmissionDecision], None] | None = None,
    ) -> Iterator[None]:
        """按用户窗口接收，再FIFO等待；异常和取消均释放持有的资源。

        Args:
            user_id: 认证层提供的实际身份。
            on_queued: 安全位置变化回调，0表示没有其他等待者排在前面。
            on_decision: 锁外报告一次实际结果，不能在模型执行异常后重复报告。

        Yields:
            实际可执行模型的区间，等待阶段不会进入此区间。
        """
        if not isinstance(user_id, UUID):
            raise ValueError("问答准入必须使用服务器UUID身份")
        ticket = None
        active = False
        decision_reported = False

        def report(status: AdmissionStatus) -> None:
            nonlocal decision_reported
            decision_reported = True
            if on_decision is not None:
                waiting = 0.0
                if ticket is not None:
                    waiting = max(0.0, (self._clock() - ticket.queued_at) * 1000)
                on_decision(AdmissionDecision(
                    status=status, queued=ticket is not None,
                    queue_wait_ms=waiting,
                ))

        try:
            check_cancellation()
            with self._condition:
                now = self._clock()
                self._prune(now)
                self._expire_waiters(now)
                if self._closed:
                    raise QuestionQueueFullError("系统暂不接收新问题")
                quota = self._quota(user_id, now)
                if quota.remaining == 0:
                    raise QuestionRateLimitError(quota.retry_after)
                if self._active < self._max_active and not self._waiting:
                    self._active += 1
                    active = True
                else:
                    if len(self._waiting) >= self._max_waiting:
                        raise QuestionQueueFullError("问答排队容量已满")
                    ticket = _Ticket(uuid4(), now + self._queue_timeout, now)
                    self._waiting.append(ticket)
                self._accepted.append((now, user_id))
                self._users.setdefault(user_id, deque()).append(now)
            if ticket is not None:
                previous_position = None
                while True:
                    check_cancellation()
                    with self._condition:
                        now = self._clock()
                        self._expire_waiters(now)
                        if self._closed:
                            raise QuestionQueueFullError(
                                "系统暂不接收新问题", was_queued=True,
                            )
                        if ticket.expired:
                            raise QuestionQueueTimeoutError("问答排队超时")
                        if self._waiting[0] is ticket:
                            if self._active < self._max_active:
                                self._waiting.popleft()
                                self._active += 1
                                active = True
                                break
                        position = self._waiting.index(ticket)
                    if position != previous_position and on_queued is not None:
                        on_queued(position)
                    previous_position = position
                    with self._condition:
                        remaining = ticket.deadline - self._clock()
                        if remaining > 0:
                            self._condition.wait(min(0.05, remaining))
            check_cancellation()
            report("admitted")
            yield
        except BaseException as error:
            if not decision_reported:
                status: AdmissionStatus = "error"
                if isinstance(error, QuestionRateLimitError):
                    status = "rate_limited"
                elif isinstance(error, QuestionQueueFullError):
                    status = "queue_full"
                elif isinstance(error, QuestionQueueTimeoutError):
                    status = "queue_timeout"
                elif isinstance(error, StreamCancelled):
                    status = "cancelled"
                try:
                    report(status)
                except Exception:
                    # 存储观察者不能覆盖已经确定的限流、超时或取消语义。
                    pass
            raise
        finally:
            with self._condition:
                if active:
                    self._active -= 1
                elif ticket is not None and ticket in self._waiting:
                    self._waiting.remove(ticket)
                self._condition.notify_all()

    def quota(self, user_id: UUID) -> QuestionQuota:
        """读取实际窗口余额，不消耗额度或创建没有提问的身份记录。

        Args:
            user_id: 已验证用户。

        Returns:
            当前滑动窗口的只读余额。
        """
        with self._condition:
            now = self._clock()
            self._prune(now)
            return self._quota(user_id, now)

    def close(self) -> None:
        """停止接收并唤醒等待者，不把排队请求推进模型执行。

        Returns:
            None。
        """
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _quota(self, user_id: UUID, now: float) -> QuestionQuota:
        accepted = self._users.get(user_id)
        used = len(accepted) if accepted is not None else 0
        remaining = max(0, self._limit - used)
        retry_after = 0
        if remaining == 0 and accepted is not None:
            retry_after = max(1, ceil(accepted[0] + self._window - now))
        return QuestionQuota(
            limit=self._limit, remaining=remaining,
            window_seconds=self._window, retry_after=retry_after,
        )

    def _prune(self, now: float) -> None:
        while self._accepted and self._accepted[0][0] <= now - self._window:
            _, user_id = self._accepted.popleft()
            accepted = self._users[user_id]
            accepted.popleft()
            if not accepted:
                del self._users[user_id]

    def _expire_waiters(self, now: float) -> None:
        while self._waiting and self._waiting[0].deadline <= now:
            ticket = self._waiting.popleft()
            ticket.expired = True


def question_slot(
    admission: QuestionAdmissionController | None, user_id: UUID | None, *,
    on_queued: Callable[[int], None] | None = None,
    on_decision: Callable[[AdmissionDecision], None] | None = None,
) -> AbstractContextManager[None]:
    """线上已认证调用共享准入，可信无身份内部调用保持离线兼容。

    Args:
        admission: 组合根提供的同一准入组件。
        user_id: 服务端身份，None不通过HTTP开放。
        on_queued: 当前请求的位置输出回调。
        on_decision: 实际准入结果的观察者。

    Returns:
        实际执行名额的上下文。
    """
    if admission is None or user_id is None:
        return nullcontext()
    return admission.acquire(
        user_id, on_queued=on_queued, on_decision=on_decision,
    )
