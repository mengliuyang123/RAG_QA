# -*- coding: utf-8 -*-
"""个人滑窗、FIFO容量与取消的真实线程行为，不调用模型或网络。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest

from app.application.admission import QuestionAdmissionController
from app.application.errors import QuestionQueueFullError
from app.application.errors import QuestionQueueTimeoutError
from app.application.errors import QuestionRateLimitError
from app.application.execution_control import StreamCancelled
from app.application.execution_control import cancellation_scope


class Clock:
    """可推进的单调时钟，测试不需要等待一分钟。"""

    def __init__(self) -> None:
        """设置固定起点。

        Returns:
            None。
        """
        self.now = 100.0

    def __call__(self) -> float:
        """返回测试时间。

        Returns:
            单调秒数。
        """
        return self.now


def test_user_sliding_window_and_retry_after_are_actual_accepted_questions() -> None:
    """窗口如果按固定分钟清零、失败退回或用户间共享都会给错额度。
    """
    clock = Clock()
    gate = QuestionAdmissionController(per_user_limit=2, clock=clock)
    first = uuid4()
    second = uuid4()
    with gate.acquire(first):
        pass
    clock.now += 0.1
    with pytest.raises(RuntimeError):
        with gate.acquire(first):
            raise RuntimeError("generation failed")
    with pytest.raises(QuestionRateLimitError) as caught:
        with gate.acquire(first):
            pytest.fail("额度耗尽不能执行")
    assert caught.value.retry_after == 60
    assert gate.quota(first).remaining == 0
    with gate.acquire(second):
        pass
    assert gate.quota(second).remaining == 1
    clock.now = 160.0
    assert gate.quota(first).remaining == 1
    with gate.acquire(first):
        pass
    assert gate.quota(first).remaining == 0
    clock.now = 220.0
    assert gate.quota(first).remaining == 2


def test_fifo_queue_does_not_let_new_arrival_jump_ahead() -> None:
    """释放名额时若新请求插队，旧等待者次序和输出会错误。
    """
    gate = QuestionAdmissionController(max_active=1, max_waiting=2)
    queued_first = Event()
    queued_second = Event()
    entered = []
    positions = []

    def run(label: str, queued: Event) -> None:
        def report(position: int) -> None:
            positions.append((label, position))
            queued.set()

        with gate.acquire(uuid4(), on_queued=report):
            entered.append(label)

    with ThreadPoolExecutor(max_workers=2) as workers:
        with gate.acquire(uuid4()):
            first = workers.submit(run, "first", queued_first)
            assert queued_first.wait(1)
            second = workers.submit(run, "second", queued_second)
            assert queued_second.wait(1)
            assert entered == []
            with pytest.raises(QuestionQueueFullError):
                with gate.acquire(uuid4()):
                    pytest.fail("满队列不能插队")
        first.result(timeout=1)
        second.result(timeout=1)
    assert entered == ["first", "second"]
    assert ("first", 0) in positions
    assert ("second", 1) in positions


def test_queue_full_does_not_charge_user_but_accepted_timeout_does() -> None:
    """未接收请求不应消耗额度，已接受的排队请求则在超时后仍计数。
    """
    clock = Clock()
    gate = QuestionAdmissionController(
        per_user_limit=2, max_active=1, max_waiting=1,
        queue_timeout_seconds=1, clock=clock,
    )
    waiting = Event()
    first = uuid4()
    rejected = uuid4()

    def wait_for_slot() -> None:
        with gate.acquire(first, on_queued=lambda position: waiting.set()):
            pytest.fail("截止时间过后不应进入执行")

    with ThreadPoolExecutor(max_workers=1) as workers:
        with gate.acquire(uuid4()):
            future = workers.submit(wait_for_slot)
            assert waiting.wait(1)
            with pytest.raises(QuestionQueueFullError):
                with gate.acquire(rejected):
                    pytest.fail("满队列不能接收")
            assert gate.quota(rejected).remaining == 2
            clock.now += 1
            with pytest.raises(QuestionQueueTimeoutError):
                future.result(timeout=1)
    assert gate.quota(first).remaining == 1


def test_cancelled_waiter_leaves_no_queue_slot_or_active_permit() -> None:
    """断连若不移除ticket，后续正常请求会被假满或永远无法执行。
    """
    gate = QuestionAdmissionController(max_active=1, max_waiting=1)
    queued = Event()
    cancelled = Event()

    def check() -> None:
        if cancelled.is_set():
            raise StreamCancelled()

    def waiting() -> None:
        with cancellation_scope(check):
            with gate.acquire(uuid4(), on_queued=lambda position: queued.set()):
                pytest.fail("已取消请求不能进入执行")

    with ThreadPoolExecutor(max_workers=1) as workers:
        with gate.acquire(uuid4()):
            future = workers.submit(waiting)
            assert queued.wait(1)
            cancelled.set()
            with pytest.raises(StreamCancelled):
                future.result(timeout=1)
        with gate.acquire(uuid4()):
            pass


def test_model_or_status_callback_exception_releases_capacity() -> None:
    """名额释放必须走finally，阶段回调失败也不能遗留等待者。
    """
    gate = QuestionAdmissionController(max_active=1, max_waiting=1)
    with pytest.raises(RuntimeError):
        with gate.acquire(uuid4()):
            raise RuntimeError("model failed")

    def broken(position: int) -> None:
        raise RuntimeError("status delivery failed")

    with gate.acquire(uuid4()):
        with pytest.raises(RuntimeError):
            with gate.acquire(uuid4(), on_queued=broken):
                pytest.fail("回调失败不能执行")
    with gate.acquire(uuid4()):
        pass


def test_close_wakes_waiters_without_starting_them() -> None:
    """生命周期关闭不能让排队请求随后进入模型。
    """
    gate = QuestionAdmissionController(max_active=1)
    queued = Event()

    def waiting() -> None:
        with gate.acquire(uuid4(), on_queued=lambda position: queued.set()):
            pytest.fail("关闭后不得执行")

    with ThreadPoolExecutor(max_workers=1) as workers:
        with gate.acquire(uuid4()):
            future = workers.submit(waiting)
            assert queued.wait(1)
            gate.close()
            with pytest.raises(QuestionQueueFullError):
                future.result(timeout=1)


def test_zero_wait_capacity_still_allows_available_execution_slot() -> None:
    """禁用排队不等于禁用执行；忙时才立即拒绝。
    """
    gate = QuestionAdmissionController(max_active=1, max_waiting=0)
    with gate.acquire(uuid4()):
        with pytest.raises(QuestionQueueFullError):
            with gate.acquire(uuid4()):
                pytest.fail("无排队名额不得等待")
