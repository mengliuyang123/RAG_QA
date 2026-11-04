# -*- coding: utf-8 -*-
"""同步用例与实时输出间的单槽、可协作取消的事件通道。"""

from collections.abc import Callable
from collections.abc import Iterator
from contextvars import copy_context
from dataclasses import dataclass
from queue import Empty
from queue import Full
from queue import Queue
from threading import Event
from threading import Lock
from threading import Thread
from typing import Generic
from typing import TypeVar

from app.application.execution_control import StreamCancelled
from app.application.execution_control import cancellation_scope

EventType = TypeVar("EventType")


@dataclass(frozen=True)
class _Delivery(Generic[EventType]):
    value: EventType
    acknowledged: Event


@dataclass(frozen=True)
class _Failure:
    error: BaseException


class EventSink(Generic[EventType]):
    """用阻塞交接施加背压，不把入队误认为消费者已接收。"""

    def __init__(
        self,
        queue: Queue[_Delivery[EventType] | _Failure],
        cancelled: Event,
    ) -> None:
        """保存当前请求独有的通道与取消信号。

        Args:
            queue: 容量固定为一的通道。
            cancelled: 关闭请求时设定的信号。

        Returns:
            None。
        """
        self._queue = queue
        self._cancelled = cancelled

    def check(self) -> None:
        """在外部调用、提交或发布之前检查取消。

        Returns:
            None。

        Raises:
            StreamCancelled: 当前请求已经关闭。
        """
        if self._cancelled.is_set():
            raise StreamCancelled()

    def emit(self, value: EventType) -> None:
        """交付一条事件，并等待消费者请求下一条才继续业务执行。

        Args:
            value: 已过滤且可对外发布的类型化事件。

        Returns:
            None。
        """
        delivery = _Delivery(value, Event())
        self._put(delivery)
        while not delivery.acknowledged.wait(timeout=0.02):
            self.check()
        self.check()

    def _put(self, delivery: _Delivery[EventType] | _Failure) -> None:
        while True:
            self.check()
            try:
                self._queue.put(delivery, timeout=0.02)
                return
            except Full:
                continue


class LiveEventStream(Iterator[EventType]):
    """每请求独立工作线程，关闭时通知取消，不强杀同步SDK调用。

    Attributes:
        poll: 为HTTP异步适配提供非阻塞消费，避免占用等待中的线程池槽位。
    """

    def __init__(self, producer: Callable[[EventSink[EventType]], None]) -> None:
        """保存用例，首次消费才启动，未使用的流没有线程副作用。

        Args:
            producer: 拥有会话锁和资源清理责任的同步生产用例。

        Returns:
            None。
        """
        self._producer = producer
        self._queue: Queue[_Delivery[EventType] | _Failure] = Queue(maxsize=1)
        self._cancelled = Event()
        self._finished = Event()
        self._start_lock = Lock()
        self._started = False
        self._previous: _Delivery[EventType] | None = None

    def __iter__(self) -> "LiveEventStream[EventType]":
        """返回本流。

        Returns:
            当前可关闭迭代器。
        """
        return self

    def __next__(self) -> EventType:
        """同步消费者等待下一条事件。

        Returns:
            下一条公开事件。

        Raises:
            StopIteration: 生产结束或流被关闭。
        """
        while True:
            result = self.poll(wait_seconds=0.02)
            if result is not None:
                return result

    def poll(self, wait_seconds: float = 0.0) -> EventType | None:
        """确认上一条交付，并在有限时间内读取新事件。

        Args:
            wait_seconds: 最长等待；HTTP异步适配使用零值。

        Returns:
            公开事件；暂时无事件时为None。

        Raises:
            StopIteration: 已关闭，或生产结束且事件已消费完毕。
        """
        if self._cancelled.is_set():
            raise StopIteration
        self._start()
        if self._previous is not None:
            self._previous.acknowledged.set()
            self._previous = None
        # 必须先读完成状态，再检查队列；反序会漏掉刚好在Empty之后入队的错误。
        finished_before_read = self._finished.is_set()
        try:
            delivery = self._queue.get(timeout=wait_seconds)
        except Empty:
            if finished_before_read:
                raise StopIteration
            return None
        if isinstance(delivery, _Failure):
            raise delivery.error
        self._previous = delivery
        return delivery.value

    def close(self) -> None:
        """线程安全地取消后续工作，不等待正在进行的同步上游调用。

        Returns:
            None。
        """
        self._cancelled.set()

    def _start(self) -> None:
        with self._start_lock:
            if self._started or self._cancelled.is_set():
                return
            self._started = True
            context = copy_context()
            Thread(
                target=context.run, args=(self._run,),
                name="rag-stream", daemon=True,
            ).start()

    def _run(self) -> None:
        sink = EventSink(self._queue, self._cancelled)
        try:
            with cancellation_scope(sink.check):
                self._producer(sink)
        except StreamCancelled:
            pass
        except BaseException as error:
            try:
                sink._put(_Failure(error))
            except StreamCancelled:
                pass
        finally:
            self._finished.set()
