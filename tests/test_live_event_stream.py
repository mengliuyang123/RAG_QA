# -*- coding: utf-8 -*-
"""有界事件通道必须在慢消费时施加背压，并能跨线程安全关闭。"""

from threading import Event
from queue import Empty

import pytest

from app.application.streaming import EventSink
from app.application.streaming import LiveEventStream


def test_slow_consumer_blocks_producer_before_following_side_effect() -> None:
    """仅放进队列不能当作已交付，否则会在末尾token被读取前提交。"""
    side_effect = Event()
    stopped = Event()

    def produce(sink: EventSink[str]) -> None:
        try:
            sink.emit("validated token")
            side_effect.set()
        finally:
            stopped.set()

    stream = LiveEventStream(produce)
    assert next(stream) == "validated token"
    assert not side_effect.wait(timeout=0.05)
    stream.close()
    assert stopped.wait(timeout=0.5)
    assert not side_effect.is_set()
    with pytest.raises(StopIteration):
        next(stream)


def test_requesting_next_event_acknowledges_previous_delivery() -> None:
    """正常消费能继续生产，并确定性结束，不依靠队列empty推断成功。"""
    side_effect = Event()

    def produce(sink: EventSink[str]) -> None:
        sink.emit("one")
        side_effect.set()
        sink.emit("two")

    stream = LiveEventStream(produce)
    assert list(stream) == ["one", "two"]
    assert side_effect.is_set()


def test_preflight_failure_preserves_original_error() -> None:
    """响应头前的范围或会话错误仍须交给HTTP层映射。"""
    def produce(sink: EventSink[str]) -> None:
        raise ValueError("scope")

    stream = LiveEventStream(produce)
    with pytest.raises(ValueError, match="scope"):
        next(stream)
    stream.close()


def test_failure_enqueued_during_empty_poll_is_not_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty和finished之间的竞争不能把已入队错误误判成正常结束。

    Args:
        monkeypatch: 只控制队列读与生产结束的竞态时序。
    """
    release = Event()

    def produce(sink: EventSink[str]) -> None:
        assert release.wait(timeout=0.5)
        raise ValueError("preflight failed")

    stream = LiveEventStream(produce)
    original_get = stream._queue.get
    first_poll = True

    def racing_get(*args: object, **kwargs: object):
        nonlocal first_poll
        if first_poll:
            first_poll = False
            try:
                original_get(block=False)
            except Empty:
                release.set()
                assert stream._finished.wait(timeout=0.5)
                raise
        return original_get(*args, **kwargs)

    monkeypatch.setattr(stream._queue, "get", racing_get)
    assert stream.poll() is None
    with pytest.raises(ValueError, match="preflight failed"):
        next(stream)
    stream.close()
