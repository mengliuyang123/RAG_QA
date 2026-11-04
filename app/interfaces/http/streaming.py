# -*- coding: utf-8 -*-
"""异步发送应用事件，避免同步next等待妨碍断连清理。"""

from collections.abc import AsyncIterator
from collections.abc import Iterator
from typing import Generic
from typing import TypeVar

import anyio
from starlette.concurrency import iterate_in_threadpool

from app.application.streaming import LiveEventStream


EventType = TypeVar("EventType")


class PreparedEventStream(Generic[EventType]):
    """持有响应头前已预取的首事件及请求结束清理责任。"""

    def __init__(self, first: EventType, iterator: Iterator[EventType]) -> None:
        """保存首事件，不重复启动或推进应用流。

        Args:
            first: 通过HTTP前置异常检查后的首事件。
            iterator: 当前请求拥有的应用迭代器。

        Returns:
            None。
        """
        self._first = first
        self._iterator = iterator

    async def events(self) -> AsyncIterator[EventType]:
        """非阻塞消费生产通道，并在取消时立即通知同步工作线程。

        Yields:
            保持原顺序的类型化应用事件。
        """
        try:
            yield self._first
            if isinstance(self._iterator, LiveEventStream):
                while True:
                    try:
                        event = self._iterator.poll()
                    except StopIteration:
                        return
                    if event is None:
                        await anyio.sleep(0.005)
                    else:
                        yield event
            else:
                # 保持旧端口替身/原始评估流兼容，不为生产入口增加旁路。
                async for event in iterate_in_threadpool(self._iterator):
                    yield event
        finally:
            self.close()

    def close(self) -> None:
        """幂等转交请求清理；生产通道的close可跨线程调用。

        Returns:
            None。
        """
        close = getattr(self._iterator, "close", None)
        if callable(close):
            close()
