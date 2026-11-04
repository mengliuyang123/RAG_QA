# -*- coding: utf-8 -*-
"""普通HTTP查询也传播断连，不能让已离开的排队请求随后调用模型。"""

from collections.abc import Callable
from threading import Event
from typing import TypeVar
from typing import cast

import anyio
from fastapi import HTTPException
from fastapi import Request

from app.application.execution_control import StreamCancelled
from app.application.execution_control import cancellation_scope


_Result = TypeVar("_Result")


async def run_cancellable_query(
    request: Request, call: Callable[[], _Result],
) -> _Result:
    """在线程池执行同步用例，同时在事件循环监听断连。

    Args:
        request: 请求正文已经由FastAPI解析完成的HTTP请求。
        call: 保留原类型与异常的同步查询用例。

    Returns:
        正常返回值；客户端已离开时以499结束，不启动后续模型调用。
    """
    cancelled = Event()
    result: _Result | None = None
    failure: BaseException | None = None

    def check() -> None:
        if cancelled.is_set():
            raise StreamCancelled()

    def execute() -> _Result:
        with cancellation_scope(check):
            return call()

    async def observe_disconnect() -> None:
        while True:
            if await request.is_disconnected():
                cancelled.set()
                return
            await anyio.sleep(0.02)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(observe_disconnect)
        try:
            # 不abandon同步SDK：其未返回前必须继续占用真实执行名额。
            result = await anyio.to_thread.run_sync(execute)
        except BaseException as error:
            failure = error
        finally:
            cancelled.set()
            tasks.cancel_scope.cancel()
    if isinstance(failure, StreamCancelled):
        raise HTTPException(status_code=499, detail="请求已断开")
    if failure is not None:
        raise failure
    return cast(_Result, result)
