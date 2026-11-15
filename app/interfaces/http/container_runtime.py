# -*- coding: utf-8 -*-
"""应用容器的热替换：配置变更后切换到新容器，旧容器在请求结束后关闭。

每个 HTTP 请求开始时"租用"当时的容器并固定到 ``request.state.container``，
请求（包括流式响应）结束才归还。替换容器只影响之后到达的请求；
旧容器在最后一个租约归还后，于后台线程中关闭。
"""

import logging
import threading
from collections.abc import Callable

from fastapi import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from app.bootstrap.container import ApplicationContainer
from app.config import Settings

logger = logging.getLogger(__name__)

ContainerFactory = Callable[[Settings], ApplicationContainer]


class ContainerRuntime:
    """持有当前容器、请求租约和待关闭的旧容器。"""

    def __init__(self, state: object, factory: ContainerFactory) -> None:
        """绑定 FastAPI ``app.state``；当前容器始终存放在 ``state.container``。

        Args:
            state: FastAPI 应用的 state 对象。
            factory: 根据配置创建新容器的工厂。

        Returns:
            None。
        """
        self._state = state
        self._factory = factory
        self._lock = threading.Lock()
        self._leases: dict[int, int] = {}
        self._retiring: dict[int, ApplicationContainer] = {}
        self._worker_started = False
        # 配置更新串行执行，避免两次保存交错写 .env、交错替换容器。
        self.update_lock = threading.Lock()

    @property
    def current(self) -> ApplicationContainer:
        """返回新请求将使用的容器。"""
        return self._state.container  # type: ignore[attr-defined]

    def acquire(self) -> ApplicationContainer:
        """为一个请求租用当前容器。"""
        with self._lock:
            container = self.current
            self._leases[id(container)] = self._leases.get(id(container), 0) + 1
            return container

    def release(self, container: ApplicationContainer) -> None:
        """归还租约；旧容器的最后一个租约归还时关闭它。"""
        to_close: ApplicationContainer | None = None
        with self._lock:
            remaining = self._leases.get(id(container), 0) - 1
            if remaining > 0:
                self._leases[id(container)] = remaining
            else:
                self._leases.pop(id(container), None)
                to_close = self._retiring.pop(id(container), None)
        if to_close is not None:
            self._close_in_background(to_close)

    def build(self, settings: Settings) -> ApplicationContainer:
        """用新配置创建容器；容器是惰性的，构造本身不连接外部服务。"""
        return self._factory(settings)

    def start_ingestion_worker(self) -> None:
        """由 HTTP 生命周期调用；之后替换上来的容器也会启动入库消费者。"""
        self._worker_started = True
        self.current.start_ingestion_worker()

    def swap(self, container: ApplicationContainer) -> None:
        """让之后的请求使用新容器，并安排旧容器关闭。

        Args:
            container: 已用新配置创建的容器。

        Returns:
            None。
        """
        with self._lock:
            old = self.current
            self._state.container = container  # type: ignore[attr-defined]
            close_now = self._leases.get(id(old), 0) == 0
            if not close_now:
                self._retiring[id(old)] = old
        if self._worker_started:
            container.start_ingestion_worker()
        if close_now:
            self._close_in_background(old)

    def close_all(self) -> None:
        """应用退出时同步关闭当前容器和所有尚未关闭的旧容器。"""
        with self._lock:
            containers = [self.current, *self._retiring.values()]
            self._retiring.clear()
        for container in containers:
            try:
                container.close()
            except Exception:
                logger.exception("关闭应用容器失败")

    @staticmethod
    def _close_in_background(container: ApplicationContainer) -> None:
        def close() -> None:
            try:
                container.close()
            except Exception:
                logger.exception("关闭被替换的应用容器失败")

        threading.Thread(target=close, name="retire-container", daemon=True).start()


class ContainerLeaseMiddleware:
    """为每个 HTTP 请求固定一个容器，直到响应完全发送结束。"""

    def __init__(self, app: ASGIApp, runtime: ContainerRuntime) -> None:
        """包装下游 ASGI 应用。"""
        self.app = app
        self.runtime = runtime

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """租用容器、写入 ``scope["state"]``，请求结束后归还。"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        container = self.runtime.acquire()
        scope.setdefault("state", {})["container"] = container
        try:
            await self.app(scope, receive, send)
        finally:
            self.runtime.release(container)


def current_container(request: Request) -> ApplicationContainer:
    """返回本请求固定使用的容器。

    没有经过租约中间件时（例如测试直接调用依赖），退回 ``app.state.container``。
    """
    container = getattr(request.state, "container", None)
    if container is None:
        container = request.app.state.container
    return container
