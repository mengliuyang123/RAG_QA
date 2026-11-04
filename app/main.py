# -*- coding: utf-8 -*-
"""FastAPI 应用工厂与容器生命周期。"""

from collections.abc import AsyncIterator
from collections.abc import Callable
from contextlib import asynccontextmanager
import logging

import anyio
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.bootstrap.container import ApplicationContainer
from app.bootstrap.container import build_application_container
from app.config import Settings
from app.config import get_settings
from app.interfaces.http.error_handlers import register_error_handlers
from app.interfaces.http.router import router
from app.utils.logger import configure_logging
from app.application.errors import TraceStorageError


ContainerFactory = Callable[[Settings], ApplicationContainer]
logger = logging.getLogger(__name__)


async def _maintain_traces(
    container: ApplicationContainer, interval_seconds: float,
) -> None:
    # 不在启动时打开数据库，保留health与无请求场景的惰性装配行为。
    while True:
        await anyio.sleep(interval_seconds)
        try:
            await anyio.to_thread.run_sync(container.prune_expired_traces)
        except TraceStorageError:
            logger.warning("追踪过期清理暂时不可用，将在下个周期重试")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """在应用停止时关闭当前应用容器。

    Args:
        application: 持有本次运行容器的 FastAPI 应用。

    Yields:
        应用处理请求期间的生命周期控制权。
    """
    try:
        application.state.container.start_ingestion_worker()
        failure: BaseException | None = None
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(
                _maintain_traces, application.state.container,
                application.state.trace_cleanup_interval_seconds,
            )
            try:
                yield
            except BaseException as error:
                # 维护任务不应把应用主体的原始错误包装为ExceptionGroup。
                failure = error
            finally:
                tasks.cancel_scope.cancel()
        if failure is not None:
            raise failure
    finally:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(application.state.container.close)


def create_app(
    settings: Settings | None = None,
    container_factory: ContainerFactory = build_application_container,
) -> FastAPI:
    """创建并装配持有惰性应用容器的 FastAPI 应用。

    Args:
        settings: 可选的显式应用配置；省略时读取进程配置。
        container_factory: 从配置创建惰性应用容器的
            可替换工厂。

    Returns:
        已注册路由、异常处理器和生命周期的 FastAPI 应用。
    """
    application_settings = settings
    if application_settings is None:
        application_settings = get_settings()
    configure_logging(application_settings.log_level)
    container = container_factory(application_settings)

    application = FastAPI(
        title="RAG 知识库问答系统",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.container = container
    application.state.trace_cleanup_interval_seconds = (
        application_settings.trace_cleanup_interval_seconds
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(application_settings.cors_allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "X-CSRF-Token"],
        expose_headers=["Retry-After"],
    )
    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(application_settings.http_allowed_hosts),
    )
    register_error_handlers(application)
    application.include_router(router)
    return application


app = create_app()
