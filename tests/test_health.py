# -*- coding: utf-8 -*-
"""FastAPI 应用工厂与健康检查测试。"""

import httpx
import pytest

from app.bootstrap.container import ApplicationContainer
from app.bootstrap.container import build_application_container
from app.bootstrap.factories import AdapterFactories
from app.config import Settings
from tests.http_auth import create_test_app as create_app


@pytest.mark.anyio
async def test_health_endpoint_returns_ok() -> None:
    """健康检查路由被移除或响应异常时，本测试应失败。"""
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_application_lifespan_closes_container_when_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """lifespan body 抛出异常时仍应关闭当前应用容器。

    Args:
        monkeypatch: pytest 提供的属性替换工具。

    Returns:
        None。
    """
    events: list[str] = []
    settings = Settings(_env_file=None, ingestion_worker_enabled=False)
    container = build_application_container(settings)
    original_close = container.close

    def _close_container() -> None:
        events.append("closed")
        original_close()

    def _container_factory(
        _settings: Settings,
    ) -> ApplicationContainer:
        return container

    monkeypatch.setattr(
        container,
        "close",
        _close_container,
    )
    application = create_app(
        settings=settings,
        container_factory=_container_factory,
    )

    with pytest.raises(RuntimeError, match="lifespan body failed"):
        async with application.router.lifespan_context(application):
            assert events == []
            raise RuntimeError("lifespan body failed")

    assert events == ["closed"]


def test_application_exposes_project_metadata() -> None:
    """应用工厂遗漏项目元数据时，本测试应失败。"""
    application = create_app()

    assert application.title == "RAG 知识库问答系统"
    assert application.version == "0.1.0"


@pytest.mark.anyio
async def test_health_during_lifespan_keeps_adapters_lazy_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """health 请求不得构造 adapter，退出 lifespan 应关闭容器。

    Args:
        monkeypatch: pytest 提供的属性替换工具。

    Returns:
        None。
    """
    settings = Settings(_env_file=None, ingestion_worker_enabled=False)
    adapter_factory_calls: list[str] = []
    container_factory_calls: list[Settings] = []
    close_events: list[str] = []

    def _record_adapter_factory(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        adapter_factory_calls.append("called")
        return object()

    factories = AdapterFactories(
        vector_store=_record_adapter_factory,
        embedding_provider=_record_adapter_factory,
        chunker=_record_adapter_factory,
        parser_resolver=_record_adapter_factory,
        keyword_index=_record_adapter_factory,
        repair_log=_record_adapter_factory,
        reranker=_record_adapter_factory,
        llm_provider=_record_adapter_factory,
    )
    container = build_application_container(
        settings,
        factories=factories,
    )

    original_close = container.close

    def _close_container() -> None:
        close_events.append("closed")
        original_close()

    def _container_factory(
        received_settings: Settings,
    ) -> ApplicationContainer:
        container_factory_calls.append(received_settings)
        return container

    monkeypatch.setattr(container, "close", _close_container)
    application = create_app(
        settings=settings,
        container_factory=_container_factory,
    )

    assert application.state.container is container
    assert container_factory_calls == [settings]
    assert adapter_factory_calls == []
    assert close_events == []

    async with application.router.lifespan_context(application):
        assert adapter_factory_calls == []
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert adapter_factory_calls == []
        assert close_events == []

    assert adapter_factory_calls == []
    assert close_events == ["closed"]


@pytest.fixture
def anyio_backend() -> str:
    """API 测试固定使用 asyncio，避免引入额外异步后端。

    Returns:
        AnyIO 使用的 asyncio 后端名称。
    """
    return "asyncio"
