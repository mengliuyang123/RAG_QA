# -*- coding: utf-8 -*-
"""模型、质量调用和整轮超时须保留可识别的边界。"""

from pathlib import Path
from uuid import UUID
from uuid import uuid4

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from openai import APITimeoutError

from app.application.errors import ConversationTimeoutError
from app.interfaces.http.dependencies import get_conversation_service
from tests.http_auth import create_test_app as create_app
from app.infrastructure.conversations.langgraph_agent import (
    LangGraphConversationAgent,
)
from tests.test_conversation_service import _FakeAgent
from tests.test_conversation_service import _service


class _TimeoutModel:
    def bind_tools(self, tools: object, *, strict: bool) -> "_TimeoutModel":
        """保持 Agent 所需的工具模型接口。

        Args:
            tools: 图注册的工具定义。
            strict: 工具参数约束。

        Returns:
            当前测试模型。
        """
        return self

    def invoke(self, messages: object) -> object:
        """在真实图节点处模拟 SDK 超时。

        Args:
            messages: 图传入的完整消息。

        Returns:
            不返回，始终抛出 SDK 超时。
        """
        raise APITimeoutError(request=httpx.Request("POST", "http://offline"))


def test_agent_preserves_sdk_timeout_category() -> None:
    """SDK 超时不能被普通 Agent 错误吞掉。"""
    agent = LangGraphConversationAgent(
        model=_TimeoutModel(), retrieval_service=object(),
        checkpointer=InMemorySaver(), top_k=1,
    )
    with pytest.raises(ConversationTimeoutError):
        agent.run(uuid4(), "问题", [])


@pytest.mark.anyio
async def test_conversation_http_timeout_is_504() -> None:
    """HTTP 只公开稳定的超时消息，不泄露上游地址。"""
    class FailingService:
        """模拟执行边界上的超时。"""

        def create(self, *, owner_id: UUID | None = None) -> None:
            """抛出需要映射的超时异常。"""
            raise ConversationTimeoutError("private upstream")

    app = create_app()
    app.dependency_overrides[get_conversation_service] = FailingService
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.post("/api/conversations")
        assert response.status_code == 504
        assert response.json()["detail"] == "会话处理超时，请稍后重试"
    finally:
        app.state.container.close()


def test_stream_timeout_does_not_save_a_successful_turn(tmp_path: Path) -> None:
    """流内超时有唯一终止错误，历史与锁均不残留成功状态。

    Args:
        tmp_path: 隔离 SQLite 的目录。
    """
    service = _service(
        tmp_path, _FakeAgent(error=ConversationTimeoutError("private")),
    )
    conversation = service.create()
    events = list(service.stream_query(conversation.id, "问题"))
    assert [event.event for event in events] == ["status", "error"]
    assert events[-1].data.model_dump() == {
        "code": "timeout", "message": "会话处理超时，请稍后重试",
    }
    assert service.get(conversation.id).turns_count == 0
    assert not service._conversation_locks


@pytest.fixture
def anyio_backend() -> str:
    """只使用已安装的 asyncio 后端。

    Returns:
        异步后端名称。
    """
    return "asyncio"
