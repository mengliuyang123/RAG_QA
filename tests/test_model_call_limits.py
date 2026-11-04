# -*- coding: utf-8 -*-
"""验证显式 SDK 边界、服务端策略和 Agent 整轮预算。"""

from uuid import uuid4
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.application.errors import ConversationAgentError
from app.bootstrap.factories import configured_adapter_factories
from app.config import Settings
from app.domain.errors import PromptContextTooLargeError
from app.domain.retrieval import RetrievalStrategy
from app.infrastructure.conversations.langgraph_agent import (
    LangGraphConversationAgent,
)
from app.infrastructure.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from app.infrastructure.generation.openai_compatible import (
    OpenAICompatibleLLMProvider,
)


@pytest.mark.parametrize("provider_type,expected", [
    (OpenAICompatibleLLMProvider, 60.0),
    (OpenAICompatibleEmbeddingProvider, 30.0),
])
def test_online_provider_defaults_are_finite(
    provider_type: (
        type[OpenAICompatibleLLMProvider]
        | type[OpenAICompatibleEmbeddingProvider]
    ),
    expected: float,
) -> None:
    """在线 provider 不得依赖 SDK 长等待和隐式重试。

    Args:
        provider_type: 被验证的真实 SDK adapter 类型。
        expected: 默认请求超时，单位为秒。
    """
    provider = provider_type(api_key="offline", base_url="http://localhost/v1")
    try:
        assert provider._client.timeout == expected
        assert provider._client.max_retries == 0
    finally:
        provider.close()


def test_configured_factories_apply_model_limits_and_hybrid_policy() -> None:
    """设置应传到真实生产 adapter，而非只定义环境变量。"""
    settings = Settings(
        _env_file=None, llm_timeout_seconds=7,
        embedding_timeout_seconds=4, model_max_retries=1,
        agent_turn_timeout_seconds=20,
    )
    factories = configured_adapter_factories(settings)
    embedder = factories.embedding_provider("offline", "http://localhost", "e", 1)
    llm = factories.llm_provider("offline", "http://localhost", "l", 0, None)
    agent = factories.conversation_agent(
        "offline", "http://localhost", "l", 0, None,
        object(), InMemorySaver(), 5, 3,
    )
    try:
        assert embedder._client.timeout == 4
        assert llm._client.timeout == 7
        assert agent._model.root_client.timeout == 7
        assert agent._model.root_client.max_retries == 1
        assert agent._retrieval_strategy is RetrievalStrategy.HYBRID
        assert factories.reranker(None, None, "qwen3-rerank", "reranks") is None
    finally:
        embedder.close()
        llm.close()
        agent.close()


def test_explicit_rerank_strategy_requires_server_configuration() -> None:
    """明确要求精排却未配服务时，应在启动配置校验时失败。"""
    with pytest.raises(ValueError, match="RERANKER_BASE_URL"):
        Settings(
            _env_file=None, agent_retrieval_strategy="hybrid_rerank",
            reranker_base_url=None,
        )


def test_configured_agent_rejects_full_prompt_before_sdk_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """生产工厂必须安装输入保护，而不只是测试时手动注入。

    Args:
        monkeypatch: 替换外部 HTTP 传输的 fixture。
    """
    def handle(request: httpx.Request) -> httpx.Response:
        raise AssertionError("超长提示词不应发送到 SDK 传输层")

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(
        "app.bootstrap.factories.httpx",
        SimpleNamespace(Client=lambda **kwargs: client),
    )
    settings = Settings(
        _env_file=None, conversation_context_tokens=8192,
        conversation_reserved_tokens=4096, llm_max_output_tokens=2048,
    )
    agent = configured_adapter_factories(settings).conversation_agent(
        "offline", "http://localhost:1/v1", "model", 0.0, None,
        object(), InMemorySaver(), 5, 3,
    )
    try:
        with pytest.raises(PromptContextTooLargeError):
            agent.run(uuid4(), "Q" * 9000, [])
    finally:
        agent.close()


def test_expired_agent_budget_stops_before_another_network_operation() -> None:
    """模型调用耗尽整轮预算后，不得再进入检索。"""
    now = [0.0]

    class SlowModel:
        """使用确定性时钟替代真实耗时调用的模型。"""

        def bind_tools(self, tools: object, *, strict: bool) -> "SlowModel":
            """保留模型接口以执行真实图。

            Args:
                tools: 工具定义。
                strict: 工具参数校验选项。

            Returns:
                当前测试模型。
            """
            return self

        def invoke(self, messages: object) -> AIMessage:
            """让模拟时间超过整轮预算。

            Args:
                messages: 提供给模型的消息。

            Returns:
                原本会触发检索的工具调用。
            """
            now[0] = 2.0
            return AIMessage(content="", tool_calls=[{
                "name": "search_knowledge_base", "args": {"query": "A"},
                "id": "call-1", "type": "tool_call",
            }])

    class UnreachableRetriever:
        """任何调用都意味着图没有遵守整轮预算。"""

        def retrieve(
            self, *args: object, **kwargs: object,
        ) -> list[object]:
            """确保预算耗尽后不再发起外部调用。

            Args:
                args: 位置参数。
                kwargs: 检索选项。

            Returns:
                不返回，调用代表测试失败。
            """
            raise AssertionError("预算耗尽后不应检索")

    agent = LangGraphConversationAgent(
        model=SlowModel(), retrieval_service=UnreachableRetriever(),
        checkpointer=InMemorySaver(), top_k=5,
        turn_timeout_seconds=1, clock=lambda: now[0],
    )
    with pytest.raises(ConversationAgentError, match="预算"):
        agent.run(uuid4(), "问题", [])
