# -*- coding: utf-8 -*-

from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError
from openai import AuthenticationError
from openai import RateLimitError

from app.application.errors import (
    EmbeddingAuthenticationError,
    EmbeddingRateLimitError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
    MissingAPIKeyError,
)
from app.infrastructure.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)


class FakeEmbeddingsAPI:
    """记录请求，并返回可控的假向量响应。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.dimension = 2
        self.drop_last = False
        self.error: Exception | None = None

    def create(self, *, input: list[str], model: str):
        """模拟 OpenAI embeddings.create 接口。"""
        if self.error is not None:
            raise self.error
        self.calls.append({"input": input, "model": model})
        data = [
            SimpleNamespace(index=index, embedding=[float(len(text)), float(index)])
            for index, text in enumerate(input)
        ]
        if self.dimension == 3 and data:
            data[-1].embedding.append(0.0)
        if self.drop_last and data:
            data.pop()
        return SimpleNamespace(data=list(reversed(data)))


class FakeOpenAIClient:
    """只暴露测试所需的 embeddings 资源。"""

    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsAPI()
        self.close_calls = 0

    def close(self) -> None:
        """记录 provider 是否尝试关闭客户端。

        Returns:
            None。
        """
        self.close_calls += 1


def test_embed_documents_batches_requests_and_preserves_order() -> None:
    """文档向量应分批请求，并按原文本顺序返回。"""
    client = FakeOpenAIClient()
    provider = OpenAICompatibleEmbeddingProvider(
        model="test-model",
        batch_size=2,
        client=client,
    )

    result = provider.embed_documents(["a", "bbbb", "cc"])

    assert client.embeddings.calls == [
        {"input": ["a", "bbbb"], "model": "test-model"},
        {"input": ["cc"], "model": "test-model"},
    ]
    assert result == [[1.0, 0.0], [4.0, 1.0], [2.0, 0.0]]


def test_embed_documents_skips_api_for_empty_input() -> None:
    """空文本列表无需发起远程请求。"""
    client = FakeOpenAIClient()
    provider = OpenAICompatibleEmbeddingProvider(client=client)

    assert provider.embed_documents([]) == []
    assert client.embeddings.calls == []


def test_embed_query_returns_single_vector() -> None:
    """查询向量接口应返回一条扁平向量。"""
    client = FakeOpenAIClient()
    provider = OpenAICompatibleEmbeddingProvider(client=client)

    assert provider.embed_query("问题") == [2.0, 0.0]


def test_embed_documents_rejects_missing_response_item() -> None:
    """响应数量与请求不一致时不得写入错位向量。"""
    client = FakeOpenAIClient()
    client.embeddings.drop_last = True
    provider = OpenAICompatibleEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingResponseError, match="数量"):
        provider.embed_documents(["a", "b"])


def test_embed_documents_rejects_inconsistent_dimensions() -> None:
    """同一批次向量维度不一致时应立即失败。"""
    client = FakeOpenAIClient()
    client.embeddings.dimension = 3
    provider = OpenAICompatibleEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingResponseError, match="维度"):
        provider.embed_documents(["a", "b"])


@pytest.mark.parametrize(
    ("upstream_error", "expected_error"),
    [
        (
            APITimeoutError(
                request=httpx.Request("POST", "http://embedding.test")
            ),
            EmbeddingTimeoutError,
        ),
        (
            RateLimitError(
                "too many requests",
                response=httpx.Response(
                    429,
                    request=httpx.Request(
                        "POST",
                        "http://embedding.test",
                    ),
                ),
                body=None,
            ),
            EmbeddingRateLimitError,
        ),
        (
            AuthenticationError(
                "unauthorized",
                response=httpx.Response(
                    401,
                    request=httpx.Request(
                        "POST",
                        "http://embedding.test",
                    ),
                ),
                body=None,
            ),
            EmbeddingAuthenticationError,
        ),
    ],
)
def test_provider_maps_known_upstream_errors(
    upstream_error: Exception,
    expected_error: type[Exception],
) -> None:
    """向量服务错误应转换为可稳定分类的应用异常。

    Args:
        upstream_error: 模拟的 OpenAI SDK 异常。
        expected_error: 预期的稳定应用异常类型。

    Returns:
        None。
    """
    client = FakeOpenAIClient()
    client.embeddings.error = upstream_error
    provider = OpenAICompatibleEmbeddingProvider(client=client)

    with pytest.raises(expected_error) as exc_info:
        provider.embed_query("问题")

    assert exc_info.value.__cause__ is upstream_error


def test_production_provider_requires_api_key() -> None:
    """未注入客户端时必须提供 API Key。"""
    with pytest.raises(MissingAPIKeyError, match="EMBEDDING_API_KEY"):
        OpenAICompatibleEmbeddingProvider(api_key=None)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_provider_rejects_invalid_batch_size(batch_size: int) -> None:
    """批量大小必须是正整数。"""
    with pytest.raises(ValueError, match="batch_size"):
        OpenAICompatibleEmbeddingProvider(
            client=FakeOpenAIClient(),
            batch_size=batch_size,
        )


def test_production_provider_passes_credentials_and_base_url(monkeypatch) -> None:
    """生产客户端应使用配置的兼容服务地址和凭据。"""
    captured: dict[str, str] = {}

    def fake_openai(*, api_key: str, base_url: str, **options: object):
        captured.update(api_key=api_key, base_url=base_url)
        return FakeOpenAIClient()

    monkeypatch.setattr(
        "app.infrastructure.embeddings.openai_compatible.OpenAI",
        fake_openai,
    )

    OpenAICompatibleEmbeddingProvider(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        model="qwen3-embedding",
    )

    assert captured == {
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
    }


def test_provider_closes_only_internally_owned_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """close 应幂等释放内部客户端但保留调用方注入客户端。

    Args:
        monkeypatch: pytest 提供的属性替换工具。

    Returns:
        None。
    """
    internal_client = FakeOpenAIClient()
    external_client = FakeOpenAIClient()

    def fake_openai(
        *, api_key: str, base_url: str, **options: object,
    ) -> FakeOpenAIClient:
        del api_key
        del base_url
        return internal_client

    monkeypatch.setattr(
        "app.infrastructure.embeddings.openai_compatible.OpenAI",
        fake_openai,
    )
    internally_owned = OpenAICompatibleEmbeddingProvider(
        api_key="test-key",
    )
    externally_owned = OpenAICompatibleEmbeddingProvider(
        client=external_client,
    )

    internally_owned.close()
    internally_owned.close()
    externally_owned.close()
    externally_owned.close()

    assert internal_client.close_calls == 1
    assert external_client.close_calls == 0
