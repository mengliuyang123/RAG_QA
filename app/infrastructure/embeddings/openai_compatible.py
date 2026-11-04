# -*- coding: utf-8 -*-
"""文本向量化抽象及 OpenAI 兼容协议实现。"""

from typing import Any

from openai import APITimeoutError
from openai import AuthenticationError
from openai import OpenAI
from openai import RateLimitError

from app.application.errors import EmbeddingAuthenticationError
from app.application.errors import EmbeddingError
from app.application.errors import EmbeddingRateLimitError
from app.application.errors import EmbeddingResponseError
from app.application.errors import EmbeddingTimeoutError
from app.application.errors import MissingAPIKeyError
from app.application.tracing import provider_call
from app.infrastructure.generation.usage import extract_token_usage


class OpenAICompatibleEmbeddingProvider:
    """通过 OpenAI 兼容的 Embeddings API 分批生成文本向量。

    Attributes:
        model: 每次请求传给远程服务的 Embedding 模型名。
        batch_size: 单次远程请求最多包含的文本数。
        _owns_client: 是否由 provider 创建并负责关闭客户端。
        _closed: 内部客户端是否已经尝试关闭。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        batch_size: int = 20,
        client: Any | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 0,
    ) -> None:
        """创建 provider；测试可注入兼容 OpenAI 接口的客户端。

        Args:
            api_key: 未注入客户端时所需的 Embedding 服务凭据。
            base_url: OpenAI 兼容服务的 API 根地址。
            model: 请求使用的 Embedding 模型名。
            batch_size: 单次 Embedding 请求中的最大文本数。
            client: 可选的 OpenAI 兼容客户端。
            timeout_seconds: 单次模型请求等待上限。
            max_retries: SDK 额外重试次数。

        Returns:
            None。

        Raises:
            ValueError: ``batch_size`` 不是正整数。
            MissingAPIKeyError: 未注入客户端且没有可用凭据。
        """
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("请求超时与重试参数无效")
        self._owns_client = client is None
        if client is None:
            if not api_key or not api_key.strip():
                raise MissingAPIKeyError(
                    "缺少 EMBEDDING_API_KEY，无法创建向量客户端"
                )
            client = OpenAI(
                api_key=api_key, base_url=base_url,
                timeout=timeout_seconds, max_retries=max_retries,
            )

        self._client = client
        self._origin = str(getattr(client, "base_url", base_url))
        self._closed = False
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """分批向量化文本，并验证响应数量、顺序和维度。

        Args:
            texts: 按期望返回顺序排列的文本列表。

        Returns:
            与 ``texts`` 一一对应且顺序一致的向量列表。

        Raises:
            EmbeddingTimeoutError: 上游向量请求超时。
            EmbeddingRateLimitError: 上游向量服务触发限流。
            EmbeddingAuthenticationError: 上游向量服务认证失败。
            EmbeddingResponseError: 远程响应数量、索引或维度不一致。
            EmbeddingError: 发生其他向量服务调用错误。
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []
        expected_dimension: int | None = None
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            try:
                with provider_call("embedding", self._origin, self.model) as call:
                    response = self._client.embeddings.create(
                        input=batch,
                        model=self.model,
                    )
                    call.usage = extract_token_usage(
                        getattr(response, "usage", None), no_generation=True,
                    )
                    call.response_model = getattr(response, "model", None)
            except APITimeoutError as exc:
                raise EmbeddingTimeoutError("向量服务请求超时") from exc
            except RateLimitError as exc:
                raise EmbeddingRateLimitError(
                    "向量服务请求过多"
                ) from exc
            except AuthenticationError as exc:
                raise EmbeddingAuthenticationError(
                    "向量服务认证失败"
                ) from exc
            except Exception as exc:
                raise EmbeddingError("向量服务调用失败") from exc

            try:
                ordered_items = sorted(
                    response.data,
                    key=lambda item: item.index,
                )
            except (AttributeError, TypeError) as exc:
                raise EmbeddingResponseError(
                    "向量服务返回了无效响应"
                ) from exc
            if len(ordered_items) != len(batch):
                raise EmbeddingResponseError(
                    "向量响应数量与请求文本数量不一致"
                )

            for expected_index, item in enumerate(ordered_items):
                if item.index != expected_index:
                    raise EmbeddingResponseError(
                        "向量响应索引与请求文本顺序不一致"
                    )
                try:
                    vector = list(item.embedding)
                except (AttributeError, TypeError) as exc:
                    raise EmbeddingResponseError(
                        "向量服务返回了无效向量"
                    ) from exc
                if expected_dimension is None:
                    expected_dimension = len(vector)
                elif len(vector) != expected_dimension:
                    raise EmbeddingResponseError("向量响应维度不一致")
                embeddings.append(vector)

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """向量化一条查询文本。

        Args:
            text: 待向量化的查询文本。

        Returns:
            查询文本对应的单个向量。

        Raises:
            EmbeddingError: 远程响应不满足单条查询的向量约定。
        """
        return self.embed_documents([text])[0]

    def close(self) -> None:
        """幂等关闭由本 provider 创建的 OpenAI 客户端。

        Returns:
            None。

        Raises:
            Exception: 内部客户端关闭失败时原样传播的异常。
        """
        if not self._owns_client or self._closed:
            return
        # 先记录关闭尝试，避免异常路径重复关闭同一连接池。
        self._closed = True
        self._client.close()
