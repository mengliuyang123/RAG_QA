# -*- coding: utf-8 -*-
"""OpenAI 聊天补全协议的语言模型生成适配器。"""

import logging
import sys
from collections.abc import Iterator
from typing import Any
from typing import Literal

from openai import APITimeoutError
from openai import AuthenticationError
from openai import OpenAI
from openai import RateLimitError

from app.application.errors import LLMAuthenticationError
from app.application.errors import LLMError
from app.application.errors import LLMGenerationError
from app.application.errors import LLMMissingAPIKeyError
from app.application.errors import LLMRateLimitError
from app.application.errors import LLMTimeoutError
from app.application.tracing import provider_call
from app.infrastructure.generation.usage import extract_token_usage


logger = logging.getLogger(__name__)


class OpenAICompatibleLLMProvider:
    """通过 OpenAI 聊天补全协议访问远程或本地语言模型。

    Attributes:
        _owns_client: 是否由 provider 创建并负责关闭客户端。
        _closed: 内部客户端是否已经尝试关闭。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        client: Any | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 0,
        max_output_tokens: int | None = None,
        output_token_parameter: Literal[
            "max_tokens", "max_completion_tokens",
        ] = "max_tokens",
    ) -> None:
        """使用独立配置装配 OpenAI 兼容客户端。

        Args:
            api_key: 兼容服务凭据；注入客户端时可为空。
            base_url: OpenAI 兼容服务根地址。
            model: 聊天补全模型名称。
            client: 测试或外部生命周期管理的兼容客户端。
            temperature: 可选生成温度；为空时沿用服务默认值。
            reasoning_effort: 可选推理强度；为空时沿用服务
                默认值。
            timeout_seconds: 单次模型请求等待上限。
            max_retries: SDK 额外重试次数。
            max_output_tokens: 可选输出上限，与完整上下文预留一致。
            output_token_parameter: 端点实际支持的上限参数，不自动重试切换。

        Returns:
            None。

        Raises:
            LLMMissingAPIKeyError: 未注入客户端且凭据为空。
        """
        if timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("请求超时与重试参数无效")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("输出上限必须为正数")
        if output_token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("输出上限参数名无效")
        self._owns_client = client is None
        if client is None:
            if api_key is None or not api_key.strip():
                raise LLMMissingAPIKeyError(
                    "未配置 LLM_API_KEY，无法创建语言模型客户端"
                )
            client = OpenAI(
                api_key=api_key, base_url=base_url,
                timeout=timeout_seconds, max_retries=max_retries,
            )
        self._client = client
        self._origin = str(getattr(client, "base_url", base_url))
        self._closed = False
        self._model = model
        self._request_options: dict[str, object] = {}
        if max_output_tokens is not None:
            self._request_options[output_token_parameter] = max_output_tokens
        if temperature is not None:
            self._request_options["temperature"] = temperature
        if reasoning_effort is not None:
            self._request_options["reasoning_effort"] = reasoning_effort

    def generate(self, messages: list[dict[str, str]]) -> str:
        """请求非流式聊天补全并转换常见上游错误。

        Args:
            messages: 已构建的聊天消息。

        Returns:
            去除首尾空白后的完整回答。

        Raises:
            LLMTimeoutError: 上游请求超时。
            LLMRateLimitError: 上游触发限流。
            LLMAuthenticationError: 上游认证失败。
            LLMGenerationError: 上游响应无效或发生其他错误。
        """
        try:
            with provider_call("generation", self._origin, self._model) as call:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    **self._request_options,
                )
                call.usage = extract_token_usage(getattr(response, "usage", None))
                call.response_model = getattr(response, "model", None)
        except APITimeoutError as exc:
            raise LLMTimeoutError("语言模型请求超时") from exc
        except RateLimitError as exc:
            raise LLMRateLimitError("语言模型服务请求过多") from exc
        except AuthenticationError as exc:
            raise LLMAuthenticationError(
                "语言模型服务认证失败"
            ) from exc
        except Exception as exc:
            raise LLMGenerationError("语言模型调用失败") from exc

        try:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason in {"length", "content_filter"}:
                raise LLMGenerationError("语言模型回答被截断或过滤")
            if getattr(response.choices[0].message, "refusal", None):
                raise LLMGenerationError("语言模型拒绝了本次生成请求")
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMGenerationError(
                "语言模型返回了无效响应"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMGenerationError("语言模型返回了空回答")
        return content.strip()

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """逐个返回非空回答增量并关闭上游流。

        Args:
            messages: 已构建的聊天消息。

        Yields:
            非空文本增量。

        Raises:
            LLMTimeoutError: 上游请求超时。
            LLMRateLimitError: 上游触发限流。
            LLMAuthenticationError: 上游认证失败。
            LLMGenerationError: 上游返回无效事件或其他错误。
        """
        upstream: object | None = None
        try:
            try:
                upstream = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    stream=True,
                    **self._request_options,
                )
                upstream_iterator = iter(upstream)
            except APITimeoutError as exc:
                raise LLMTimeoutError("语言模型请求超时") from exc
            except RateLimitError as exc:
                raise LLMRateLimitError(
                    "语言模型服务请求过多"
                ) from exc
            except AuthenticationError as exc:
                raise LLMAuthenticationError(
                    "语言模型服务认证失败"
                ) from exc
            except LLMError:
                raise
            except Exception as exc:
                raise LLMGenerationError(
                    "语言模型流式调用失败"
                ) from exc

            while True:
                try:
                    chunk = next(upstream_iterator)
                    content = chunk.choices[0].delta.content
                except StopIteration:
                    break
                except APITimeoutError as exc:
                    raise LLMTimeoutError(
                        "语言模型请求超时"
                    ) from exc
                except RateLimitError as exc:
                    raise LLMRateLimitError(
                        "语言模型服务请求过多"
                    ) from exc
                except AuthenticationError as exc:
                    raise LLMAuthenticationError(
                        "语言模型服务认证失败"
                    ) from exc
                except LLMError:
                    raise
                except Exception as exc:
                    raise LLMGenerationError(
                        "语言模型流式调用失败"
                    ) from exc

                if isinstance(content, str) and content:
                    yield content
        finally:
            active_error = sys.exception()
            close = getattr(upstream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    logger.exception("语言模型上游流关闭失败")
                    if active_error is None:
                        raise LLMGenerationError(
                            "语言模型流式响应关闭失败"
                        ) from exc

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
