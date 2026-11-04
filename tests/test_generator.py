# -*- coding: utf-8 -*-

from collections.abc import Iterator
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, AuthenticationError, RateLimitError

from app.application.errors import (
    LLMAuthenticationError,
    LLMGenerationError,
    LLMMissingAPIKeyError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.application.query.generator import AnswerGenerator
from app.application.query.prompting import PromptBuilder
from app.domain.retrieval import RetrievalResult
from app.infrastructure.generation.openai_compatible import (
    OpenAICompatibleLLMProvider,
)


def make_result(
    index: int,
    *,
    filename: str = "fastapi.pdf",
    page: int | None = 12,
    text: str | None = None,
) -> RetrievalResult:
    """创建提示词测试使用的检索结果。"""
    return RetrievalResult(
        chunk_id=f"doc-a:{index}",
        doc_id="doc-a",
        text=text or f"文本块 {index}",
        filename=filename,
        page=page,
        chunk_index=index,
        score=0.9 - index * 0.1,
    )


class FakeCompletionsAPI:
    """记录聊天补全参数并返回可控内容。"""

    def __init__(self, content: str | None = "生成的回答[1]") -> None:
        self.content = content
        self.finish_reason = "stop"
        self.refusal: str | None = None
        self.calls: list[dict[str, object]] = []
        self.error: Exception | None = None

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        **options: object,
    ) -> SimpleNamespace:
        """模拟 OpenAI 兼容的聊天补全接口。"""
        call: dict[str, object] = {
            "model": model,
            "messages": messages,
        }
        call.update(options)
        self.calls.append(call)
        if self.error is not None:
            raise self.error
        message = SimpleNamespace(content=self.content, refusal=self.refusal)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=message, finish_reason=self.finish_reason,
        )])


class FakeOpenAIClient:
    """只暴露生成模块需要的聊天补全资源。"""

    def __init__(self, content: str | None = "生成的回答[1]") -> None:
        self.completions = FakeCompletionsAPI(content)
        self.chat = SimpleNamespace(completions=self.completions)
        self.close_calls = 0

    def close(self) -> None:
        """记录 provider 是否尝试关闭客户端。

        Returns:
            None。
        """
        self.close_calls += 1


@pytest.mark.parametrize("parameter", ["max_tokens", "max_completion_tokens"])
def test_llm_provider_passes_output_cap_to_request(parameter: str) -> None:
    """完整上下文为输出预留的空间必须落实为明确的请求参数。

    Args:
        parameter: 端点实际支持的输出上限参数。
    """
    client = FakeOpenAIClient()
    provider = OpenAICompatibleLLMProvider(
        client=client, max_output_tokens=77, output_token_parameter=parameter,
    )
    provider.generate([{"role": "user", "content": "问题"}])
    assert client.completions.calls[0][parameter] == 77
    other = (
        "max_tokens" if parameter == "max_completion_tokens"
        else "max_completion_tokens"
    )
    assert other not in client.completions.calls[0]


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_llm_provider_rejects_incomplete_generation(finish_reason: str) -> None:
    """达到输出上限时，即使剩余文本有引用也不能冒充完整答案。

    Args:
        finish_reason: 上游声明的截断或过滤原因。
    """
    client = FakeOpenAIClient("看似完整但被截断的回答[1]")
    client.completions.finish_reason = finish_reason
    provider = OpenAICompatibleLLMProvider(client=client)
    with pytest.raises(LLMGenerationError):
        provider.generate([{"role": "user", "content": "问题"}])


def test_llm_provider_rejects_refusal_with_nonempty_content() -> None:
    """上游明确拒绝生成时不能仅检查正文是否非空。"""
    client = FakeOpenAIClient("不应发布的正文[1]")
    client.completions.refusal = "refused"
    with pytest.raises(LLMGenerationError):
        OpenAICompatibleLLMProvider(client=client).generate([
            {"role": "user", "content": "问题"},
        ])


class FakeChatStream:
    """模拟 OpenAI 流式响应并记录关闭状态。

    Attributes:
        contents: 依次返回的增量内容。
        closed: 上游流是否已被关闭。
    """

    def __init__(self, contents: list[str | None]) -> None:
        """保存待返回的增量内容。

        Args:
            contents: 依次返回的增量内容。
        """
        self.contents = contents
        self.closed = False

    def __iter__(self) -> Iterator[SimpleNamespace]:
        """返回完整的 OpenAI 兼容流式分块。

        Yields:
            包含 choices、delta 与 content 字段的响应分块。
        """
        for content in self.contents:
            delta = SimpleNamespace(content=content)
            choice = SimpleNamespace(delta=delta)
            yield SimpleNamespace(choices=[choice])

    def close(self) -> None:
        """记录提供方在完成后关闭了上游流。"""
        self.closed = True


class FailingChatStream:
    """在迭代时失败的 OpenAI 流式响应。"""

    def __init__(self, error: Exception) -> None:
        """保存将在迭代期间抛出的异常。

        Args:
            error: 迭代时抛出的上游异常。
        """
        self.error = error
        self.closed = False

    def __iter__(self) -> Iterator[SimpleNamespace]:
        """在开始产生分块时抛出指定异常。

        Yields:
            此测试流不会产生任何分块。

        Raises:
            Exception: 保存的上游异常。
        """
        yield from ()
        raise self.error

    def close(self) -> None:
        """记录提供方在迭代失败后关闭了上游流。"""
        self.closed = True


class CloseFailingChatStream(FakeChatStream):
    """在关闭时抛出指定异常的 OpenAI 流式响应。"""

    def __init__(
        self,
        contents: list[str | None],
        close_error: Exception,
    ) -> None:
        """保存回答增量与关闭异常。

        Args:
            contents: 依次返回的增量内容。
            close_error: 关闭上游流时抛出的异常。
        """
        super().__init__(contents)
        self.close_error = close_error

    def close(self) -> None:
        """记录关闭请求后抛出预设异常。

        Raises:
            Exception: 构造时注入的关闭异常。
        """
        self.closed = True
        raise self.close_error


class FailingAndCloseFailingChatStream(FailingChatStream):
    """迭代与关闭都会失败的 OpenAI 流式响应。"""

    def __init__(
        self,
        iteration_error: Exception,
        close_error: Exception,
    ) -> None:
        """保存迭代异常与关闭异常。

        Args:
            iteration_error: 迭代上游流时抛出的异常。
            close_error: 关闭上游流时抛出的异常。
        """
        super().__init__(iteration_error)
        self.close_error = close_error

    def close(self) -> None:
        """记录关闭请求后抛出预设异常。

        Raises:
            Exception: 构造时注入的关闭异常。
        """
        self.closed = True
        raise self.close_error


class FakeStreamingCompletionsAPI:
    """记录流式聊天补全参数并返回预设上游流。"""

    def __init__(self, stream: object) -> None:
        """保存待返回的上游流。

        Args:
            stream: 模拟聊天补全创建接口返回的上游流。
        """
        self.stream = stream
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **options: object,
    ) -> object:
        """记录请求并返回模拟上游流。

        Args:
            model: 请求使用的模型名称。
            messages: 请求使用的聊天消息。
            stream: 是否请求流式响应。

        Returns:
            预设的流式响应对象。
        """
        call: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        call.update(options)
        self.calls.append(call)
        return self.stream


class FakeStreamingOpenAIClient:
    """只暴露流式生成所需的 OpenAI 资源。"""

    def __init__(self, stream: object) -> None:
        """以预设上游流装配流式补全资源。

        Args:
            stream: 模拟聊天补全创建接口返回的上游流。
        """
        self.completions = FakeStreamingCompletionsAPI(stream)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self) -> list[dict[str, object]]:
        """返回全部流式聊天补全请求参数。

        Returns:
            流式聊天补全请求参数列表。
        """
        return self.completions.calls


class FakeStreamingProvider:
    """记录消息并返回固定文本增量的语言模型提供方。"""

    def __init__(self) -> None:
        """初始化空的流式请求记录。"""
        self.messages: list[list[dict[str, str]]] = []

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """记录消息并返回两个固定文本增量。

        Args:
            messages: 答案生成器构建的聊天消息。

        Yields:
            用于断言流式转发行为的文本增量。
        """
        self.messages.append(messages)
        yield "答"
        yield "案[1]"


def test_prompt_builder_numbers_context_and_preserves_sources() -> None:
    """提示词应按检索顺序编号，并包含来源、页码、正文和问题。"""
    messages = PromptBuilder().build(
        "FastAPI 如何声明路径参数？",
        [
            make_result(0, text="使用花括号声明路径参数。"),
            make_result(1, filename="typing.md", page=None),
        ],
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "只能根据给定上下文" in messages[0]["content"]
    assert "根据现有知识库，我无法回答这个问题" in messages[0]["content"]
    user_prompt = messages[1]["content"]
    assert "[1]" in user_prompt
    assert "文件名：fastapi.pdf" in user_prompt
    assert "页码：12" in user_prompt
    assert "使用花括号声明路径参数。" in user_prompt
    assert "[2]" in user_prompt
    assert "文件名：typing.md" in user_prompt
    assert "页码：无" in user_prompt
    assert "问题：FastAPI 如何声明路径参数？" in user_prompt


def test_prompt_builder_rejects_empty_results() -> None:
    """空上下文不得被交给语言模型生成答案。"""
    with pytest.raises(ValueError, match="上下文不能为空"):
        PromptBuilder().build("问题", [])


def test_openai_compatible_provider_sends_model_and_messages() -> None:
    """生成服务应把模型名和完整消息原样发送给兼容接口。"""
    client = FakeOpenAIClient()
    provider = OpenAICompatibleLLMProvider(model="local-chat", client=client)
    messages = [{"role": "user", "content": "问题"}]

    answer = provider.generate(messages)

    assert answer == "生成的回答[1]"
    assert client.completions.calls == [
        {"model": "local-chat", "messages": messages}
    ]


def test_provider_forwards_explicit_temperature() -> None:
    """显式温度必须进入非流式请求以保证评估可复现。"""
    client = FakeOpenAIClient()
    provider = OpenAICompatibleLLMProvider(
        model="local-chat",
        client=client,
        temperature=0.0,
    )
    messages = [{"role": "user", "content": "问题"}]

    provider.generate(messages)

    assert client.completions.calls == [
        {
            "model": "local-chat",
            "messages": messages,
            "temperature": 0.0,
        }
    ]


def test_provider_forwards_explicit_reasoning_effort() -> None:
    """非流式请求必须传递显式推理强度以避免模型只返回思考。"""
    client = FakeOpenAIClient()
    provider = OpenAICompatibleLLMProvider(
        model="qwen3.5:4b",
        client=client,
        reasoning_effort="none",
    )
    messages = [{"role": "user", "content": "问题"}]

    provider.generate(messages)

    assert client.completions.calls == [
        {
            "model": "qwen3.5:4b",
            "messages": messages,
            "reasoning_effort": "none",
        }
    ]


def test_provider_streams_non_empty_tokens_and_closes_upstream() -> None:
    """流式提供方应过滤空增量、传递请求参数并关闭上游流。"""
    stream = FakeChatStream(["你", None, "好"])
    client = FakeStreamingOpenAIClient(stream)
    provider = OpenAICompatibleLLMProvider(
        model="qwen3",
        client=client,
    )
    messages = [{"role": "user", "content": "问题"}]

    tokens = list(provider.stream(messages))

    assert tokens == ["你", "好"]
    assert client.calls == [
        {"model": "qwen3", "messages": messages, "stream": True}
    ]
    assert stream.closed is True


def test_provider_forwards_temperature_to_stream_request() -> None:
    """流式请求必须与非流式请求使用同一显式温度。"""
    stream = FakeChatStream(["答案"])
    client = FakeStreamingOpenAIClient(stream)
    provider = OpenAICompatibleLLMProvider(
        model="qwen3",
        client=client,
        temperature=0.2,
    )
    messages = [{"role": "user", "content": "问题"}]

    assert list(provider.stream(messages)) == ["答案"]
    assert client.calls == [
        {
            "model": "qwen3",
            "messages": messages,
            "stream": True,
            "temperature": 0.2,
        }
    ]


def test_provider_forwards_reasoning_effort_to_stream_request() -> None:
    """流式请求必须与非流式请求使用同一显式推理强度。"""
    stream = FakeChatStream(["答案"])
    client = FakeStreamingOpenAIClient(stream)
    provider = OpenAICompatibleLLMProvider(
        model="qwen3.5:4b",
        client=client,
        reasoning_effort="none",
    )
    messages = [{"role": "user", "content": "问题"}]

    assert list(provider.stream(messages)) == ["答案"]
    assert client.calls == [
        {
            "model": "qwen3.5:4b",
            "messages": messages,
            "stream": True,
            "reasoning_effort": "none",
        }
    ]


def test_provider_maps_mid_stream_failure_and_closes_upstream() -> None:
    """流式迭代失败应映射领域异常并关闭已创建的上游流。"""
    stream = FailingChatStream(RuntimeError("private upstream detail"))
    provider = OpenAICompatibleLLMProvider(
        model="qwen3",
        client=FakeStreamingOpenAIClient(stream),
    )

    with pytest.raises(LLMGenerationError):
        list(provider.stream([{"role": "user", "content": "问题"}]))

    assert stream.closed is True


def test_provider_maps_close_only_failure_to_generation_error() -> None:
    """正常耗尽后的关闭失败应转换为稳定生成异常。"""
    close_error = RuntimeError("private close detail")
    stream = CloseFailingChatStream(["答案"], close_error)
    provider = OpenAICompatibleLLMProvider(
        model="qwen3",
        client=FakeStreamingOpenAIClient(stream),
    )

    with pytest.raises(LLMGenerationError) as exc_info:
        list(provider.stream([{"role": "user", "content": "问题"}]))

    assert exc_info.value.__cause__ is close_error
    assert "private close detail" not in str(exc_info.value)
    assert stream.closed is True


def test_provider_preserves_generation_failure_when_close_fails() -> None:
    """已发生的生成失败不得被随后发生的关闭异常替换。"""
    iteration_error = RuntimeError("private generation detail")
    close_error = RuntimeError("private close detail")
    stream = FailingAndCloseFailingChatStream(
        iteration_error,
        close_error,
    )
    provider = OpenAICompatibleLLMProvider(
        model="qwen3",
        client=FakeStreamingOpenAIClient(stream),
    )

    with pytest.raises(LLMGenerationError) as exc_info:
        list(provider.stream([{"role": "user", "content": "问题"}]))

    assert exc_info.value.__cause__ is iteration_error
    assert "private close detail" not in str(exc_info.value)
    assert stream.closed is True


def test_provider_close_preserves_generator_exit_when_close_fails() -> None:
    """上游关闭异常不得替换消费端触发的 GeneratorExit。"""
    stream = CloseFailingChatStream(
        ["第一个", "第二个"],
        RuntimeError("private close detail"),
    )
    provider = OpenAICompatibleLLMProvider(
        model="qwen3",
        client=FakeStreamingOpenAIClient(stream),
    )
    iterator = provider.stream(
        [{"role": "user", "content": "问题"}]
    )

    assert next(iterator) == "第一个"
    iterator.close()

    assert stream.closed is True


def test_provider_preserves_consumer_exception_when_close_fails() -> None:
    """消费端异常不得被映射或被上游关闭异常替换。"""
    stream = CloseFailingChatStream(
        ["第一个", "第二个"],
        RuntimeError("private close detail"),
    )
    provider = OpenAICompatibleLLMProvider(
        model="qwen3",
        client=FakeStreamingOpenAIClient(stream),
    )
    iterator = provider.stream(
        [{"role": "user", "content": "问题"}]
    )
    injected_error = RuntimeError("consumer detail")

    assert next(iterator) == "第一个"
    with pytest.raises(RuntimeError) as exc_info:
        iterator.throw(injected_error)

    assert exc_info.value is injected_error
    assert stream.closed is True


def test_production_provider_passes_independent_llm_configuration(monkeypatch) -> None:
    """生产客户端必须使用 LLM 专用地址和凭据。"""
    captured: dict[str, str] = {}

    def fake_openai(*, api_key: str, base_url: str, **options: object):
        captured.update(api_key=api_key, base_url=base_url)
        return FakeOpenAIClient()

    monkeypatch.setattr(
        "app.infrastructure.generation.openai_compatible.OpenAI",
        fake_openai,
    )

    OpenAICompatibleLLMProvider(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        model="qwen3",
    )

    assert captured == {
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
    }


def test_llm_provider_closes_only_internally_owned_client(
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
        "app.infrastructure.generation.openai_compatible.OpenAI",
        fake_openai,
    )
    internally_owned = OpenAICompatibleLLMProvider(
        api_key="test-key",
    )
    externally_owned = OpenAICompatibleLLMProvider(
        client=external_client,
    )

    internally_owned.close()
    internally_owned.close()
    externally_owned.close()
    externally_owned.close()

    assert internal_client.close_calls == 1
    assert external_client.close_calls == 0


def test_production_provider_requires_llm_api_key() -> None:
    """没有注入测试客户端时必须显式提供已解析的 LLM 凭据。"""
    with pytest.raises(LLMMissingAPIKeyError, match="LLM_API_KEY"):
        OpenAICompatibleLLMProvider(api_key=None)


def test_provider_rejects_empty_response_content() -> None:
    """兼容服务返回空内容时不得伪装成成功回答。"""
    provider = OpenAICompatibleLLMProvider(client=FakeOpenAIClient("  "))

    with pytest.raises(LLMGenerationError, match="空回答"):
        provider.generate([{"role": "user", "content": "问题"}])


@pytest.mark.parametrize(
    ("upstream_error", "expected_error"),
    [
        (
            APITimeoutError(request=httpx.Request("POST", "http://llm.test")),
            LLMTimeoutError,
        ),
        (
            RateLimitError(
                "too many requests",
                response=httpx.Response(
                    429,
                    request=httpx.Request("POST", "http://llm.test"),
                ),
                body=None,
            ),
            LLMRateLimitError,
        ),
        (
            AuthenticationError(
                "unauthorized",
                response=httpx.Response(
                    401,
                    request=httpx.Request("POST", "http://llm.test"),
                ),
                body=None,
            ),
            LLMAuthenticationError,
        ),
    ],
)
def test_provider_maps_known_upstream_errors(
    upstream_error: Exception,
    expected_error: type[Exception],
) -> None:
    """超时、限流和认证失败应转换为稳定领域异常并保留异常链。"""
    client = FakeOpenAIClient()
    client.completions.error = upstream_error
    provider = OpenAICompatibleLLMProvider(client=client)

    with pytest.raises(expected_error) as exc_info:
        provider.generate([{"role": "user", "content": "问题"}])

    assert exc_info.value.__cause__ is upstream_error


def test_answer_generator_combines_prompt_builder_and_llm_provider() -> None:
    """答案生成器应只调用一次 LLM，并返回其文本结果。"""
    client = FakeOpenAIClient("组合后的答案[1]")
    generator = AnswerGenerator(
        prompt_builder=PromptBuilder(),
        llm_provider=OpenAICompatibleLLMProvider(client=client),
    )

    answer = generator.generate("问题", [make_result(0)])

    assert answer == "组合后的答案[1]"
    assert len(client.completions.calls) == 1


def test_answer_generator_stream_reuses_numbered_prompt_context() -> None:
    """流式生成应传递 PromptBuilder 构建的编号上下文并转发增量。"""
    provider = FakeStreamingProvider()
    generator = AnswerGenerator(
        prompt_builder=PromptBuilder(),
        llm_provider=provider,
    )
    results = [make_result(0, text="使用花括号声明路径参数。")]

    tokens = list(generator.stream("FastAPI 如何声明路径参数？", results))

    assert tokens == ["答", "案[1]"]
    assert len(provider.messages) == 1
    user_message = provider.messages[0][1]["content"]
    assert (
        "[1]\n文件名：fastapi.pdf\n页码：12\n内容：\n"
        "使用花括号声明路径参数。"
    ) in user_message
    assert "问题：FastAPI 如何声明路径参数？" in user_message
