# -*- coding: utf-8 -*-
"""流式事件模型与 RAG Pipeline 行为测试。"""

from collections.abc import Iterator

import pytest

from app.application.errors import LLMGenerationError
from app.application.query.events import ErrorStreamData
from app.application.query.events import ErrorStreamEvent
from app.application.query.events import TokenStreamData
from app.application.query.events import TokenStreamEvent
from app.application.query.prompting import INSUFFICIENT_CONTEXT_ANSWER
from app.application.query.service import QueryService
from app.domain.errors import RetrievalError
from app.domain.retrieval import RetrievalResult, RetrievalStrategy


def make_result(index: int) -> RetrievalResult:
    """创建流式 Pipeline 测试使用的检索结果。

    Args:
        index: 文本块的零基下标。

    Returns:
        带完整元数据的检索结果。
    """
    return RetrievalResult(
        chunk_id=f"doc-a:{index}",
        doc_id="doc-a",
        text=f"文本块 {index}",
        filename="fastapi.pdf",
        page=12,
        chunk_index=index,
        score=0.9 - index * 0.1,
    )


class FakeRetriever:
    """返回预设检索结果。"""

    def __init__(self, results: list[RetrievalResult]) -> None:
        """保存预设检索结果。

        Args:
            results: 查询时返回的有序检索结果。
        """
        self.results = results

    def retrieve(
        self,
        question: str,
        strategy: RetrievalStrategy | str,
        top_k: int,
    ) -> list[RetrievalResult]:
        """返回预设检索结果。

        Args:
            question: 用户问题；本测试替身不使用。
            strategy: 检索策略；本测试替身不使用。
            top_k: 返回结果数量上限。

        Returns:
            截断到 ``top_k`` 的预设结果。
        """
        del question
        del strategy
        return self.results[:top_k]


class FakeStreamingAnswerGenerator:
    """按顺序返回预设回答增量。"""

    def __init__(self, tokens: list[str]) -> None:
        """保存预设回答增量。

        Args:
            tokens: 流式返回的文本增量。
        """
        self.tokens = tokens
        self.stream_calls: list[tuple[str, list[RetrievalResult]]] = []

    def stream(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> Iterator[str]:
        """逐个返回预设回答增量。

        Args:
            question: 用户问题；本测试替身不使用。
            results: 检索结果；本测试替身不使用。

        Yields:
            预设文本增量。
        """
        self.stream_calls.append((question, results))
        yield from self.tokens


class FakeAnswerGenerator:
    """返回预设的非流式回答。"""

    def __init__(self, answer: str) -> None:
        """保存预设回答。

        Args:
            answer: 非流式生成时返回的完整回答。
        """
        self._answer = answer

    def generate(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> str:
        """返回预设回答。

        Args:
            question: 用户问题；本测试替身不使用。
            results: 检索结果；本测试替身不使用。

        Returns:
            构造时设置的完整回答。
        """
        del question
        del results
        return self._answer


class FailingStreamingAnswerGenerator(FakeStreamingAnswerGenerator):
    """返回预设增量后抛出指定异常。"""

    def __init__(self, tokens: list[str], error: Exception) -> None:
        """保存预设增量与异常。

        Args:
            tokens: 抛出异常前返回的文本增量。
            error: 增量耗尽后抛出的异常。
        """
        super().__init__(tokens)
        self.error = error

    def stream(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> Iterator[str]:
        """返回增量后抛出指定异常。

        Args:
            question: 用户问题。
            results: 检索结果。

        Yields:
            抛出异常前的预设文本增量。

        Raises:
            Exception: 构造时注入的异常。
        """
        self.stream_calls.append((question, results))
        yield from self.tokens
        raise self.error


class FailingRetriever:
    """检索时抛出指定异常。"""

    def __init__(self, error: Exception) -> None:
        """保存检索时抛出的异常。

        Args:
            error: 检索时抛出的异常。
        """
        self.error = error

    def retrieve(
        self,
        question: str,
        strategy: RetrievalStrategy | str,
        top_k: int,
    ) -> list[RetrievalResult]:
        """抛出构造时注入的异常。

        Args:
            question: 用户问题；本测试替身不使用。
            strategy: 检索策略；本测试替身不使用。
            top_k: 结果上限；本测试替身不使用。

        Returns:
            本测试替身不会正常返回。

        Raises:
            Exception: 构造时注入的异常。
        """
        del question
        del strategy
        del top_k
        raise self.error


class SequenceClock:
    """依次返回预设时间点。"""

    def __init__(self, *values: float) -> None:
        """保存预设时间点。

        Args:
            *values: 依次返回的单调时间值。
        """
        self._values = iter(values)

    def __call__(self) -> float:
        """返回下一个时间点。

        Returns:
            下一个预设时间点。
        """
        return next(self._values)


class FailingCallClock:
    """在指定调用次数抛出异常的测试时钟。"""

    def __init__(self, fail_on_call: int) -> None:
        """保存应失败的调用次数。

        Args:
            fail_on_call: 从一开始计数的失败调用次数。
        """
        self._fail_on_call = fail_on_call
        self._call_count = 0

    def __call__(self) -> float:
        """返回递增时间或在指定次数抛出异常。

        Returns:
            当前调用次数对应的浮点时间。

        Raises:
            RuntimeError: 当前调用次数是预设失败点。
        """
        self._call_count += 1
        if self._call_count == self._fail_on_call:
            raise RuntimeError("private clock detail")
        return float(self._call_count)


class ClosableTokenIterator:
    """记录是否收到关闭请求的文本增量迭代器。"""

    def __init__(
        self,
        tokens: list[str],
        *,
        iteration_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        """保存预设增量并标记为尚未关闭。

        Args:
            tokens: 依次返回的文本增量。
            iteration_error: 增量耗尽后抛出的可选异常。
            close_error: 关闭迭代器时抛出的可选异常。
        """
        self._tokens = iter(tokens)
        self._iteration_error = iteration_error
        self._close_error = close_error
        self.closed = False

    def __iter__(self) -> Iterator[str]:
        """返回当前迭代器。

        Returns:
            当前文本增量迭代器。
        """
        return self

    def __next__(self) -> str:
        """返回下一个文本增量。

        Returns:
            下一个文本增量。

        Raises:
            Exception: 增量耗尽后注入的可选异常。
        """
        try:
            return next(self._tokens)
        except StopIteration:
            if self._iteration_error is not None:
                raise self._iteration_error
            raise

    def close(self) -> None:
        """记录关闭请求并按需抛出预设异常。

        Raises:
            Exception: 构造时注入的关闭异常。
        """
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class IteratorAnswerGenerator:
    """直接返回注入的文本增量迭代器。"""

    def __init__(self, iterator: ClosableTokenIterator) -> None:
        """保存文本增量迭代器。

        Args:
            iterator: 流式生成时返回的可关闭迭代器。
        """
        self.iterator = iterator

    def stream(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> Iterator[str]:
        """返回注入的文本增量迭代器。

        Args:
            question: 用户问题；本测试替身不使用。
            results: 检索结果；本测试替身不使用。

        Returns:
            注入的文本增量迭代器。
        """
        del question
        del results
        return self.iterator


def make_streaming_pipeline(
    upstream: ClosableTokenIterator,
) -> QueryService:
    """创建使用可关闭上游迭代器的 Pipeline。

    Args:
        upstream: 应接收关闭请求的文本增量迭代器。

    Returns:
        装配测试依赖的流式 Pipeline。
    """
    return QueryService(
        retriever=FakeRetriever([make_result(0)]),
        answer_generator=IteratorAnswerGenerator(upstream),
        clock=SequenceClock(1.0, 1.01),
    )


def test_stream_event_models_expose_stable_payloads() -> None:
    """事件模型应暴露稳定的 SSE 负载结构。"""
    token_event = TokenStreamEvent(data=TokenStreamData(token="你"))
    error_event = ErrorStreamEvent(
        data=ErrorStreamData(
            code="generation_error",
            message="语言模型服务暂时不可用",
        )
    )

    assert token_event.model_dump() == {
        "event": "token",
        "data": {"token": "你"},
    }
    assert error_event.model_dump() == {
        "event": "error",
        "data": {
            "code": "generation_error",
            "message": "语言模型服务暂时不可用",
        },
    }


def test_stream_query_emits_tokens_then_matching_complete() -> None:
    """回答增量应按顺序发送并原样拼接到完成事件。"""
    pipeline = QueryService(
        retriever=FakeRetriever([make_result(0)]),
        answer_generator=FakeStreamingAnswerGenerator(["你", "好[1]"]),
        clock=SequenceClock(1.0, 1.01, 1.03),
    )

    events = list(pipeline.stream_query("问题", strategy="vector", top_k=1))

    assert [event.event for event in events] == [
        "token",
        "token",
        "complete",
    ]
    assert [event.data.token for event in events[:2]] == ["你", "好[1]"]
    complete = events[-1].data
    assert complete.answer == "你好[1]"
    assert complete.citations[0].chunk_text == "文本块 0"
    assert complete.latency_ms == 30.0
    assert complete.retrieval_strategy is RetrievalStrategy.VECTOR


def test_stream_complete_matches_non_streaming_query_payload() -> None:
    """完整流的负载必须与 ``query()`` 的结果一致。"""
    results = [make_result(0)]
    non_streaming_pipeline = QueryService(
        retriever=FakeRetriever(results),
        answer_generator=FakeAnswerGenerator("答案[1]"),
        clock=SequenceClock(1.0, 1.01, 1.03),
    )
    streaming_pipeline = QueryService(
        retriever=FakeRetriever(results),
        answer_generator=FakeStreamingAnswerGenerator(["答案", "[1]"]),
        clock=SequenceClock(1.0, 1.01, 1.03),
    )

    query_result = non_streaming_pipeline.query("问题", top_k=1)
    stream_events = list(streaming_pipeline.stream_query("问题", top_k=1))
    complete_data = stream_events[-1].data

    assert complete_data.model_dump() == query_result.model_dump()
    assert complete_data.answer == "".join(
        event.data.token
        for event in stream_events
        if event.event == "token"
    )


def test_stream_query_skips_llm_when_retrieval_is_empty() -> None:
    """空检索应跳过模型并发送标准知识不足完成事件。"""
    generator = FakeStreamingAnswerGenerator(["不得调用"])
    pipeline = QueryService(
        retriever=FakeRetriever([]),
        answer_generator=generator,
        clock=SequenceClock(1.0, 1.01, 1.02),
    )

    events = list(pipeline.stream_query("未知问题", top_k=5))

    assert generator.stream_calls == []
    assert [event.event for event in events] == ["complete"]
    assert events[0].data.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert events[0].data.citations == []


def test_stream_query_emits_error_without_complete_after_failure() -> None:
    """部分增量后的模型失败应只以稳定错误终止。"""
    generator = FailingStreamingAnswerGenerator(
        tokens=["部分"],
        error=LLMGenerationError("private upstream detail"),
    )
    pipeline = QueryService(
        retriever=FakeRetriever([make_result(0)]),
        answer_generator=generator,
        clock=SequenceClock(1.0, 1.01),
    )

    events = list(pipeline.stream_query("问题", top_k=1))

    assert [event.event for event in events] == ["token", "error"]
    assert events[-1].data.model_dump() == {
        "code": "generation_error",
        "message": "语言模型服务暂时不可用",
    }


def test_stream_query_maps_retrieval_failure_to_stable_error() -> None:
    """检索领域异常应映射为不泄露细节的稳定错误。"""
    pipeline = QueryService(
        retriever=FailingRetriever(
            RetrievalError("private vector database detail")
        ),
        answer_generator=FakeStreamingAnswerGenerator(["不得调用"]),
        clock=SequenceClock(1.0),
    )

    events = list(pipeline.stream_query("问题", top_k=1))

    assert [event.event for event in events] == ["error"]
    assert events[0].data.model_dump() == {
        "code": "retrieval_error",
        "message": "检索服务暂时不可用",
    }
    assert "private vector database detail" not in str(
        events[0].model_dump()
    )


def test_stream_query_maps_empty_upstream_to_stable_error() -> None:
    """空模型流应只发送空生成稳定错误。"""
    pipeline = QueryService(
        retriever=FakeRetriever([make_result(0)]),
        answer_generator=FakeStreamingAnswerGenerator([]),
        clock=SequenceClock(1.0, 1.01),
    )

    events = list(pipeline.stream_query("问题", top_k=1))

    assert [event.event for event in events] == ["error"]
    assert events[0].data.model_dump() == {
        "code": "empty_generation",
        "message": "语言模型未返回有效内容",
    }


def test_stream_query_maps_unknown_failure_without_leaking_detail() -> None:
    """未知生成异常应映射为不泄露细节的内部错误。"""
    pipeline = QueryService(
        retriever=FakeRetriever([make_result(0)]),
        answer_generator=FailingStreamingAnswerGenerator(
            tokens=[],
            error=RuntimeError("private implementation detail"),
        ),
        clock=SequenceClock(1.0, 1.01),
    )

    events = list(pipeline.stream_query("问题", top_k=1))

    assert [event.event for event in events] == ["error"]
    assert events[0].data.model_dump() == {
        "code": "internal_error",
        "message": "流式查询处理失败",
    }
    assert "private implementation detail" not in str(
        events[0].model_dump()
    )


def test_closing_stream_query_closes_upstream_iterator() -> None:
    """关闭 Pipeline 流应把取消信号传给答案增量迭代器。"""
    upstream = ClosableTokenIterator(["第一个", "第二个"])
    pipeline = make_streaming_pipeline(upstream)
    event_iterator = pipeline.stream_query("问题", top_k=1)

    first_event = next(event_iterator)
    event_iterator.close()

    assert first_event.event == "token"
    assert upstream.closed is True


def test_stream_query_maps_close_only_failure_to_internal_error() -> None:
    """成功终止前的关闭失败应替换为唯一内部错误。"""
    upstream = ClosableTokenIterator(
        ["答案[1]"],
        close_error=RuntimeError("private close detail"),
    )
    pipeline = QueryService(
        retriever=FakeRetriever([make_result(0)]),
        answer_generator=IteratorAnswerGenerator(upstream),
        clock=SequenceClock(1.0, 1.01, 1.02),
    )

    events = list(pipeline.stream_query("问题", top_k=1))

    assert [event.event for event in events] == ["token", "error"]
    assert events[-1].data.model_dump() == {
        "code": "internal_error",
        "message": "流式查询处理失败",
    }
    assert "private close detail" not in str(events[-1].model_dump())
    assert upstream.closed is True


def test_stream_query_preserves_generation_error_when_close_fails() -> None:
    """待发送的生成错误不得被迭代器关闭异常替换。"""
    upstream = ClosableTokenIterator(
        ["部分"],
        iteration_error=LLMGenerationError(
            "private generation detail"
        ),
        close_error=RuntimeError("private close detail"),
    )
    pipeline = make_streaming_pipeline(upstream)

    events = list(pipeline.stream_query("问题", top_k=1))

    assert [event.event for event in events] == ["token", "error"]
    assert events[-1].data.model_dump() == {
        "code": "generation_error",
        "message": "语言模型服务暂时不可用",
    }
    assert "private close detail" not in str(events[-1].model_dump())
    assert upstream.closed is True


def test_closing_pipeline_stream_preserves_generator_exit_on_close_error(
) -> None:
    """答案关闭异常不得替换 Pipeline 的 GeneratorExit。"""
    upstream = ClosableTokenIterator(
        ["第一个", "第二个"],
        close_error=RuntimeError("private close detail"),
    )
    pipeline = make_streaming_pipeline(upstream)
    event_iterator = pipeline.stream_query("问题", top_k=1)

    assert next(event_iterator).event == "token"
    event_iterator.close()

    assert upstream.closed is True


def test_pipeline_preserves_consumer_exception_when_close_fails() -> None:
    """消费端异常不得被答案迭代器关闭异常替换。"""
    upstream = ClosableTokenIterator(
        ["第一个", "第二个"],
        close_error=RuntimeError("private close detail"),
    )
    pipeline = make_streaming_pipeline(upstream)
    event_iterator = pipeline.stream_query("问题", top_k=1)
    injected_error = RuntimeError("consumer detail")

    assert next(event_iterator).event == "token"
    with pytest.raises(RuntimeError) as exc_info:
        event_iterator.throw(injected_error)

    assert exc_info.value is injected_error
    assert upstream.closed is True


@pytest.mark.parametrize(
    ("results", "fail_on_call"),
    [
        ([make_result(0)], 1),
        ([make_result(0)], 2),
        ([], 3),
    ],
)
def test_stream_query_maps_clock_failure_to_one_internal_error(
    results: list[RetrievalResult],
    fail_on_call: int,
) -> None:
    """任一流式计时失败都应只发送稳定内部错误。

    Args:
        results: 本用例使用的检索结果。
        fail_on_call: 时钟抛出异常的调用次数。
    """
    pipeline = QueryService(
        retriever=FakeRetriever(results),
        answer_generator=FakeStreamingAnswerGenerator(["答案[1]"]),
        clock=FailingCallClock(fail_on_call),
    )

    events = list(pipeline.stream_query("问题", top_k=1))

    assert [event.event for event in events] == ["error"]
    assert events[0].data.model_dump() == {
        "code": "internal_error",
        "message": "流式查询处理失败",
    }
    assert "private clock detail" not in str(events[0].model_dump())


def test_complete_terminal_does_not_catch_consumer_exception() -> None:
    """完成事件悬停点注入的消费端异常必须原样传播。"""
    pipeline = QueryService(
        retriever=FakeRetriever([make_result(0)]),
        answer_generator=FakeStreamingAnswerGenerator(["答案[1]"]),
        clock=SequenceClock(1.0, 1.01, 1.02),
    )
    event_iterator = pipeline.stream_query("问题", top_k=1)

    assert next(event_iterator).event == "token"
    assert next(event_iterator).event == "complete"
    injected_error = RuntimeError("consumer detail")

    with pytest.raises(RuntimeError) as exc_info:
        event_iterator.throw(injected_error)

    assert exc_info.value is injected_error


def test_error_terminal_does_not_catch_consumer_exception() -> None:
    """错误事件悬停点注入的消费端异常必须原样传播。"""
    pipeline = QueryService(
        retriever=FakeRetriever([make_result(0)]),
        answer_generator=FakeStreamingAnswerGenerator([]),
        clock=SequenceClock(1.0, 1.01),
    )
    event_iterator = pipeline.stream_query("问题", top_k=1)

    assert next(event_iterator).event == "error"
    injected_error = RuntimeError("consumer detail")

    with pytest.raises(RuntimeError) as exc_info:
        event_iterator.throw(injected_error)

    assert exc_info.value is injected_error
