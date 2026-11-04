# -*- coding: utf-8 -*-
"""RAG 查询 Pipeline 的行为测试。"""

import hashlib
import logging

import pytest

from app.application.query.prompting import INSUFFICIENT_CONTEXT_ANSWER
from app.application.query.service import QueryService
from app.domain.answers import Citation
from app.domain.retrieval import RetrievalResult, RetrievalStrategy


def make_result(index: int, *, page: int | None = 12) -> RetrievalResult:
    """创建 RAG 编排测试使用的检索结果。

    Args:
        index: 文本块的零基下标。
        page: 来源页码。

    Returns:
        带完整元数据的检索结果。
    """
    return RetrievalResult(
        chunk_id=f"doc-a:{index}",
        doc_id="doc-a",
        text=f"文本块 {index}",
        filename="fastapi.pdf",
        page=page,
        chunk_index=index,
        score=0.9 - index * 0.1,
    )


class FakeRetriever:
    """记录查询参数并返回预设结果。

    Attributes:
        results: 查询时返回的预设结果。
        calls: 已接收的查询参数。
    """

    def __init__(self, results: list[RetrievalResult]) -> None:
        """保存预设检索结果。

        Args:
            results: 查询时返回的有序检索结果。
        """
        self.results = results
        self.calls: list[tuple[str, RetrievalStrategy | str, int]] = []

    def retrieve(
        self,
        question: str,
        strategy: RetrievalStrategy | str,
        top_k: int,
    ) -> list[RetrievalResult]:
        """模拟统一检索器接口。

        Args:
            question: 用户问题。
            strategy: 本次检索策略。
            top_k: 返回结果数量上限。

        Returns:
            截断到 ``top_k`` 的预设检索结果。
        """
        self.calls.append((question, strategy, top_k))
        return self.results[:top_k]


class SpyAnswerGenerator:
    """记录生成调用并返回固定回答。

    Attributes:
        answer: 每次生成时返回的固定回答。
        calls: 已接收的问题和检索上下文。
    """

    def __init__(self, answer: str = "答案[1]") -> None:
        """保存固定回答。

        Args:
            answer: 每次生成调用返回的文本。
        """
        self.answer = answer
        self.calls: list[tuple[str, list[RetrievalResult]]] = []

    def generate(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> str:
        """记录本次生成使用的问题和上下文。

        Args:
            question: 用户问题。
            results: 提供给生成器的检索上下文。

        Returns:
            构造时设置的固定回答。
        """
        self.calls.append((question, results))
        return self.answer


class SequenceClock:
    """依次返回预设时间点以便断言毫秒延迟。

    Attributes:
        _values: 预设时间点的迭代器。
    """

    def __init__(self, *values: float) -> None:
        """保存测试时钟依次返回的时间点。

        Args:
            *values: 依次返回的单调时间值。
        """
        self._values = iter(values)

    def __call__(self) -> float:
        """返回下一个单调时钟值。

        Returns:
            下一个预设时间点。
        """
        return next(self._values)


def test_pipeline_returns_answer_citations_and_latency() -> None:
    """生成一次答案，并只返回回答实际引用的来源。"""
    results = [make_result(0), make_result(1, page=None)]
    retriever = FakeRetriever(results)
    generator = SpyAnswerGenerator("FastAPI 使用路径占位符[1]。")
    pipeline = QueryService(
        retriever=retriever,
        answer_generator=generator,
        clock=SequenceClock(10.0, 10.025, 10.125),
    )

    response = pipeline.query(
        "路径参数如何声明？",
        strategy=RetrievalStrategy.VECTOR,
        top_k=2,
    )

    assert retriever.calls == [
        ("路径参数如何声明？", RetrievalStrategy.VECTOR, 2)
    ]
    assert generator.calls == [("路径参数如何声明？", results)]
    assert response.answer == "FastAPI 使用路径占位符[1]。"
    assert response.latency_ms == 125.0
    assert response.retrieval_strategy is RetrievalStrategy.VECTOR
    assert [citation.model_dump() for citation in response.citations] == [
        {
            "doc_id": "doc-a",
            "filename": "fastapi.pdf",
            "page": 12,
            "chunk_text": "文本块 0",
            "position": 1,
            "chunk_id": "doc-a:0", "kb_id": "default", "doc_updated_at": None,
        },
    ]


def test_pipeline_preserves_first_citation_order_from_answer() -> None:
    """来源顺序必须与回答中的首次引用顺序一致。"""
    results = [make_result(0), make_result(1)]
    pipeline = QueryService(
        retriever=FakeRetriever(results),
        answer_generator=SpyAnswerGenerator(
            "第二项[2]，第一项[1]，重复[2]。"
        ),
        clock=SequenceClock(1.0, 1.01, 1.02),
    )

    response = pipeline.query("问题", top_k=2)

    assert [item.chunk_text for item in response.citations] == [
        "文本块 1",
        "文本块 0",
    ]


def test_pipeline_logs_missing_citations_for_uncited_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """正常回答没有有效编号时必须记录结构化日志。

    Args:
        caplog: pytest 提供的日志捕获器。
    """
    pipeline = QueryService(
        retriever=FakeRetriever([make_result(0)]),
        answer_generator=SpyAnswerGenerator(
            "这是没有引用编号的事实回答。"
        ),
        clock=SequenceClock(1.0, 1.01, 1.02),
    )

    with caplog.at_level(
        logging.WARNING,
        logger="app.application.query.service",
    ):
        response = pipeline.query("问题", top_k=1)

    assert response.citations == []
    warning_records = [
        record
        for record in caplog.records
        if record.getMessage() == "生成回答缺少有效引用"
    ]
    assert len(warning_records) == 1
    assert warning_records[0].missing_citations is True


def test_pipeline_does_not_flag_insufficient_context_as_missing_citation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """知识不足拒答不是事实回答，不应计入缺失引用。

    Args:
        caplog: pytest 提供的日志捕获器。
    """
    pipeline = QueryService(
        retriever=FakeRetriever([make_result(0)]),
        answer_generator=SpyAnswerGenerator(INSUFFICIENT_CONTEXT_ANSWER),
        clock=SequenceClock(1.0, 1.01, 1.02),
    )

    with caplog.at_level(
        logging.WARNING,
        logger="app.application.query.service",
    ):
        response = pipeline.query("问题", top_k=1)

    assert response.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert response.citations == []
    missing_records = [
        record
        for record in caplog.records
        if record.getMessage() == "生成回答缺少有效引用"
    ]
    assert missing_records == []


def test_pipeline_uses_injected_citation_extractor() -> None:
    """Pipeline 应使用调用方注入的引用提取策略。"""

    def select_second_citation(
        answer: str,
        results: list[RetrievalResult],
    ) -> list[Citation]:
        """返回第二条来源以证明 Pipeline 使用注入边界。

        Args:
            answer: 生成回答；本测试策略不解析其内容。
            results: Pipeline 限制后的检索上下文。

        Returns:
            只包含第二条检索结果的来源列表。
        """
        del answer
        result = results[1]
        return [
            Citation(
                doc_id=result.doc_id,
                filename=result.filename,
                page=result.page,
                chunk_text=result.text,
            )
        ]

    pipeline = QueryService(
        retriever=FakeRetriever([make_result(0), make_result(1)]),
        answer_generator=SpyAnswerGenerator("答案[1]"),
        clock=SequenceClock(1.0, 1.01, 1.02),
        citation_extractor=select_second_citation,
    )

    response = pipeline.query("问题", top_k=2)

    assert [item.chunk_text for item in response.citations] == ["文本块 1"]


def test_pipeline_skips_llm_when_retrieval_is_empty() -> None:
    """知识库没有命中时应直接拒答并跳过模型调用。"""
    retriever = FakeRetriever([])
    generator = SpyAnswerGenerator()
    pipeline = QueryService(
        retriever=retriever,
        answer_generator=generator,
        clock=SequenceClock(2.0, 2.0005, 2.001),
    )

    response = pipeline.query("未知问题", strategy="vector", top_k=5)

    assert generator.calls == []
    assert response.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert response.citations == []
    assert response.latency_ms == 1.0
    assert response.retrieval_strategy is RetrievalStrategy.VECTOR


def test_pipeline_limits_generation_context_to_requested_top_k() -> None:
    """生成上下文不得超过调用方请求的 ``top_k``。"""

    class OverReturningRetriever(FakeRetriever):
        """模拟违反结果数量协议的自定义检索器。"""

        def retrieve(
            self,
            question: str,
            strategy: RetrievalStrategy | str,
            top_k: int,
        ) -> list[RetrievalResult]:
            """忽略 ``top_k`` 并返回全部结果。

            Args:
                question: 用户问题。
                strategy: 本次检索策略。
                top_k: 调用方请求的结果上限；测试故意忽略。

            Returns:
                未截断的全部预设检索结果。
            """
            self.calls.append((question, strategy, top_k))
            return self.results

    retriever = OverReturningRetriever(
        [make_result(0), make_result(1), make_result(2)]
    )
    generator = SpyAnswerGenerator("答案[1][2]")
    pipeline = QueryService(
        retriever=retriever,
        answer_generator=generator,
        clock=SequenceClock(1.0, 1.005, 1.01),
    )

    response = pipeline.query("问题", top_k=2)

    assert len(generator.calls[0][1]) == 2
    assert len(response.citations) == 2


def test_pipeline_logs_retrieval_generation_and_total_timings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """成功查询应输出检索、生成与总耗时结构化字段。"""
    pipeline = QueryService(
        retriever=FakeRetriever([make_result(0)]),
        answer_generator=SpyAnswerGenerator(),
        clock=SequenceClock(5.0, 5.01, 5.03),
    )

    with caplog.at_level(
        logging.INFO,
        logger="app.application.query.service",
    ):
        response = pipeline.query(
            "问题",
            strategy="hybrid_rerank",
            top_k=1,
        )

    timing_records = [
        record
        for record in caplog.records
        if record.getMessage() == "RAG 查询阶段耗时"
    ]
    assert len(timing_records) == 1
    record = timing_records[0]
    assert record.retrieval_strategy == "hybrid_rerank"
    assert record.retrieval_ms >= 0
    assert record.generation_ms >= 0
    assert record.total_ms >= 0
    assert response.latency_ms == record.total_ms


def test_pipeline_evaluate_query_returns_one_retrieval_trace() -> None:
    """评估查询应复用一次检索生成答案并返回脱敏追踪字段。"""
    results = [make_result(0), make_result(1, page=None)]
    retriever = FakeRetriever(results)
    generator = SpyAnswerGenerator("答案[2]")
    pipeline = QueryService(
        retriever=retriever,
        answer_generator=generator,
        clock=SequenceClock(10.0, 10.025, 10.125),
    )

    response = pipeline.evaluate_query(
        "评估问题",
        strategy=RetrievalStrategy.HYBRID,
        top_k=2,
    )

    assert retriever.calls == [
        ("评估问题", RetrievalStrategy.HYBRID, 2)
    ]
    assert generator.calls[0][1][0] is results[0]
    assert generator.calls[0][1][1] is results[1]
    assert response.answer == "答案[2]"
    assert response.retrieval_ms == 25.0
    assert response.generation_ms == 100.0
    assert response.latency_ms == 125.0
    assert response.retrieval_strategy is RetrievalStrategy.HYBRID
    assert response.retrieved[0].rank == 1
    assert response.retrieved[0].text == "文本块 0"
    assert response.retrieved[0].text_sha256 == hashlib.sha256(
        "文本块 0".encode("utf-8")
    ).hexdigest()
    assert "text" not in response.model_dump()["retrieved"][0]


def test_pipeline_query_keeps_existing_public_schema() -> None:
    """新增评估接口不得向普通查询结果泄露追踪字段。"""
    pipeline = QueryService(
        retriever=FakeRetriever([make_result(0)]),
        answer_generator=SpyAnswerGenerator(),
        clock=SequenceClock(1.0, 1.01, 1.02),
    )

    response = pipeline.query("普通问题")

    assert set(response.model_dump()) == {
        "answer",
        "citations",
        "latency_ms",
        "retrieval_strategy",
        "answer_mode", "retrieval_calls", "metadata", "created_at",
    }


def test_pipeline_close_releases_retriever_resources() -> None:
    """Pipeline 关闭时应把资源释放请求传给检索器。"""

    class ClosableRetriever(FakeRetriever):
        """记录资源是否被 Pipeline 释放。"""

        def __init__(self) -> None:
            """创建尚未关闭的测试检索器。"""
            super().__init__([])
            self.closed = False

        def close(self) -> None:
            self.closed = True

    retriever = ClosableRetriever()
    pipeline = QueryService(retriever, SpyAnswerGenerator())

    pipeline.close()

    assert retriever.closed is True
