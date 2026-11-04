# -*- coding: utf-8 -*-
"""编排检索、答案生成与查询结果组装。"""

import hashlib
import logging
import sys
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from app.application.errors import EmbeddingError
from app.application.errors import LLMError
from app.application.errors import RerankerUnavailableError
from app.application.errors import VectorStoreError
from app.application.query.citations import extract_citations
from app.application.query.events import CompleteStreamData
from app.application.query.events import CompleteStreamEvent
from app.application.query.events import ErrorStreamData
from app.application.query.events import ErrorStreamEvent
from app.application.query.events import StreamEvent
from app.application.query.events import TokenStreamData
from app.application.query.events import TokenStreamEvent
from app.application.query.prompting import INSUFFICIENT_CONTEXT_ANSWER
from app.domain.answers import Citation
from app.domain.answers import EvaluationQueryResult
from app.domain.answers import QueryResult
from app.domain.answers import RetrievedChunkTrace
from app.domain.errors import RetrievalError
from app.domain.errors import UnsupportedRetrievalStrategyError
from app.domain.retrieval import RetrievalResult
from app.domain.retrieval import RetrievalStrategy


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _QueryExecution:
    answer: str
    citations: list[Citation]
    results: list[RetrievalResult]
    retrieval_ms: float
    generation_ms: float
    latency_ms: float
    retrieval_strategy: RetrievalStrategy


class RetrieverProtocol(Protocol):
    """定义查询编排层依赖的最小检索接口。"""

    def retrieve(
        self,
        question: str,
        strategy: RetrievalStrategy | str,
        top_k: int,
        *,
        kb_ids: tuple[str, ...] | None = None,
    ) -> list[RetrievalResult]:
        """返回与问题相关的有序文本块。

        Args:
            question: 用户问题。
            strategy: 本次使用的检索策略。
            top_k: 最终结果数量上限。
            kb_ids: 两路召回共同施加的实际知识库范围。

        Returns:
            按相关性排序的检索结果。
        """


class AnswerGeneratorProtocol(Protocol):
    """定义查询编排层依赖的最小答案生成接口。"""

    def generate(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> str:
        """根据问题和检索结果生成完整答案。

        Args:
            question: 用户问题。
            results: 提供给语言模型的有序检索上下文。

        Returns:
            语言模型生成的完整回答。
        """

    def stream(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> Iterator[str]:
        """根据问题和检索结果逐个返回回答增量。

        Args:
            question: 用户问题。
            results: 提供给语言模型的有序检索上下文。

        Returns:
            逐个产生非空文本增量的迭代器。
        """


CitationExtractor = Callable[
    [str, list[RetrievalResult]],
    list[Citation],
]


class QueryService:
    """执行非流式、评估和流式知识库查询用例。"""

    def __init__(
        self,
        retriever: RetrieverProtocol,
        answer_generator: AnswerGeneratorProtocol,
        clock: Callable[[], float] = perf_counter,
        citation_extractor: CitationExtractor = extract_citations,
    ) -> None:
        """装配查询链路依赖。

        Args:
            retriever: 提供统一策略入口的检索器。
            answer_generator: 根据检索上下文生成回答的组件。
            clock: 返回单调时间的可替换函数。
            citation_extractor: 把回答编号映射为来源的纯函数。
        """
        self._retriever = retriever
        self._answer_generator = answer_generator
        self._clock = clock
        self._citation_extractor = citation_extractor

    def query(
        self,
        question: str,
        strategy: RetrievalStrategy | str = RetrievalStrategy.VECTOR,
        top_k: int = 5,
    ) -> QueryResult:
        """检索上下文、按需生成答案并返回可信来源。

        Args:
            question: 用户问题。
            strategy: 本次使用的检索策略。
            top_k: 参与生成的文本块数量上限。

        Returns:
            包含回答、实际引用来源和耗时的查询结果。
        """
        execution = self._execute_query(
            question,
            strategy,
            top_k,
        )
        return QueryResult(
            answer=execution.answer,
            citations=execution.citations,
            latency_ms=execution.latency_ms,
            retrieval_strategy=execution.retrieval_strategy,
        )

    def evaluate_query(
        self,
        question: str,
        strategy: RetrievalStrategy | str = RetrievalStrategy.VECTOR,
        top_k: int = 5,
    ) -> EvaluationQueryResult:
        """执行一次查询并返回评估所需的有序检索追踪。

        Args:
            question: 用户问题。
            strategy: 本次使用的检索策略。
            top_k: 参与生成的文本块数量上限。

        Returns:
            包含回答、来源、文本哈希和阶段耗时的评估
            结果。
        """
        execution = self._execute_query(question, strategy, top_k)
        retrieved: list[RetrievedChunkTrace] = []
        for rank, result in enumerate(execution.results, start=1):
            text_sha256 = hashlib.sha256(
                result.text.encode("utf-8")
            ).hexdigest()
            retrieved.append(
                RetrievedChunkTrace(
                    rank=rank,
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    text=result.text,
                    text_sha256=text_sha256,
                    filename=result.filename,
                    page=result.page,
                    chunk_index=result.chunk_index,
                    score=result.score,
                )
            )
        return EvaluationQueryResult(
            answer=execution.answer,
            citations=execution.citations,
            retrieved=retrieved,
            retrieval_ms=execution.retrieval_ms,
            generation_ms=execution.generation_ms,
            latency_ms=execution.latency_ms,
            retrieval_strategy=execution.retrieval_strategy,
        )

    def _execute_query(
        self,
        question: str,
        strategy: RetrievalStrategy | str,
        top_k: int,
    ) -> _QueryExecution:
        """执行普通查询与评估查询共用的非流式链路。"""
        started_at = self._clock()
        selected_strategy = self._select_strategy(strategy)
        bounded_results = self._retrieve_bounded(
            question,
            selected_strategy,
            top_k,
        )
        retrieval_finished_at = self._clock()
        retrieval_ms = max(
            0.0,
            round((retrieval_finished_at - started_at) * 1000, 3),
        )
        if bounded_results:
            answer = self._answer_generator.generate(
                question,
                bounded_results,
            )
            citations = self._citation_extractor(answer, bounded_results)
            if not citations and answer != INSUFFICIENT_CONTEXT_ANSWER:
                logger.warning(
                    "生成回答缺少有效引用",
                    extra={"missing_citations": True},
                )
        else:
            answer = INSUFFICIENT_CONTEXT_ANSWER
            citations = []

        finished_at = self._clock()
        generation_ms = 0.0
        if bounded_results:
            generation_ms = max(
                0.0,
                round(
                    (finished_at - retrieval_finished_at) * 1000,
                    3,
                ),
            )
        latency_ms = max(
            0.0,
            round((finished_at - started_at) * 1000, 3),
        )
        logger.info(
            "RAG 查询阶段耗时",
            extra={
                "retrieval_strategy": selected_strategy.value,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": latency_ms,
            },
        )
        return _QueryExecution(
            answer=answer,
            citations=citations,
            results=bounded_results,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            latency_ms=latency_ms,
            retrieval_strategy=selected_strategy,
        )

    def stream_query(
        self,
        question: str,
        strategy: RetrievalStrategy | str = RetrievalStrategy.VECTOR,
        top_k: int = 5,
    ) -> Iterator[StreamEvent]:
        """检索上下文并逐个返回回答增量与完成事件。

        Args:
            question: 用户问题。
            strategy: 本次使用的检索策略。
            top_k: 参与生成的文本块数量上限。

        Yields:
            回答增量以及唯一的完成事件。
        """
        selected_strategy = self._select_strategy(strategy)
        terminal_event = yield from self._stream_query_events(
            question,
            selected_strategy,
            top_k,
        )
        yield terminal_event

    def _stream_query_events(
        self,
        question: str,
        selected_strategy: RetrievalStrategy,
        top_k: int,
    ) -> Generator[StreamEvent, None, StreamEvent]:
        """执行受控操作并返回唯一终止事件。"""
        try:
            started_at = self._clock()
        except Exception:
            logger.exception("RAG 流式查询初始化计时失败")
            return self._internal_stream_error()

        try:
            bounded_results = self._retrieve_bounded(
                question,
                selected_strategy,
                top_k,
            )
        except (
            EmbeddingError,
            RetrievalError,
            RerankerUnavailableError,
            VectorStoreError,
        ):
            logger.exception("RAG 流式查询检索失败")
            return ErrorStreamEvent(
                data=ErrorStreamData(
                    code="retrieval_error",
                    message="检索服务暂时不可用",
                ),
            )
        except Exception:
            logger.exception("RAG 流式查询发生未知检索错误")
            return self._internal_stream_error()

        try:
            retrieval_finished_at = self._clock()
        except Exception:
            logger.exception("RAG 流式查询检索计时失败")
            return self._internal_stream_error()

        if not bounded_results:
            try:
                finished_at = self._clock()
                latency_ms = max(
                    0.0,
                    round((finished_at - started_at) * 1000, 3),
                )
                return CompleteStreamEvent(
                    data=CompleteStreamData(
                        answer=INSUFFICIENT_CONTEXT_ANSWER,
                        citations=[],
                        latency_ms=latency_ms,
                        retrieval_strategy=selected_strategy,
                    )
                )
            except Exception:
                logger.exception("RAG 流式查询完成计时失败")
                return self._internal_stream_error()

        return (yield from self._stream_answer_events(
            question,
            bounded_results,
            selected_strategy,
            started_at,
            retrieval_finished_at,
        ))

    def _stream_answer_events(
        self,
        question: str,
        bounded_results: list[RetrievalResult],
        selected_strategy: RetrievalStrategy,
        started_at: float,
        retrieval_finished_at: float,
    ) -> Generator[StreamEvent, None, StreamEvent]:
        """发送回答增量并把依赖失败转换为终止事件。"""
        try:
            answer_iterator = self._answer_generator.stream(
                question,
                bounded_results,
            )
        except LLMError:
            logger.exception("RAG 流式查询生成失败")
            return self._generation_stream_error()
        except Exception:
            logger.exception("RAG 流式查询发生未知生成错误")
            return self._internal_stream_error()

        close_failed = False
        try:
            terminal_event = yield from self._consume_answer_iterator(
                answer_iterator,
                bounded_results,
                selected_strategy,
                started_at,
                retrieval_finished_at,
            )
        finally:
            active_error = sys.exception()
            close = getattr(answer_iterator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.exception("RAG 答案增量迭代器关闭失败")
                    if active_error is None:
                        close_failed = True

        if close_failed and terminal_event.event == "complete":
            return self._internal_stream_error()
        return terminal_event

    def _consume_answer_iterator(
        self,
        answer_iterator: Iterator[str],
        bounded_results: list[RetrievalResult],
        selected_strategy: RetrievalStrategy,
        started_at: float,
        retrieval_finished_at: float,
    ) -> Generator[StreamEvent, None, StreamEvent]:
        """消费答案迭代器并返回待发送的终止事件。"""
        tokens: list[str] = []
        while True:
            try:
                token = next(answer_iterator)
            except StopIteration:
                break
            except LLMError:
                logger.exception("RAG 流式查询生成失败")
                return self._generation_stream_error()
            except Exception:
                logger.exception("RAG 流式查询发生未知生成错误")
                return self._internal_stream_error()

            try:
                token_event = TokenStreamEvent(
                    data=TokenStreamData(token=token)
                )
            except Exception:
                logger.exception("RAG 流式查询构建增量事件失败")
                return self._internal_stream_error()

            tokens.append(token)
            yield token_event

        if not tokens:
            return ErrorStreamEvent(
                data=ErrorStreamData(
                    code="empty_generation",
                    message="语言模型未返回有效内容",
                ),
            )

        try:
            answer = "".join(tokens)
            citations = self._citation_extractor(
                answer,
                bounded_results,
            )
            finished_at = self._clock()
            latency_ms = max(
                0.0,
                round((finished_at - started_at) * 1000, 3),
            )
            logger.info(
                "RAG 流式查询阶段耗时",
                extra={
                    "retrieval_strategy": selected_strategy.value,
                    "retrieval_ms": max(
                        0.0,
                        round(
                            (retrieval_finished_at - started_at)
                            * 1000,
                            3,
                        ),
                    ),
                    "generation_ms": max(
                        0.0,
                        round(
                            (finished_at - retrieval_finished_at)
                            * 1000,
                            3,
                        ),
                    ),
                    "total_ms": latency_ms,
                },
            )
            return CompleteStreamEvent(
                data=CompleteStreamData(
                    answer=answer,
                    citations=citations,
                    latency_ms=latency_ms,
                    retrieval_strategy=selected_strategy,
                )
            )
        except LLMError:
            logger.exception("RAG 流式查询生成失败")
            return self._generation_stream_error()
        except Exception:
            logger.exception("RAG 流式查询发生未知生成错误")
            return self._internal_stream_error()

    @staticmethod
    def _generation_stream_error() -> ErrorStreamEvent:
        """构建不泄露上游细节的生成错误事件。"""
        return ErrorStreamEvent(
            data=ErrorStreamData(
                code="generation_error",
                message="语言模型服务暂时不可用",
            )
        )

    @staticmethod
    def _internal_stream_error() -> ErrorStreamEvent:
        """构建不泄露实现细节的内部错误事件。"""
        return ErrorStreamEvent(
            data=ErrorStreamData(
                code="internal_error",
                message="流式查询处理失败",
            )
        )

    def _select_strategy(
        self,
        strategy: RetrievalStrategy | str,
    ) -> RetrievalStrategy:
        """把策略输入转换为领域枚举。"""
        try:
            return RetrievalStrategy(strategy)
        except ValueError as exc:
            raise UnsupportedRetrievalStrategyError(
                f"不支持的检索策略：{strategy}"
            ) from exc

    def _retrieve_bounded(
        self,
        question: str,
        strategy: RetrievalStrategy,
        top_k: int,
    ) -> list[RetrievalResult]:
        """检索并防御性限制返回给生成器的上下文数量。"""
        results = self._retriever.retrieve(
            question,
            strategy=strategy,
            top_k=top_k,
        )
        # 自定义检索器可能违反数量协议，因此在生成边界
        # 再次限制上下文。
        return results[:top_k]

    def close(self) -> None:
        """释放检索链路中由生产组件持有的外部资源。

        Returns:
            None。
        """
        close = getattr(self._retriever, "close", None)
        if callable(close):
            close()
