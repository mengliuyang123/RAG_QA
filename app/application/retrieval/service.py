# -*- coding: utf-8 -*-
"""编排向量、关键词和候选重排检索用例。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from time import perf_counter

from app.application.errors import (
    MissingKeywordRetrieverError,
    MissingRerankerError,
    VectorStoreError,
)
from app.application.errors import IngestionJobStorageError
from app.application.execution_control import check_cancellation
from app.application.ports.embedding import EmbeddingProvider
from app.application.ports.keyword_index import KeywordIndex
from app.application.ports.ingestion_jobs import DocumentPublication
from app.application.ports.reranking import Reranker
from app.application.ports.vector_store import VectorStoreReader
from app.application.retrieval.fusion import reciprocal_rank_fusion
from app.domain.errors import (
    InvalidQuestionError,
    InvalidTopKError,
    UnsupportedRetrievalStrategyError,
)
from app.domain.retrieval import (
    RetrievalResult,
    RetrievalStrategy,
    VectorSearchResult,
)
from app.domain.knowledge_bases import validate_kb_ids


logger = logging.getLogger(__name__)


class VectorRetriever:
    """将问题向量化并查询只读向量存储。

    Attributes:
        _vector_store: 提供向量相似性搜索的只读存储。
        _embedder: 生成查询向量的提供者。
    """

    def __init__(
        self,
        vector_store: VectorStoreReader,
        embedder: EmbeddingProvider,
    ) -> None:
        """装配只读向量存储与查询向量器。

        Args:
            vector_store: 提供向量相似性查询的存储。
            embedder: 生成查询文本向量的提供者。

        Returns:
            None。
        """
        self._vector_store = vector_store
        self._embedder = embedder

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        *,
        kb_ids: tuple[str, ...] | None = None,
        excluded_doc_ids: tuple[str, ...] = (),
    ) -> list[RetrievalResult]:
        """返回与问题最相关的文本块，并保留存储既有顺序。

        Args:
            question: 待检索的用户问题。
            top_k: 需要从向量存储读取的候选数量。
            kb_ids: 在向量召回前强制施加的范围。
            excluded_doc_ids: 在向量Top-K之前排除的文档身份。

        Returns:
            已转换为统一结果模型的向量检索命中。

        Raises:
            InvalidQuestionError: 问题仅包含空白字符。
            InvalidTopKError: 候选数量不是正数。
        """
        normalized_question = self._validate_question(question)
        self._validate_candidate_k(top_k)
        validate_kb_ids(kb_ids)
        if kb_ids == ():
            return []
        check_cancellation()
        query_vector = self._embedder.embed_query(normalized_question)
        check_cancellation()
        if excluded_doc_ids:
            matches = self._vector_store.query_by_embedding(
                query_vector, top_k, kb_ids=kb_ids,
                excluded_doc_ids=excluded_doc_ids,
            )
        elif kb_ids is None:
            matches = self._vector_store.query_by_embedding(query_vector, top_k)
        else:
            matches = self._vector_store.query_by_embedding(
                query_vector, top_k, kb_ids=kb_ids,
            )
        for match in matches:
            if kb_ids is not None and match.chunk.kb_id not in kb_ids:
                raise VectorStoreError("向量召回违反知识库范围")
            if match.chunk.doc_id in excluded_doc_ids:
                raise VectorStoreError("向量召回返回了未发布文档")
        return [self._to_retrieval_result(match) for match in matches]

    @staticmethod
    def _validate_question(question: str) -> str:
        """清理问题首尾空白，并拒绝空问题。"""
        normalized_question = question.strip()
        if not normalized_question:
            raise InvalidQuestionError("检索问题不能为空")
        return normalized_question

    @staticmethod
    def _validate_candidate_k(top_k: int) -> None:
        """确保内部向量候选数量为正数。"""
        if top_k <= 0:
            raise InvalidTopKError("top_k 必须大于 0")

    @staticmethod
    def _to_retrieval_result(match: VectorSearchResult) -> RetrievalResult:
        """将向量距离转换为统一的相关性结果。"""
        chunk = match.chunk
        return RetrievalResult(
            chunk_id=chunk.id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            filename=chunk.filename,
            page=chunk.page,
            chunk_index=chunk.chunk_index,
            score=1.0 - match.distance,
            kb_id=chunk.kb_id,
            doc_updated_at=chunk.doc_updated_at,
        )


class RetrievalService:
    """统一校验请求，并编排纯向量、混合和重排检索。

    Attributes:
        _vector_retriever: 负责向量候选召回的组件。
        _keyword_retriever: 可选的关键词索引端口。
        _reranker: 可选的候选重排端口。
        _top_k_rerank: 重排前保留的融合候选数量。
        _clock: 可替换的单调时钟函数。
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordIndex | None = None,
        reranker: Reranker | None = None,
        top_k_rerank: int = 20,
        clock: Callable[[], float] = perf_counter,
        *,
        publication: DocumentPublication | None = None,
    ) -> None:
        """装配检索策略用到的端口实现。

        Args:
            vector_retriever: 执行向量候选召回的组件。
            keyword_retriever: hybrid 策略可选的关键词索引。
            reranker: hybrid_rerank 策略可选的候选重排器。
            top_k_rerank: 重排前保留的融合候选数。
            clock: 返回单调时间的可替换函数。
            publication: 只读发布登记；生产装配必须提供。

        Returns:
            None。
        """
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        self._reranker = reranker
        self._top_k_rerank = top_k_rerank
        self._clock = clock
        self._publication = publication

    def retrieve(
        self,
        question: str,
        strategy: RetrievalStrategy | str = RetrievalStrategy.VECTOR,
        top_k: int = 5,
        *,
        kb_ids: tuple[str, ...] | None = None,
    ) -> list[RetrievalResult]:
        """校验策略并返回结构统一的检索结果。

        Args:
            question: 待检索的用户问题。
            strategy: vector、hybrid 或 hybrid_rerank 策略。
            top_k: 最终返回结果的数量上限。
            kb_ids: 向量和关键词两路共同使用的实际范围。

        Returns:
            按选定策略生成的有序检索结果。

        Raises:
            InvalidQuestionError: 问题仅包含空白字符。
            InvalidTopKError: 结果数量不在支持范围内。
            UnsupportedRetrievalStrategyError: 策略值不受支持。
            MissingKeywordRetrieverError: hybrid 策略缺少关键词索引。
            MissingRerankerError: hybrid_rerank 策略缺少重排器。
        """
        check_cancellation()
        try:
            selected_strategy = RetrievalStrategy(strategy)
        except ValueError as exc:
            raise UnsupportedRetrievalStrategyError(
                f"不支持的检索策略：{strategy}"
            ) from exc

        normalized_question = VectorRetriever._validate_question(question)
        self._validate_top_k(top_k)
        validate_kb_ids(kb_ids)
        if kb_ids == ():
            return []
        excluded_doc_ids = self._unpublished_documents()
        timings = {
            "vector_ms": 0.0,
            "bm25_ms": 0.0,
            "fusion_ms": 0.0,
            "rerank_ms": 0.0,
        }
        if selected_strategy is RetrievalStrategy.VECTOR:
            started_at = self._clock()
            results = self._vector_search(
                normalized_question, top_k, kb_ids, excluded_doc_ids,
            )
            self._check_publication_snapshot(excluded_doc_ids)
            timings["vector_ms"] = self._elapsed_ms(started_at)
        elif selected_strategy is RetrievalStrategy.HYBRID:
            results = self._hybrid_search(
                normalized_question,
                top_k,
                timings,
                kb_ids,
                excluded_doc_ids,
            )
        elif selected_strategy is RetrievalStrategy.HYBRID_RERANK:
            if self._reranker is None:
                raise MissingRerankerError("hybrid_rerank 策略缺少重排器")
            if self._top_k_rerank < top_k:
                raise InvalidTopKError("top_k_rerank 必须大于或等于 top_k")
            candidates = self._hybrid_search(
                normalized_question,
                self._top_k_rerank,
                timings,
                kb_ids,
                excluded_doc_ids,
            )
            started_at = self._clock()
            check_cancellation()
            results = self._rerank(normalized_question, candidates, top_k)
            timings["rerank_ms"] = self._elapsed_ms(started_at)
        else:
            raise UnsupportedRetrievalStrategyError(
                f"不支持的检索策略：{selected_strategy}"
            )

        logger.info(
            "检索阶段耗时",
            extra={
                "retrieval_strategy": selected_strategy.value,
                **timings,
            },
        )
        for result in results:
            if kb_ids is not None and result.kb_id not in kb_ids:
                raise VectorStoreError("检索结果违反知识库范围")
            if result.doc_id in excluded_doc_ids:
                raise VectorStoreError("检索结果包含未发布文档")
        return results

    def _unpublished_documents(self) -> tuple[str, ...]:
        check_cancellation()
        if self._publication is None:
            return ()
        try:
            identities = self._publication.unpublished_document_ids()
        except IngestionJobStorageError as error:
            raise VectorStoreError("文档发布状态暂时不可用") from error
        return tuple(str(identity) for identity in identities)

    def _check_publication_snapshot(
        self, excluded_doc_ids: tuple[str, ...],
    ) -> None:
        # 查询向量生成包含远程等待；期间可能新增待发布文档。如果仍沿用
        # 旧排除集，新的部分块可能已挤占Top-K。拒绝此轮不一致结果，不能
        # 静默后过滤，也不能把它们发送给重排／生成等外部服务。
        current = set(self._unpublished_documents())
        if current.difference(excluded_doc_ids):
            raise VectorStoreError("文档发布状态已变化，请重试查询")

    def close(self) -> None:
        """释放重排器自行持有的远程连接池。

        Returns:
            None。
        """
        if self._reranker is None:
            return
        close = getattr(self._reranker, "close", None)
        if callable(close):
            close()

    def _vector_search(
        self,
        question: str,
        top_k: int,
        kb_ids: tuple[str, ...] | None = None,
        excluded_doc_ids: tuple[str, ...] = (),
    ) -> list[RetrievalResult]:
        """执行一次向量召回。"""
        if excluded_doc_ids:
            return self._vector_retriever.retrieve(
                question, top_k, kb_ids=kb_ids,
                excluded_doc_ids=excluded_doc_ids,
            )
        if kb_ids is None:
            return self._vector_retriever.retrieve(question, top_k)
        return self._vector_retriever.retrieve(question, top_k, kb_ids=kb_ids)

    def _bm25_search(
        self,
        question: str,
        top_k: int,
        kb_ids: tuple[str, ...] | None = None,
        excluded_doc_ids: tuple[str, ...] = (),
    ) -> list[RetrievalResult]:
        """执行一次 BM25 召回，并检查关键词依赖。"""
        if self._keyword_retriever is None:
            raise MissingKeywordRetrieverError("混合检索策略缺少关键词检索器")
        if excluded_doc_ids:
            results = self._keyword_retriever.search(
                question, top_k, kb_ids=kb_ids,
                excluded_doc_ids=excluded_doc_ids,
            )
        elif kb_ids is None:
            return self._keyword_retriever.search(question, top_k)
        else:
            results = self._keyword_retriever.search(
                question, top_k, kb_ids=kb_ids,
            )
        for result in results:
            if kb_ids is not None and result.kb_id not in kb_ids:
                raise VectorStoreError("关键词召回违反知识库范围")
            if result.doc_id in excluded_doc_ids:
                raise VectorStoreError("关键词召回返回了未发布文档")
        return results

    @staticmethod
    def _rrf_fuse(
        vector_results: list[RetrievalResult],
        keyword_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """复用 RRF 纯函数融合两路排名。"""
        return reciprocal_rank_fusion([vector_results, keyword_results])

    def _hybrid_search(
        self,
        question: str,
        top_k: int,
        timings: dict[str, float],
        kb_ids: tuple[str, ...] | None = None,
        excluded_doc_ids: tuple[str, ...] = (),
    ) -> list[RetrievalResult]:
        """扩大两路候选集，融合后限制为指定数量。"""
        candidate_k = max(top_k * 4, 20)
        started_at = self._clock()
        vector_results = self._vector_search(
            question, candidate_k, kb_ids, excluded_doc_ids,
        )
        timings["vector_ms"] = self._elapsed_ms(started_at)
        started_at = self._clock()
        check_cancellation()
        keyword_results = self._bm25_search(
            question, candidate_k, kb_ids, excluded_doc_ids,
        )
        self._check_publication_snapshot(excluded_doc_ids)
        timings["bm25_ms"] = self._elapsed_ms(started_at)
        started_at = self._clock()
        results = self._rrf_fuse(vector_results, keyword_results)[:top_k]
        timings["fusion_ms"] = self._elapsed_ms(started_at)
        return results

    def _rerank(
        self,
        question: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """使用已装配的重排器精排融合候选。"""
        if self._reranker is None:
            raise MissingRerankerError("hybrid_rerank 策略缺少重排器")
        return self._reranker.rerank(question, candidates, top_k)

    def _elapsed_ms(self, started_at: float) -> float:
        """把单调时钟差转换为非负毫秒数。"""
        return max(0.0, round((self._clock() - started_at) * 1000, 3))

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        """把公共请求的结果数限制在 API 支持范围内。"""
        if not 1 <= top_k <= 20:
            raise InvalidTopKError("top_k 必须在 1 到 20 之间")
