# -*- coding: utf-8 -*-
"""面向精确术语召回的内存 BM25 关键词索引。"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.domain.documents import DocumentChunk
from app.domain.knowledge_bases import validate_kb_ids
from app.domain.retrieval import RetrievalResult


TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+|[\u3400-\u4dbf\u4e00-\u9fff]"
)


@dataclass(frozen=True)
class _BM25State:
    """一次完整构建且可被查询原子读取的索引状态。"""

    chunks: tuple[DocumentChunk, ...]
    tokenized_corpus: tuple[tuple[str, ...], ...]
    index: BM25Okapi | None


def tokenize(text: str) -> list[str]:
    """提取英文标识符、数字和中日韩统一表意字符。

    Args:
        text: 需要分词的原始文本。

    Returns:
        统一为小写的检索 token 列表。
    """
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


class BM25Index:
    """保存文本块的可全量重建 BM25 内存索引。

    Attributes:
        _chunk_loader: 查询前读取共享持久化文本块的可选函数。
    """

    def __init__(
        self,
        chunk_loader: Callable[[], list[DocumentChunk]] | None = None,
    ) -> None:
        """创建空索引。

        Args:
            chunk_loader: 查询前读取共享持久化文本块的可选
                函数。

        Returns:
            None。
        """
        self._state = _BM25State((), (), None)
        self._chunk_loader = chunk_loader

    def rebuild(self, chunks: list[DocumentChunk]) -> None:
        """用可分词文本块替换当前索引内容。

        Args:
            chunks: 用于构建新一代索引的完整文本块快照。

        Returns:
            None。
        """
        searchable: list[tuple[DocumentChunk, list[str]]] = []
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            if tokens:
                searchable.append((chunk, tokens))

        new_chunks = tuple(chunk for chunk, _ in searchable)
        tokenized_corpus = [tokens for _, tokens in searchable]
        new_index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

        # 局部构建完成后才发布，避免并发查询读到
        # 混合代际。
        self._state = _BM25State(
            chunks=new_chunks,
            tokenized_corpus=tuple(
                tuple(tokens)
                for tokens in tokenized_corpus
            ),
            index=new_index,
        )

    def search(
        self, query: str, top_k: int, *,
        kb_ids: tuple[str, ...] | None = None,
        excluded_doc_ids: tuple[str, ...] = (),
    ) -> list[RetrievalResult]:
        """返回至少命中一个查询 token 的最高分文本块。

        Args:
            query: 用户输入的关键词查询。
            top_k: 返回结果数量上限。
            kb_ids: 排序和Top-K前的实际范围，空tuple无结果。
            excluded_doc_ids: 排序和Top-K前排除的未发布文档。

        Returns:
            按 BM25 分数降序排列的检索结果。

        Raises:
            ValueError: ``top_k`` 不是正整数。
        """
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        validate_kb_ids(kb_ids)
        if kb_ids == ():
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        if self._chunk_loader is not None:
            # Chroma 是跨 worker 的事实来源；当前语料较小，
            # 查询前刷新可避免其他 worker 长期使用旧索引。
            self.rebuild(self._chunk_loader())
        state = self._state
        if state.index is None:
            return []

        query_token_set = set(query_tokens)
        selected_kbs = set(kb_ids) if kb_ids is not None else None
        excluded = set(excluded_doc_ids)
        matched_indices = []
        for index, document_tokens in enumerate(state.tokenized_corpus):
            if state.chunks[index].doc_id in excluded:
                continue
            if selected_kbs is not None:
                if state.chunks[index].kb_id not in selected_kbs:
                    continue
            if query_token_set.intersection(document_tokens):
                matched_indices.append(index)
        # 针对已过滤的候选计算批次分数，不为每个查询重建分库索引。
        batch_scores = state.index.get_batch_scores(query_tokens, matched_indices)
        scores = dict(zip(matched_indices, batch_scores, strict=True))
        ranked_indices = sorted(
            matched_indices,
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]
        return [
            self._to_retrieval_result(
                state.chunks[index],
                float(scores[index]),
            )
            for index in ranked_indices
        ]

    @staticmethod
    def _to_retrieval_result(
        chunk: DocumentChunk,
        score: float,
    ) -> RetrievalResult:
        """把索引文本块转换为统一检索结果。"""
        return RetrievalResult(
            chunk_id=chunk.id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            filename=chunk.filename,
            page=chunk.page,
            chunk_index=chunk.chunk_index,
            score=score,
            kb_id=chunk.kb_id,
            doc_updated_at=chunk.doc_updated_at,
        )
