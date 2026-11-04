# -*- coding: utf-8 -*-
"""声明检索候选重排端口。"""

from typing import Protocol

from app.domain.retrieval import RetrievalResult


class Reranker(Protocol):
    """定义按查询相关性重新排序候选的能力。"""

    def rerank(
        self,
        question: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """返回相关性重新排序后的候选。

        Args:
            question: 用户问题。
            candidates: 待重新排序的候选结果。
            top_k: 返回结果数量上限。

        Returns:
            按相关性重新排序并截断后的候选结果。
        """
