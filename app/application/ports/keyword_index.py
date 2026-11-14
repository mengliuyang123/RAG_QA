# -*- coding: utf-8 -*-
"""声明关键词索引端口。"""

from typing import Protocol

from app.domain.documents import DocumentChunk
from app.domain.retrieval import RetrievalResult


class KeywordIndex(Protocol):
    """关键词查询及旧派生索引的重建接口。

    数据库适配器的 maintained_with_chunks=True 表示索引随块事务维护，
    调用者无需读取全量语料或调用 rebuild。
    """

    def rebuild(self, chunks: list[DocumentChunk]) -> None:
        """原子重建派生索引；随块事务维护的数据库实现忽略快照。

        Args:
            chunks: 用于构建新一代索引的完整文本块快照。

        Returns:
            None。
        """

    def search(
        self, query: str, top_k: int, *,
        kb_ids: tuple[str, ...] | None = None,
        excluded_doc_ids: tuple[str, ...] = (),
    ) -> list[RetrievalResult]:
        """按关键词相关性返回统一检索结果。

        Args:
            query: 用户输入的关键词查询。
            top_k: 返回结果数量上限。
            kb_ids: Top-K选取前的范围；空tuple不得回退全库。
            excluded_doc_ids: Top-K前排除的未发布文档。

        Returns:
            按关键词相关性排序的检索结果。
        """
