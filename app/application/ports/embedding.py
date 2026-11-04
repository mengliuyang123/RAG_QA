# -*- coding: utf-8 -*-
"""声明文本向量化端口。"""

from typing import Protocol


class EmbeddingProvider(Protocol):
    """定义文档入库与查询共同依赖的向量化能力。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """按输入顺序返回多段文本向量。

        Args:
            texts: 待向量化的文本列表。

        Returns:
            与输入顺序一致的文本向量列表。
        """

    def embed_query(self, text: str) -> list[float]:
        """返回一条查询文本的向量。

        Args:
            text: 待向量化的查询文本。

        Returns:
            查询文本的向量。
        """
