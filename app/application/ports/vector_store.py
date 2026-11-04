# -*- coding: utf-8 -*-
"""声明向量存储读、写和管理端口。"""

from typing import Protocol

from app.domain.documents import DocumentChunk
from app.domain.retrieval import VectorSearchResult


class VectorStoreReader(Protocol):
    """定义检索和文档管理所需的向量存储读取能力。"""

    def query_by_embedding(
        self,
        vector: list[float],
        top_k: int,
        *,
        kb_ids: tuple[str, ...] | None = None,
        excluded_doc_ids: tuple[str, ...] = (),
    ) -> list[VectorSearchResult]:
        """按查询向量返回数据库既有顺序的命中结果。

        Args:
            vector: 查询文本的向量。
            top_k: 返回命中数量上限。
            kb_ids: 召回前的范围；None兼容全库，空tuple不返回结果。
            excluded_doc_ids: Top-K前必须排除的未发布文档。

        Returns:
            按向量存储既有顺序排列的命中结果。
        """

    def list_chunks(self) -> list[DocumentChunk]:
        """稳定列出当前保存的全部文本块。

        Returns:
            按稳定顺序排列的全部文本块。
        """


class VectorStoreWriter(Protocol):
    """定义入库和补偿所需的向量存储写入能力。"""

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """写入一批文本块及对应向量。

        Args:
            chunks: 待持久化的文本块。
            embeddings: 与文本块一一对应的向量。

        Returns:
            None。
        """

    def delete_document(self, doc_id: str) -> None:
        """删除指定文档的全部向量记录。

        Args:
            doc_id: 待删除文档的稳定唯一标识。

        Returns:
            None。
        """

    def list_chunks(self) -> list[DocumentChunk]:
        """稳定列出当前保存的全部文本块。

        Returns:
            按稳定顺序排列的全部文本块。
        """


class VectorStoreAdmin(Protocol):
    """定义向量存储运行维护所需的管理能力。"""

    def collection_metadata(self) -> dict[str, object]:
        """返回当前 collection 元数据的防御性副本。

        Returns:
            当前 collection 的元数据。
        """

    def update_collection_metadata(
        self,
        metadata: dict[str, str | int],
    ) -> None:
        """合并并持久化 collection 元数据。

        Args:
            metadata: 要覆盖或追加的字符串、整数元数据。

        Returns:
            None。
        """

    def delete_collection(self) -> None:
        """删除构造时绑定的精确 collection。

        Returns:
            None。
        """

    def count(self) -> int:
        """返回 collection 当前记录数。

        Returns:
            collection 中的记录数量。
        """
