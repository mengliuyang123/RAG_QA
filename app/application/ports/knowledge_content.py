# -*- coding: utf-8 -*-
"""声明知识库目录及浏览所需的只读内容投影，不依赖向量模型。"""

from typing import Protocol
from uuid import UUID

from app.domain.documents import DocumentChunk
from app.domain.knowledge_views import IndexedChunkHeader
from app.domain.knowledge_views import IndexedDocumentHeader


class KnowledgeContentReader(Protocol):
    """将目录元数据统计与昂贵的检索／全文载荷隔离。"""

    def count_documents_by_kb(
        self, *, excluded_doc_ids: tuple[str, ...] = (),
    ) -> dict[str, int]:
        """仅用分页元数据统计各库去重文档数，不把chunk数当文档数。

        Args:
            excluded_doc_ids: 计数前排除的未发布文档身份。

        Returns:
            有文档知识库的精确计数，空库由权威目录补零。
        """

    def list_document_headers(self, kb_id: str | None = None) -> tuple[IndexedDocumentHeader, ...]:
        """按库读取文档元数据，不加载正文。

        Args:
            kb_id: 已验证知识库ID，None供管理列表读取全部元数据。

        Returns:
            稳定有序的文档头。
        """

    def list_chunk_headers(self, doc_id: UUID) -> tuple[IndexedChunkHeader, ...]:
        """只读取指定文档元数据以选择原文窗口。

        Args:
            doc_id: 目标文档UUID。

        Returns:
            同源、同版本的实际块顺序。
        """

    def read_chunks(self, headers: tuple[IndexedChunkHeader, ...]) -> tuple[DocumentChunk, ...]:
        """只对已选定的最多七块加载原文。

        Args:
            headers: 可信元数据窗口。

        Returns:
            经再次归属校验的当前正文。
        """
