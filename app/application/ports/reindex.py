# -*- coding: utf-8 -*-
"""声明按新分块规则原地重建已发布文档所需的存储端口。"""

from typing import Protocol

from app.domain.documents import DocumentChunk, ReindexCandidate


class DocumentReindexStore(Protocol):
    """列出可重建文档，并在一个事务内整体替换某篇文档的文本块。"""

    def candidates(
        self, kb_ids: tuple[str, ...] | None = None,
    ) -> list[ReindexCandidate]:
        """列出已发布、未删除、没有替换任务在进行的文档。

        Args:
            kb_ids: 限定的知识库；None 表示全部。

        Returns:
            按知识库、文件名稳定排序的候选；没有入库任务记录的文档不出现。
        """

    def documents_without_job(
        self, kb_ids: tuple[str, ...] | None = None,
    ) -> list[tuple[str, str]]:
        """列出已发布但找不到入库任务（因而找不到源文件）的文档。

        Returns:
            ``(kb_id, filename)`` 列表。
        """

    def swap_chunks(
        self, candidate: ReindexCandidate,
        chunks: list[DocumentChunk], embeddings: list[list[float]],
    ) -> bool:
        """在业务写锁下确认文档仍是同一已发布版本，再整体替换其文本块。

        Args:
            candidate: 规划时的文档快照。
            chunks: 新的子块，``doc_id`` 必须与候选一致。
            embeddings: 与子块一一对应的向量。

        Returns:
            True 表示已替换；False 表示文档期间被删除、替换或内容变化，未做任何修改。
        """
