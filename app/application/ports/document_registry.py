# -*- coding: utf-8 -*-
"""documents 表登记端口：查重、替换清理与按 doc_id 删除。"""

from typing import Protocol
from uuid import UUID

from app.domain.documents import StoredDocument


class DocumentRegistry(Protocol):
    """登记事实与分块删除在同一数据库事务内完成。"""

    def get(self, doc_id: UUID) -> StoredDocument | None:
        """读取一篇未删除的登记文档。

        Args:
            doc_id: 文档 UUID。

        Returns:
            未删除的登记，没有时为 None。
        """

    def list_live(self, kb_id: str | None = None) -> tuple[StoredDocument, ...]:
        """列出未删除的登记文档。

        Args:
            kb_id: 可选知识库过滤。

        Returns:
            按上传时间倒序的登记。
        """

    def replaced_by(self, doc_id: UUID) -> StoredDocument | None:
        """读取被 doc_id 这篇新版本替换掉（已删除）的旧文档。

        Args:
            doc_id: 已发布的新版本。

        Returns:
            已被删除的旧版本登记；没有替换时为 None。
        """

    def delete_documents(self, targets: dict[UUID, str]) -> None:
        """同一事务按 doc_id 删除分块、标记登记删除并记录删除事实。

        Args:
            targets: doc_id 到所属知识库的映射。

        Returns:
            None。
        """
