# -*- coding: utf-8 -*-
"""声明文档删除修复记录端口。"""

from typing import Protocol
from uuid import UUID


class RepairLogWriter(Protocol):
    """定义持久化文档删除部分失败记录的能力。"""

    def append_document_deletion(self, doc_id: UUID, filename: str) -> None:
        """追加一条不含路径和正文的删除修复记录。

        Args:
            doc_id: 已完成索引删除的文档 UUID。
            filename: 不含目录的安全原始文件名。

        Returns:
            None。
        """
