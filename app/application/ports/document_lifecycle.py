# -*- coding: utf-8 -*-
"""区分已知删除请求和从未存在的来源，不保存正文。"""

from typing import Protocol
from uuid import UUID


class DocumentLifecycle(Protocol):
    """持久化删除前确认过的文档身份，当前是否存在仍由索引判断。"""

    def record_deletion(self, doc_id: UUID, kb_id: str) -> None:
        """在已验证文档存在后、删除索引前记录操作事实。

        Args:
            doc_id: 已确认存在的文档UUID。
            kb_id: 实际来源知识库。

        Returns:
            None；失败必须阻止尚未执行的索引删除。
        """

    def was_deleted(self, doc_id: UUID) -> bool:
        """读取已知删除请求，不能用它覆盖仍存在的索引正文。

        Args:
            doc_id: 当前索引已经确认不存在的文档。

        Returns:
            是否存在持久化删除记录；存储不可读时必须抛错。
        """
