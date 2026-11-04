# -*- coding: utf-8 -*-
"""管理员文档只读投影，不暴露索引写入或模型能力。"""

from typing import Protocol

from app.domain.documents import DocumentSummary


class DocumentListing(Protocol):
    """提供现有文档外壳可复用的管理元数据。"""

    def list_documents(self, kb_id: str | None = None) -> list[DocumentSummary]:
        """按实际知识库筛选已存在索引的文档，不读取正文。

        Args:
            kb_id: 可选实际知识库过滤。

        Returns:
            保持旧排序和字段的文档摘要。
        """
