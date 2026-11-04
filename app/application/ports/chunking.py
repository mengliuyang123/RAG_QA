# -*- coding: utf-8 -*-
"""声明文档分块端口。"""

from collections.abc import Iterable, Iterator
from typing import Protocol

from app.domain.documents import DocumentChunk, DocumentSegment


class Chunker(Protocol):
    """定义把文档片段转换为可检索文本块的能力。"""

    def split(
        self,
        doc_id: str,
        segments: Iterable[DocumentSegment],
    ) -> list[DocumentChunk]:
        """切分小型文档，并收集全部文本块。

        Args:
            doc_id: 来源文档的稳定唯一标识。
            segments: 待切分的源文本片段。

        Returns:
            按文档顺序生成的文本块。
        """

    def split_iter(
        self,
        doc_id: str,
        segments: Iterable[DocumentSegment],
    ) -> Iterator[DocumentChunk]:
        """惰性切分文档片段。

        Args:
            doc_id: 来源文档的稳定唯一标识。
            segments: 待切分的源文本片段。

        Returns:
            逐个产出文本块的迭代器。
        """
