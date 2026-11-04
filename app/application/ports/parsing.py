# -*- coding: utf-8 -*-
"""声明文档解析和解析器选择端口。"""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from app.domain.documents import DocumentSegment


class DocumentParser(Protocol):
    """定义把单个文档转换为文本片段的能力。"""

    def parse(self, path: str | Path) -> list[DocumentSegment]:
        """解析小型文档，并收集全部片段。

        Args:
            path: 待解析文档的路径。

        Returns:
            文档中按来源顺序排列的文本片段。
        """

    def parse_iter(self, path: str | Path) -> Iterator[DocumentSegment]:
        """惰性解析文档。

        Args:
            path: 待解析文档的路径。

        Returns:
            逐个产出文本片段的迭代器。
        """


class ParserResolver(Protocol):
    """定义按文档路径选择适用解析器的能力。"""

    def resolve(self, path: str | Path) -> DocumentParser:
        """返回与文档路径匹配的解析器。

        Args:
            path: 待解析文档的路径。

        Returns:
            可解析该路径的文档解析器。
        """
