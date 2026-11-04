# -*- coding: utf-8 -*-
"""提供 Markdown 文档解析适配器。"""

from collections.abc import Iterator
from pathlib import Path

from app.domain.documents import DocumentSegment
from app.infrastructure.parsing.base import BaseParser


class MarkdownParser(BaseParser):
    """将 Markdown 文件作为一个逻辑源文本片段读取。"""

    def parse_iter(self, path: str | Path) -> Iterator[DocumentSegment]:
        """惰性产出 Markdown 文件的逻辑文本片段。

        Args:
            path: 待解析 Markdown 文件的路径。

        Returns:
            一个非空源文本片段，或空迭代器。
        """
        document_path = self._validate_path(path)
        text = self._read_utf8_text(document_path)
        if not text.strip():
            return
        yield DocumentSegment(
            text=text,
            page=None,
            filename=document_path.name,
        )
