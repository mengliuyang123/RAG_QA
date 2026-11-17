# -*- coding: utf-8 -*-
"""提供 Markdown 文档解析适配器。"""

from collections.abc import Iterator
from pathlib import Path

from app.domain.documents import DocumentSegment
from app.application.ingestion.markdown_sections import split_markdown_sections
from app.infrastructure.parsing.base import BaseParser


class MarkdownParser(BaseParser):
    """按 ATX 标题把 Markdown 切成片段，每个片段带所属标题路径。

    标题行本身不进入正文；围栏代码块里的 ``#`` 行按正文处理。
    """

    def parse_iter(self, path: str | Path) -> Iterator[DocumentSegment]:
        """惰性产出 Markdown 文件中每个章节的正文片段。

        Args:
            path: 待解析 Markdown 文件的路径。

        Returns:
            按文档顺序排列、正文非空的片段；没有标题时只有一个片段。
        """
        document_path = self._validate_path(path)
        text = self._read_utf8_text(document_path)
        for heading_path, body in split_markdown_sections(text):
            yield DocumentSegment(
                text=body,
                page=None,
                filename=document_path.name,
                heading_path=heading_path,
            )
