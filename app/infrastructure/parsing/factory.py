# -*- coding: utf-8 -*-
"""根据文件扩展名选择文档解析适配器。"""

from pathlib import Path

from app.application.errors import UnsupportedFileTypeError
from app.application.ports.parsing import DocumentParser
from app.infrastructure.parsing.base import BaseParser
from app.infrastructure.parsing.markdown import MarkdownParser
from app.infrastructure.parsing.pdf import PDFParser
from app.infrastructure.parsing.text import TextParser


class ParserFactory:
    """根据文档扩展名创建对应的解析器。"""

    _PARSERS: dict[str, type[BaseParser]] = {
        ".pdf": PDFParser,
        ".md": MarkdownParser,
        ".markdown": MarkdownParser,
        ".txt": TextParser,
    }

    @classmethod
    def create(cls, path: str | Path) -> DocumentParser:
        """返回路径对应的解析器。

        Args:
            path: 待解析文档的路径。

        Returns:
            能解析该文件类型的文档解析器。

        Raises:
            UnsupportedFileTypeError: 文件扩展名没有对应解析器。
        """
        extension = Path(path).suffix.lower()
        parser_type = cls._PARSERS.get(extension)
        if parser_type is None:
            raise UnsupportedFileTypeError(
                f"不支持的文档类型：{extension or '<无扩展名>'}"
            )
        return parser_type()

    def resolve(self, path: str | Path) -> DocumentParser:
        """通过解析器选择端口返回路径对应的解析器。

        Args:
            path: 待解析文档的路径。

        Returns:
            能解析该文件类型的文档解析器。

        Raises:
            UnsupportedFileTypeError: 文件扩展名没有对应解析器。
        """
        return self.create(path)
