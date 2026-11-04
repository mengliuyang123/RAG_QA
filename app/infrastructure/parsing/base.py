# -*- coding: utf-8 -*-
"""提供格式解析器共享的路径和文本处理能力。"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from app.application.errors import DocumentParsingError
from app.domain.documents import DocumentSegment


class BaseParser(ABC):
    """所有格式解析器共享的实现基类。"""

    def parse(self, path: str | Path) -> list[DocumentSegment]:
        """解析小型文档，并收集全部片段。

        Args:
            path: 待解析文档的路径。

        Returns:
            文档中按来源顺序排列的文本片段。
        """
        return list(self.parse_iter(path))

    @abstractmethod
    def parse_iter(self, path: str | Path) -> Iterator[DocumentSegment]:
        """惰性解析文档。

        Args:
            path: 待解析文档的路径。

        Returns:
            逐个产出文本片段的迭代器。
        """

    @staticmethod
    def _validate_path(path: str | Path) -> Path:
        document_path = Path(path)
        if not document_path.exists():
            raise FileNotFoundError(f"文档不存在：{document_path}")
        if not document_path.is_file():
            raise FileNotFoundError(f"文档路径不是文件：{document_path}")
        return document_path

    @staticmethod
    def _read_utf8_text(path: Path) -> str:
        """读取 UTF-8 文本，并统一转换解码异常。

        Args:
            path: 已验证存在的文本文件路径。

        Returns:
            解码后的文本内容。

        Raises:
            DocumentParsingError: 文件不能按 UTF-8 解码。
        """
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentParsingError(
                f"无法解析 UTF-8 文本文档：{path.name}"
            ) from exc
