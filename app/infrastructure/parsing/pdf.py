# -*- coding: utf-8 -*-
"""提供 PDF 文档解析适配器。"""

from collections.abc import Iterator
from pathlib import Path

import pymupdf as fitz

from app.application.errors import DocumentParsingError
from app.domain.documents import DocumentSegment
from app.infrastructure.parsing.base import BaseParser


class PDFParser(BaseParser):
    """为 PDF 的每一页提取一个文本片段。"""

    def parse_iter(self, path: str | Path) -> Iterator[DocumentSegment]:
        """按页惰性解析 PDF，避免复制完整文件到 Python 内存。

        Args:
            path: 待解析 PDF 文件的路径。

        Returns:
            逐页产出的非空文本片段。

        Raises:
            DocumentParsingError: PDF 无法由 PyMuPDF 解析。
        """
        document_path = self._validate_path(path)
        document = None
        parse_error: DocumentParsingError | None = None
        try:
            document = fitz.open(document_path)
            for page_number, page in enumerate(document, start=1):
                text = page.get_text()
                if text.strip():
                    yield DocumentSegment(
                        text=text,
                        page=page_number,
                        filename=document_path.name,
                    )
        except Exception as exc:
            # 清除 PyMuPDF traceback，避免其持有临时文件导致 Windows 删除失败。
            safe_cause = type(exc)(str(exc))
            parse_error = DocumentParsingError(
                f"无法解析 PDF 文档：{document_path.name}"
            )
            parse_error.__cause__ = safe_cause
            parse_error.__suppress_context__ = True
        finally:
            if document is not None:
                document.close()

        if parse_error is not None:
            raise parse_error
