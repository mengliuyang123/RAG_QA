# -*- coding: utf-8 -*-
"""提供基于递归字符分隔符的文档分块适配器。"""

from collections.abc import Iterable, Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.domain.documents import DocumentChunk, DocumentSegment


DEFAULT_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    "、",
    ".",
    "!",
    "?",
    ";",
    ",",
    " ",
    "",
]


class RecursiveDocumentChunker:
    """使用递归字符策略切分文档片段。"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """创建并校验递归字符分块器。

        Args:
            chunk_size: 每个文本块的最大字符数。
            chunk_overlap: 相邻文本块保留的重叠字符数。

        Raises:
            ValueError: 块大小或重叠大小不满足约束。
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能小于 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=DEFAULT_SEPARATORS,
        )

    def split(
        self,
        doc_id: str,
        segments: Iterable[DocumentSegment],
    ) -> list[DocumentChunk]:
        """把小型片段序列收集为文本块列表。

        Args:
            doc_id: 来源文档的稳定唯一标识。
            segments: 待切分的源文本片段。

        Returns:
            按来源顺序生成的文本块。
        """
        return list(self.split_iter(doc_id, segments))

    def split_iter(
        self,
        doc_id: str,
        segments: Iterable[DocumentSegment],
    ) -> Iterator[DocumentChunk]:
        """惰性产出带稳定 ID 的文本块。

        Args:
            doc_id: 来源文档的稳定唯一标识。
            segments: 待切分的源文本片段。

        Returns:
            逐个产出含来源元数据的非空文本块。
        """
        chunk_index = 0
        for segment in segments:
            if not segment.text.strip():
                continue

            for text in self._splitter.split_text(segment.text):
                if not text.strip():
                    continue
                yield DocumentChunk(
                    id=f"{doc_id}:{chunk_index}",
                    doc_id=doc_id,
                    text=text,
                    filename=segment.filename,
                    page=segment.page,
                    chunk_index=chunk_index,
                )
                chunk_index += 1
