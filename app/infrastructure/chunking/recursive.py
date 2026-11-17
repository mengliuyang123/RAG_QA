# -*- coding: utf-8 -*-
"""提供基于递归字符分隔符的父子两级文档分块适配器。"""

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
    """先按章节和页切出父块，再在父块内部切出不重叠的子块。

    父块是交给生成模型的上下文单位，由同一 ``parent_index`` 的子块按序拼接而成，
    不单独存储；子块是检索单位。片段本身已在标题和页码处断开，所以父块不跨章节、不跨页。
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        parent_chunk_size: int | None = None,
    ) -> None:
        """创建并校验分块器。

        Args:
            chunk_size: 子块的最大字符数。
            chunk_overlap: 同一片段内相邻父块之间的重叠字符数。
            parent_chunk_size: 父块的最大字符数；为空时等于 ``chunk_size``，
                即每个父块只有一个子块，行为与旧的单级分块一致。

        Raises:
            ValueError: 块大小或重叠大小不满足约束。
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能小于 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        parent_size = chunk_size if parent_chunk_size is None else parent_chunk_size
        if parent_size < chunk_size:
            raise ValueError("parent_chunk_size 必须大于或等于 chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_size
        self._parent_splitter = self._splitter(parent_size, chunk_overlap)
        # 子块之间不重叠，按序拼接即可还原父块（切分处的空白除外）。
        self._child_splitter = self._splitter(chunk_size, 0)

    @staticmethod
    def _splitter(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
            separators=DEFAULT_SEPARATORS,
        )

    def split(
        self,
        doc_id: str,
        segments: Iterable[DocumentSegment],
    ) -> list[DocumentChunk]:
        """把小型片段序列收集为子块列表。

        Args:
            doc_id: 来源文档的稳定唯一标识。
            segments: 待切分的源文本片段。

        Returns:
            按来源顺序生成的子块。
        """
        return list(self.split_iter(doc_id, segments))

    def split_iter(
        self,
        doc_id: str,
        segments: Iterable[DocumentSegment],
    ) -> Iterator[DocumentChunk]:
        """惰性产出带稳定 ID、父块序号和标题路径的子块。

        Args:
            doc_id: 来源文档的稳定唯一标识。
            segments: 待切分的源文本片段。

        Returns:
            逐个产出含来源元数据的非空子块；``chunk_index`` 和 ``parent_index``
            在整篇文档内从 0 开始连续编号。
        """
        chunk_index = 0
        parent_index = 0
        for segment in segments:
            if not segment.text.strip():
                continue

            for parent in self._parent_splitter.split_text(segment.text):
                children = [
                    text for text in self._child_splitter.split_text(parent)
                    if text.strip()
                ]
                if not children:
                    continue
                for text in children:
                    yield DocumentChunk(
                        id=f"{doc_id}:{chunk_index}",
                        doc_id=doc_id,
                        text=text,
                        filename=segment.filename,
                        page=segment.page,
                        chunk_index=chunk_index,
                        parent_index=parent_index,
                        heading_path=segment.heading_path,
                    )
                    chunk_index += 1
                parent_index += 1
