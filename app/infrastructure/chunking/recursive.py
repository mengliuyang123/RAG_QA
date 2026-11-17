# -*- coding: utf-8 -*-
"""提供基于递归字符分隔符的文档分块适配器。"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

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


# Markdown 围栏代码块：最多 3 个空格缩进，``` 或 ~~~ 至少 3 个。
_FENCE_OPEN = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")


@dataclass(frozen=True)
class _CodeBlock:
    """一个围栏代码块，保留原始开闭围栏行。"""

    opening: str
    lines: list[str]
    closing: str


def _blocks(text: str) -> Iterator[str | _CodeBlock]:
    """把文本拆成普通文字与围栏代码块，保持原有顺序。

    Args:
        text: 一个源片段的全文。

    Returns:
        依次产出普通文字字符串或代码块；未闭合的围栏视为代码一直延续到末尾。
    """
    prose: list[str] = []
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        match = _FENCE_OPEN.match(lines[index])
        if not match or (match.group(2)[0] == "`" and "`" in match.group(3)):
            prose.append(lines[index])
            index += 1
            continue
        if prose:
            yield "\n".join(prose)
            prose = []
        fence = match.group(2)
        closer = re.compile(rf"^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*$")
        body: list[str] = []
        index += 1
        closing = fence
        while index < len(lines) and not closer.match(lines[index]):
            body.append(lines[index])
            index += 1
        if index < len(lines):
            closing = lines[index]
            index += 1
        yield _CodeBlock(match.group(0), body, closing)
    if prose:
        yield "\n".join(prose)


def _split_code(block: _CodeBlock, chunk_size: int) -> Iterator[str]:
    """按行切分代码块，每块都重新包上原围栏，绝不在行内或语句中间切开。

    能放下时整块保留；否则优先在空行（函数、段落之间）处切，单行超长时才硬切。

    Args:
        block: 待切分的代码块。
        chunk_size: 文本块最大字符数，包含围栏。

    Returns:
        依次产出带围栏的代码文本块。
    """
    def wrap(lines: list[str]) -> str:
        return "\n".join([block.opening, *lines, block.closing])

    if not any(line.strip() for line in block.lines):
        return
    whole = wrap(_trim(block.lines))
    if len(whole) <= chunk_size:
        yield whole
        return
    budget = max(chunk_size - len(block.opening) - len(block.closing) - 2, 1)
    current: list[str] = []
    size = 0
    for line in block.lines:
        pieces = [line[start:start + budget] for start in range(0, len(line), budget)] or [""]
        for piece in pieces:
            added = len(piece) + (1 if current else 0)
            if current and size + added > budget:
                # 回退到当前块里最后一个空行，让函数/段落保持完整。
                cut = max((i for i, text in enumerate(current) if not text.strip()), default=0)
                if cut * 2 < len(current):
                    cut = len(current)
                head, current = current[:cut], current[cut:]
                while current and not current[0].strip():
                    current.pop(0)
                if any(text.strip() for text in head):
                    yield wrap(_trim(head))
                size = len("\n".join(current))
                added = len(piece) + (1 if current else 0)
            current.append(piece)
            size += added
    if any(text.strip() for text in current):
        yield wrap(_trim(current))


def _trim(lines: list[str]) -> list[str]:
    """去掉代码片段首尾的空行。"""
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


class RecursiveDocumentChunker:
    """使用递归字符策略切分文档片段，围栏代码块单独按行切分。"""

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

            for text in self._texts(segment.text):
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

    def _texts(self, text: str) -> Iterator[str]:
        """普通文字走递归分隔符，代码块保持完整或按行切分。

        Args:
            text: 一个源片段的全文。

        Returns:
            按原文顺序产出的文本块正文。
        """
        for block in _blocks(text):
            if isinstance(block, _CodeBlock):
                yield from _split_code(block, self.chunk_size)
            else:
                yield from self._splitter.split_text(block)
