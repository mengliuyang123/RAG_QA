from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from app.domain.documents import DocumentSegment
from app.infrastructure.chunking.recursive import RecursiveDocumentChunker


def test_split_preserves_source_metadata() -> None:
    """分块器遗漏来源字段或确定性 ID 时，本测试应失败。"""
    chunker = RecursiveDocumentChunker(chunk_size=20, chunk_overlap=5)
    segments = [
        DocumentSegment(
            text="第一段。第二段。",
            page=3,
            filename="guide.pdf",
        )
    ]

    chunks = chunker.split("doc-1", segments)

    assert chunks[0].doc_id == "doc-1"
    assert chunks[0].filename == "guide.pdf"
    assert chunks[0].page == 3
    assert chunks[0].chunk_index == 0
    assert chunks[0].id == "doc-1:0"


def test_split_iter_is_lazy() -> None:
    """创建分块迭代器时不应立即消费上游解析结果。"""
    consumed = 0

    def segments() -> Iterator[DocumentSegment]:
        nonlocal consumed
        consumed += 1
        yield DocumentSegment(text="hello", page=None, filename="notes.txt")

    chunks = RecursiveDocumentChunker(chunk_size=20, chunk_overlap=5).split_iter(
        "doc-1",
        segments(),
    )

    assert isinstance(chunks, Iterator)
    assert consumed == 0
    assert next(chunks).text == "hello"
    assert consumed == 1


def test_split_is_list_convenience_wrapper() -> None:
    """小文档仍可通过 split 一次取得完整列表。"""
    segment = DocumentSegment(text="hello", page=None, filename="notes.txt")
    chunker = RecursiveDocumentChunker(chunk_size=20, chunk_overlap=5)

    assert chunker.split("doc-1", [segment]) == list(
        chunker.split_iter("doc-1", [segment])
    )


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_chunker_rejects_invalid_sizes(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """无效长度配置不应进入底层分割器。"""
    with pytest.raises((ValueError, ValidationError)):
        RecursiveDocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_split_ignores_blank_segments() -> None:
    """空白解析结果不应产生空 chunk 或占用索引。"""
    segments = [
        DocumentSegment(text=" \n\t", page=1, filename="guide.pdf"),
        DocumentSegment(text="content", page=2, filename="guide.pdf"),
    ]

    chunks = RecursiveDocumentChunker(chunk_size=20, chunk_overlap=5).split(
        "doc-1",
        segments,
    )

    assert [chunk.text for chunk in chunks] == ["content"]
    assert chunks[0].chunk_index == 0


def test_chunk_indexes_are_continuous_across_segments() -> None:
    """整篇文档的 chunk_index 应跨 segment 连续递增。"""
    segments = [
        DocumentSegment(text="page-one", page=1, filename="guide.pdf"),
        DocumentSegment(text="page-two", page=2, filename="guide.pdf"),
    ]

    chunks = RecursiveDocumentChunker(chunk_size=20, chunk_overlap=5).split(
        "doc-1",
        segments,
    )

    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert [chunk.id for chunk in chunks] == ["doc-1:0", "doc-1:1"]


def test_overlap_does_not_cross_pdf_pages() -> None:
    """上一页末尾文本不得成为下一页 chunk 的 overlap。"""
    segments = [
        DocumentSegment(text="AAAAABBBBBCCCCC", page=1, filename="guide.pdf"),
        DocumentSegment(text="DDDDDEEEEEFFFFF", page=2, filename="guide.pdf"),
    ]

    chunks = RecursiveDocumentChunker(chunk_size=10, chunk_overlap=3).split(
        "doc-1",
        segments,
    )
    page_two_chunks = [chunk.text for chunk in chunks if chunk.page == 2]

    assert page_two_chunks
    assert all("C" not in text for text in page_two_chunks)
    assert page_two_chunks[0].startswith("D")


def test_chunks_respect_character_size_limit_for_chinese_and_english() -> None:
    """中英文文本块都不得超过字符长度上限。"""
    segment = DocumentSegment(
        text="第一段包含中文标点。Second paragraph has English words.第三段继续说明。",
        page=None,
        filename="notes.md",
    )

    chunks = RecursiveDocumentChunker(chunk_size=16, chunk_overlap=4).split(
        "doc-1",
        [segment],
    )

    assert len(chunks) > 1
    assert all(0 < len(chunk.text) <= 16 for chunk in chunks)


def test_overlap_is_reproducible_for_text_without_separators() -> None:
    """没有自然分隔符时，相邻块应保留配置长度的重叠文本。"""
    segment = DocumentSegment(
        text="abcdefghijklmnop",
        page=None,
        filename="notes.txt",
    )

    chunks = RecursiveDocumentChunker(chunk_size=10, chunk_overlap=3).split(
        "doc-1",
        [segment],
    )

    assert [chunk.text for chunk in chunks] == ["abcdefghij", "hijklmnop"]


def _texts(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    chunker = RecursiveDocumentChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    segment = DocumentSegment(text=text, page=None, filename="guide.md")
    return [chunk.text for chunk in chunker.split("doc-1", [segment])]


def test_short_code_block_stays_whole() -> None:
    """代码里的逗号、句号、空格不能被当作分隔符切开。"""
    code = "```python\ndef connect(host, port):\n    return pool.get(host, port, timeout=3.0)\n```"
    texts = _texts(f"连接数据库：\n\n{code}\n\n完成后检查日志。", chunk_size=120)

    assert code in texts
    assert all(text.count("```") in (0, 2) for text in texts)


def test_long_code_block_splits_on_lines_and_keeps_fences() -> None:
    """超长代码按行切分，每块都带原语言围栏，且不切断任何一行。"""
    functions = [
        f"def handler_{i}(event, context):\n    value = compute(event, {i})\n    return value"
        for i in range(8)
    ]
    body = "\n\n".join(functions)
    texts = _texts(f"```python\n{body}\n```", chunk_size=160)

    assert len(texts) > 1
    original_lines = set(body.split("\n"))
    for text in texts:
        lines = text.split("\n")
        assert lines[0] == "```python" and lines[-1] == "```"
        assert len(text) <= 160
        assert set(lines[1:-1]) <= original_lines
    # 在空行处切：每个函数都完整落在某一块里。
    for function in functions:
        assert any(function in text for text in texts)


def test_overlong_single_line_is_hard_split_within_limit() -> None:
    line = "x = '" + "a" * 400 + "'"
    texts = _texts(f"~~~\n{line}\n~~~", chunk_size=100)

    assert len(texts) > 1
    assert all(len(text) <= 100 and text.startswith("~~~\n") and text.endswith("\n~~~") for text in texts)
    assert "".join(text[4:-4] for text in texts) == line


def test_unclosed_fence_treats_rest_as_code() -> None:
    texts = _texts("说明\n```bash\nexport A=1, B=2. C=3\n", chunk_size=200)

    assert texts == ["说明", "```bash\nexport A=1, B=2. C=3\n```"]


def test_inline_backticks_are_not_fences() -> None:
    text = "用 ```code``` 表示行内代码，后面是普通段落。"

    assert _texts(text) == [text]
