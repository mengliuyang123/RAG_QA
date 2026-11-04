from pathlib import Path
from collections.abc import Iterator

import pymupdf as fitz
import pytest

import app.infrastructure.parsing.pdf as parser_module

from app.application.errors import (
    DocumentParsingError,
    UnsupportedFileTypeError,
)
from app.infrastructure.parsing.factory import ParserFactory
from app.infrastructure.parsing.markdown import MarkdownParser
from app.infrastructure.parsing.pdf import PDFParser
from app.infrastructure.parsing.text import TextParser


def test_markdown_parser_reads_utf8_text(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# 标题\n内容", encoding="utf-8")

    segments = MarkdownParser().parse(path)

    assert len(segments) == 1
    assert segments[0].text == "# 标题\n内容"
    assert segments[0].page is None
    assert segments[0].filename == "notes.md"


def test_text_parser_reads_utf8_text(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("第一行\n第二行", encoding="utf-8")

    segments = TextParser().parse(path)

    assert segments[0].text == "第一行\n第二行"
    assert segments[0].filename == "notes.txt"


def test_pdf_parser_preserves_one_based_page_numbers(tmp_path: Path) -> None:
    path = tmp_path / "guide.pdf"
    document = fitz.open()
    # PyMuPDF 默认内置字体不包含中文字形，这里使用 ASCII 文本测试解析行为。
    for text in ("Page One", "Page Two"):
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()

    segments = PDFParser().parse(path)

    assert [segment.page for segment in segments] == [1, 2]
    assert all(segment.filename == "guide.pdf" for segment in segments)
    assert "Page One" in segments[0].text
    assert "Page Two" in segments[1].text


def test_pdf_parse_iter_is_lazy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """创建迭代器时不应立即打开 PDF，消费时才开始解析。"""
    path = tmp_path / "guide.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Page One")
    document.save(path)
    document.close()

    real_open = parser_module.fitz.open
    open_calls = 0

    def tracking_open(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(parser_module.fitz, "open", tracking_open)

    segments = PDFParser().parse_iter(path)

    assert isinstance(segments, Iterator)
    assert open_calls == 0
    assert next(segments).text.startswith("Page One")
    assert open_calls == 1
    segments.close()


def test_pdf_parse_iter_does_not_read_entire_file_into_python_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PDF 流式解析不得调用 Path.read_bytes 复制整个文件。"""
    path = tmp_path / "guide.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Page One")
    document.save(path)
    document.close()

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("parse_iter 不应调用 Path.read_bytes")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    assert next(PDFParser().parse_iter(path)).page == 1


def test_parse_collects_parse_iter_results(tmp_path: Path) -> None:
    """原有 parse 接口应保留，并收集流式接口的全部结果。"""
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    parser = TextParser()

    assert parser.parse(path) == list(parser.parse_iter(path))


def test_factory_dispatches_by_extension(tmp_path: Path) -> None:
    assert isinstance(ParserFactory.create(tmp_path / "a.md"), MarkdownParser)
    assert isinstance(ParserFactory.create(tmp_path / "a.markdown"), MarkdownParser)
    assert isinstance(ParserFactory.create(tmp_path / "a.txt"), TextParser)
    assert isinstance(ParserFactory.create(tmp_path / "a.pdf"), PDFParser)


def test_factory_resolves_parser_through_resolver_port(tmp_path: Path) -> None:
    """解析器工厂必须通过 Resolver 端口提供同样的扩展名分派行为。"""
    resolver = ParserFactory()

    parser = resolver.resolve(tmp_path / "notes.md")

    assert isinstance(parser, MarkdownParser)


def test_factory_rejects_unsupported_extension(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        ParserFactory.create(tmp_path / "a.docx")


def test_factory_accepts_uppercase_pdf_extension(tmp_path: Path) -> None:
    """扩展名大小写不应影响解析器选择。"""
    assert isinstance(ParserFactory.create(tmp_path / "A.PDF"), PDFParser)


@pytest.mark.parametrize("suffix", [".txt", ".md", ".markdown"])
def test_text_parsers_return_no_segments_for_blank_file(
    tmp_path: Path,
    suffix: str,
) -> None:
    """纯空白文本不应进入后续知识库流程。"""
    path = tmp_path / f"blank{suffix}"
    path.write_text("  \n\t", encoding="utf-8")

    parser = ParserFactory.create(path)

    assert parser.parse(path) == []


def test_pdf_parser_wraps_corrupted_file_error(tmp_path: Path) -> None:
    """损坏 PDF 应转换为包含文件名的领域异常并保留原异常。"""
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf")

    with pytest.raises(DocumentParsingError, match="broken.pdf") as exc_info:
        PDFParser().parse(path)

    assert exc_info.value.__cause__ is not None


@pytest.mark.parametrize("suffix", [".txt", ".md"])
def test_text_parsers_wrap_invalid_utf8_error(
    tmp_path: Path,
    suffix: str,
) -> None:
    """无法按 UTF-8 解码的文本应转换为领域异常。"""
    path = tmp_path / f"invalid{suffix}"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(DocumentParsingError, match=path.name) as exc_info:
        ParserFactory.create(path).parse(path)

    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


def test_parser_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TextParser().parse(tmp_path / "missing.txt")
