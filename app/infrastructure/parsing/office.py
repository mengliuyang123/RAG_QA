# -*- coding: utf-8 -*-
"""Word（.docx）与 PowerPoint（.pptx）解析适配器。

只用标准库读取 Office Open XML 压缩包，不引入额外依赖。旧版二进制 .doc/.ppt 不支持，
需要先另存为 .docx/.pptx。压缩包成员数与解压后大小设有上限，防止压缩炸弹。
"""

import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree

from app.application.errors import DocumentParsingError
from app.domain.documents import DocumentSegment
from app.infrastructure.parsing.base import BaseParser


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"

MAX_MEMBERS = 10_000
MAX_PART_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024


class _Package:
    """带大小上限的 OOXML 压缩包读取器。"""

    def __init__(self, archive: zipfile.ZipFile, label: str) -> None:
        self._archive = archive
        self._label = label
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS or sum(info.file_size for info in infos) > MAX_TOTAL_BYTES:
            raise DocumentParsingError(f"{label} 文档过大或结构异常")
        self._names = {info.filename for info in infos}

    def has(self, name: str) -> bool:
        return name in self._names

    def xml(self, name: str) -> ElementTree.Element:
        info = self._archive.getinfo(name)
        if info.file_size > MAX_PART_BYTES:
            raise DocumentParsingError(f"{self._label} 文档内容过大")
        with self._archive.open(info) as member:
            # 声明大小可以伪造，读取时再按上限截断判断。
            data = member.read(MAX_PART_BYTES + 1)
        if len(data) > MAX_PART_BYTES:
            raise DocumentParsingError(f"{self._label} 文档内容过大")
        return ElementTree.fromstring(data)


def _open(path: Path, label: str, required: str) -> tuple[zipfile.ZipFile, _Package]:
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise DocumentParsingError(
            f"无法解析 {label} 文档：{path.name}（旧版 .doc/.ppt 请先另存为 .docx/.pptx）"
        ) from exc
    try:
        package = _Package(archive, label)
        if not package.has(required):
            raise DocumentParsingError(f"无法解析 {label} 文档：{path.name}")
    except BaseException:
        archive.close()
        raise
    return archive, package


def _walk(node: ElementTree.Element) -> Iterator[ElementTree.Element]:
    """深度优先遍历，跳过 mc:Fallback：文本框等内容会同时存一份旧格式副本，否则文字重复。"""
    yield node
    for child in node:
        if child.tag != MC_FALLBACK:
            yield from _walk(child)


def _clean(lines: list[str]) -> str:
    text = "\n".join(line.rstrip() for line in lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _word_paragraph(paragraph: ElementTree.Element) -> str:
    parts: list[str] = []
    for node in _walk(paragraph):
        if node.tag == W + "t" and node.text:
            parts.append(node.text)
        elif node.tag == W + "tab":
            parts.append("\t")
        elif node.tag in (W + "br", W + "cr"):
            parts.append("\n")
    return "".join(parts)


def _word_blocks(container: ElementTree.Element) -> Iterator[str]:
    """按文档顺序产出段落与表格行；表格单元格用 | 分隔，保留行内对应关系。"""
    for child in container:
        if child.tag == W + "p":
            yield _word_paragraph(child)
        elif child.tag == W + "tbl":
            for row in child.iter(W + "tr"):
                cells = [
                    " ".join(filter(None, (_word_paragraph(p).strip() for p in cell.iter(W + "p"))))
                    for cell in row.iter(W + "tc")
                ]
                if any(cells):
                    yield " | ".join(cells)
            yield ""
        elif child.tag == W + "sdt":
            content = child.find(W + "sdtContent")
            if content is not None:
                yield from _word_blocks(content)


class DocxParser(BaseParser):
    """把 Word 正文（段落、表格、页眉页脚除外）读成一个逻辑片段。

    .docx 没有固定分页，与 Markdown 一样不提供页码。
    """

    def parse_iter(self, path: str | Path) -> Iterator[DocumentSegment]:
        document_path = self._validate_path(path)
        archive, package = _open(document_path, "Word", "word/document.xml")
        try:
            try:
                root = package.xml("word/document.xml")
            except (ElementTree.ParseError, KeyError, zipfile.BadZipFile, OSError) as exc:
                raise DocumentParsingError(f"无法解析 Word 文档：{document_path.name}") from exc
        finally:
            archive.close()
        body = root.find(W + "body")
        text = _clean(list(_word_blocks(body))) if body is not None else ""
        if text:
            yield DocumentSegment(text=text, page=None, filename=document_path.name)


def _slide_order(package: _Package) -> list[str]:
    """按演示文稿中的放映顺序返回幻灯片部件路径，而不是按文件名排序。"""
    presentation = package.xml("ppt/presentation.xml")
    relations = package.xml("ppt/_rels/presentation.xml.rels")
    targets = {
        rel.get("Id"): rel.get("Target", "")
        for rel in relations.iter(REL + "Relationship")
    }
    slides: list[str] = []
    for slide in presentation.iter(P + "sldId"):
        target = targets.get(slide.get(R + "id"), "")
        name = target.lstrip("/") if target.startswith("/") else "ppt/" + target
        if package.has(name):
            slides.append(name)
    return slides


def _drawing_text(root: ElementTree.Element) -> list[str]:
    lines: list[str] = []
    for paragraph in root.iter(A + "p"):
        parts = []
        for node in _walk(paragraph):
            if node.tag == A + "t" and node.text:
                parts.append(node.text)
            elif node.tag == A + "br":
                parts.append("\n")
        line = "".join(parts)
        if line.strip():
            lines.append(line)
    return lines


def _notes_part(package: _Package, slide: str) -> str | None:
    folder, _, filename = slide.rpartition("/")
    rels_name = f"{folder}/_rels/{filename}.rels"
    if not package.has(rels_name):
        return None
    for rel in package.xml(rels_name).iter(REL + "Relationship"):
        if rel.get("Type", "").endswith("/notesSlide"):
            target = rel.get("Target", "")
            parts = (folder + "/" + target).split("/")
            resolved: list[str] = []
            for part in parts:
                if part == "..":
                    if resolved:
                        resolved.pop()
                elif part and part != ".":
                    resolved.append(part)
            name = "/".join(resolved)
            return name if package.has(name) else None
    return None


class PptxParser(BaseParser):
    """每张幻灯片一个片段，页码即幻灯片序号；演讲者备注附在该页之后。"""

    def parse_iter(self, path: str | Path) -> Iterator[DocumentSegment]:
        document_path = self._validate_path(path)
        archive, package = _open(document_path, "PowerPoint", "ppt/presentation.xml")
        slides: list[tuple[int, str]] = []
        try:
            try:
                for number, slide in enumerate(_slide_order(package), start=1):
                    lines = _drawing_text(package.xml(slide))
                    notes = _notes_part(package, slide)
                    if notes:
                        # 备注页里也有幻灯片缩略图占位，只取正文占位符的文字。
                        note_lines = [
                            line for shape in package.xml(notes).iter(P + "sp")
                            if _is_body_placeholder(shape)
                            for line in _drawing_text(shape)
                        ]
                        if note_lines:
                            lines += ["", "备注：", *note_lines]
                    text = _clean(lines)
                    if text:
                        slides.append((number, text))
            except (ElementTree.ParseError, KeyError, zipfile.BadZipFile, OSError) as exc:
                raise DocumentParsingError(f"无法解析 PowerPoint 文档：{document_path.name}") from exc
        finally:
            archive.close()
        for number, text in slides:
            yield DocumentSegment(text=text, page=number, filename=document_path.name)


def _is_body_placeholder(shape: ElementTree.Element) -> bool:
    placeholder = shape.find(f"{P}nvSpPr/{P}nvPr/{P}ph")
    return placeholder is not None and placeholder.get("type") == "body"
