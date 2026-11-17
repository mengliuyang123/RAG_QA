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
from app.infrastructure.parsing.base import BaseParser, HeadingTracker


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


_HEADING_NAME = re.compile(r"^(?:heading|标题)\s*([1-9])$", re.IGNORECASE)


def _outline_level(properties: ElementTree.Element | None) -> int | None:
    """读取 w:outlineLvl（0 基，9 表示正文），返回 1 基标题级别。"""
    if properties is None:
        return None
    node = properties.find(W + "outlineLvl")
    if node is None:
        return None
    try:
        value = int(node.get(W + "val", ""))
    except ValueError:
        return None
    return value + 1 if 0 <= value <= 8 else None


def _heading_styles(package: _Package) -> dict[str, int]:
    """把样式 ID 映射为标题级别：样式名为 Heading N / 标题 N，或样式设置了大纲级别。"""
    if not package.has("word/styles.xml"):
        return {}
    levels: dict[str, int] = {}
    for style in package.xml("word/styles.xml").iter(W + "style"):
        if style.get(W + "type") != "paragraph":
            continue
        style_id = style.get(W + "styleId")
        if not style_id:
            continue
        name = style.find(W + "name")
        match = _HEADING_NAME.match(" ".join((name.get(W + "val", "") if name is not None else "").split()))
        level = int(match.group(1)) if match else _outline_level(style.find(W + "pPr"))
        if level is not None:
            levels[style_id] = level
    return levels


def _paragraph_heading_level(
    paragraph: ElementTree.Element, styles: dict[str, int],
) -> int | None:
    """段落自身的大纲级别优先，其次是段落样式对应的标题级别。"""
    properties = paragraph.find(W + "pPr")
    direct = _outline_level(properties)
    if direct is not None:
        return direct
    style = properties.find(W + "pStyle") if properties is not None else None
    style_id = style.get(W + "val") if style is not None else None
    if not style_id:
        return None
    if style_id in styles:
        return styles[style_id]
    match = _HEADING_NAME.match(style_id)
    return int(match.group(1)) if match else None


def _word_blocks(
    container: ElementTree.Element, styles: dict[str, int],
) -> Iterator[tuple[int | None, str]]:
    """按文档顺序产出 ``(标题级别, 文本)``：正文和表格行的级别为 None。

    表格单元格用 | 分隔，保留行内对应关系；表格里的段落不作为标题。
    """
    for child in container:
        if child.tag == W + "p":
            yield _paragraph_heading_level(child, styles), _word_paragraph(child)
        elif child.tag == W + "tbl":
            for row in child.iter(W + "tr"):
                cells = [
                    " ".join(filter(None, (_word_paragraph(p).strip() for p in cell.iter(W + "p"))))
                    for cell in row.iter(W + "tc")
                ]
                if any(cells):
                    yield None, " | ".join(cells)
            yield None, ""
        elif child.tag == W + "sdt":
            content = child.find(W + "sdtContent")
            if content is not None:
                yield from _word_blocks(content, styles)


class DocxParser(BaseParser):
    """按标题样式把 Word 正文（段落、表格，页眉页脚除外）切成章节片段。

    .docx 没有固定分页，与 Markdown 一样不提供页码；标题段落只进入标题路径，不进入正文。
    """

    def parse_iter(self, path: str | Path) -> Iterator[DocumentSegment]:
        document_path = self._validate_path(path)
        archive, package = _open(document_path, "Word", "word/document.xml")
        try:
            try:
                root = package.xml("word/document.xml")
                styles = _heading_styles(package)
            except (ElementTree.ParseError, KeyError, zipfile.BadZipFile, OSError) as exc:
                raise DocumentParsingError(f"无法解析 Word 文档：{document_path.name}") from exc
        finally:
            archive.close()
        body = root.find(W + "body")
        if body is None:
            return
        tracker = HeadingTracker()
        path_now: tuple[str, ...] = ()
        lines: list[str] = []
        for level, text in _word_blocks(body, styles):
            if level is not None and text.strip():
                section = _clean(lines)
                if section:
                    yield DocumentSegment(
                        text=section, page=None, filename=document_path.name,
                        heading_path=path_now,
                    )
                lines = []
                tracker.enter(level, text)
                path_now = tracker.path
                continue
            lines.append(text)
        section = _clean(lines)
        if section:
            yield DocumentSegment(
                text=section, page=None, filename=document_path.name,
                heading_path=path_now,
            )


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
        slides: list[tuple[int, str, tuple[str, ...]]] = []
        try:
            try:
                for number, slide in enumerate(_slide_order(package), start=1):
                    slide_root = package.xml(slide)
                    lines = _drawing_text(slide_root)
                    title = _slide_title(slide_root)
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
                        slides.append((number, text, (title,) if title else ()))
            except (ElementTree.ParseError, KeyError, zipfile.BadZipFile, OSError) as exc:
                raise DocumentParsingError(f"无法解析 PowerPoint 文档：{document_path.name}") from exc
        finally:
            archive.close()
        for number, text, heading_path in slides:
            yield DocumentSegment(
                text=text, page=number, filename=document_path.name,
                heading_path=heading_path,
            )


def _placeholder_type(shape: ElementTree.Element) -> str | None:
    placeholder = shape.find(f"{P}nvSpPr/{P}nvPr/{P}ph")
    return None if placeholder is None else placeholder.get("type")


def _is_body_placeholder(shape: ElementTree.Element) -> bool:
    return _placeholder_type(shape) == "body"


def _slide_title(root: ElementTree.Element) -> str:
    """返回幻灯片标题占位符（title / ctrTitle）的文字，合并为一行；没有时为空。

    标题文字仍保留在幻灯片正文里，与改动前一致。
    """
    for shape in root.iter(P + "sp"):
        if _placeholder_type(shape) in ("title", "ctrTitle"):
            return " ".join(" ".join(_drawing_text(shape)).split())
    return ""
