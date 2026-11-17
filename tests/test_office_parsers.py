# -*- coding: utf-8 -*-
"""Word（.docx）与 PowerPoint（.pptx）解析：正文顺序、表格、幻灯片顺序、备注与异常输入。"""

import io
import zipfile
from pathlib import Path

import pytest

import app.infrastructure.parsing.office as office
from app.application.errors import DocumentParsingError
from app.application.ingestion.service import UploadService
from app.infrastructure.parsing.factory import ParserFactory
from app.infrastructure.parsing.office import DocxParser
from app.infrastructure.parsing.office import PptxParser
from tests.helpers import make_test_pipeline


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NOTES_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"


def _zip(parts: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _p(text: str) -> str:
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def docx_bytes(body: str) -> bytes:
    return _zip({
        "[Content_Types].xml": "<Types/>",
        "word/document.xml": f'<w:document xmlns:w="{W_NS}"><w:body>{body}</w:body></w:document>',
    })


def _slide(*paragraphs: str) -> str:
    runs = "".join(f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p>" for text in paragraphs)
    return (
        f'<p:sld xmlns:p="{P_NS}" xmlns:a="{A_NS}"><p:cSld><p:spTree>'
        f"<p:sp><p:txBody>{runs}</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
    )


def pptx_bytes() -> bytes:
    """三张幻灯片：文件名顺序与放映顺序不同，第二张无文字，第一张带备注。"""
    rels = (
        f'<Relationships xmlns="{REL_NS}">'
        '<Relationship Id="rId2" Target="slides/slide2.xml"/>'
        '<Relationship Id="rId3" Target="slides/slide1.xml"/>'
        '<Relationship Id="rId4" Target="/ppt/slides/slide3.xml"/>'
        "</Relationships>"
    )
    presentation = (
        f'<p:presentation xmlns:p="{P_NS}" xmlns:r="{R_NS}"><p:sldIdLst>'
        '<p:sldId id="256" r:id="rId2"/><p:sldId id="257" r:id="rId3"/><p:sldId id="258" r:id="rId4"/>'
        "</p:sldIdLst></p:presentation>"
    )
    notes = (
        f'<p:notes xmlns:p="{P_NS}" xmlns:a="{A_NS}"><p:cSld><p:spTree>'
        '<p:sp><p:nvSpPr><p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr>'
        "<p:txBody><a:p><a:r><a:t>缩略图</a:t></a:r></a:p></p:txBody></p:sp>"
        '<p:sp><p:nvSpPr><p:nvPr><p:ph type="body"/></p:nvPr></p:nvSpPr>'
        "<p:txBody><a:p><a:r><a:t>讲到这里演示回滚</a:t></a:r></a:p></p:txBody></p:sp>"
        "</p:spTree></p:cSld></p:notes>"
    )
    return _zip({
        "ppt/presentation.xml": presentation,
        "ppt/_rels/presentation.xml.rels": rels,
        "ppt/slides/slide2.xml": _slide("灰度发布流程", "先放量 5%"),
        "ppt/slides/_rels/slide2.xml.rels": (
            f'<Relationships xmlns="{REL_NS}">'
            f'<Relationship Id="rId1" Type="{NOTES_TYPE}" Target="../notesSlides/notesSlide1.xml"/>'
            "</Relationships>"
        ),
        "ppt/notesSlides/notesSlide1.xml": notes,
        "ppt/slides/slide1.xml": _slide(),
        "ppt/slides/slide3.xml": _slide("回滚条件"),
    })


def test_docx_keeps_paragraph_and_table_order(tmp_path: Path) -> None:
    path = tmp_path / "手册.docx"
    path.write_bytes(docx_bytes(
        _p("部署手册")
        + "<w:p><w:r><w:t>第一行</w:t><w:br/><w:t>第二行</w:t><w:tab/><w:t>尾</w:t></w:r></w:p>"
        + "<w:tbl><w:tr><w:tc>" + _p("参数") + "</w:tc><w:tc>" + _p("默认值") + "</w:tc></w:tr>"
        + "<w:tr><w:tc>" + _p("timeout") + "</w:tc><w:tc>" + _p("30") + "</w:tc></w:tr></w:tbl>"
        + "<w:sdt><w:sdtContent>" + _p("目录里的文字") + "</w:sdtContent></w:sdt>"
        + _p("结束")
    ))

    segments = DocxParser().parse(path)

    assert len(segments) == 1
    assert segments[0].page is None
    assert segments[0].filename == "手册.docx"
    assert segments[0].text == (
        "部署手册\n第一行\n第二行\t尾\n参数 | 默认值\ntimeout | 30\n\n目录里的文字\n结束"
    )


def _styled(style_id: str, text: str) -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="{style_id}"/></w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>'


def _styles(*styles: tuple[str, str, int | None]) -> str:
    """生成 styles.xml：每项为 (样式 ID, 样式名, 可选的 0 基大纲级别)。"""
    entries = []
    for style_id, name, outline in styles:
        level = f'<w:pPr><w:outlineLvl w:val="{outline}"/></w:pPr>' if outline is not None else ""
        entries.append(
            f'<w:style w:type="paragraph" w:styleId="{style_id}"><w:name w:val="{name}"/>{level}</w:style>'
        )
    return f'<w:styles xmlns:w="{W_NS}">{"".join(entries)}</w:styles>'


def styled_docx_bytes(body: str, styles: str) -> bytes:
    return _zip({
        "[Content_Types].xml": "<Types/>",
        "word/document.xml": f'<w:document xmlns:w="{W_NS}"><w:body>{body}</w:body></w:document>',
        "word/styles.xml": styles,
    })


def test_docx_heading_styles_split_sections(tmp_path: Path) -> None:
    """英文内置名、中文样式名、样式大纲级别和段落直接大纲级别都识别为标题；表格归属所在章节。"""
    path = tmp_path / "制度.docx"
    path.write_bytes(styled_docx_bytes(
        _p("封面说明")
        + _styled("1", "报销制度")
        + _p("总则")
        + _styled("Heading2", "差旅费")
        + "<w:tbl><w:tr><w:tc>" + _p("城市") + "</w:tc><w:tc>" + _p("标准") + "</w:tc></w:tr></w:tbl>"
        + _styled("MyTitle", "住宿")
        + _p("每晚 500 元")
        + '<w:p><w:pPr><w:outlineLvl w:val="1"/></w:pPr><w:r><w:t>招待费</w:t></w:r></w:p>'
        + _p("招待正文"),
        _styles(("1", "标题 1", None), ("Heading2", "heading 2", None), ("MyTitle", "我的标题", 2)),
    ))

    segments = DocxParser().parse(path)

    assert [(s.heading_path, s.text) for s in segments] == [
        ((), "封面说明"),
        (("报销制度",), "总则"),
        (("报销制度", "差旅费"), "城市 | 标准"),
        (("报销制度", "差旅费", "住宿"), "每晚 500 元"),
        (("报销制度", "招待费"), "招待正文"),
    ]


def test_docx_without_heading_styles_stays_one_segment(tmp_path: Path) -> None:
    path = tmp_path / "plain.docx"
    path.write_bytes(styled_docx_bytes(
        _styled("Normal", "第一段") + _p("第二段"), _styles(("Normal", "Normal", None)),
    ))

    segments = DocxParser().parse(path)

    assert [(s.heading_path, s.text) for s in segments] == [((), "第一段\n第二段")]


def test_docx_text_box_fallback_copy_is_not_duplicated(tmp_path: Path) -> None:
    mc = "http://schemas.openxmlformats.org/markup-compatibility/2006"
    path = tmp_path / "box.docx"
    path.write_bytes(docx_bytes(
        f'<w:p xmlns:mc="{mc}"><w:r><mc:AlternateContent>'
        "<mc:Choice><w:t>文本框内容</w:t></mc:Choice>"
        "<mc:Fallback><w:t>文本框内容</w:t></mc:Fallback>"
        "</mc:AlternateContent></w:r></w:p>"
    ))
    assert DocxParser().parse(path)[0].text == "文本框内容"


def test_empty_docx_yields_nothing(tmp_path: Path) -> None:
    path = tmp_path / "empty.docx"
    path.write_bytes(docx_bytes("<w:p/>"))
    assert DocxParser().parse(path) == []


def test_pptx_uses_presentation_order_page_numbers_and_notes(tmp_path: Path) -> None:
    path = tmp_path / "发布.pptx"
    path.write_bytes(pptx_bytes())

    segments = PptxParser().parse(path)

    # 空白的第 2 页被跳过，但第 3 页仍保留真实页码，引用能对上幻灯片。
    assert [segment.page for segment in segments] == [1, 3]
    assert segments[0].text == "灰度发布流程\n先放量 5%\n\n备注：\n讲到这里演示回滚"
    assert "缩略图" not in segments[0].text
    assert segments[1].text == "回滚条件"
    assert {segment.filename for segment in segments} == {"发布.pptx"}
    # 这些幻灯片没有标题占位符。
    assert all(segment.heading_path == () for segment in segments)


def test_pptx_title_placeholder_becomes_heading_path(tmp_path: Path) -> None:
    def slide(title_type: str, title: str, body: str) -> str:
        return (
            f'<p:sld xmlns:p="{P_NS}" xmlns:a="{A_NS}"><p:cSld><p:spTree>'
            f'<p:sp><p:nvSpPr><p:nvPr><p:ph type="{title_type}"/></p:nvPr></p:nvSpPr>'
            f"<p:txBody><a:p><a:r><a:t>{title}</a:t></a:r></a:p></p:txBody></p:sp>"
            f"<p:sp><p:txBody><a:p><a:r><a:t>{body}</a:t></a:r></a:p></p:txBody></p:sp>"
            "</p:spTree></p:cSld></p:sld>"
        )

    path = tmp_path / "titles.pptx"
    path.write_bytes(_zip({
        "ppt/presentation.xml": (
            f'<p:presentation xmlns:p="{P_NS}" xmlns:r="{R_NS}"><p:sldIdLst>'
            '<p:sldId id="256" r:id="rId1"/><p:sldId id="257" r:id="rId2"/></p:sldIdLst></p:presentation>'
        ),
        "ppt/_rels/presentation.xml.rels": (
            f'<Relationships xmlns="{REL_NS}">'
            '<Relationship Id="rId1" Target="slides/slide1.xml"/>'
            '<Relationship Id="rId2" Target="slides/slide2.xml"/></Relationships>'
        ),
        "ppt/slides/slide1.xml": slide("ctrTitle", "发布手册", "内部资料"),
        "ppt/slides/slide2.xml": slide("title", "灰度流程", "先放量 5%"),
    }))

    segments = PptxParser().parse(path)

    assert [(s.page, s.heading_path, s.text) for s in segments] == [
        (1, ("发布手册",), "发布手册\n内部资料"),
        (2, ("灰度流程",), "灰度流程\n先放量 5%"),
    ]


@pytest.mark.parametrize("parser,name", [(DocxParser, "old.docx"), (PptxParser, "old.pptx")])
def test_legacy_or_corrupt_files_raise_parsing_error(tmp_path: Path, parser, name) -> None:
    path = tmp_path / name
    path.write_bytes(b"\xd0\xcf\x11\xe0 legacy binary office file")
    with pytest.raises(DocumentParsingError, match="另存为"):
        parser().parse(path)


def test_zip_without_expected_part_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fake.docx"
    path.write_bytes(_zip({"ppt/presentation.xml": "<x/>"}))
    with pytest.raises(DocumentParsingError):
        DocxParser().parse(path)
    broken = tmp_path / "broken.docx"
    broken.write_bytes(_zip({"word/document.xml": "<w:document"}))
    with pytest.raises(DocumentParsingError):
        DocxParser().parse(broken)


def test_oversized_parts_are_rejected(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "big.docx"
    path.write_bytes(docx_bytes(_p("x" * 5000)))
    monkeypatch.setattr(office, "MAX_PART_BYTES", 1000)
    with pytest.raises(DocumentParsingError, match="过大"):
        DocxParser().parse(path)
    monkeypatch.setattr(office, "MAX_PART_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(office, "MAX_TOTAL_BYTES", 1000)
    with pytest.raises(DocumentParsingError, match="过大"):
        DocxParser().parse(path)


def test_factory_selects_office_parsers_case_insensitively(tmp_path: Path) -> None:
    assert isinstance(ParserFactory.create(tmp_path / "A.DOCX"), DocxParser)
    assert isinstance(ParserFactory.create(tmp_path / "deck.pptx"), PptxParser)


def test_upload_ingests_docx_and_pptx_into_chunks(tmp_path: Path) -> None:
    pipeline, _ = make_test_pipeline()
    service = UploadService(docs_path=tmp_path, max_upload_size=1024 * 1024, ingestion_pipeline=pipeline)

    word = service.save_and_parse("guide.docx", docx_bytes(_p("Redis 缓存配置说明")))
    deck = service.save_and_parse("deck.pptx", pptx_bytes())

    assert word.filename == "guide.docx" and word.chunks_count >= 1
    assert deck.filename == "deck.pptx" and deck.chunks_count >= 2
