from pathlib import Path

import pytest

from app.application.errors import (
    DocumentParsingError,
    EmptyDocumentError,
    FileTooLargeError,
    MissingFilenameError,
    UnsupportedFileTypeError,
)
from app.application.ingestion.service import UploadService
from app.domain.documents import UploadResult
from tests.helpers import make_test_pipeline


@pytest.fixture
def service(tmp_path: Path) -> UploadService:
    """创建使用临时文档目录的上传服务。"""
    pipeline, _ = make_test_pipeline()
    return UploadService(
        docs_path=tmp_path,
        max_upload_size=16,
        ingestion_pipeline=pipeline,
    )


def test_upload_service_strips_client_path(service: UploadService) -> None:
    """客户端路径不得参与服务器端文件路径构造。"""
    result = service.save_and_parse("../../notes.txt", b"hello")

    assert result.filename == "notes.txt"
    assert result.chunks_count == 1
    assert len(result.doc_id) == 36


def test_upload_service_returns_domain_result(tmp_path: Path) -> None:
    """上传用例不得把 HTTP 响应模型泄漏到应用层边界。"""
    pipeline, _ = make_test_pipeline()
    upload_service = UploadService(tmp_path, 8, pipeline)

    result = upload_service.save_and_parse("notes.txt", b"hello")

    assert isinstance(result, UploadResult)
    assert result.filename == "notes.txt"


def test_upload_service_saves_file_with_document_id_prefix(
    service: UploadService,
    tmp_path: Path,
) -> None:
    """不同文档使用唯一前缀保存，避免同名文件相互覆盖。"""
    result = service.save_and_parse("notes.txt", b"hello")

    saved_files = list(tmp_path.iterdir())
    assert [path.name for path in saved_files] == [f"{result.doc_id}_notes.txt"]
    assert saved_files[0].read_bytes() == b"hello"


def test_upload_service_rejects_missing_filename(service: UploadService) -> None:
    """缺少文件名时不应创建服务器文件。"""
    with pytest.raises(MissingFilenameError):
        service.save_and_parse(None, b"hello")


def test_upload_service_rejects_empty_content(service: UploadService) -> None:
    """空上传内容不应进入解析流程。"""
    with pytest.raises(EmptyDocumentError):
        service.save_and_parse("notes.txt", b"")


def test_upload_service_rejects_content_over_size_limit(
    service: UploadService,
) -> None:
    """超过配置上限的上传不应写入磁盘。"""
    with pytest.raises(FileTooLargeError):
        service.save_and_parse("notes.txt", b"x" * 17)


def test_upload_service_rejects_unsupported_extension(
    service: UploadService,
) -> None:
    """不支持的格式应在落盘前被拒绝。"""
    with pytest.raises(UnsupportedFileTypeError):
        service.save_and_parse("notes.docx", b"content")


def test_upload_service_removes_file_when_parsing_fails(
    service: UploadService,
    tmp_path: Path,
) -> None:
    """解析失败后不应留下未登记的原始文件。"""
    with pytest.raises(DocumentParsingError):
        service.save_and_parse("broken.pdf", b"not a pdf")

    assert list(tmp_path.iterdir()) == []


def test_upload_service_rejects_document_without_extractable_text(
    service: UploadService,
    tmp_path: Path,
) -> None:
    """只有空白内容的文档不应被视为有效知识库文档。"""
    with pytest.raises(EmptyDocumentError):
        service.save_and_parse("blank.txt", b"  \n")

    assert list(tmp_path.iterdir()) == []


def test_upload_service_stores_original_safe_filename(tmp_path: Path) -> None:
    """向量元数据应保留原文件名，而不是带 UUID 的磁盘保存名。"""
    pipeline, store = make_test_pipeline()
    service = UploadService(tmp_path, 16, pipeline)

    service.save_and_parse("../notes.txt", b"hello")

    assert [chunk.filename for chunk in store.chunks] == ["notes.txt"]
