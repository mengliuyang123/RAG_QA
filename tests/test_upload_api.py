# -*- coding: utf-8 -*-
"""文档上传 HTTP 接口测试。"""

from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn

import httpx
import pytest
from fastapi import FastAPI

from app.application.errors import FileTooLargeError
from app.application.ingestion.service import UploadService
from app.interfaces.http.dependencies import get_upload_service
from app.interfaces.http.routes.uploads import upload_document
from tests.http_auth import create_test_app as create_app
from tests.helpers import make_test_pipeline


@pytest.fixture
def anyio_backend() -> str:
    """API 测试固定使用 asyncio。

    Returns:
        AnyIO 使用的 asyncio 后端名称。
    """
    return "asyncio"


@pytest.fixture
def app(tmp_path: Path) -> Iterator[FastAPI]:
    """创建使用临时上传目录的测试应用。

    Args:
        tmp_path: pytest 提供的临时上传目录。

    Yields:
        注入离线上传服务的 FastAPI 应用。
    """
    application = create_app()
    pipeline, _ = make_test_pipeline()
    application.dependency_overrides[get_upload_service] = lambda: UploadService(
        docs_path=tmp_path,
        max_upload_size=8,
        ingestion_pipeline=pipeline,
    )
    yield application
    application.dependency_overrides.clear()


@pytest.mark.anyio
async def test_upload_txt_returns_parse_summary(app: FastAPI) -> None:
    """上传成功时应返回文档标识、原文件名和片段数。

    Args:
        app: 注入离线上传服务的 FastAPI 应用。
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["chunks_count"] == 1
    assert "segments_count" not in body
    assert len(body["doc_id"]) == 36


@pytest.mark.anyio
async def test_upload_rejects_unsupported_extension(
    app: FastAPI,
) -> None:
    """不支持的扩展名应映射为 HTTP 415。

    Args:
        app: 注入离线上传服务的 FastAPI 应用。
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/upload",
            files={"file": ("notes.docx", b"hello", "application/octet-stream")},
        )

    assert response.status_code == 415
    assert "不支持" in response.json()["detail"]


@pytest.mark.anyio
async def test_upload_rejects_empty_file(app: FastAPI) -> None:
    """空文件应映射为 HTTP 422。

    Args:
        app: 注入离线上传服务的 FastAPI 应用。
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"", "text/plain")},
        )

    assert response.status_code == 422
    assert "为空" in response.json()["detail"]


@pytest.mark.anyio
async def test_upload_rejects_file_over_limit(app: FastAPI) -> None:
    """超过大小限制的文件应映射为 HTTP 413。

    Args:
        app: 注入离线上传服务的 FastAPI 应用。
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"x" * 9, "text/plain")},
        )

    assert response.status_code == 413
    assert "大小限制" in response.json()["detail"]


@pytest.mark.anyio
async def test_upload_rejects_corrupted_pdf(app: FastAPI) -> None:
    """损坏文档应映射为 HTTP 422，并隐藏底层库异常。

    Args:
        app: 注入离线上传服务的 FastAPI 应用。
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/upload",
            files={"file": ("broken.pdf", b"bad pdf", "application/pdf")},
        )

    assert response.status_code == 422
    assert "无法解析" in response.json()["detail"]


@pytest.mark.anyio
async def test_upload_rejects_invalid_utf8_text(app: FastAPI) -> None:
    """无法解码的文本文件应映射为 HTTP 422。

    Args:
        app: 注入离线上传服务的 FastAPI 应用。
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/upload",
            files={"file": ("invalid.txt", b"\xff\xfe", "text/plain")},
        )

    assert response.status_code == 422
    assert "无法解析" in response.json()["detail"]


@pytest.mark.anyio
async def test_upload_hides_unexpected_internal_error(
    app: FastAPI,
) -> None:
    """未知内部错误应返回稳定消息，不得暴露异常细节。

    Args:
        app: 注入离线上传服务的 FastAPI 应用。
    """

    class _FailingUploadService:
        """模拟发生未知内部错误的上传服务。"""

        def save_and_parse(
            self,
            filename: str | None,
            content: bytes,
        ) -> NoReturn:
            """模拟上传服务的未知异常。

            Args:
                filename: 上传文件名。
                content: 上传文件内容。

            Raises:
                RuntimeError: 固定的未知内部错误。
            """
            del filename
            del content
            raise RuntimeError("private implementation detail")

    app.dependency_overrides[get_upload_service] = _FailingUploadService
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "文档上传处理失败"


@pytest.mark.anyio
async def test_upload_reads_at_most_one_byte_over_configured_limit(
    tmp_path: Path,
) -> None:
    """HTTP 层不得在校验大小前把任意大的上传完整读入内存。

    Args:
        tmp_path: pytest 提供的临时上传目录。
    """

    class _SizeTrackingFile:
        """记录路由请求的最大读取字节数。"""

        filename = "large.pdf"

        def __init__(self) -> None:
            self.requested_size: int | None = None
            self.closed = False

        async def read(self, size: int) -> bytes:
            """返回请求大小的内容并记录读取上限。

            Args:
                size: 路由请求读取的最大字节数。

            Returns:
                精确达到请求大小的测试内容。
            """
            self.requested_size = size
            return b"x" * size

        async def close(self) -> None:
            """记录 HTTP 层已经关闭上传文件。

            Returns:
                None。
            """
            self.closed = True

    file = _SizeTrackingFile()
    pipeline, _ = make_test_pipeline()
    service = UploadService(
        docs_path=tmp_path,
        max_upload_size=8,
        ingestion_pipeline=pipeline,
    )

    with pytest.raises(FileTooLargeError):
        await upload_document(file=file, service=service)

    assert file.requested_size == 9
    assert file.closed is True
