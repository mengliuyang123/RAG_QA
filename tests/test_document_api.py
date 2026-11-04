# -*- coding: utf-8 -*-
"""文档列表和删除 HTTP 契约测试。"""

from datetime import datetime, timezone
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from app.application.errors import (
    DocumentDeletePartialError,
    DocumentNotFoundError,
    DocumentStorageCorruptionError,
)
from app.domain.documents import (
    DocumentDeleteResult,
    DocumentFileStatus,
    DocumentSummary,
)
from app.interfaces.http.dependencies import get_document_service
from tests.http_auth import create_test_app as create_app


DOCUMENT_ID = UUID("11111111-1111-4111-8111-111111111111")


class FakeDocumentService:
    """为路由测试返回固定结果或稳定领域异常。"""

    def __init__(
        self,
        documents: list[DocumentSummary] | None = None,
        error: Exception | None = None,
    ) -> None:
        """保存固定文档、异常和删除调用记录。

        Args:
            documents: 列表接口应返回的固定文档。
            error: 服务方法应抛出的可选异常。

        Returns:
            None。
        """
        self.documents = documents or []
        self.error = error
        self.deleted_ids: list[UUID] = []

    def list_documents(self) -> list[DocumentSummary]:
        """返回固定列表或注入异常。

        Returns:
            固定的文档摘要列表。

        Raises:
            Exception: 测试场景注入的稳定或未知异常。
        """
        if self.error is not None:
            raise self.error
        return self.documents

    def delete_document(self, doc_id: UUID) -> DocumentDeleteResult:
        """记录规范 UUID 并返回固定删除结果。

        Args:
            doc_id: HTTP 层传入的规范文档 UUID。

        Returns:
            固定的成功删除结果。

        Raises:
            Exception: 测试场景注入的稳定或未知异常。
        """
        self.deleted_ids.append(doc_id)
        if self.error is not None:
            raise self.error
        return DocumentDeleteResult(doc_id=doc_id)


@pytest.fixture
def anyio_backend() -> str:
    """API 测试固定使用 asyncio。

    Returns:
        AnyIO 使用的 asyncio 后端名称。
    """
    return "asyncio"


def _app(service: FakeDocumentService) -> FastAPI:
    """创建注入指定文档服务的 FastAPI 应用。

    Args:
        service: 当前测试场景使用的文档服务。

    Returns:
        注入指定文档服务的 FastAPI 应用。
    """
    application = create_app()
    application.dependency_overrides[get_document_service] = lambda: service
    return application


@pytest.mark.anyio
async def test_get_documents_returns_stable_list_and_total() -> None:
    """GET 应返回 UTC 时间、文件状态和精确 total。"""
    service = FakeDocumentService(
        documents=[
            DocumentSummary(
                doc_id=DOCUMENT_ID,
                filename="notes.md",
                chunks_count=2,
                created_at=datetime(
                    2026,
                    8,
                    20,
                    1,
                    2,
                    3,
                    tzinfo=timezone.utc,
                ),
                file_status=DocumentFileStatus.AVAILABLE,
            )
        ]
    )
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/documents")

    assert response.status_code == 200
    assert response.json() == {
        "documents": [
            {
                "doc_id": str(DOCUMENT_ID),
                "filename": "notes.md",
                "chunks_count": 2,
                "created_at": "2026-08-20T01:02:03Z",
                "file_status": "available",
                "kb_id": None,
                "file_type": None,
                "size_bytes": None,
                "uploaded_by": None,
                "ingestion_status": None,
                "published": None,
                "citations_7d": None,
                "unattributed_citations_7d": None,
            }
        ],
        "total": 1,
    }


@pytest.mark.anyio
async def test_get_documents_returns_empty_list() -> None:
    """空知识库应返回空列表和零 total。"""
    transport = httpx.ASGITransport(app=_app(FakeDocumentService()))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/documents")

    assert response.status_code == 200
    assert response.json() == {"documents": [], "total": 0}


@pytest.mark.anyio
async def test_get_documents_hides_storage_corruption_details() -> None:
    """存储损坏应返回稳定 500，不泄露元数据内容。"""
    service = FakeDocumentService(
        error=DocumentStorageCorruptionError("private /absolute/path")
    )
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/documents")

    assert response.status_code == 500
    assert response.json()["detail"] == "文档存储状态异常"
    assert "absolute" not in response.text


@pytest.mark.anyio
async def test_delete_document_passes_canonical_uuid_to_service() -> None:
    """合法路径 UUID 应进入服务并返回固定成功结构。"""
    service = FakeDocumentService()
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.delete(f"/api/documents/{DOCUMENT_ID}")

    assert response.status_code == 200
    assert response.json() == {
        "doc_id": str(DOCUMENT_ID),
        "deleted": True,
    }
    assert service.deleted_ids == [DOCUMENT_ID]


@pytest.mark.anyio
async def test_delete_document_rejects_malformed_uuid_before_service() -> None:
    """非法 UUID 应由 FastAPI 返回 422，且不调用服务。"""
    service = FakeDocumentService()
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.delete("/api/documents/not-a-uuid")

    assert response.status_code == 422
    assert service.deleted_ids == []


@pytest.mark.anyio
async def test_delete_document_maps_missing_to_404() -> None:
    """未知文档应映射为不含内部信息的 HTTP 404。"""
    service = FakeDocumentService(error=DocumentNotFoundError("private"))
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.delete(f"/api/documents/{DOCUMENT_ID}")

    assert response.status_code == 404
    assert response.json()["detail"] == "文档不存在"


@pytest.mark.anyio
async def test_delete_document_maps_partial_cleanup_to_structured_500() -> None:
    """索引已删除后的文件失败应返回可修复结构化 500。"""
    service = FakeDocumentService(error=DocumentDeletePartialError())
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.delete(f"/api/documents/{DOCUMENT_ID}")

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "document_delete_partial",
        "message": "文档索引已删除，但原文件清理失败",
        "repair_required": True,
    }


@pytest.mark.anyio
async def test_delete_document_hides_unexpected_error() -> None:
    """未知删除错误不得把原始异常正文返回客户端。"""
    service = FakeDocumentService(
        error=RuntimeError("private database response")
    )
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.delete(f"/api/documents/{DOCUMENT_ID}")

    assert response.status_code == 500
    assert response.json()["detail"] == "文档删除失败"
    assert "private" not in response.text
