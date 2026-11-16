# -*- coding: utf-8 -*-
"""文档列表与删除 HTTP 路由。"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status
from fastapi import Query
from fastapi import Response
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.application.documents.service import DocumentService
from app.application.errors import DocumentDeletePartialError
from app.application.errors import DocumentNotFoundError
from app.application.errors import DocumentStorageCorruptionError
from app.application.errors import KnowledgeContentError
from app.application.errors import DocumentIngestionInProgressError
from app.application.errors import VectorStoreError
from app.interfaces.http.dependencies import get_document_service
from app.interfaces.http.schemas import DocumentDeleteResponse
from app.interfaces.http.schemas import DocumentListResponse
from app.domain.knowledge_bases import KB_ID_PATTERN
from app.domain.knowledge_access import KbPermissions
from app.interfaces.http.kb_permissions import ensure_editor
from app.interfaces.http.kb_permissions import ensure_unrestricted
from app.interfaces.http.kb_permissions import ensure_visible
from app.interfaces.http.kb_permissions import kb_permissions


router = APIRouter()


def _ensure_documents_editable(
    service: DocumentService, permissions: KbPermissions, doc_ids: tuple[UUID, ...],
) -> None:
    """非管理员只能删除自己是 editor 的知识库里的文档。"""
    if permissions.unrestricted:
        return
    owners = {str(document.doc_id): document.kb_id for document in service.list_documents()}
    for doc_id in doc_ids:
        ensure_editor(permissions, owners.get(str(doc_id)) or "")
logger = logging.getLogger(__name__)


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    description=(
        "返回当前向量存储中的文档聚合列表。\n\n"
        "Args:\n"
        "    service: 共享生产依赖装配的文档服务。\n\n"
        "Returns:\n"
        "    文档列表及精确总数。"
    ),
)
def list_documents(
    service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
    response: Response,
    permissions: Annotated[KbPermissions, Depends(kb_permissions)],
    kb_id: Annotated[str | None, Query(pattern=KB_ID_PATTERN)] = None,
) -> DocumentListResponse:
    """返回当前向量存储中的文档聚合列表。

    Args:
        service: 生产或测试注入的文档应用服务。
        response: 禁止缓存的响应。
        kb_id: 可选实际索引知识库过滤。

    Returns:
        文档列表及精确总数。

    Raises:
        HTTPException: 列表读取发生未知内部错误。
        DocumentStorageCorruptionError: 存储元数据损坏。
    """
    if kb_id is None:
        ensure_unrestricted(permissions)
    else:
        ensure_editor(permissions, kb_id)
    try:
        response.headers["Cache-Control"] = "no-store"
        documents = service.list_documents() if kb_id is None else service.list_documents(kb_id=kb_id)
        serialized_documents = []
        for document in documents:
            serialized_documents.append(document.model_dump())
        return DocumentListResponse(
            documents=serialized_documents,
            total=len(documents),
        )
    except (DocumentStorageCorruptionError, KnowledgeContentError):
        raise
    except Exception as error:
        logger.exception("文档列表读取失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档列表读取失败",
        ) from error


class BatchDeleteRequest(BaseModel):
    """批量删除请求，一次最多 100 篇。

    Attributes:
        doc_ids: 待删除文档 UUID。
    """

    model_config = ConfigDict(extra="forbid")

    doc_ids: list[UUID] = Field(min_length=1, max_length=100)


class BatchDeleteResponse(BaseModel):
    """批量删除结果。

    Attributes:
        deleted: 已删除的文档。
        not_found: 不存在的文档。
    """

    model_config = ConfigDict(extra="forbid")

    deleted: list[UUID]
    not_found: list[UUID]


@router.post("/documents/batch-delete", response_model=BatchDeleteResponse)
def delete_documents(
    body: BatchDeleteRequest,
    service: Annotated[DocumentService, Depends(get_document_service)],
    response: Response,
    permissions: Annotated[KbPermissions, Depends(kb_permissions)],
) -> BatchDeleteResponse:
    """按一组 doc_id 删除分块、登记与源文件，入库失败的文档同样可删。

    Args:
        body: 待删除的文档 UUID 列表。
        service: 文档应用服务。
        response: 禁止缓存的响应。

    Returns:
        已删除与不存在的文档。
    """
    response.headers["Cache-Control"] = "no-store"
    _ensure_documents_editable(service, permissions, tuple(body.doc_ids))
    result = service.delete_documents(tuple(body.doc_ids))
    return BatchDeleteResponse(deleted=list(result.deleted), not_found=list(result.not_found))


@router.delete(
    "/documents/{doc_id}",
    response_model=DocumentDeleteResponse,
    description=(
        "删除一个文档的索引和受控目录内源文件。\n\n"
        "Args:\n"
        "    doc_id: FastAPI 已校验的规范文档 UUID。\n"
        "    service: 与上传共享生产依赖的文档服务。\n\n"
        "Returns:\n"
        "    删除成功的文档 UUID 和布尔标记。"
    ),
)
def delete_document(
    doc_id: UUID,
    service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
    permissions: Annotated[KbPermissions, Depends(kb_permissions)],
) -> DocumentDeleteResponse:
    """删除一个文档的索引和受控目录内源文件。

    Args:
        doc_id: FastAPI 已校验的规范文档 UUID。
        service: 生产或测试注入的文档应用服务。

    Returns:
        删除成功的文档 UUID 和布尔标记。

    Raises:
        HTTPException: 删除发生未知内部错误。
        DocumentNotFoundError: 文档不存在。
        DocumentDeletePartialError: 索引已删除但源文件清理失败。
        DocumentStorageCorruptionError: 存储元数据损坏。
    """
    _ensure_documents_editable(service, permissions, (doc_id,))
    try:
        result = service.delete_document(doc_id)
        return DocumentDeleteResponse.model_validate(
            result.model_dump()
        )
    except (
        DocumentNotFoundError,
        DocumentDeletePartialError,
        DocumentStorageCorruptionError,
        KnowledgeContentError,
        DocumentIngestionInProgressError,
        VectorStoreError,
    ):
        raise
    except Exception as error:
        logger.exception("文档删除失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档删除失败",
        ) from error
