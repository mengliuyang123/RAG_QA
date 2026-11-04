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


router = APIRouter()
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
