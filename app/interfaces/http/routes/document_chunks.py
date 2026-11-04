# -*- coding: utf-8 -*-
"""仅管理员可访问的分块正文诊断协议。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi import Response

from app.application.documents.chunks import AdminDocumentChunksService
from app.domain.auth import User
from app.domain.document_diagnostics import DiagnosticChunkPage
from app.interfaces.http.auth import require_admin


router = APIRouter(prefix="/admin/documents", tags=["documents"])


def get_admin_document_chunks(request: Request) -> AdminDocumentChunksService:
    """从组合根取得不依赖模型的诊断用例。

    Args:
        request: 当前HTTP请求。

    Returns:
        同一运行时的分页用例。
    """
    return request.app.state.container.admin_document_chunks


@router.get("/{doc_id}/chunks", response_model=DiagnosticChunkPage)
def list_chunks(
    doc_id: UUID, response: Response,
    user: Annotated[User, Depends(require_admin)],
    service: Annotated[AdminDocumentChunksService, Depends(get_admin_document_chunks)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(max_length=2048)] = None,
) -> DiagnosticChunkPage:
    """校验角色与游标后返回经审计的当前正文页。

    Args:
        doc_id: 目标文档。
        response: 禁止缓存的响应。
        user: 已认证的管理员。
        service: 组合根提供的正文诊断用例。
        limit: 每页至多100块。
        cursor: 上页游标。

    Returns:
        含正式引用统计的当前正文页。
    """
    response.headers["Cache-Control"] = "no-store"
    try:
        return service.list(user.id, doc_id, limit=limit, cursor=cursor)
    except ValueError as error:
        raise HTTPException(status_code=422, detail="分块游标或参数无效") from error
