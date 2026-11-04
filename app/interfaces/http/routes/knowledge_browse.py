# -*- coding: utf-8 -*-
"""已登录用户的文档浏览和原文窗口，不复用管理员运维列表。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Path
from fastapi import Query
from fastapi import Request
from fastapi import Response

from app.application.knowledge.browse import KnowledgeBrowseService
from app.domain.knowledge_bases import KB_ID_PATTERN
from app.domain.knowledge_views import DocumentContext
from app.domain.knowledge_views import KnowledgeDocumentPage
from app.domain.suggestions import SuggestedQuestions


router = APIRouter(tags=["knowledge-browse"])


def get_knowledge_browse(request: Request) -> KnowledgeBrowseService:
    """只通过当前组合根获取范围受控的只读服务。

    Args:
        request: 已经通过路由登录检查的请求。

    Returns:
        本应用共享的浏览服务。
    """
    return request.app.state.container.knowledge_browse_service


@router.get("/suggested-questions", response_model=SuggestedQuestions)
def suggested_questions(
    response: Response,
    service: Annotated[KnowledgeBrowseService, Depends(get_knowledge_browse)],
    kb_id: Annotated[str | None, Query(pattern=KB_ID_PATTERN)] = None,
    limit: Annotated[int, Query(ge=1, le=20)] = 3,
) -> SuggestedQuestions:
    """只显示管理员明确发布的预设，不自动公开真实历史提问。

    Args:
        response: 禁止缓存。
        service: 范围与预设用例。
        kb_id: 可选知识库范围。
        limit: 1–20条，默认3条。

    Returns:
        已发布预设，未配置时为空。
    """
    response.headers["Cache-Control"] = "no-store"
    return service.suggested_questions(kb_id, limit)


@router.get("/knowledge-bases/{kb_id}/documents", response_model=KnowledgeDocumentPage)
def list_knowledge_documents(
    kb_id: Annotated[str, Path(pattern=KB_ID_PATTERN)], response: Response,
    service: Annotated[KnowledgeBrowseService, Depends(get_knowledge_browse)],
    q: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(max_length=2048)] = None,
) -> KnowledgeDocumentPage:
    """提供成员只读的元数据页，不公开正文或运维状态。

    Args:
        kb_id: 实际知识库范围。
        response: 禁止缓存。
        service: 范围验证后的浏览用例。
        q: 字面文件名子串。
        limit: 页大小。
        cursor: 与查询条件绑定的游标。

    Returns:
        文档元数据和可选下一页游标。
    """
    response.headers["Cache-Control"] = "no-store"
    try:
        return service.list_documents(kb_id, q=q, limit=limit, cursor=cursor)
    except ValueError as error:
        raise HTTPException(status_code=422, detail="文档分页参数无效") from error


@router.get("/documents/{doc_id}/chunks/{chunk_id:path}", response_model=DocumentContext)
def get_document_context(
    doc_id: UUID, chunk_id: Annotated[str, Path(min_length=1, max_length=2048)],
    response: Response,
    service: Annotated[KnowledgeBrowseService, Depends(get_knowledge_browse)],
    context: Annotated[int, Query(ge=0, le=3)] = 1,
) -> DocumentContext:
    """加载引用目标及其实际邻居，块ID作为完整不透明字符串处理。

    Args:
        doc_id: 来源文档UUID。
        chunk_id: 允许编码分隔符的不透明ID。
        response: 禁止缓存认证后的正文。
        service: 范围受控的只读用例。
        context: 前后各取0–3块。

    Returns:
        最多七块原文及明确目标标记。
    """
    response.headers["Cache-Control"] = "no-store"
    return service.get_context(doc_id, chunk_id, context)
