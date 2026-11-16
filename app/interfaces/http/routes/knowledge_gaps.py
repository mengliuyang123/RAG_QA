# -*- coding: utf-8 -*-
"""知识缺口工作台的状态接口。

缺口候选来自现有的拒答聚类与点踩反馈接口，补充文档走知识库上传接口，
复测走 /api/query；这里只保存每个缺口的处理状态与最近一次复测结果。
"""

from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Path
from fastapi import Request
from fastapi import Response
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.application.knowledge_gaps import GapVerificationError
from app.application.knowledge_gaps import KnowledgeGapService
from app.domain.auth import User
from app.domain.knowledge_bases import KB_ID_PATTERN
from app.domain.knowledge_gaps import GAP_KEY_PATTERN
from app.domain.knowledge_gaps import MAX_GAP_QUESTIONS
from app.domain.knowledge_gaps import GapCheck
from app.domain.knowledge_gaps import GapSource
from app.domain.knowledge_gaps import GapStatus
from app.domain.knowledge_gaps import KnowledgeGap
from app.interfaces.http.auth import require_admin
from app.interfaces.http.container_runtime import current_container


router = APIRouter(prefix="/admin/gaps", tags=["insights"])


class SaveGapRequest(BaseModel):
    """缺口的完整状态；PUT 整体替换。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: GapSource
    title: str = Field(min_length=1, max_length=8000)
    questions: tuple[Annotated[str, Field(min_length=1, max_length=8000)], ...] = Field(
        min_length=1, max_length=MAX_GAP_QUESTIONS,
    )
    kb_id: str | None = Field(default=None, pattern=KB_ID_PATTERN)
    status: GapStatus
    doc_id: str | None = Field(default=None, max_length=64)
    last_check: GapCheck | None = None


def get_gap_service(request: Request) -> KnowledgeGapService:
    """从组合根获取缺口状态用例。"""
    return current_container(request).knowledge_gap_service


@router.get("")
def list_gaps(
    response: Response,
    service: Annotated[KnowledgeGapService, Depends(get_gap_service)],
) -> dict[str, list[KnowledgeGap]]:
    """列出已跟踪的缺口状态。"""
    response.headers["Cache-Control"] = "no-store"
    return {"gaps": service.list()}


@router.put("/{gap_key}")
def save_gap(
    gap_key: Annotated[str, Path(pattern=GAP_KEY_PATTERN)],
    body: SaveGapRequest, response: Response,
    user: Annotated[User, Depends(require_admin)],
    service: Annotated[KnowledgeGapService, Depends(get_gap_service)],
) -> KnowledgeGap:
    """保存状态；verified 必须附带全部通过的复测结果。"""
    response.headers["Cache-Control"] = "no-store"
    try:
        return service.save(
            user.id, gap_key, source=body.source, title=body.title,
            questions=body.questions, kb_id=body.kb_id, status=body.status,
            doc_id=body.doc_id, last_check=body.last_check,
        )
    except GapVerificationError as error:
        raise HTTPException(
            422, {"code": "gap_not_verified", "message": str(error)},
        ) from error
