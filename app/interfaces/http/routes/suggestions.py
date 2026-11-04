# -*- coding: utf-8 -*-
"""管理员明确发布预设的入口，不提供自动私人历史转公开的流程。"""

from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Path
from fastapi import Response

from app.application.knowledge.browse import KnowledgeBrowseService
from app.domain.knowledge_bases import KB_ID_PATTERN
from app.domain.suggestions import CuratedQuestionsUpdate
from app.domain.suggestions import SuggestedQuestions
from app.interfaces.http.routes.knowledge_browse import get_knowledge_browse


router = APIRouter(prefix="/admin/knowledge-bases", tags=["knowledge-bases"])


@router.put("/{kb_id}/suggested-questions", response_model=SuggestedQuestions)
def replace_curated_questions(
    kb_id: Annotated[str, Path(pattern=KB_ID_PATTERN)],
    body: CuratedQuestionsUpdate, response: Response,
    service: Annotated[KnowledgeBrowseService, Depends(get_knowledge_browse)],
) -> SuggestedQuestions:
    """管理员与CSRF校验后原子替换，空集合表示明确清空。

    Args:
        kb_id: 已注册知识库。
        body: 最多50个不重复的公开问题。
        response: 禁止缓存。
        service: 生产预设用例。

    Returns:
        发布后的稳定预设集合，重复相同内容保持ID。
    """
    response.headers["Cache-Control"] = "no-store"
    return service.replace_questions(kb_id, body)
