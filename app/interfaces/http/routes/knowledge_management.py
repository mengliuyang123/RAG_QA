# -*- coding: utf-8 -*-
"""知识库访问设置（管理员）、文档复审与标准答案（库 editor 或管理员）。"""

import logging
from datetime import date
from typing import Annotated
from typing import Literal
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Path
from fastapi import Request
from fastapi import Response
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.application.documents.service import DocumentService
from app.application.errors import DocumentAlreadyExistsError
from app.application.errors import DocumentNameConflictError
from app.application.errors import DocumentReplaceConflictError
from app.application.errors import KnowledgeBaseNotFoundError
from app.application.ingestion.jobs import IngestionJobService
from app.application.knowledge.access import FAQ_FILENAME
from app.application.knowledge.access import FaqNotFoundError
from app.application.knowledge.access import FaqService
from app.domain.auth import User
from app.domain.knowledge_access import DocumentReview
from app.domain.knowledge_access import FaqEntry
from app.domain.knowledge_access import KbAccess
from app.domain.knowledge_access import KbMember
from app.domain.knowledge_access import KbMemberRole
from app.domain.knowledge_access import KbPermissions
from app.domain.knowledge_bases import KB_ID_PATTERN
from app.interfaces.http.auth import require_admin
from app.interfaces.http.auth import require_user
from app.interfaces.http.container_runtime import current_container
from app.interfaces.http.dependencies import get_document_service
from app.interfaces.http.kb_permissions import ensure_editor
from app.interfaces.http.kb_permissions import kb_permissions
from app.interfaces.http.routes.ingestion_jobs import get_job_service


router = APIRouter(prefix="/admin/knowledge-bases", tags=["knowledge-bases"])
logger = logging.getLogger(__name__)
KbId = Annotated[str, Path(pattern=KB_ID_PATTERN)]


def _ensure_kb_exists(request: Request, kb_id: str) -> None:
    if all(kb.id != kb_id for kb in current_container(request).knowledge_base_repository.list()):
        raise KnowledgeBaseNotFoundError("知识库不存在")


def _editable_kb(
    request: Request, kb_id: str, permissions: KbPermissions,
) -> None:
    ensure_editor(permissions, kb_id)
    _ensure_kb_exists(request, kb_id)


# ---------------------------------------------------------------- 访问设置

class MemberInput(BaseModel):
    """一名成员。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: UUID
    role: KbMemberRole


class AccessRequest(BaseModel):
    """访问设置，整体替换。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    restricted: bool
    members: tuple[MemberInput, ...] = Field(default=(), max_length=500)


@router.get("/{kb_id}/access")
def get_access(
    kb_id: KbId, request: Request, response: Response,
    _admin: Annotated[User, Depends(require_admin)],
) -> KbAccess:
    """读取知识库访问设置（仅管理员）。"""
    response.headers["Cache-Control"] = "no-store"
    _ensure_kb_exists(request, kb_id)
    return current_container(request).knowledge_access_service.get(kb_id)


@router.put("/{kb_id}/access")
def save_access(
    kb_id: KbId, body: AccessRequest, request: Request, response: Response,
    admin: Annotated[User, Depends(require_admin)],
) -> KbAccess:
    """设置是否受限及成员（仅管理员）；变更在成员的下一次请求生效。"""
    response.headers["Cache-Control"] = "no-store"
    container = current_container(request)
    _ensure_kb_exists(request, kb_id)
    known = {user.id for user in container.auth_service.list_users()}
    if any(member.user_id not in known for member in body.members):
        raise HTTPException(422, "成员中有不存在的用户")
    try:
        saved = container.knowledge_access_service.save(
            kb_id, restricted=body.restricted,
            members=tuple(KbMember(user_id=m.user_id, role=m.role) for m in body.members),
        )
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    logger.info(
        "管理员 %s 修改了知识库 %s 的访问设置：restricted=%s members=%d",
        admin.id, kb_id, saved.restricted, len(saved.members),
    )
    return saved


# ---------------------------------------------------------------- 文档复审

class ReviewRequest(BaseModel):
    """复审信息。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review_due_on: date | None = None
    outdated: bool = False
    note: str | None = Field(default=None, max_length=500)


@router.get("/{kb_id}/document-reviews")
def list_reviews(
    kb_id: KbId, request: Request, response: Response,
    permissions: Annotated[KbPermissions, Depends(kb_permissions)],
) -> dict[str, list[DocumentReview]]:
    """列出复审记录。"""
    response.headers["Cache-Control"] = "no-store"
    _editable_kb(request, kb_id, permissions)
    return {"reviews": current_container(request).document_review_service.list(kb_id)}


@router.put("/{kb_id}/document-reviews/{doc_id}")
def save_review(
    kb_id: KbId, doc_id: UUID, body: ReviewRequest, request: Request, response: Response,
    user: Annotated[User, Depends(require_user)],
    permissions: Annotated[KbPermissions, Depends(kb_permissions)],
    documents: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentReview:
    """设置复审日期或已过时；已过时的文档立即停止参与检索。"""
    response.headers["Cache-Control"] = "no-store"
    _editable_kb(request, kb_id, permissions)
    if all(str(document.doc_id) != str(doc_id) for document in documents.list_documents(kb_id=kb_id)):
        raise HTTPException(404, "文档不存在")
    return current_container(request).document_review_service.save(
        user.id, kb_id, str(doc_id),
        review_due_on=body.review_due_on, outdated=body.outdated, note=body.note,
    )


# ---------------------------------------------------------------- 标准答案

class FaqRequest(BaseModel):
    """一条标准答案。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1, max_length=500)
    answer: str = Field(min_length=1, max_length=5000)


class FaqPublication(BaseModel):
    """把标准答案发布为知识库文档的结果。

    Attributes:
        status: queued 已提交入库；unchanged 内容未变；removed 已删除空文档；
            none 没有可发布内容；busy 上一版仍在入库，需稍后重新发布。
        job_id: 入库任务。
    """

    status: Literal["queued", "unchanged", "removed", "none", "busy"]
    job_id: str | None = None


class FaqResponse(BaseModel):
    """标准答案列表与最近一次发布结果。"""

    entries: list[FaqEntry]
    filename: str = FAQ_FILENAME
    publication: FaqPublication | None = None


def _publish(
    request: Request, kb_id: str, user: User, faq: FaqService,
    jobs: IngestionJobService, documents: DocumentService,
) -> FaqPublication:
    entries = faq.list(kb_id)
    existing = next(
        (doc for doc in documents.list_documents(kb_id=kb_id) if doc.filename == FAQ_FILENAME), None,
    )
    if not entries:
        if existing is None:
            return FaqPublication(status="none")
        documents.delete_document(UUID(str(existing.doc_id)))
        return FaqPublication(status="removed")
    content = faq.render(entries).encode("utf-8")
    worker = current_container(request).ingestion_worker
    worker.ensure_running()
    try:
        try:
            job = jobs.submit(kb_id, FAQ_FILENAME, content, user.id)
        except DocumentNameConflictError as conflict:
            job = jobs.submit(
                kb_id, FAQ_FILENAME, content, user.id,
                conflict="replace", replace_doc_id=UUID(str(conflict.existing.doc_id)),
            )
    except DocumentAlreadyExistsError:
        return FaqPublication(status="unchanged")
    except DocumentReplaceConflictError:
        return FaqPublication(status="busy")
    worker.wake()
    return FaqPublication(status="queued", job_id=job.job_id)


def _faq_response(
    request: Request, kb_id: str, user: User, publish: bool,
    jobs: IngestionJobService, documents: DocumentService,
) -> FaqResponse:
    faq = current_container(request).faq_service
    publication = _publish(request, kb_id, user, faq, jobs, documents) if publish else None
    return FaqResponse(entries=faq.list(kb_id), publication=publication)


@router.get("/{kb_id}/faq")
def list_faq(
    kb_id: KbId, request: Request, response: Response,
    user: Annotated[User, Depends(require_user)],
    permissions: Annotated[KbPermissions, Depends(kb_permissions)],
    jobs: Annotated[IngestionJobService, Depends(get_job_service)],
    documents: Annotated[DocumentService, Depends(get_document_service)],
) -> FaqResponse:
    """列出标准答案。"""
    response.headers["Cache-Control"] = "no-store"
    _editable_kb(request, kb_id, permissions)
    return _faq_response(request, kb_id, user, False, jobs, documents)


@router.post("/{kb_id}/faq", status_code=201)
def create_faq(
    kb_id: KbId, body: FaqRequest, request: Request, response: Response,
    user: Annotated[User, Depends(require_user)],
    permissions: Annotated[KbPermissions, Depends(kb_permissions)],
    jobs: Annotated[IngestionJobService, Depends(get_job_service)],
    documents: Annotated[DocumentService, Depends(get_document_service)],
) -> FaqResponse:
    """新增一条标准答案并重新发布「常见问题」文档。"""
    response.headers["Cache-Control"] = "no-store"
    _editable_kb(request, kb_id, permissions)
    current_container(request).faq_service.save(
        user.id, kb_id, entry_id=None, question=body.question, answer=body.answer,
    )
    return _faq_response(request, kb_id, user, True, jobs, documents)


@router.put("/{kb_id}/faq/{entry_id}")
def update_faq(
    kb_id: KbId, entry_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")],
    body: FaqRequest, request: Request, response: Response,
    user: Annotated[User, Depends(require_user)],
    permissions: Annotated[KbPermissions, Depends(kb_permissions)],
    jobs: Annotated[IngestionJobService, Depends(get_job_service)],
    documents: Annotated[DocumentService, Depends(get_document_service)],
) -> FaqResponse:
    """修改一条标准答案并重新发布。"""
    response.headers["Cache-Control"] = "no-store"
    _editable_kb(request, kb_id, permissions)
    try:
        current_container(request).faq_service.save(
            user.id, kb_id, entry_id=entry_id, question=body.question, answer=body.answer,
        )
    except FaqNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    return _faq_response(request, kb_id, user, True, jobs, documents)


@router.delete("/{kb_id}/faq/{entry_id}")
def delete_faq(
    kb_id: KbId, entry_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")],
    request: Request, response: Response,
    user: Annotated[User, Depends(require_user)],
    permissions: Annotated[KbPermissions, Depends(kb_permissions)],
    jobs: Annotated[IngestionJobService, Depends(get_job_service)],
    documents: Annotated[DocumentService, Depends(get_document_service)],
) -> FaqResponse:
    """删除一条标准答案并重新发布；删空后移除「常见问题」文档。"""
    response.headers["Cache-Control"] = "no-store"
    _editable_kb(request, kb_id, permissions)
    try:
        current_container(request).faq_service.delete(kb_id, entry_id)
    except FaqNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    return _faq_response(request, kb_id, user, True, jobs, documents)


@router.post("/{kb_id}/faq/publish")
def publish_faq(
    kb_id: KbId, request: Request, response: Response,
    user: Annotated[User, Depends(require_user)],
    permissions: Annotated[KbPermissions, Depends(kb_permissions)],
    jobs: Annotated[IngestionJobService, Depends(get_job_service)],
    documents: Annotated[DocumentService, Depends(get_document_service)],
) -> FaqResponse:
    """手动重新发布（例如上次发布时上一版仍在入库）。"""
    response.headers["Cache-Control"] = "no-store"
    _editable_kb(request, kb_id, permissions)
    return _faq_response(request, kb_id, user, True, jobs, documents)
