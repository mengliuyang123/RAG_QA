# -*- coding: utf-8 -*-
"""管理员异步上传与持久任务进度；旧upload仍保留201契约。"""

from datetime import datetime
from functools import partial
from typing import Annotated
from typing import Literal
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import Path
from fastapi import Request
from fastapi import Response
from fastapi import UploadFile
from pydantic import BaseModel
from pydantic import ConfigDict
from starlette.concurrency import run_in_threadpool

from app.interfaces.http.container_runtime import current_container
from app.application.errors import DocumentAlreadyExistsError
from app.application.ingestion.jobs import IngestionJobService
from app.domain.auth import User
from app.domain.ingestion_jobs import IngestionJob
from app.interfaces.http.auth import require_user
from app.domain.knowledge_access import KbPermissions
from app.interfaces.http.kb_permissions import ensure_editor
from app.interfaces.http.kb_permissions import ensure_unrestricted
from app.interfaces.http.kb_permissions import ensure_visible
from app.interfaces.http.kb_permissions import kb_permissions


router = APIRouter(prefix="/admin")


class ExistingDocument(BaseModel):
    """已登记文档的公开字段。

    Attributes:
        doc_id: 已有文档标识。
        kb_id: 所属知识库。
        filename: 文件名。
        status: pending、ready 或 failed。
        uploaded_at: 登记时间。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: UUID
    kb_id: str
    filename: str
    status: Literal["pending", "ready", "failed"]
    uploaded_at: datetime


class QueuedIngestionResponse(BaseModel):
    """仅承诺已经持久入队，不把HTTP接收成功当作索引完成。

    Attributes:
        job_id: 后续轮询标识；内容已存在时为 None。
        doc_id: 新文档标识，或内容已存在时的已有文档。
        status: 新入队为 queued；内容完全相同时为 exists。
        existing: 内容已存在时的已有文档。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str | None
    doc_id: UUID
    status: Literal["queued", "exists"] = "queued"
    existing: ExistingDocument | None = None


def get_job_service(request: Request) -> IngestionJobService:
    """读取任务不需要启动模型或持有文档写锁。

    Args:
        request: 当前已认证管理员请求。

    Returns:
        持久入队与进度服务。
    """
    return current_container(request).ingestion_job_service


@router.post(
    "/knowledge-bases/{kb_id}/documents", status_code=202,
    response_model=QueuedIngestionResponse, response_model_exclude_unset=True,
)
async def submit_ingestion(
    request: Request, response: Response,
    kb_id: Annotated[str, Path(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")],
    file: Annotated[UploadFile, File()],
    user: Annotated[User, Depends(require_user)],
    permissions: Annotated[KbPermissions, Depends(kb_permissions)],
    service: Annotated[IngestionJobService, Depends(get_job_service)],
    on_conflict: Annotated[Literal["replace", "keep_both"] | None, Form()] = None,
    replace_doc_id: Annotated[UUID | None, Form()] = None,
) -> QueuedIngestionResponse:
    """有限读取上传，文件同步与事务入队在线程池完成，不执行入库。

    Args:
        request: 带实际后台生命周期的应用请求。
        response: 设置不可缓存响应。
        kb_id: 路径指定的目标知识库。
        file: 框架管理的上传文件。
        user: 服务器认证的管理员，上传者不可由表单指定。
        service: 持久化任务应用服务。
        on_conflict: 同名不同内容时用户的选择；不给时返回409让前端询问。
        replace_doc_id: 选择替换时被替换的旧文档。

    Returns:
        202对应的queued标识，实际进度通过GET读取；内容相同则200返回已有文档。
    """
    try:
        ensure_editor(permissions, kb_id)
        worker = current_container(request).ingestion_worker
        worker.ensure_running()
        content = await file.read(service.max_upload_size + 1)
        response.headers["Cache-Control"] = "no-store"
        try:
            queued = await run_in_threadpool(
                partial(
                    service.submit, kb_id, file.filename, content, user.id,
                    conflict=on_conflict, replace_doc_id=replace_doc_id,
                ),
            )
        except DocumentAlreadyExistsError as error:
            response.status_code = 200
            existing = ExistingDocument.model_validate(error.existing.model_dump(
                include={"doc_id", "kb_id", "filename", "status", "uploaded_at"},
            ))
            return QueuedIngestionResponse(
                job_id=None, doc_id=existing.doc_id, status="exists", existing=existing,
            )
        worker.wake()
        return QueuedIngestionResponse(
            job_id=queued.job_id, doc_id=queued.doc_id, status="queued",
        )
    finally:
        await file.close()


@router.get("/ingestion-jobs/{job_id}", response_model=IngestionJob)
def get_ingestion_progress(
    response: Response,
    job_id: Annotated[str, Path(min_length=1, max_length=128)],
    service: Annotated[IngestionJobService, Depends(get_job_service)],
    permissions: Annotated[KbPermissions, Depends(kb_permissions)],
) -> IngestionJob:
    """不等文档写锁读取真实持久状态，不暴露摘要、路径和租约。

    Args:
        response: 不可缓存响应。
        job_id: 后端返回的稳定任务标识。
        service: 持久任务读取能力。

    Returns:
        当前公开状态和安全失败分类。
    """
    response.headers["Cache-Control"] = "no-store"
    if not permissions.unrestricted and not permissions.editable:
        # 不是任何知识库 editor 的成员在查询前就拒绝，不透露任务是否存在。
        ensure_unrestricted(permissions)
    job = service.get(job_id)
    ensure_editor(permissions, job.kb_id)
    return job
