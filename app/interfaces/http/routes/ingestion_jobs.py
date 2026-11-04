# -*- coding: utf-8 -*-
"""管理员异步上传与持久任务进度；旧upload仍保留201契约。"""

from typing import Annotated
from typing import Literal
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Path
from fastapi import Request
from fastapi import Response
from fastapi import UploadFile
from pydantic import BaseModel
from pydantic import ConfigDict
from starlette.concurrency import run_in_threadpool

from app.application.ingestion.jobs import IngestionJobService
from app.domain.auth import User
from app.domain.ingestion_jobs import IngestionJob
from app.interfaces.http.auth import require_admin


router = APIRouter(prefix="/admin")


class QueuedIngestionResponse(BaseModel):
    """仅承诺已经持久入队，不把HTTP接收成功当作索引完成。

    Attributes:
        job_id: 后续轮询标识。
        doc_id: 新文档标识。
        status: 接收时固定为queued。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    doc_id: UUID
    status: Literal["queued"] = "queued"


def get_job_service(request: Request) -> IngestionJobService:
    """读取任务不需要启动模型或持有文档写锁。

    Args:
        request: 当前已认证管理员请求。

    Returns:
        持久入队与进度服务。
    """
    return request.app.state.container.ingestion_job_service


@router.post(
    "/knowledge-bases/{kb_id}/documents", status_code=202,
    response_model=QueuedIngestionResponse,
)
async def submit_ingestion(
    request: Request, response: Response,
    kb_id: Annotated[str, Path(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")],
    file: Annotated[UploadFile, File()],
    user: Annotated[User, Depends(require_admin)],
    service: Annotated[IngestionJobService, Depends(get_job_service)],
) -> QueuedIngestionResponse:
    """有限读取上传，文件同步与事务入队在线程池完成，不执行入库。

    Args:
        request: 带实际后台生命周期的应用请求。
        response: 设置不可缓存响应。
        kb_id: 路径指定的目标知识库。
        file: 框架管理的上传文件。
        user: 服务器认证的管理员，上传者不可由表单指定。
        service: 持久化任务应用服务。

    Returns:
        202对应的queued标识，实际进度通过GET读取。
    """
    try:
        worker = request.app.state.container.ingestion_worker
        worker.ensure_running()
        content = await file.read(service.max_upload_size + 1)
        queued = await run_in_threadpool(
            service.submit, kb_id, file.filename, content, user.id,
        )
        worker.wake()
        response.headers["Cache-Control"] = "no-store"
        return QueuedIngestionResponse(
            job_id=queued.job_id, doc_id=queued.doc_id,
        )
    finally:
        await file.close()


@router.get("/ingestion-jobs/{job_id}", response_model=IngestionJob)
def get_ingestion_progress(
    response: Response,
    job_id: Annotated[str, Path(min_length=1, max_length=128)],
    service: Annotated[IngestionJobService, Depends(get_job_service)],
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
    return service.get(job_id)
