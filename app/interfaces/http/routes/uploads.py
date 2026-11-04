# -*- coding: utf-8 -*-
"""文档上传 HTTP 路由。"""

import logging
from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi import status
from starlette.concurrency import run_in_threadpool

from app.application.errors import DocumentParsingError
from app.application.errors import EmptyDocumentError
from app.application.errors import FileTooLargeError
from app.application.errors import MissingFilenameError
from app.application.errors import UnsupportedFileTypeError
from app.application.ingestion.service import UploadService
from app.interfaces.http.dependencies import get_upload_service
from app.interfaces.http.schemas import UploadParseResponse


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/upload",
    response_model=UploadParseResponse,
    status_code=status.HTTP_201_CREATED,
    description="接收文档上传并返回向量入库结果摘要。",
)
async def upload_document(
    file: Annotated[
        UploadFile,
        File(description="PDF、Markdown 或 TXT 文档"),
    ],
    service: Annotated[
        UploadService,
        Depends(get_upload_service),
    ],
) -> UploadParseResponse:
    """接收文档上传并返回向量入库结果摘要。

    Args:
        file: FastAPI 管理的上传文件。
        service: 生产或测试注入的上传应用服务。

    Returns:
        文档标识、清理后文件名和入库文本块数量。

    Raises:
        HTTPException: 上传发生未知内部错误。
        UnsupportedFileTypeError: 文件类型不受支持。
        FileTooLargeError: 文件超过配置上限。
        MissingFilenameError: 上传缺少安全文件名。
        EmptyDocumentError: 文件为空或没有可提取文本。
        DocumentParsingError: 文档无法解析。
    """
    try:
        # 多读取 1 字节即可判断超限，避免先把任意大文件
        # 完整载入内存。
        content = await file.read(service.max_upload_size + 1)
        result = await run_in_threadpool(
            service.save_and_parse, file.filename, content,
        )
        return UploadParseResponse.model_validate(result.model_dump())
    except (
        UnsupportedFileTypeError,
        FileTooLargeError,
        MissingFilenameError,
        EmptyDocumentError,
        DocumentParsingError,
    ):
        raise
    except Exception as error:
        logger.exception("文档上传处理失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档上传处理失败",
        ) from error
    finally:
        await file.close()
