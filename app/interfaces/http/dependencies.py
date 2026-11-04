# -*- coding: utf-8 -*-
"""从当前 FastAPI 应用容器解析应用服务。"""

from fastapi import Request

from app.application.documents.service import DocumentService
from app.application.conversations.service import ConversationService
from app.application.ingestion.service import UploadService
from app.application.query.routed_service import RoutedQueryService
from app.bootstrap.container import ApplicationContainer


def get_conversation_service(request: Request) -> ConversationService:
    """返回当前容器共享的多轮会话应用服务。"""
    container: ApplicationContainer = request.app.state.container
    return container.conversation_service


def get_query_service(request: Request) -> RoutedQueryService:
    """返回当前应用容器共享的查询服务。

    Args:
        request: 当前 FastAPI 请求。

    Returns:
        当前应用惰性创建并共享的查询服务。
    """
    container: ApplicationContainer = request.app.state.container
    return container.query_service


def get_upload_service(request: Request) -> UploadService:
    """返回当前应用容器共享的上传服务。

    Args:
        request: 当前 FastAPI 请求。

    Returns:
        当前应用惰性创建并共享的上传服务。
    """
    container: ApplicationContainer = request.app.state.container
    return container.upload_service


def get_document_service(request: Request) -> DocumentService:
    """返回当前应用容器共享的文档服务。

    Args:
        request: 当前 FastAPI 请求。

    Returns:
        当前应用惰性创建并共享的文档服务。
    """
    container: ApplicationContainer = request.app.state.container
    return container.document_service
