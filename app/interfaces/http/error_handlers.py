# -*- coding: utf-8 -*-
"""集中映射稳定应用与领域异常到 HTTP 响应。"""

import logging

from fastapi import FastAPI
from fastapi import Request
from fastapi import status
from fastapi.responses import JSONResponse

from app.application.errors import DocumentDeletePartialError
from app.application.errors import DocumentIngestionInProgressError
from app.application.errors import ConversationAgentError
from app.application.errors import ConversationBusyError
from app.application.errors import ConversationDeleteError
from app.application.errors import ConversationDeletingError
from app.application.errors import ConversationNotFoundError
from app.application.errors import ConversationStorageError
from app.application.errors import ConversationTimeoutError
from app.application.errors import DocumentNotFoundError
from app.application.errors import DocumentRevisionChangedError
from app.application.errors import DocumentParsingError
from app.application.errors import DocumentStorageCorruptionError
from app.application.errors import EmptyDocumentError
from app.application.errors import EmbeddingError
from app.application.errors import FileTooLargeError
from app.application.errors import LLMError
from app.application.errors import LLMTimeoutError
from app.application.errors import MissingFilenameError
from app.application.errors import RerankerUnavailableError
from app.application.errors import UnsupportedFileTypeError
from app.application.errors import VectorStoreError
from app.domain.errors import RetrievalError
from app.domain.errors import HistoryContextTooLargeError
from app.domain.errors import PromptContextTooLargeError
from app.domain.errors import InvalidKnowledgeScopeError
from app.application.errors import KnowledgeBaseStorageError
from app.application.errors import KnowledgeBaseNotFoundError
from app.application.errors import KnowledgeBaseConflictError
from app.application.errors import KnowledgeBaseNotEmptyError
from app.application.errors import KnowledgeBaseProtectedError
from app.application.errors import KnowledgeBaseIngestionBusyError
from app.application.errors import KnowledgeBaseDeletionInProgressError
from app.application.errors import KnowledgeBaseDeletePartialError
from app.application.errors import KnowledgeContentError
from app.application.errors import ChunkNotFoundError
from app.application.errors import DocumentGoneError
from app.application.errors import IngestionJobStorageError
from app.application.errors import IngestionJobNotFoundError
from app.application.errors import IngestionJobConflictError
from app.application.errors import IngestionQueueFullError
from app.application.errors import IngestionSourceError
from app.application.errors import IngestionWorkerUnavailableError
from app.application.errors import QuestionRateLimitError
from app.application.errors import QuestionQueueFullError
from app.application.errors import QuestionQueueTimeoutError
from app.application.errors import TraceStorageError
from app.application.errors import TraceNotFoundError
from app.application.errors import FeedbackNotFoundError
from app.application.errors import FeedbackChangedError
from app.application.errors import InsightStorageError
from app.application.errors import RefusalClustersBusyError
from app.application.errors import RefusalClustersChangedError
from app.application.errors import RefusalClustersTimeoutError
from app.application.errors import TurnNotFoundError
from app.application.errors import ShareExpiredError
from app.application.errors import ShareNotFoundError
from app.application.errors import ShareStorageError
from app.application.errors import EvaluationReportNotFoundError
from app.application.errors import EvaluationReportStorageError
from app.interfaces.http.auth_errors import register_auth_error_handlers
from app.interfaces.http.error_contract import install_error_contract


logger = logging.getLogger(__name__)


async def _handle_refusal_cluster_error(
    request: Request, error: Exception,
) -> JSONResponse:
    code, message, status_code = (
        "refusal_clusters_changed", "拒答数据已变化，请刷新首页", 409,
    )
    if isinstance(error, RefusalClustersBusyError):
        code, message = "refusal_clusters_busy", "拒答聚类正在运行，请稍后重试"
    elif isinstance(error, RefusalClustersTimeoutError):
        code, message, status_code = (
            "refusal_clusters_timeout", "拒答聚类超时，请缩小窗口重试", 504,
        )
    return JSONResponse(
        status_code=status_code, headers={"Cache-Control": "no-store"},
        content={"detail": message, "error": {
            "code": code, "message": message, "details": {},
        }},
    )


async def _handle_evaluation_report_error(
    request: Request, error: Exception,
) -> JSONResponse:
    code, message, status_code = (
        "evaluation_unavailable", "评估报告暂时不可用", 503,
    )
    if isinstance(error, EvaluationReportNotFoundError):
        code, message, status_code = "no_evaluation", "尚无已发布评估", 404
    return JSONResponse(
        status_code=status_code, headers={"Cache-Control": "no-store"},
        content={
            "detail": message,
            "error": {"code": code, "message": message, "details": {}},
        },
    )


async def _handle_share_error(
    request: Request, error: Exception,
) -> JSONResponse:
    code, message, status_code = "share_unavailable", "分享暂时不可用", 503
    if isinstance(error, ShareNotFoundError):
        code, message, status_code = "share_not_found", "分享不存在", 404
    elif isinstance(error, ShareExpiredError):
        code, message, status_code = "share_expired", "分享已过期", 410
    return JSONResponse(
        status_code=status_code,
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
        content={
            "detail": message,
            "error": {"code": code, "message": message, "details": {}},
        },
    )


async def _handle_document_ingestion_busy(
    request: Request, error: DocumentIngestionInProgressError,
) -> JSONResponse:
    message = "文档仍在入库，请等待任务结束后删除"
    return JSONResponse(status_code=409, headers={"Cache-Control": "no-store"}, content={
        "detail": message,
        "error": {
            "code": "document_ingestion_in_progress", "message": message,
            "details": {},
        },
    })


async def _handle_question_admission_error(
    request: Request, error: Exception,
) -> JSONResponse:
    status_code, code = 503, "queue_full"
    message = "当前提问人数较多，请稍后重试"
    details = {}
    headers = {"Cache-Control": "no-store"}
    if isinstance(error, QuestionRateLimitError):
        status_code, code = 429, "rate_limited"
        message = "提问过于频繁，请稍后重试"
        details["retry_after"] = error.retry_after
        headers["Retry-After"] = str(error.retry_after)
    elif isinstance(error, QuestionQueueTimeoutError):
        code, message = "queue_timeout", "排队等待超时，请稍后重试"
    return JSONResponse(status_code=status_code, headers=headers, content={
        "detail": message,
        "error": {"code": code, "message": message, "details": details},
    })


async def _handle_ingestion_job_error(
    request: Request, error: Exception,
) -> JSONResponse:
    code = "ingestion_unavailable"
    message = "入库任务暂时不可用，请稍后重试"
    status_code = 503
    if isinstance(error, IngestionJobNotFoundError):
        code, message, status_code = "ingestion_job_not_found", "入库任务不存在", 404
    elif isinstance(error, IngestionJobConflictError):
        code, message, status_code = "ingestion_conflict", "入库任务状态已变化", 409
    elif isinstance(error, IngestionQueueFullError):
        code, message = "ingestion_queue_full", "入库队列已满，请稍后重试"
    elif isinstance(error, IngestionSourceError):
        code, message = "ingestion_source_unavailable", "入库源文件暂时不可用"
    return JSONResponse(
        status_code=status_code, headers={"Cache-Control": "no-store"},
        content={
            "detail": message,
            "error": {"code": code, "message": message, "details": {}},
        },
    )


async def _handle_feedback_admin_error(request: Request, error: Exception) -> JSONResponse:
    code, message, status = "insight_unavailable", "洞察数据暂时不可用", 503
    if isinstance(error, FeedbackNotFoundError):
        code, message, status = "feedback_not_found", "反馈不存在", 404
    elif isinstance(error, FeedbackChangedError):
        code, message, status = "feedback_changed", "反馈已变化，请刷新后重试", 409
    return JSONResponse(status_code=status, headers={"Cache-Control": "no-store"}, content={
        "detail": message, "error": {"code": code, "message": message, "details": {}},
    })


async def _handle_trace_not_found(
    request: Request, error: TraceNotFoundError,
) -> JSONResponse:
    return JSONResponse(status_code=404, headers={"Cache-Control": "no-store"}, content={
        "detail": "追踪不存在", "error": {
            "code": "trace_not_found", "message": "追踪不存在", "details": {},
        },
    })


async def _handle_turn_not_found(
    request: Request, error: TurnNotFoundError,
) -> JSONResponse:
    return JSONResponse(status_code=404, headers={"Cache-Control": "no-store"}, content={
        "detail": "轮次不存在", "error": {
            "code": "turn_not_found", "message": "轮次不存在", "details": {},
        },
    })


async def _handle_trace_storage_error(
    request: Request, error: TraceStorageError,
) -> JSONResponse:
    del request
    del error
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "追踪存储暂时不可用"},
    )


async def _handle_conversation_not_found(
    request: Request,
    error: ConversationNotFoundError,
) -> JSONResponse:
    del request
    del error
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": "会话不存在"},
    )


async def _handle_context_too_large(
    request: Request,
    error: HistoryContextTooLargeError,
) -> JSONResponse:
    del request
    del error
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": "问题超过模型输入预算，请缩短问题或调整配置"},
    )


async def _handle_prompt_too_large(
    request: Request, error: PromptContextTooLargeError,
) -> JSONResponse:
    del request
    del error
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": "系统、历史与检索证据超过模型输入预算，请调整配置"},
    )


async def _handle_invalid_scope(
    request: Request, error: InvalidKnowledgeScopeError,
) -> JSONResponse:
    del request
    del error
    return JSONResponse(status_code=422, content={"detail": "指定的知识库不存在"})


async def _handle_catalog_storage_error(
    request: Request, error: KnowledgeBaseStorageError,
) -> JSONResponse:
    del request
    del error
    message = "知识库目录暂时不可用"
    return JSONResponse(status_code=503, headers={"Cache-Control": "no-store"}, content={
        "detail": message, "error": {
            "code": "knowledge_base_unavailable", "message": message, "details": {},
        },
    })


async def _handle_catalog_resource_error(request: Request, error: Exception) -> JSONResponse:
    code, message, status_code = "knowledge_base_not_found", "知识库不存在", 404
    if isinstance(error, KnowledgeBaseConflictError):
        code, message, status_code = "knowledge_base_exists", "知识库ID已存在", 409
    return JSONResponse(status_code=status_code, headers={"Cache-Control": "no-store"}, content={
        "detail": message, "error": {"code": code, "message": message, "details": {}},
    })


async def _handle_knowledge_delete_error(request: Request, error: Exception) -> JSONResponse:
    code, message, status_code = "knowledge_base_not_empty", "知识库仍有文档，请确认后使用force删除", 409
    details = {}
    if isinstance(error, KnowledgeBaseProtectedError):
        code, message = "knowledge_base_protected", "默认知识库不可删除，请按文档清理"
    elif isinstance(error, KnowledgeBaseIngestionBusyError):
        code, message = "knowledge_base_ingestion_in_progress", "知识库仍在入库，请等待任务结束后重试"
    elif isinstance(error, KnowledgeBaseDeletionInProgressError):
        code, message = "knowledge_base_deleting", "知识库删除尚未完成，请按原force请求重试"
    elif isinstance(error, KnowledgeBaseDeletePartialError):
        code, message, status_code = "knowledge_base_delete_partial", "知识库删除尚未完成，请按原请求重试", 500
        details = {"retryable": True, "repair_required": True}
    return JSONResponse(status_code=status_code, headers={"Cache-Control": "no-store"}, content={
        "detail": message, "error": {"code": code, "message": message, "details": details},
    })


async def _handle_content_resource_error(request: Request, error: Exception) -> JSONResponse:
    code, message, status_code = "knowledge_content_unavailable", "知识库内容暂时不可读", 503
    if isinstance(error, ChunkNotFoundError):
        code, message, status_code = "chunk_not_found", "来源文档与块组合不存在", 404
    elif isinstance(error, DocumentGoneError):
        code, message, status_code = "document_deleted", "来源文档已删除，请查看回答保存的引用片段", 410
    return JSONResponse(status_code=status_code, headers={"Cache-Control": "no-store"}, content={
        "detail": message, "error": {"code": code, "message": message, "details": {}},
    })


async def _handle_conversation_conflict(
    request: Request,
    error: ConversationBusyError | ConversationDeletingError,
) -> JSONResponse:
    del request
    detail = "该会话已有请求正在运行"
    if isinstance(error, ConversationDeletingError):
        detail = "会话正在删除"
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": detail},
    )


async def _handle_conversation_agent_error(
    request: Request,
    error: ConversationAgentError,
) -> JSONResponse:
    del request
    logger.error("会话 Agent 执行失败", exc_info=error)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": "会话生成服务暂时不可用"},
    )


async def _handle_conversation_timeout(
    request: Request,
    error: ConversationTimeoutError,
) -> JSONResponse:
    del request
    del error
    logger.warning("会话执行超时")
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"detail": "会话处理超时，请稍后重试"},
    )


async def _handle_conversation_storage_error(
    request: Request,
    error: ConversationStorageError,
) -> JSONResponse:
    del request
    logger.error("会话业务存储操作失败", exc_info=error)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "会话存储暂时不可用"},
    )


async def _handle_conversation_delete_error(
    request: Request,
    error: ConversationDeleteError,
) -> JSONResponse:
    del request
    logger.error("会话删除未完整完成", exc_info=error)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "会话删除未完成"},
    )


async def _handle_unsupported_file_type(
    request: Request,
    error: UnsupportedFileTypeError,
) -> JSONResponse:
    del request
    return JSONResponse(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        content={"detail": str(error)},
    )


async def _handle_file_too_large(
    request: Request,
    error: FileTooLargeError,
) -> JSONResponse:
    del request
    return JSONResponse(
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        content={"detail": str(error)},
    )


async def _handle_unprocessable_content(
    request: Request,
    error: (
        MissingFilenameError
        | EmptyDocumentError
        | DocumentParsingError
        | RetrievalError
    ),
) -> JSONResponse:
    del request
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": str(error)},
    )


async def _handle_document_not_found(
    request: Request,
    error: DocumentNotFoundError,
) -> JSONResponse:
    del request
    del error
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": "文档不存在"},
    )


async def _handle_document_revision_changed(
    request: Request, error: DocumentRevisionChangedError,
) -> JSONResponse:
    del request
    del error
    return JSONResponse(
        status_code=409,
        content={
            "detail": "来源已变化，请重新加载首页",
            "error": {
                "code": "document_changed",
                "message": "来源已变化，请重新加载首页",
                "details": {},
            },
        },
        headers={"Cache-Control": "no-store"},
    )


async def _handle_document_delete_partial(
    request: Request,
    error: DocumentDeletePartialError,
) -> JSONResponse:
    del request
    del error
    logger.error(
        "文档索引已删除但源文件清理失败",
        extra={"repair_required": True},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": {
                "code": "document_delete_partial",
                "message": "文档索引已删除，但原文件清理失败",
                "repair_required": True,
            }
        },
    )


async def _handle_document_storage_corruption(
    request: Request,
    error: DocumentStorageCorruptionError,
) -> JSONResponse:
    del request
    logger.error("文档存储状态异常", exc_info=error)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "文档存储状态异常"},
    )


async def _handle_llm_timeout(
    request: Request,
    error: LLMTimeoutError,
) -> JSONResponse:
    del request
    del error
    logger.warning("语言模型服务响应超时")
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"detail": "语言模型服务响应超时"},
    )


async def _handle_llm_error(
    request: Request,
    error: LLMError,
) -> JSONResponse:
    del request
    logger.error("语言模型服务调用失败", exc_info=error)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": "语言模型服务暂时不可用"},
    )


async def _handle_reranker_unavailable(
    request: Request,
    error: RerankerUnavailableError,
) -> JSONResponse:
    del request
    logger.error("重排服务调用失败", exc_info=error)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": "重排服务暂时不可用"},
    )


async def _handle_retrieval_backend_error(
    request: Request,
    error: EmbeddingError | VectorStoreError,
) -> JSONResponse:
    del request
    logger.error("检索后端调用失败", exc_info=error)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": "检索服务暂时不可用"},
    )


def register_error_handlers(application: FastAPI) -> None:
    """为应用注册稳定的领域与应用异常映射。

    Args:
        application: 需要注册异常处理器的 FastAPI 应用。

    Returns:
        None。
    """
    register_auth_error_handlers(application)
    for error_type in (
        RefusalClustersBusyError, RefusalClustersChangedError,
        RefusalClustersTimeoutError,
    ):
        application.add_exception_handler(error_type, _handle_refusal_cluster_error)
    for error_type in (
        EvaluationReportNotFoundError, EvaluationReportStorageError,
    ):
        application.add_exception_handler(error_type, _handle_evaluation_report_error)
    for error_type in (ShareNotFoundError, ShareExpiredError, ShareStorageError):
        application.add_exception_handler(error_type, _handle_share_error)
    for error_type in (
        KnowledgeBaseNotEmptyError, KnowledgeBaseProtectedError,
        KnowledgeBaseIngestionBusyError, KnowledgeBaseDeletionInProgressError,
        KnowledgeBaseDeletePartialError,
    ):
        application.add_exception_handler(error_type, _handle_knowledge_delete_error)
    application.add_exception_handler(
        DocumentRevisionChangedError, _handle_document_revision_changed,
    )
    application.add_exception_handler(
        DocumentIngestionInProgressError, _handle_document_ingestion_busy,
    )
    for error_type in (
        QuestionRateLimitError, QuestionQueueFullError, QuestionQueueTimeoutError,
    ):
        application.add_exception_handler(error_type, _handle_question_admission_error)
    for error_type in (
        IngestionJobStorageError, IngestionJobNotFoundError,
        IngestionJobConflictError, IngestionQueueFullError,
        IngestionSourceError, IngestionWorkerUnavailableError,
    ):
        application.add_exception_handler(error_type, _handle_ingestion_job_error)
    for error in (FeedbackNotFoundError, FeedbackChangedError, InsightStorageError):
        application.add_exception_handler(error, _handle_feedback_admin_error)
    application.add_exception_handler(TraceNotFoundError, _handle_trace_not_found)
    application.add_exception_handler(TurnNotFoundError, _handle_turn_not_found)
    application.add_exception_handler(
        EmbeddingError,
        _handle_retrieval_backend_error,
    )
    application.add_exception_handler(
        HistoryContextTooLargeError,
        _handle_context_too_large,
    )
    application.add_exception_handler(
        PromptContextTooLargeError,
        _handle_prompt_too_large,
    )
    application.add_exception_handler(InvalidKnowledgeScopeError, _handle_invalid_scope)
    application.add_exception_handler(KnowledgeBaseStorageError, _handle_catalog_storage_error)
    application.add_exception_handler(KnowledgeBaseNotFoundError, _handle_catalog_resource_error)
    application.add_exception_handler(KnowledgeBaseConflictError, _handle_catalog_resource_error)
    application.add_exception_handler(KnowledgeContentError, _handle_content_resource_error)
    application.add_exception_handler(ChunkNotFoundError, _handle_content_resource_error)
    application.add_exception_handler(DocumentGoneError, _handle_content_resource_error)
    application.add_exception_handler(TraceStorageError, _handle_trace_storage_error)
    application.add_exception_handler(
        VectorStoreError,
        _handle_retrieval_backend_error,
    )
    application.add_exception_handler(
        UnsupportedFileTypeError,
        _handle_unsupported_file_type,
    )
    application.add_exception_handler(
        FileTooLargeError,
        _handle_file_too_large,
    )
    application.add_exception_handler(
        MissingFilenameError,
        _handle_unprocessable_content,
    )
    application.add_exception_handler(
        EmptyDocumentError,
        _handle_unprocessable_content,
    )
    application.add_exception_handler(
        DocumentParsingError,
        _handle_unprocessable_content,
    )
    application.add_exception_handler(
        RetrievalError,
        _handle_unprocessable_content,
    )
    application.add_exception_handler(
        DocumentNotFoundError,
        _handle_document_not_found,
    )
    application.add_exception_handler(
        DocumentDeletePartialError,
        _handle_document_delete_partial,
    )
    application.add_exception_handler(
        DocumentStorageCorruptionError,
        _handle_document_storage_corruption,
    )
    application.add_exception_handler(
        LLMTimeoutError,
        _handle_llm_timeout,
    )
    application.add_exception_handler(
        LLMError,
        _handle_llm_error,
    )
    application.add_exception_handler(
        RerankerUnavailableError,
        _handle_reranker_unavailable,
    )
    application.add_exception_handler(
        ConversationNotFoundError,
        _handle_conversation_not_found,
    )
    application.add_exception_handler(
        ConversationBusyError,
        _handle_conversation_conflict,
    )
    application.add_exception_handler(
        ConversationDeletingError,
        _handle_conversation_conflict,
    )
    application.add_exception_handler(
        ConversationAgentError,
        _handle_conversation_agent_error,
    )
    application.add_exception_handler(
        ConversationTimeoutError,
        _handle_conversation_timeout,
    )
    application.add_exception_handler(
        ConversationStorageError,
        _handle_conversation_storage_error,
    )
    application.add_exception_handler(
        ConversationDeleteError,
        _handle_conversation_delete_error,
    )
    install_error_contract(application, {
        TraceStorageError: "trace_unavailable",
        ConversationNotFoundError: "conversation_not_found",
        HistoryContextTooLargeError: "history_context_too_large",
        PromptContextTooLargeError: "prompt_context_too_large",
        InvalidKnowledgeScopeError: "invalid_knowledge_scope",
        ConversationBusyError: "conversation_busy",
        ConversationDeletingError: "conversation_deleting",
        ConversationAgentError: "conversation_generation_unavailable",
        ConversationTimeoutError: "conversation_timeout",
        ConversationStorageError: "conversation_storage_unavailable",
        ConversationDeleteError: "conversation_delete_failed",
        UnsupportedFileTypeError: "unsupported_file_type",
        FileTooLargeError: "file_too_large",
        MissingFilenameError: "missing_filename",
        EmptyDocumentError: "empty_document",
        DocumentParsingError: "document_parsing_failed",
        RetrievalError: "invalid_retrieval_request",
        DocumentNotFoundError: "document_not_found",
        DocumentDeletePartialError: "document_delete_partial",
        DocumentStorageCorruptionError: "document_storage_corruption",
        LLMTimeoutError: "generation_timeout",
        LLMError: "generation_unavailable",
        RerankerUnavailableError: "reranker_unavailable",
        EmbeddingError: "retrieval_unavailable",
        VectorStoreError: "retrieval_unavailable",
    })
