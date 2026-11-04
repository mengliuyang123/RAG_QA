# -*- coding: utf-8 -*-
"""普通与 SSE 知识库查询路由。"""

import json
import logging
from collections.abc import AsyncIterator
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from fastapi.responses import EventSourceResponse
from fastapi.sse import ServerSentEvent

from app.application.errors import EmbeddingError
from app.application.errors import LLMError
from app.application.errors import RerankerUnavailableError
from app.application.errors import VectorStoreError
from app.application.query.events import StreamEvent
from app.application.query.routed_service import RoutedQueryService
from app.application.errors import ConversationAgentError
from app.application.errors import KnowledgeBaseStorageError
from app.application.errors import TraceStorageError
from app.application.errors import QuestionRateLimitError
from app.application.errors import QuestionQueueFullError
from app.application.errors import QuestionQueueTimeoutError
from app.domain.errors import InvalidKnowledgeScopeError
from app.domain.errors import RetrievalError
from app.domain.errors import PromptContextTooLargeError
from app.domain.auth import User
from app.interfaces.http.auth import require_admin
from app.interfaces.http.dependencies import get_query_service
from app.interfaces.http.schemas import QueryRequest
from app.interfaces.http.schemas import QueryResponse
from app.interfaces.http.schemas import flatten_answer_payload
from app.interfaces.http.streaming import PreparedEventStream
from app.interfaces.http.execution import run_cancellable_query


router = APIRouter()
logger = logging.getLogger(__name__)


def get_stream_query_iterator(
    request: QueryRequest,
    user: Annotated[User, Depends(require_admin)],
    query_service: Annotated[
        RoutedQueryService,
        Depends(get_query_service),
    ],
) -> Iterator[PreparedEventStream[StreamEvent]]:
    """创建由 FastAPI 请求退出栈负责关闭的流式查询迭代器。

    Args:
        request: 已通过 Pydantic 校验的查询请求。
        query_service: 生产或测试注入的共享查询服务。
        user: 服务器验证的管理员身份。

    Yields:
        当前请求的应用事件迭代器；响应结束后关闭。
    """
    stream_iterator = query_service.stream_query(
        question=request.question,
        strategy=request.strategy,
        top_k=request.top_k,
        user_id=user.id,
        **_execution_options(request),
    )
    try:
        first = next(stream_iterator)

        yield PreparedEventStream(first, stream_iterator)
    finally:
        close = getattr(stream_iterator, "close", None)
        if callable(close):
            close()


@router.post(
    "/query",
    response_model=QueryResponse,
    description=(
        "执行非流式知识库问答，"
        "并把领域异常映射为稳定 HTTP 响应。"
    ),
)
async def query_knowledge_base(
    request: QueryRequest,
    transport_request: Request,
    user: Annotated[User, Depends(require_admin)],
    query_service: Annotated[
        RoutedQueryService,
        Depends(get_query_service),
    ],
) -> QueryResponse:
    """执行非流式知识库问答并映射 HTTP 响应。

    Args:
        request: 已通过 Pydantic 校验的查询请求。
        query_service: 生产或测试注入的共享查询服务。
        user: 服务器验证的管理员身份。
        transport_request: 用于监听普通HTTP断连的传输请求。

    Returns:
        保持稳定字段结构的查询响应。

    Raises:
        HTTPException: 查询发生未知内部错误。
        LLMError: 交由集中错误处理器映射的模型错误。
        RerankerUnavailableError: 交由集中错误处理器映射的
            重排错误。
        RetrievalError: 交由集中错误处理器映射的检索错误。
    """
    try:
        result = await run_cancellable_query(
            transport_request,
            lambda: query_service.query(
                question=request.question,
                strategy=request.strategy,
                top_k=request.top_k,
                user_id=user.id,
                **_execution_options(request),
            ),
        )
        return QueryResponse.model_validate(flatten_answer_payload(result))
    except (
        HTTPException,
        EmbeddingError,
        LLMError,
        RerankerUnavailableError,
        RetrievalError,
        PromptContextTooLargeError,
        ConversationAgentError,
        InvalidKnowledgeScopeError,
        KnowledgeBaseStorageError,
        TraceStorageError,
        VectorStoreError,
        QuestionRateLimitError,
        QuestionQueueFullError,
        QuestionQueueTimeoutError,
    ):
        raise
    except Exception as error:
        logger.exception("知识库查询处理失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="知识库查询处理失败",
        ) from error


@router.post(
    "/query/stream",
    response_class=EventSourceResponse,
    description=(
        "把流式 RAG 领域事件编码为原生 SSE。\n\n"
        "Args:\n"
        "    stream_iterator: "
        "请求退出栈持有的领域事件迭代器。\n\n"
        "Yields:\n"
        "    带事件名和 JSON payload 的原生 SSE 事件。"
    ),
)
async def stream_query_knowledge_base(
    stream_iterator: Annotated[
        PreparedEventStream[StreamEvent],
        Depends(get_stream_query_iterator, scope="request"),
    ],
) -> AsyncIterator[ServerSentEvent]:
    """把流式查询应用事件编码为原生 SSE。

    Args:
        stream_iterator: 请求退出栈持有的应用事件迭代器。

    Yields:
        带事件名和紧凑 JSON payload 的原生 SSE 事件。
    """
    async for stream_event in stream_iterator.events():
        if stream_event.event == "complete":
            payload = QueryResponse.model_validate(
                flatten_answer_payload(stream_event.data),
            ).model_dump(mode="json")
        else:
            payload = stream_event.data.model_dump(mode="json")
        raw_data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        yield ServerSentEvent(
            event=stream_event.event,
            raw_data=raw_data,
        )


def _execution_options(request: QueryRequest) -> dict[str, object]:
    options = {}
    if request.kb_scope is not None:
        options["kb_scope"] = request.kb_scope
    if request.lane != "auto":
        options["lane"] = request.lane
    if request.debug:
        options["debug"] = True
    return options
