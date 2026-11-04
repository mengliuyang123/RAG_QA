# -*- coding: utf-8 -*-
"""多轮 Agentic RAG 会话 CRUD、查询与安全 SSE 路由。"""

import json
from collections.abc import AsyncIterator
from collections.abc import Iterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi import Response
from fastapi import status
from fastapi.responses import EventSourceResponse
from fastapi.sse import ServerSentEvent

from app.application.conversations.events import ConversationStreamEvent
from app.application.conversations.service import ConversationService
from app.domain.conversations import Conversation
from app.domain.conversations import ConversationQueryResult
from app.domain.conversations import ConversationSummary
from app.domain.auth import User
from app.domain.errors import InvalidConversationCursorError
from app.interfaces.http.auth import require_user
from app.interfaces.http.dependencies import get_conversation_service
from app.interfaces.http.schemas import ConversationCreateResponse
from app.interfaces.http.schemas import ConversationCreateRequest
from app.interfaces.http.schemas import ConversationPatchRequest
from app.interfaces.http.schemas import ConversationDeleteResponse
from app.interfaces.http.schemas import ConversationDetailResponse
from app.interfaces.http.schemas import ConversationListResponse
from app.interfaces.http.schemas import ConversationQueryRequest
from app.interfaces.http.schemas import ConversationQueryResponse
from app.interfaces.http.schemas import ConversationSummaryResponse
from app.interfaces.http.schemas import ConversationTurnResponse
from app.interfaces.http.schemas import flatten_answer_payload
from app.interfaces.http.streaming import PreparedEventStream
from app.interfaces.http.execution import run_cancellable_query


router = APIRouter(prefix="/conversations", tags=["conversations"])


def _summary_response(
    conversation: Conversation | ConversationSummary,
) -> ConversationSummaryResponse:
    return ConversationSummaryResponse.model_validate(
        conversation.model_dump()
    )


def _detail_response(
    conversation: Conversation, next_cursor: str | None = None,
) -> ConversationDetailResponse:
    turns = []
    for turn in conversation.turns:
        payload = flatten_answer_payload(turn)
        payload.pop("conversation_id", None)
        turns.append(ConversationTurnResponse.model_validate(payload))
    return ConversationDetailResponse(
        **_summary_response(conversation).model_dump(),
        turns=turns,
        turns_next_cursor=next_cursor,
    )


def _query_response(
    result: ConversationQueryResult,
) -> ConversationQueryResponse:
    payload = flatten_answer_payload(result)
    return ConversationQueryResponse.model_validate(payload)


def get_conversation_stream_iterator(
    conversation_id: UUID,
    request: ConversationQueryRequest,
    user: Annotated[User, Depends(require_user)],
    conversation_service: Annotated[
        ConversationService,
        Depends(get_conversation_service),
    ],
) -> Iterator[PreparedEventStream[ConversationStreamEvent]]:
    """预取首事件，在响应头发送前检查归属和忙碌状态。

    Args:
        conversation_id: 待查询会话。
        request: 当前问题与可选范围。
        user: 已验证身份。
        conversation_service: 组合根提供的服务。

    Yields:
        已预取事件的安全迭代器。
    """
    stream_iterator = conversation_service.stream_query(
        conversation_id,
        request.question,
        owner_id=user.id,
        **({"kb_scope": request.kb_scope} if request.kb_scope is not None else {}),
    )
    try:
        first_event = next(stream_iterator)

        yield PreparedEventStream(first_event, stream_iterator)
    finally:
        close = getattr(stream_iterator, "close", None)
        if callable(close):
            close()


@router.post(
    "",
    response_model=ConversationCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    user: Annotated[User, Depends(require_user)],
    conversation_service: Annotated[
        ConversationService,
        Depends(get_conversation_service),
    ],
    request: ConversationCreateRequest | None = None,
) -> ConversationCreateResponse:
    """创建会话并保存可选默认范围。

    Args:
        conversation_service: 组合根提供的会话服务。
        request: 可选的初始设置，未提供时保持旧行为。
        user: 已验证的创建者。

    Returns:
        新会话摘要。
    """
    if request is None:
        conversation = conversation_service.create(owner_id=user.id)
    else:
        conversation = conversation_service.create(
            kb_scope=request.kb_scope, owner_id=user.id,
        )
    return ConversationCreateResponse.model_validate(
        _summary_response(conversation).model_dump()
    )


@router.patch("/{conversation_id}", response_model=ConversationSummaryResponse)
def update_conversation(
    conversation_id: UUID,
    request: ConversationPatchRequest,
    user: Annotated[User, Depends(require_user)],
    conversation_service: Annotated[
        ConversationService,
        Depends(get_conversation_service),
    ],
) -> ConversationSummaryResponse:
    """修改标题及默认范围，不改变已有轮次。

    Args:
        conversation_id: 目标会话。
        request: 需要修改的设置。
        conversation_service: 组合根提供的会话服务。
        user: 当前用户，不赋予管理员读取他人会话的例外。

    Returns:
        更新后的会话摘要。
    """
    return _summary_response(conversation_service.update(
        conversation_id, title=request.title, kb_scope=request.kb_scope,
        owner_id=user.id,
    ))


@router.get("", response_model=ConversationListResponse)
def list_conversations(
    response: Response,
    user: Annotated[User, Depends(require_user)],
    conversation_service: Annotated[
        ConversationService,
        Depends(get_conversation_service),
    ],
    limit: Annotated[int | None, Query(ge=1, le=100)] = None,
    cursor: Annotated[str | None, Query(min_length=1, max_length=2048)] = None,
) -> ConversationListResponse:
    """只列出当前用户会话。

    Args:
        user: 已验证身份。
        conversation_service: 组合根提供的服务。
        response: 禁止缓存私人会话。
        limit: 显式分页大小；未传分页参数时保留旧全量响应。
        cursor: 后续页位置，单独传入时每页20条。

    Returns:
        当前用户会话列表与数量。
    """
    response.headers["Cache-Control"] = "no-store"
    if limit is not None or cursor is not None:
        try:
            page = conversation_service.list_page(
                owner_id=user.id, limit=limit or 20, cursor=cursor,
            )
        except InvalidConversationCursorError as error:
            raise HTTPException(422, "会话分页参数无效") from error
        return ConversationListResponse(
            conversations=[_summary_response(item) for item in page.conversations],
            total=page.total, next_cursor=page.next_cursor,
        )
    conversations = [
        _summary_response(item)
        for item in conversation_service.list(owner_id=user.id)
    ]
    return ConversationListResponse(
        conversations=conversations,
        total=len(conversations),
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
)
def get_conversation(
    conversation_id: UUID,
    response: Response,
    user: Annotated[User, Depends(require_user)],
    conversation_service: Annotated[
        ConversationService,
        Depends(get_conversation_service),
    ],
    turns_limit: Annotated[int | None, Query(ge=1, le=100)] = None,
    turns_cursor: Annotated[
        str | None, Query(min_length=1, max_length=2048),
    ] = None,
) -> ConversationDetailResponse:
    """在后端检查归属后返回正式历史。

    Args:
        conversation_id: 目标标识。
        user: 已验证身份。
        conversation_service: 组合根提供的服务。
        response: 禁止缓存私人历史。
        turns_limit: 当前页轮数；不传分页参数时保持旧全量读取。
        turns_cursor: 更早页的位置，单独传入默认20轮。

    Returns:
        本人会话详情。
    """
    response.headers["Cache-Control"] = "no-store"
    if turns_limit is not None or turns_cursor is not None:
        try:
            page = conversation_service.get_page(
                conversation_id, owner_id=user.id, limit=turns_limit or 20,
                cursor=turns_cursor,
            )
        except InvalidConversationCursorError as error:
            raise HTTPException(422, "会话分页参数无效") from error
        return _detail_response(page, page.turns_next_cursor)
    return _detail_response(conversation_service.get(
        conversation_id, owner_id=user.id,
    ))


@router.delete(
    "/{conversation_id}",
    response_model=ConversationDeleteResponse,
)
def delete_conversation(
    conversation_id: UUID,
    user: Annotated[User, Depends(require_user)],
    conversation_service: Annotated[
        ConversationService,
        Depends(get_conversation_service),
    ],
) -> ConversationDeleteResponse:
    """只删除本人的会话与checkpoint。

    Args:
        conversation_id: 目标标识。
        user: 已验证身份。
        conversation_service: 组合根提供的服务。

    Returns:
        成功删除标记。
    """
    conversation_service.delete(conversation_id, owner_id=user.id)
    return ConversationDeleteResponse(
        conversation_id=conversation_id,
        deleted=True,
    )


@router.post(
    "/{conversation_id}/query",
    response_model=ConversationQueryResponse,
)
async def query_conversation(
    conversation_id: UUID,
    request: ConversationQueryRequest,
    transport_request: Request,
    user: Annotated[User, Depends(require_user)],
    conversation_service: Annotated[
        ConversationService,
        Depends(get_conversation_service),
    ],
) -> ConversationQueryResponse:
    """为已验证用户执行一轮安全会话问答。

    Args:
        conversation_id: 目标标识。
        request: 原始问题与可选范围。
        transport_request: 普通HTTP断连监听，不改变会话请求体。
        user: 已验证身份。
        conversation_service: 组合根提供的服务。

    Returns:
        已审核且已提交的回答。
    """
    result = await run_cancellable_query(
        transport_request,
        lambda: conversation_service.query(
            conversation_id, request.question,
            owner_id=user.id,
            **({"kb_scope": request.kb_scope} if request.kb_scope is not None else {}),
        ),
    )
    return _query_response(result)


@router.post(
    "/{conversation_id}/query/stream",
    response_class=EventSourceResponse,
)
async def stream_conversation(
    stream_iterator: Annotated[
        PreparedEventStream[ConversationStreamEvent],
        Depends(get_conversation_stream_iterator, scope="request"),
    ],
) -> AsyncIterator[ServerSentEvent]:
    """输出已校验的本人会话事件。

    Args:
        stream_iterator: 响应头前已检查归属的事件流。

    Yields:
        阶段、已校验正文或唯一终止事件。
    """
    async for event in stream_iterator.events():
        if event.event == "complete":
            payload = _query_response(event.data).model_dump(mode="json")
        else:
            payload = event.data.model_dump(mode="json")
        yield ServerSentEvent(
            event=event.event,
            raw_data=json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
