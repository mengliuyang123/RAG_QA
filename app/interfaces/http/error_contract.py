# -*- coding: utf-8 -*-
"""只适配异常响应，保留旧协议且不缓冲正常或SSE响应。"""

import json
from collections.abc import Awaitable
from collections.abc import Callable
from functools import wraps

from fastapi import FastAPI
from fastapi import Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.responses import Response


ErrorHandler = Callable[[Request, Exception], Awaitable[Response]]
_STATUS_CODES = {
    400: "bad_request", 401: "unauthenticated", 403: "forbidden",
    404: "not_found", 405: "method_not_allowed", 409: "conflict",
    410: "gone", 413: "payload_too_large", 415: "unsupported_media_type",
    422: "validation_error", 429: "rate_limited", 500: "internal_error",
    502: "upstream_unavailable", 503: "service_unavailable", 504: "timeout",
}


def _extend_response(response: Response, code: str | None) -> Response:
    if not isinstance(response, JSONResponse):
        return response
    payload = json.loads(response.body)
    if not isinstance(payload, dict) or "error" in payload:
        return response
    detail = payload.get("detail")
    details = {}
    message = "请求处理失败"
    resolved_code = code or _STATUS_CODES.get(response.status_code, "http_error")
    if isinstance(detail, str):
        message = detail
    elif isinstance(detail, dict):
        if isinstance(detail.get("message"), str):
            message = detail["message"]
        if code is None and isinstance(detail.get("code"), str):
            resolved_code = detail["code"]
        for key, value in detail.items():
            if key not in ("code", "message"):
                details[key] = value
    elif isinstance(detail, list):
        message = "请求参数无效"
        fields = []
        for field in detail:
            if isinstance(field, dict):
                # 旧detail照常保留，新增摘要不复制input、ctx或自定义消息。
                fields.append({
                    "loc": field.get("loc", []),
                    "type": field.get("type", "invalid"),
                })
        details["fields"] = fields
    payload["error"] = {
        "code": resolved_code, "message": message, "details": details,
    }
    result = JSONResponse(
        payload, status_code=response.status_code,
        background=response.background,
    )
    # 不能用dict折叠重复头；只重新计算追加正文后的长度。
    result.raw_headers = [
        (key, value) for key, value in response.raw_headers
        if key.lower() != b"content-length"
    ]
    result.headers["content-length"] = str(len(result.body))
    if "cache-control" not in result.headers:
        result.headers["cache-control"] = "no-store"
    return result


def _compatible(handler: ErrorHandler, code: str | None) -> ErrorHandler:
    @wraps(handler)
    async def adapted(request: Request, error: Exception) -> Response:
        response = await handler(request, error)
        return _extend_response(response, code)

    return adapted


async def _unexpected(request: Request, error: Exception) -> JSONResponse:
    return JSONResponse(
        {"detail": "Internal Server Error"}, status_code=500,
    )


def install_error_contract(
    application: FastAPI, legacy_codes: dict[type[Exception], str],
) -> None:
    """在既有业务映射之后追加协议，不改旧处理器的异常选择。

    Args:
        application: 已注册业务错误处理器的应用。
        legacy_codes: 旧异常对应的稳定机器码。

    Returns:
        None。
    """
    for error_type, code in legacy_codes.items():
        handler = application.exception_handlers[error_type]
        application.add_exception_handler(error_type, _compatible(handler, code))
    application.add_exception_handler(
        HTTPException, _compatible(http_exception_handler, None),
    )
    application.add_exception_handler(
        RequestValidationError,
        _compatible(request_validation_exception_handler, "validation_error"),
    )
    application.add_exception_handler(
        Exception, _compatible(_unexpected, "internal_error"),
    )
