# -*- coding: utf-8 -*-
"""HTTP错误统一追加字段，同时保护旧detail、状态与响应头。"""

import httpx
import pytest
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from app.application.errors import ConversationBusyError
from app.application.errors import ConversationNotFoundError
from app.application.errors import ConversationTimeoutError
from app.application.errors import DocumentDeletePartialError
from app.application.errors import EmbeddingError
from app.application.errors import QuestionRateLimitError
from app.interfaces.http.error_handlers import register_error_handlers
from app.interfaces.http.error_contract import install_error_contract


@pytest.mark.anyio
@pytest.mark.parametrize("error,status_code,code,detail", [
    (ConversationNotFoundError("private"), 404, "conversation_not_found", "会话不存在"),
    (ConversationBusyError("private"), 409, "conversation_busy", "该会话已有请求正在运行"),
    (ConversationTimeoutError("private"), 504, "conversation_timeout", "会话处理超时，请稍后重试"),
    (EmbeddingError("private"), 502, "retrieval_unavailable", "检索服务暂时不可用"),
    (DocumentDeletePartialError(), 500, "document_delete_partial", {
        "code": "document_delete_partial", "message": "文档索引已删除，但原文件清理失败",
        "repair_required": True,
    }),
])
async def test_business_errors_keep_legacy_payload_and_add_codes(
    error: Exception, status_code: int, code: str, detail: str | dict,
) -> None:
    """缺少兼容字段、状态变化或泄漏上游异常均会失败。

    Args:
        error: 实际注册的业务异常。
        status_code: 旧公开状态。
        code: 稳定机器码。
        detail: 旧detail精确投影。
    """
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/failure")
    def fail() -> None:
        """触发真实异常映射。"""
        raise error

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        response = await client.get("/failure")
    body = response.json()
    assert response.status_code == status_code
    assert body["detail"] == detail
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str)
    assert isinstance(body["error"]["details"], dict)
    assert "private" not in response.text
    if isinstance(detail, dict):
        assert body["error"]["details"] == {"repair_required": True}


class Input(BaseModel):
    """真实请求校验输入。

    Attributes:
        count: 要求整数的输入字段。
    """

    count: int


@pytest.mark.anyio
async def test_default_errors_and_existing_contract_keep_headers() -> None:
    """默认错误也要统一；已有码和Retry-After不能被适配器覆盖。"""
    app = FastAPI()
    register_error_handlers(app)

    @app.post("/input")
    def accept(body: Input) -> dict[str, int]:
        """接收正常请求。

        Args:
            body: 校验后的输入。

        Returns:
            正常响应。
        """
        return {"count": body.count}

    @app.get("/limited")
    def limited() -> None:
        """触发已有结构化限流响应。"""
        raise QuestionRateLimitError(23)

    @app.get("/http")
    def http_error() -> None:
        """标准HTTP异常保留自定义响应头。"""
        raise HTTPException(429, "稍后重试", headers={"Retry-After": "7"})

    @app.get("/unknown")
    def unknown() -> None:
        """不允许未知错误正文泄漏。"""
        raise RuntimeError("secret_server_path")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        invalid = await client.post("/input", json={"count": "sensitive_input"})
        assert invalid.status_code == 422
        assert invalid.json()["detail"][0]["loc"] == ["body", "count"]
        assert invalid.json()["error"]["code"] == "validation_error"
        assert "sensitive_input" not in str(invalid.json()["error"])
        missing = await client.get("/absent")
        assert missing.status_code == 404
        assert missing.json()["detail"] == "Not Found"
        assert missing.json()["error"]["code"] == "not_found"
        method = await client.get("/input")
        assert method.status_code == 405
        assert method.headers["allow"] == "POST"
        assert method.json()["error"]["code"] == "method_not_allowed"
        limited_response = await client.get("/limited")
        assert limited_response.headers["retry-after"] == "23"
        assert limited_response.json()["error"] == {
            "code": "rate_limited", "message": "提问过于频繁，请稍后重试",
            "details": {"retry_after": 23},
        }
        retry = await client.get("/http")
        assert retry.headers["retry-after"] == "7"
        assert retry.json()["detail"] == "稍后重试"
        assert retry.json()["error"]["code"] == "rate_limited"
        unknown_response = await client.get("/unknown")
        assert unknown_response.status_code == 500
        assert unknown_response.json()["error"]["code"] == "internal_error"
        assert "secret_server_path" not in unknown_response.text
        success = await client.post("/input", json={"count": 1})
        assert success.json() == {"count": 1}


@pytest.mark.anyio
async def test_error_adaptation_keeps_duplicate_headers_and_bodyless_status() -> None:
    """复制响应时不得折叠Set-Cookie、丢失后台任务或给204加正文。"""
    app = FastAPI()
    completed = []

    class CustomError(Exception):
        """具有自定义响应头的应用异常。"""

    async def handle(request: Request, error: CustomError) -> JSONResponse:
        response = JSONResponse(
            {"detail": "conflict"}, status_code=409,
            background=BackgroundTask(completed.append, "finished"),
        )
        response.set_cookie("first", "one")
        response.set_cookie("second", "two")
        return response

    app.add_exception_handler(CustomError, handle)
    install_error_contract(app, {CustomError: "custom_conflict"})

    @app.get("/conflict")
    def conflict() -> None:
        """返回自定义错误。"""
        raise CustomError()

    @app.get("/empty")
    def empty() -> None:
        """保持禁止正文的HTTP状态。"""
        raise HTTPException(204, headers={"X-Request-State": "empty"})

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        response = await client.get("/conflict")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "custom_conflict"
        assert len(response.headers.get_list("set-cookie")) == 2
        assert response.cookies["first"] == "one"
        assert response.cookies["second"] == "two"
        assert int(response.headers["content-length"]) == len(response.content)
        assert completed == ["finished"]
        no_content = await client.get("/empty")
        assert no_content.status_code == 204
        assert no_content.content == b""
        assert no_content.headers["x-request-state"] == "empty"
