# -*- coding: utf-8 -*-
"""旧HTTP契约测试显式注入身份；认证安全测试始终使用生产依赖。"""

from typing import Any
from uuid import UUID

from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.domain.auth import User
from app.interfaces.http.auth import require_user
from app.main import create_app


TEST_USER_ID = UUID("a042bb72-44bb-4664-a72f-14a1ae23ebbb")


def allow_test_identity(application: FastAPI) -> FastAPI:
    """只在测试进程覆盖身份，不给生产服务增加免登录开关。

    Args:
        application: 尚未处理请求的测试应用。

    Returns:
        注入固定身份和测试Host的原应用。
    """
    application.dependency_overrides[require_user] = lambda: User(
        id=TEST_USER_ID, name="离线契约测试", role="admin",
    )
    for middleware in application.user_middleware:
        if middleware.cls is TrustedHostMiddleware:
            middleware.kwargs["allowed_hosts"] = ["test"]
    return application


def create_test_app(**kwargs: Any) -> FastAPI:
    """为关注旧字段和错误映射的测试创建应用。

    Args:
        kwargs: 原生产应用工厂参数。

    Returns:
        显式注入离线身份的真实应用。
    """
    return allow_test_identity(create_app(**kwargs))
