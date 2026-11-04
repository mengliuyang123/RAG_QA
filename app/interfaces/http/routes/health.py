# -*- coding: utf-8 -*-
"""应用基础健康检查路由。"""

from fastapi import APIRouter


router = APIRouter()


@router.get(
    "/health",
    description="返回应用进程的基础健康状态。",
)
def health_check() -> dict[str, str]:
    """返回应用进程的基础健康状态。

    Returns:
        固定的健康状态响应。
    """
    return {"status": "ok"}
