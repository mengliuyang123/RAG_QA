# -*- coding: utf-8 -*-
"""知识库级权限：受限库对非成员视同不存在；只有 editor 和管理员可以管理库内容。"""

from typing import Annotated

from fastapi import Depends
from fastapi import Request

from app.application.errors import ForbiddenError
from app.application.errors import KnowledgeBaseNotFoundError
from app.domain.auth import User
from app.domain.knowledge_access import KbPermissions
from app.interfaces.http.auth import require_user
from app.interfaces.http.container_runtime import current_container


def kb_permissions(
    request: Request, user: Annotated[User, Depends(require_user)],
) -> KbPermissions:
    """当前用户的知识库权限，每个请求从数据库重新计算。

    Args:
        request: 当前请求。
        user: 已验证身份。

    Returns:
        有效权限。
    """
    return current_container(request).knowledge_access_service.permissions(user)


def ensure_visible(permissions: KbPermissions, kb_id: str) -> None:
    """不可见的知识库返回404，不泄露其存在。"""
    if not permissions.can_view(kb_id):
        raise KnowledgeBaseNotFoundError("知识库不存在")


def ensure_editor(permissions: KbPermissions, kb_id: str) -> None:
    """管理库内容需要 editor 或管理员。"""
    if not permissions.can_edit(kb_id):
        raise ForbiddenError("没有该知识库的管理权限")


def ensure_unrestricted(permissions: KbPermissions) -> None:
    """跨知识库的操作只允许管理员。"""
    if not permissions.unrestricted:
        raise ForbiddenError("此操作仅限管理员")
