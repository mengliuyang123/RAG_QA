# -*- coding: utf-8 -*-
"""目录卡片与管理员元数据写接口，具体存储只由组合根提供。"""

from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Path
from fastapi import Request
from fastapi import Response
from fastapi import Query
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from app.application.knowledge.catalog import KnowledgeCatalogService
from app.application.knowledge.deletion import KnowledgeBaseDeletionService
from app.domain.auth import User
from app.domain.knowledge_deletion import KnowledgeBaseDeleteResult
from app.interfaces.http.auth import require_admin
from app.domain.knowledge_bases import KB_ID_PATTERN
from app.domain.knowledge_bases import KnowledgeBase
from app.domain.knowledge_bases import KnowledgeBasePatch
from app.domain.knowledge_views import AdminKnowledgeBaseCard
from app.domain.knowledge_views import KnowledgeBaseList


read_router = APIRouter(tags=["knowledge-bases"])
admin_router = APIRouter(prefix="/admin/knowledge-bases", tags=["knowledge-bases"])


class KnowledgeBaseCreateRequest(BaseModel):
    """创建请求不接受操作者、时间或文档计数等服务端字段。

    Attributes:
        id: 稳定知识库ID。
        name: 可见名称。
        description: 自动路由读取的范围说明。
        icon: 前端图标名。
        owner_name: 管理员维护的负责人展示名。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str = Field(pattern=KB_ID_PATTERN)
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=4000)
    icon: str | None = Field(default=None, max_length=64)
    owner_name: str | None = Field(default=None, max_length=200)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """共用领域可见名称规则，不让空白名称在用例内部才失败。

        Args:
            value: 严格校验后的名称。

        Returns:
            非空白规范名称。
        """
        return KnowledgeBase.require_visible_name(value)


def get_knowledge_catalog(request: Request) -> KnowledgeCatalogService:
    """解析当前应用的共享元数据服务。

    Args:
        request: 当前已通过路由权限检查的请求。

    Returns:
        生产组合根创建的目录服务。
    """
    return request.app.state.container.knowledge_catalog_service


def get_knowledge_deletion(request: Request) -> KnowledgeBaseDeletionService:
    """只通过组合根取得跨存储恢复删除用例。

    Args:
        request: 当前已鉴权请求。

    Returns:
        同运行共享的知识库删除服务。
    """
    return request.app.state.container.knowledge_deletion_service


@admin_router.delete("/{kb_id}", response_model=KnowledgeBaseDeleteResult)
def delete_knowledge_base(
    kb_id: Annotated[str, Path(pattern=KB_ID_PATTERN)], response: Response,
    user: Annotated[User, Depends(require_admin)],
    service: Annotated[KnowledgeBaseDeletionService, Depends(get_knowledge_deletion)],
    force: Annotated[bool, Query()] = False,
) -> KnowledgeBaseDeleteResult:
    """只有force可清理文档，Cookie角色与CSRF由真实依赖强制检查。

    Args:
        kb_id: 目标知识库的稳定ID。
        response: 禁止缓存的响应。
        user: 实际管理员身份。
        service: 持久计划与同一文档维护用例。
        force: 用户明确授权清理整库文档。

    Returns:
        清理全部完成后的真实结果。
    """
    response.headers["Cache-Control"] = "no-store"
    return service.delete(user.id, kb_id, force=force)


@read_router.get("/knowledge-bases", response_model=KnowledgeBaseList)
def list_knowledge_bases(
    response: Response,
    service: Annotated[KnowledgeCatalogService, Depends(get_knowledge_catalog)],
) -> KnowledgeBaseList:
    """向已登录用户提供目录，不暴露负责人和文档正文。

    Args:
        response: 禁止共享缓存已认证资源。
        service: 元数据目录用例。

    Returns:
        含空库及真实去重文档数的卡片集合。
    """
    response.headers["Cache-Control"] = "no-store"
    return service.list()


@admin_router.post("", status_code=201, response_model=AdminKnowledgeBaseCard)
def create_knowledge_base(
    body: KnowledgeBaseCreateRequest, response: Response,
    service: Annotated[KnowledgeCatalogService, Depends(get_knowledge_catalog)],
) -> AdminKnowledgeBaseCard:
    """管理员与CSRF检查后创建目录，不接收客户端计数与时间。

    Args:
        body: 可创建的元数据字段。
        response: 禁止缓存。
        service: 目录应用服务。

    Returns:
        新目录的管理员投影。
    """
    response.headers["Cache-Control"] = "no-store"
    return service.create(KnowledgeBase(**body.model_dump()))


@admin_router.patch("/{kb_id}", response_model=AdminKnowledgeBaseCard)
def patch_knowledge_base(
    kb_id: Annotated[str, Path(pattern=KB_ID_PATTERN)],
    body: KnowledgeBasePatch, response: Response,
    service: Annotated[KnowledgeCatalogService, Depends(get_knowledge_catalog)],
) -> AdminKnowledgeBaseCard:
    """部分更新仅改变实际传入字段，不能修改ID或创建时间。

    Args:
        kb_id: 稳定ID。
        body: 严格部分更新。
        response: 禁止缓存。
        service: 目录应用服务。

    Returns:
        已提交的新目录元数据。
    """
    response.headers["Cache-Control"] = "no-store"
    return service.patch(kb_id, body)
