# -*- coding: utf-8 -*-
"""管理员在线查看和修改模型配置；保存后写入 .env 并热替换应用容器。"""

import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from pydantic import BaseModel
from pydantic import Field

from app.application.errors import EmbeddingError
from app.application.errors import VectorStoreError
from app.application.settings.model_settings import EMBEDDING_FIELDS
from app.application.settings.model_settings import ModelSettingsError
from app.application.settings.model_settings import build_settings
from app.application.settings.model_settings import changed_fields
from app.application.settings.model_settings import env_values
from app.application.settings.model_settings import normalize_updates
from app.application.settings.model_settings import settings_view
from app.bootstrap.container import ApplicationContainer
from app.config import Settings
from app.domain.auth import User
from app.bootstrap.settings_store import open_env_file
from app.interfaces.http.auth import require_admin
from app.interfaces.http.container_runtime import ContainerRuntime

router = APIRouter(prefix="/admin/settings", tags=["settings"])
logger = logging.getLogger(__name__)

EMBEDDING_PROBE_TEXT = "连接测试"


class ModelSettingsUpdate(BaseModel):
    """只需提交要修改的字段；值为 null 或空字符串表示恢复默认。"""

    values: dict[str, str | int | None] = Field(min_length=1, max_length=64)


def _env_path(request: Request) -> Path:
    override = getattr(request.app.state, "env_file_path", None)
    if override is not None:
        return Path(override)
    return Path(str(Settings.model_config.get("env_file", ".env")))


def _stored_dimensions(container: ApplicationContainer) -> int | None:
    reader = getattr(container.vector_store, "stored_dimensions", None)
    if reader is None:
        return None
    try:
        return reader()
    except VectorStoreError:
        return None


def _view(container: ApplicationContainer) -> dict[str, object]:
    return settings_view(container.settings, os.environ, _stored_dimensions(container))


def _probe_embedding(container: ApplicationContainer) -> int:
    try:
        return len(container.embedding_provider.embed_query(EMBEDDING_PROBE_TEXT))
    except EmbeddingError as exc:
        raise HTTPException(422, f"向量模型连接失败：{exc}") from exc
    except Exception as exc:
        logger.warning("向量模型连接测试失败：%s", type(exc).__name__)
        raise HTTPException(422, "向量模型连接失败，请检查地址、模型名和 API Key") from exc


@router.get("/models")
def get_model_settings(request: Request, response: Response) -> dict[str, object]:
    """返回模型配置（含 API Key 明文）和当前向量索引维度。"""
    response.headers["Cache-Control"] = "no-store"
    runtime: ContainerRuntime = request.app.state.container_runtime
    return _view(runtime.current)


@router.put("/models")
def update_model_settings(
    body: ModelSettingsUpdate,
    request: Request,
    response: Response,
    user: Annotated[User, Depends(require_admin)],
) -> dict[str, object]:
    """校验并保存模型配置，之后到达的请求立即使用新配置。

    向量模型相关字段变化时，先用新配置请求一次向量接口；
    新维度与已入库向量的维度不同则拒绝保存，避免检索全部失败。
    """
    response.headers["Cache-Control"] = "no-store"
    runtime: ContainerRuntime = request.app.state.container_runtime
    with runtime.update_lock:
        current = runtime.current
        try:
            updates = normalize_updates(body.values)
            updated = build_settings(current.settings, updates)
        except ModelSettingsError as exc:
            raise HTTPException(422, str(exc)) from exc

        changed = changed_fields(current.settings, updated)
        if not changed:
            return _view(current)

        candidate = runtime.build(updated)
        try:
            if EMBEDDING_FIELDS.intersection(changed):
                dimensions = _probe_embedding(candidate)
                stored = _stored_dimensions(current)
                if stored is not None and dimensions != stored:
                    raise HTTPException(
                        409,
                        f"新向量模型输出 {dimensions} 维，已入库的向量是 {stored} 维，"
                        "两者不兼容。请保持原维度，或清空知识库后重新入库。",
                    )
            open_env_file(_env_path(request)).update(
                env_values({name: updates[name] for name in changed if name in updates}),
            )
        except BaseException:
            candidate.close()
            raise

        runtime.swap(candidate)
        # 只记录改了哪些键，不记录值，避免 API Key 进入日志。
        logger.info("管理员 %s 修改了模型配置：%s", user.id, ", ".join(changed))
        return _view(candidate)
