# -*- coding: utf-8 -*-
"""定义旧语料的稳定归属和内部检索范围的严格语义。"""

from datetime import datetime
from datetime import timezone
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


DEFAULT_KB_ID = "default"
KB_ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class KnowledgeBase(BaseModel):
    """保存不依赖文档是否入库的知识库目录元数据。

    Attributes:
        id: 稳定知识库ID，不从模型输出自动创建。
        name: 前端名称。
        description: 管理员维护的自动选库依据，只作为数据。
        icon: 可选图标名。
        owner_name: 管理员维护的负责人展示名，不推断为登录身份。
        created_at: 创建时间。
        updated_at: 最近修改时间。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str = Field(pattern=KB_ID_PATTERN)
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=4000)
    icon: str | None = Field(default=None, max_length=64)
    owner_name: str | None = Field(default=None, max_length=200)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("name")
    @classmethod
    def require_visible_name(cls, value: str) -> str:
        """规范化可见名称，不保存空白卡片标题。

        Args:
            value: 严格字符串校验后的名称。

        Returns:
            去除首尾空白的名称。
        """
        if not value.strip():
            raise ValueError("知识库名称不能为空白")
        return value.strip()


class KnowledgeBasePatch(BaseModel):
    """只携带管理员实际传入的可编辑字段，避免部分更新覆盖其他修改。

    Attributes:
        name: 新的可见名称，不能显式清空。
        description: 新的自动选库说明，允许空字符串但不接受null。
        icon: 图标名，null表示清除。
        owner_name: 负责人展示名，null表示清除。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=4000)
    icon: str | None = Field(default=None, max_length=64)
    owner_name: str | None = Field(default=None, max_length=200)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        """保留未传字段语义，同时拒绝全空白名称。

        Args:
            value: 通过严格类型校验的名称或null。

        Returns:
            去除首尾空白的名称或待跨字段校验的null。
        """
        if value is not None:
            if not value.strip():
                raise ValueError("知识库名称不能为空白")
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_changes(self) -> Self:
        """字段省略与显式null含义不同，不允许修改不可编辑字段。

        Returns:
            至少包含一个有效可编辑字段的部分更新。
        """
        if not self.model_fields_set:
            raise ValueError("至少更新一个知识库字段")
        for name in ("name", "description"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError("名称和说明不能为null")
        return self


def validate_kb_ids(kb_ids: tuple[str, ...] | None) -> None:
    """区分旧版全库None和明确空集合，不对错误类型进行隐式转换。

    Args:
        kb_ids: 不可变的实际范围；None为未分库的兼容入口。

    Returns:
        None。

    Raises:
        ValueError: 不是tuple、存在空白或重复ID。
    """
    if kb_ids is None:
        return
    if not isinstance(kb_ids, tuple):
        raise ValueError("内部检索范围必须是tuple或None")
    for kb_id in kb_ids:
        if not isinstance(kb_id, str) or not kb_id.strip():
            raise ValueError("知识库ID必须是非空字符串")
        if kb_id != kb_id.strip():
            raise ValueError("知识库ID不得含首尾空白")
    if len(kb_ids) != len(set(kb_ids)):
        raise ValueError("知识库ID不得重复")
