# -*- coding: utf-8 -*-
"""将用户范围和模型建议解析为不能越界的确定性检索计划。"""

from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator
from pydantic import model_validator

from app.application.ports.knowledge_bases import KnowledgeBaseRepository
from app.domain.errors import InvalidKnowledgeScopeError
from app.domain.knowledge_bases import validate_kb_ids
from app.domain.routing import KnowledgeScope


class ScopePlan(BaseModel):
    """保存本轮目录快照，使后续扩查不受模型或目录变化影响。

    Attributes:
        request: 用户原始范围。
        initial_kb_ids: 首次实际查询的范围。
        all_kb_ids: 本轮可用目录快照中的全部ID。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    request: KnowledgeScope
    initial_kb_ids: tuple[str, ...]
    all_kb_ids: tuple[str, ...]

    @field_validator("initial_kb_ids", "all_kb_ids", mode="before")
    @classmethod
    def restore_arrays(cls, value: object) -> object:
        """只转换JSON数组容器，不强制转换其中的ID。

        Args:
            value: 快照数组。

        Returns:
            不可变序列或待拒绝的输入。
        """
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def enforce_subset(self) -> Self:
        """拒绝伪造目录、空范围和对显式选择的篡改。

        Returns:
            可执行的范围计划。
        """
        validate_kb_ids(self.initial_kb_ids)
        validate_kb_ids(self.all_kb_ids)
        if not self.initial_kb_ids or not self.all_kb_ids:
            raise ValueError("执行范围和目录不能为空")
        if not set(self.initial_kb_ids).issubset(self.all_kb_ids):
            raise ValueError("执行范围必须属于可用目录")
        if self.request.mode == "explicit":
            if set(self.initial_kb_ids) != set(self.request.kb_ids):
                raise ValueError("显式范围不得被更改")
        return self

    @property
    def expansion_kb_ids(self) -> tuple[str, ...] | None:
        """仅自动且尚未覆盖全部时允许一次扩大范围。

        Returns:
            可扩查的全部范围；无需或禁止扩大时为空。
        """
        if self.request.mode == "explicit":
            return None
        if set(self.initial_kb_ids) == set(self.all_kb_ids):
            return None
        return self.all_kb_ids


class ScopeResolver:
    """目录是范围事实来源，模型建议只有通过校验后才有效。"""

    def __init__(self, catalog: KnowledgeBaseRepository) -> None:
        """保存目录读取端口。

        Args:
            catalog: 后端提供的知识库目录，不由请求端伪造。
        """
        self._catalog = catalog

    def resolve(
        self, scope: KnowledgeScope, *, preferred_kb_ids: tuple[str, ...] = (),
    ) -> ScopePlan:
        """验证显式范围，或保守处理自动选择。

        Args:
            scope: 用户请求的范围。
            preferred_kb_ids: 理解层的可选建议，非法时回退全部可用库。

        Returns:
            后续通道共用的范围快照。

        Raises:
            InvalidKnowledgeScopeError: 用户指定的知识库不存在。
        """
        all_ids = tuple(sorted(kb.id for kb in self._catalog.list()))
        if not all_ids:
            raise InvalidKnowledgeScopeError("没有可用知识库")
        if scope.mode == "explicit":
            if not set(scope.kb_ids).issubset(all_ids):
                raise InvalidKnowledgeScopeError("指定的知识库不存在")
            initial = scope.kb_ids
        else:
            try:
                validate_kb_ids(preferred_kb_ids)
                valid = bool(preferred_kb_ids) and set(preferred_kb_ids).issubset(all_ids)
            except ValueError:
                valid = False
            initial = preferred_kb_ids if valid else all_ids
        return ScopePlan(
            request=scope, initial_kb_ids=initial, all_kb_ids=all_ids,
        )
