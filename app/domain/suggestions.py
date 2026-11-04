# -*- coding: utf-8 -*-
"""仅管理员明确预设的公开示例问题，不从私人会话自动提取。"""

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from app.domain.knowledge_bases import KB_ID_PATTERN


class SuggestedQuestion(BaseModel):
    """可供全体已登录用户查看的已预设问题。

    Attributes:
        id: 不透明稳定标识。
        kb_id: 实际知识库。
        text: 管理员明确发布的可见问题。
        source: P0固定curated，不采集popular或私人历史。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    kb_id: str = Field(pattern=KB_ID_PATTERN)
    text: str = Field(min_length=1, max_length=500)
    source: Literal["curated"] = "curated"


class SuggestedQuestions(BaseModel):
    """建议列表不包含操作者身份或私人提问数据。

    Attributes:
        questions: 已发布预设，未配置时为空。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    questions: tuple[SuggestedQuestion, ...]


class CuratedQuestionsUpdate(BaseModel):
    """管理员明确替换某库预设，空集合表示清空。

    Attributes:
        questions: 最多50个规范化、不重复的公开问题。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    questions: tuple[str, ...] = Field(max_length=50)

    @field_validator("questions", mode="before")
    @classmethod
    def accept_json_array(cls, value: object) -> object:
        """只转换容器，不隐式转换问题元素类型。

        Args:
            value: 原始JSON数组或tuple。

        Returns:
            不可变容器或待拒绝的非法类型。
        """
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """去除首尾空白，拒绝空问题和大小写重复。

        Args:
            value: 严格字符串元组。

        Returns:
            可持久化的规范公开问题。
        """
        normalized = []
        seen = set()
        for question in value:
            text = question.strip()
            if not 1 <= len(text) <= 500 or text.casefold() in seen:
                raise ValueError("预设问题为空、过长或重复")
            seen.add(text.casefold())
            normalized.append(text)
        return tuple(normalized)
