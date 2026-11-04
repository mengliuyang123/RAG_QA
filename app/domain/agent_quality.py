# -*- coding: utf-8 -*-
"""定义可替换的 Agent 质量判断数据契约。"""

from typing import Self

from pydantic import BaseModel
from pydantic import AwareDatetime
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from app.domain.knowledge_bases import DEFAULT_KB_ID
from app.domain.knowledge_bases import KB_ID_PATTERN

class EvidenceItem(BaseModel):
    """保存本轮证据的稳定编号、正文和可信来源。

    Attributes:
        number: 首次分配的稳定引用编号。
        chunk_id: 实际检索命中的块身份。
        doc_id: 来源文档身份。
        filename: 来源名称。
        page: 可选来源页。
        text: 实际块正文。
        kb_id: 实际知识库归属。
        doc_updated_at: 命中版本的明确时间，未知保持None。
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    number: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    page: int | None
    text: str = Field(min_length=1)
    kb_id: str = Field(default=DEFAULT_KB_ID, pattern=KB_ID_PATTERN)
    doc_updated_at: AwareDatetime | None = None


class EvidenceAssessment(BaseModel):
    """描述证据是否足够、相关条目及尚缺的信息。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sufficient: bool
    relevant_numbers: list[int]
    missing_information: list[str]
    next_query: str | None

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """拒绝重复、非法编号及自相矛盾的充分性判断。

        Returns:
            满足协议不变量的判断。
        """
        numbers = self.relevant_numbers
        if len(numbers) != len(set(numbers)):
            raise ValueError("相关证据编号不得重复")
        if any(number < 1 for number in numbers):
            raise ValueError("相关证据编号必须为正数")
        if self.sufficient:
            if not numbers or self.missing_information or self.next_query:
                raise ValueError("充分证据判断与缺口或编号不一致")
        elif not self.missing_information:
            raise ValueError("证据不足时必须说明信息缺口")
        if any(not item.strip() for item in self.missing_information):
            raise ValueError("信息缺口不得为空白")
        if self.next_query is not None and not self.next_query.strip():
            raise ValueError("补检索查询不得为空白")
        return self


class AnswerAssessment(BaseModel):
    """分别描述回答的事实支持性和任务覆盖度。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    supported: bool
    complete: bool
    issues: list[str]

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """拒绝缺少反馈或与通过结论矛盾的判断。

        Returns:
            满足协议不变量的判断。
        """
        passed = self.supported and self.complete
        if passed == bool(self.issues):
            raise ValueError("答案判断与问题清单不一致")
        if any(not item.strip() for item in self.issues):
            raise ValueError("答案问题说明不得为空白")
        return self
