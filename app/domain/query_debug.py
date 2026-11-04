# -*- coding: utf-8 -*-
"""管理员可见的实际召回观察与最终证据判断，不含生成草稿。"""

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class RetrievalObservation(BaseModel):
    """记录单次召回中的实际名次和分数。

    Attributes:
        attempt: 本轮第几次检索。
        rank: 本次最终Top-K中的名次，不是累计证据编号。
        position: 对应证据注册表的原始编号。
        chunk_id: 实际命中的块。
        score: 本次检索策略给出的最终分数。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    attempt: int = Field(ge=1, le=3)
    rank: int = Field(ge=1, le=20)
    position: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    score: float


class RetrievalDebugItem(RetrievalObservation):
    """将召回观察关联到最终批准状态及可信来源。

    Attributes:
        doc_id: 来源文档身份。
        filename: 来源文件名。
        page: 可选页码。
        kb_id: 实际来源知识库。
        chunk_text: 本轮证据注册表中的正文。
        approved: 最后一次证据裁判是否批准此块。
        cited: 已发布答案是否实际引用此块。
    """

    doc_id: str
    filename: str
    page: int | None
    kb_id: str
    chunk_text: str
    approved: bool
    cited: bool
