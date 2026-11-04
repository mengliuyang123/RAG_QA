# -*- coding: utf-8 -*-
"""检索策略、候选和向量命中的领域模型。"""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from pydantic import AwareDatetime

from app.domain.documents import DocumentChunk
from app.domain.knowledge_bases import DEFAULT_KB_ID
from app.domain.knowledge_bases import KB_ID_PATTERN


class RetrievalStrategy(StrEnum):
    """定义系统支持的检索策略。"""

    VECTOR = "vector"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class RetrievalResult(BaseModel):
    """保存供生成、引用和重排使用的统一检索结果。

    Attributes:
        chunk_id: 文本块唯一标识。
        doc_id: 来源文档唯一标识。
        text: 非空文本块正文。
        filename: 来源文件名。
        page: 来源页码；非分页文档为 ``None``。
        chunk_index: 文本块在来源文档中的零基序号。
        score: 检索链路产生的相关性分数。
        kb_id: 持久化元数据中的知识库归属。
        doc_updated_at: 实际命中的索引版本时间，旧索引未知为None。
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str = Field(min_length=1)
    filename: str
    page: int | None
    chunk_index: int = Field(ge=0)
    score: float
    kb_id: str = Field(default=DEFAULT_KB_ID, pattern=KB_ID_PATTERN)
    doc_updated_at: AwareDatetime | None = None


@dataclass(frozen=True)
class VectorSearchResult:
    """保存一次向量检索命中的文本块及其距离。

    Attributes:
        chunk: 命中的领域文本块。
        distance: 向量存储返回的距离。
    """

    chunk: DocumentChunk
    distance: float
