# -*- coding: utf-8 -*-
"""问答结果、引用和评估追踪的领域模型。"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.retrieval import RetrievalStrategy
from app.domain.answer_metadata import AnswerMetadata
from app.domain.conversations import AnswerMode
from app.domain.knowledge_bases import DEFAULT_KB_ID
from app.domain.query_debug import RetrievalDebugItem


class Citation(BaseModel):
    """保存回答中可追溯的可信文档来源。

    Attributes:
        doc_id: 来源文档的唯一标识。
        filename: 来源文件名。
        page: 来源页码；非分页文档为 ``None``。
        chunk_text: 被引用文本块的正文。
        position: 回答使用的原始引用编号；旧调用方可暂未指定。
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    filename: str
    page: int | None
    chunk_text: str
    position: int | None = Field(
        default=None, ge=1, description="正文 [n] 的原始引用编号",
    )
    chunk_id: str | None = None
    kb_id: str = DEFAULT_KB_ID
    doc_updated_at: datetime | None = None


class QueryResult(BaseModel):
    """保存一次非流式知识库问答的完整结果。

    Attributes:
        answer: 语言模型生成的回答。
        citations: 回答实际引用的可信来源。
        latency_ms: 查询链路总耗时，单位为毫秒。
        retrieval_strategy: 本次实际执行的检索策略。
    """

    model_config = ConfigDict(frozen=True)

    answer: str
    citations: list[Citation]
    latency_ms: float = Field(ge=0)
    retrieval_strategy: RetrievalStrategy
    answer_mode: AnswerMode | None = None
    retrieval_calls: int = Field(default=0, ge=0, le=3)
    metadata: AnswerMetadata | None = None
    created_at: datetime | None = None
    retrieved: tuple[RetrievalDebugItem, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )


class RetrievedChunkTrace(BaseModel):
    """保存评估运行中一个有序检索文本块的追踪信息。

    Attributes:
        rank: 从一开始的检索名次。
        chunk_id: 文本块唯一标识。
        doc_id: 来源文档唯一标识。
        text: 仅供运行时指标使用的正文，序列化时排除。
        text_sha256: 正文 UTF-8 字节的 SHA-256。
        filename: 来源文件名。
        page: 来源页码；非分页文档为 ``None``。
        chunk_index: 文本块在来源文档中的零基序号。
        score: 检索链路给出的最终分数。
    """

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    chunk_id: str
    doc_id: str
    text: str = Field(min_length=1, exclude=True, repr=False)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    filename: str
    page: int | None
    chunk_index: int = Field(ge=0)
    score: float


class EvaluationQueryResult(BaseModel):
    """保存评估查询的回答、来源、上下文和阶段耗时。

    Attributes:
        answer: 语言模型生成的回答。
        citations: 回答实际引用的可信来源。
        retrieved: 实际交给生成器的有序文本块追踪。
        retrieval_ms: 检索阶段耗时，单位为毫秒。
        generation_ms: 生成阶段耗时，单位为毫秒。
        latency_ms: 查询链路总耗时，单位为毫秒。
        retrieval_strategy: 本次实际执行的检索策略。
    """

    model_config = ConfigDict(frozen=True)

    answer: str
    citations: list[Citation]
    retrieved: list[RetrievedChunkTrace]
    retrieval_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    retrieval_strategy: RetrievalStrategy
