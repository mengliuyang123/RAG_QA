# -*- coding: utf-8 -*-
"""HTTP API 使用的请求与响应模型。"""

from datetime import datetime
from uuid import UUID
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from pydantic import field_validator

from app.domain.documents import DocumentFileStatus
from app.domain.ingestion_jobs import IngestionJobStatus
from app.domain.conversations import AnswerMode
from app.domain.query_debug import RetrievalDebugItem
from app.domain.conversations import ConversationStatus
from app.domain.retrieval import RetrievalStrategy
from app.domain.routing import KnowledgeScope
from app.domain.routing import Lane
from app.domain.answer_metadata import ScopeInfo
from app.domain.answer_metadata import EvidenceStats
from app.domain.answer_metadata import RefusalInfo
from app.domain.answer_metadata import ClarificationInfo


MAX_QUESTION_LENGTH = 8_000


class UploadParseResponse(BaseModel):
    """文档成功保存并写入向量库后的响应模型。

    Attributes:
        doc_id: 文档的唯一标识。
        filename: 清理后的原始文件名。
        chunks_count: 写入向量库的文本块数量。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "文档成功保存并写入向量库后的响应模型。"
            ),
        }
    )

    doc_id: str = Field(description="文档的唯一标识")
    filename: str = Field(description="清理后的原始文件名")
    chunks_count: int = Field(
        ge=1,
        description="写入向量库的文本块数量",
    )


class QueryRequest(BaseModel):
    """非流式和流式知识库问答请求。

    Attributes:
        question: 清理首尾空白后的非空问题。
        strategy: 检索策略。
        top_k: 参与生成的文本块上限。
    """

    model_config = ConfigDict(
        json_schema_extra={"description": "非流式知识库问答请求。"}
    )

    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="需要查询知识库的问题",
    )
    strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.VECTOR,
        description=(
            "检索策略：纯向量、BM25 + 向量 RRF 混合检索，"
            "或混合召回后远程重排"
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="参与生成的文本块上限",
    )
    kb_scope: KnowledgeScope | None = None
    lane: Literal["auto", "fast", "agent"] = "auto"
    debug: bool = False

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        """清理问题首尾空白，并拒绝仅含空白符的请求。

        Args:
            value: 客户端提交的原始问题。

        Returns:
            清理首尾空白后的问题。

        Raises:
            ValueError: 问题仅包含空白符。
        """
        normalized = value.strip()
        if not normalized:
            raise ValueError("问题不能为空")
        return normalized


class CitationResponse(BaseModel):
    """查询响应中可追溯的文档来源。

    Attributes:
        doc_id: 来源文档的唯一标识。
        filename: 来源文件名。
        page: 来源页码；非分页文档为 ``None``。
        chunk_text: 被引用文本块的正文。
        position: 正文 [n] 的原始编号，不是列表下标。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": "查询响应中可追溯的文档来源。",
        }
    )

    doc_id: str
    filename: str
    page: int | None
    chunk_text: str
    position: int | None = Field(
        default=None, ge=1, description="正文 [n] 的原始引用编号",
    )
    chunk_id: str | None = None
    kb_id: str = "default"
    doc_updated_at: datetime | None = None


class RoutedResponseFields(BaseModel):
    """旧外壳统一追加的元数据，旧记录未知信息允许为空。"""

    model_config = ConfigDict(frozen=True)
    lane: Lane | None = None
    outcome: Literal["answered", "refused", "clarification"] | None = None
    grounding: Literal["grounded", "partial", "ungrounded"] | None = None
    scope_expanded: bool = False
    escalated: bool = False
    kb_scope: ScopeInfo | None = None
    evidence_stats: EvidenceStats | None = None
    refusal: RefusalInfo | None = None
    clarification: ClarificationInfo | None = None
    trace_id: str | None = None
    feedback: Literal["up", "down"] | None = None
    created_at: datetime | None = None


def flatten_answer_payload(result: BaseModel) -> dict[str, object]:
    """仅展开公共元数据，不把执行快照或模型上下文暴露给客户端。

    Args:
        result: 查询结果或正式历史外壳。

    Returns:
        保留旧字段名称的扁平响应。
    """
    payload = result.model_dump()
    metadata = payload.pop("metadata", None)
    if metadata is not None:
        payload.update(metadata)
    return payload


class QueryResponse(RoutedResponseFields):
    """非流式知识库问答响应。

    Attributes:
        answer: 语言模型生成的回答。
        citations: 回答实际引用的可信来源。
        latency_ms: 查询链路总耗时，单位为毫秒。
        retrieval_strategy: 本次实际执行的检索策略。
    """

    model_config = ConfigDict(
        json_schema_extra={"description": "非流式知识库问答响应。"}
    )

    answer: str
    citations: list[CitationResponse]
    latency_ms: float = Field(ge=0)
    retrieval_strategy: RetrievalStrategy
    answer_mode: AnswerMode | None = None
    retrieval_calls: int = Field(default=0, ge=0, le=3)
    retrieved: tuple[RetrievalDebugItem, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )


class DocumentSummaryResponse(BaseModel):
    """文档列表中的聚合来源信息。

    Attributes:
        doc_id: 文档的规范 UUID。
        filename: 来源文件名。
        chunks_count: 文档对应的文本块数量。
        created_at: 源文件最后修改时间；文件缺失时为
            ``None``。
        file_status: 当前源文件状态。
        kb_id: 实际知识库，兼容旧调用未报告时为空。
        file_type: 原始扩展名规范小写。
        size_bytes: 当前实际源文件大小，未知为空。
        uploaded_by: 有持久任务时的实际管理员身份，未知为空。
        ingestion_status: 真实任务状态，无登记时为空。
        published: 是否已发布，不等于存在部分索引。
        citations_7d: 七个UTC日历日的正式引用行数，不是问题数。
        unattributed_citations_7d: 缺少块ID但明确归属此文档的旧引用数。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": "文档列表中的聚合来源信息。",
        }
    )

    doc_id: UUID
    filename: str
    chunks_count: int = Field(ge=1)
    created_at: datetime | None
    file_status: DocumentFileStatus
    kb_id: str | None = None
    file_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    uploaded_by: UUID | None = None
    ingestion_status: IngestionJobStatus | None = None
    published: bool | None = None
    citations_7d: int | None = Field(default=None, ge=0)
    unattributed_citations_7d: int | None = Field(default=None, ge=0)


class DocumentListResponse(BaseModel):
    """文档列表和精确总数响应。

    Attributes:
        documents: 当前文档聚合列表。
        total: 文档总数。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": "文档列表和精确总数响应。",
        }
    )

    documents: list[DocumentSummaryResponse]
    total: int = Field(ge=0)


class DocumentDeleteResponse(BaseModel):
    """文档索引和可用源文件删除成功响应。

    Attributes:
        doc_id: 已删除文档的规范 UUID。
        deleted: 删除成功标记。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "文档索引和可用源文件删除成功响应。"
            ),
        }
    )

    doc_id: UUID
    deleted: bool


class ConversationCreateRequest(BaseModel):
    """可选的会话创建设置，空请求仍然兼容旧客户端。"""

    model_config = ConfigDict(extra="forbid")

    kb_scope: KnowledgeScope = Field(default_factory=KnowledgeScope)


class ConversationPatchRequest(BaseModel):
    """修改默认设置而不改写历史轮次。"""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=100)
    kb_scope: KnowledgeScope | None = None

    @model_validator(mode="after")
    def validate_changes(self) -> "ConversationPatchRequest":
        """拒绝无内容修改或仅包含空白的标题。

        Returns:
            已验证的修改请求。
        """
        if self.title is None and self.kb_scope is None:
            raise ValueError("至少提供标题或知识库范围")
        if self.title is not None and not self.title.strip():
            raise ValueError("标题不能为空")
        return self


class ConversationQueryRequest(BaseModel):
    """只允许提交本轮原始问题的多轮会话请求。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1, max_length=MAX_QUESTION_LENGTH,
    )
    kb_scope: KnowledgeScope | None = None

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("问题不能为空")
        return normalized


class ConversationCitationResponse(BaseModel):
    """返回与正文编号一致的引用及可信来源。"""

    position: int = Field(ge=1)
    doc_id: str
    filename: str
    page: int | None
    chunk_text: str
    chunk_id: str | None = None
    kb_id: str = "default"
    doc_updated_at: datetime | None = None


class ConversationSummaryResponse(BaseModel):
    """会话列表中的稳定摘要。"""

    id: UUID
    title: str
    status: ConversationStatus
    created_at: datetime
    updated_at: datetime
    turns_count: int = Field(ge=0)
    kb_scope: KnowledgeScope = Field(default_factory=KnowledgeScope)


class ConversationCreateResponse(ConversationSummaryResponse):
    """创建会话后的元数据响应。"""


class ConversationListResponse(BaseModel):
    """active 会话列表及精确总数。"""

    conversations: list[ConversationSummaryResponse]
    total: int = Field(ge=0)
    next_cursor: str | None = None


class ConversationTurnResponse(RoutedResponseFields):
    """一轮正式历史，不包含图 scratchpad 或工具消息。"""

    id: UUID
    user_message: str
    assistant_message: str
    answer_mode: AnswerMode
    retrieval_calls: int = Field(ge=0, le=3)
    citations: list[ConversationCitationResponse]
    latency_ms: float = Field(ge=0)
    created_at: datetime


class ConversationDetailResponse(ConversationSummaryResponse):
    """会话元数据及正式轮次，可选有限页。"""

    turns: list[ConversationTurnResponse]
    turns_next_cursor: str | None = None


class ConversationQueryResponse(RoutedResponseFields):
    """一次多轮查询的已验证正式结果。"""

    conversation_id: UUID
    turn_id: UUID
    answer: str
    answer_mode: AnswerMode
    retrieval_calls: int = Field(ge=0, le=3)
    citations: list[ConversationCitationResponse]
    latency_ms: float = Field(ge=0)


class ConversationDeleteResponse(BaseModel):
    """会话及对应 checkpoint 已完整删除的响应。"""

    conversation_id: UUID
    deleted: bool
