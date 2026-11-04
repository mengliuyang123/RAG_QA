# -*- coding: utf-8 -*-
"""文档导入、管理和检索共享的领域模型。"""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from pydantic import AwareDatetime

from app.domain.knowledge_bases import DEFAULT_KB_ID
from app.domain.knowledge_bases import KB_ID_PATTERN
from app.domain.ingestion_jobs import IngestionJobStatus

class DocumentSegment(BaseModel):
    """保存源文本片段及其引用元数据。

    Attributes:
        text: 源文本内容。
        page: 来源页码；非分页文档为 ``None``。
        filename: 来源文件名。
    """

    text: str
    page: int | None
    filename: str


class DocumentChunk(BaseModel):
    """保存可写入检索索引的文本块及来源元数据。

    Attributes:
        id: 文本块唯一标识。
        doc_id: 来源文档唯一标识。
        text: 非空文本块正文。
        filename: 来源文件名。
        page: 来源页码；非分页文档为 ``None``。
        chunk_index: 文本块在来源文档中的零基序号。
        kb_id: 来源知识库；旧语料默认归入default。
        doc_updated_at: 已明确记录的文档版本时间，旧索引未知时为None。
    """

    id: str
    doc_id: str
    text: str = Field(min_length=1)
    filename: str
    page: int | None
    chunk_index: int = Field(ge=0)
    kb_id: str = Field(default=DEFAULT_KB_ID, pattern=KB_ID_PATTERN)
    doc_updated_at: AwareDatetime | None = None


class UploadResult(BaseModel):
    """保存文档成功入库后的稳定结果。

    Attributes:
        doc_id: 新入库文档的唯一标识。
        filename: 清理后的原始文件名。
        chunks_count: 已写入索引的文本块数量。
    """

    doc_id: str
    filename: str
    chunks_count: int = Field(ge=1)


class DocumentFileStatus(StrEnum):
    """表示上传源文件在受控目录中的可用状态。"""

    AVAILABLE = "available"
    MISSING = "missing"


class DocumentSummary(BaseModel):
    """保存按文档标识聚合后的文档信息。

    Attributes:
        doc_id: 文档 UUID。
        filename: 向量元数据中唯一的安全原始文件名。
        chunks_count: 当前保存的文本块数量。
        created_at: 可用源文件 mtime 对应的 UTC 时间。
        file_status: 源文件可用或缺失状态。
        kb_id: 管理元数据中的实际知识库。
        file_type: 安全文件名的扩展名。
        size_bytes: 当前源文件实际大小，文件缺失时为None。
        uploaded_by: 任务保存的管理员UUID，旧文档未知为None。
        ingestion_status: 当前真实任务状态，旧文档未知为None。
        published: 是否满足既有发布边界，旧调用未报告时为None。
        citations_7d: 最近七个UTC日历日的正式引用行数。
        unattributed_citations_7d: 已属此文档但缺少块ID的引用数。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: UUID
    filename: str = Field(min_length=1)
    chunks_count: int = Field(ge=1)
    created_at: datetime | None
    file_status: DocumentFileStatus
    kb_id: str | None = Field(default=None, pattern=KB_ID_PATTERN)
    file_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    uploaded_by: UUID | None = None
    ingestion_status: IngestionJobStatus | None = None
    published: bool | None = None
    citations_7d: int | None = Field(default=None, ge=0)
    unattributed_citations_7d: int | None = Field(default=None, ge=0)


class DocumentDeleteResult(BaseModel):
    """保存文档索引和可用源文件均已删除的结果。

    Attributes:
        doc_id: 已删除文档 UUID。
        deleted: 固定为真，表示删除流程成功。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: UUID
    deleted: bool = True
