# -*- coding: utf-8 -*-
"""知识库目录的公开投影与管理员投影，不把运维数据混进卡片。"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import AwareDatetime

from app.domain.knowledge_bases import KB_ID_PATTERN


class KnowledgeBaseCard(BaseModel):
    """已登录用户可见的目录卡片，不含负责人或源文件状态。

    Attributes:
        id: 稳定知识库ID。
        name: 可见名称。
        description: 管理员维护的范围说明。
        icon: 可选前端图标名。
        documents_count: 当前实际去重文档数，空库为零。
        updated_at: 目录元数据最近修改时间。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=KB_ID_PATTERN)
    name: str
    description: str
    icon: str | None
    documents_count: int = Field(ge=0)
    updated_at: datetime


class AdminKnowledgeBaseCard(KnowledgeBaseCard):
    """仅管理员写入响应额外提供负责人和原始创建时间。

    Attributes:
        owner_name: 负责人展示名，不自动等同账号。
        created_at: 创建时间，PATCH不允许改写。
    """

    owner_name: str | None
    created_at: datetime


class KnowledgeBaseList(BaseModel):
    """返回包含空库的稳定有序目录。

    Attributes:
        knowledge_bases: 不含内部负责人字段的卡片集合。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    knowledge_bases: tuple[KnowledgeBaseCard, ...]


class IndexedDocumentHeader(BaseModel):
    """由索引元数据聚合的文档身份，不包含正文或源文件运维信息。

    Attributes:
        doc_id: 来源文档UUID。
        kb_id: 索引中的实际归属。
        filename: 原始安全文件名。
        chunks_count: 文档块数。
        updated_at: 已知版本时间，旧索引为None。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    doc_id: UUID
    kb_id: str = Field(pattern=KB_ID_PATTERN)
    filename: str = Field(min_length=1)
    chunks_count: int = Field(ge=1)
    updated_at: AwareDatetime | None


class IndexedChunkHeader(BaseModel):
    """只读元数据身份，用于验证请求归属和选取相邻块。

    Attributes:
        chunk_id: 不透明文本块ID，不从中解析索引。
        doc_id: 文档UUID。
        kb_id: 实际知识库。
        filename: 安全来源文件名。
        chunk_index: 块的实际顺序。
        page: 可选页码。
        doc_updated_at: 可选已知版本时间。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_id: str = Field(min_length=1)
    doc_id: UUID
    kb_id: str = Field(pattern=KB_ID_PATTERN)
    filename: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    page: int | None = Field(ge=1)
    doc_updated_at: AwareDatetime | None


class KnowledgeDocument(BaseModel):
    """普通用户的文档浏览卡片，未知时间不得补当前时间。

    Attributes:
        doc_id: 稳定文档UUID。
        filename: 显示文件名。
        file_type: 文件扩展名小写形式。
        chunks_count: 实际块数。
        updated_at: 可选已知版本时间。
        is_stale: P0固定False，时效管理另行接入。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    doc_id: UUID
    filename: str
    file_type: str
    chunks_count: int = Field(ge=1)
    updated_at: AwareDatetime | None
    is_stale: bool = False


class KnowledgeDocumentPage(BaseModel):
    """公开浏览的稳定游标页，不公开源文件状态。

    Attributes:
        documents: 当前页文档。
        next_cursor: 有下一页时的不透明游标。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    documents: tuple[KnowledgeDocument, ...]
    next_cursor: str | None = None


class DocumentBrowseCursor(BaseModel):
    """分页位置只绑定检索条件，不充当权限凭证。

    Attributes:
        last_doc_id: 上一页末尾的稳定排序键。
        fingerprint: 知识库与规范搜索词的指纹。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    last_doc_id: UUID
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class ContextChunk(BaseModel):
    """来源面板中的单块正文与是否为引用目标。

    Attributes:
        chunk_id: 不透明块ID。
        chunk_index: 文档中的实际顺序。
        page: 可选来源页。
        text: 当前索引版本的正文。
        is_target: 是否为用户点击的目标块。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_id: str
    chunk_index: int = Field(ge=0)
    page: int | None
    text: str
    is_target: bool


class DocumentContext(BaseModel):
    """范围校验后返回的有限原文窗口。

    Attributes:
        doc_id: 当前文档UUID。
        kb_id: 当前知识库归属。
        filename: 当前文档名。
        doc_updated_at: 已知版本时间，旧索引为None。
        chunks: 最多目标与前后三块，实际顺序不依赖ID格式。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    doc_id: UUID
    kb_id: str
    filename: str
    doc_updated_at: AwareDatetime | None
    chunks: tuple[ContextChunk, ...] = Field(min_length=1, max_length=7)
