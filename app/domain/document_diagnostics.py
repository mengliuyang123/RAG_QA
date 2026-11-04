# -*- coding: utf-8 -*-
"""分块诊断与正式引用行统计，不包含提问者或聊天历史。"""

from uuid import UUID

from pydantic import AwareDatetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class DocumentCitationTotals(BaseModel):
    """明确归属文档的引用行总数及块归属未知数。

    Attributes:
        total: 每个turn_id/position计一次。
        unattributed: 缺少chunk_id的旧引用行。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    total: int = Field(ge=0)
    unattributed: int = Field(ge=0)


class ChunkCitationCount(BaseModel):
    """一个精确块身份对应的已知引用行计数。

    Attributes:
        chunk_id: 不透明块ID。
        count: 仅该doc_id和chunk_id组合的计数。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_id: str = Field(min_length=1)
    count: int = Field(ge=0)


class ChunkCitationTotals(BaseModel):
    """当前页的已知计数及该文档不可分配的旧引用。

    Attributes:
        counts: 仅所选页中的块。
        unattributed: 文档级缺chunk_id的引用数。
        window_started_at: 实际UTC统计起点，含边界。
        window_ended_at: 实际UTC统计截止，含边界。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    counts: tuple[ChunkCitationCount, ...]
    unattributed: int = Field(ge=0)
    window_started_at: AwareDatetime
    window_ended_at: AwareDatetime


class DiagnosticChunk(BaseModel):
    """后台单块当前正文和不猜测未知归属的计数。

    Attributes:
        chunk_id: 不透明块标识。
        chunk_index: 实际顺序，允许不连续。
        page: 可选来源页。
        text: 当前索引真实正文。
        citations_7d: 有旧未知归属时为None，否则为完整已知计数。
        known_citations_7d: 始终保留精确身份的已知数，不冒充完整数。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_id: str
    chunk_index: int = Field(ge=0)
    page: int | None
    text: str
    citations_7d: int | None = Field(ge=0)
    known_citations_7d: int = Field(ge=0)


class DiagnosticChunkPage(BaseModel):
    """管理员正文页；published=false不能误当作用户可检索资料。

    Attributes:
        doc_id: 实际文档。
        kb_id: 实际知识库。
        filename: 索引保存的来源名。
        doc_updated_at: 明确保存的版本时间，旧索引未知为None。
        published: 当前任务发布事实。
        chunks: 当前页真实正文。
        total: 全文档实际块数。
        next_cursor: 绑定文档和元数据修订的续页。
        unattributed_citations_7d: 不可分配到具体块的旧引用行数。
        window_started_at: 引用统计UTC起点。
        window_ended_at: 引用统计UTC截止。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    doc_id: UUID
    kb_id: str
    filename: str
    doc_updated_at: AwareDatetime | None
    published: bool
    chunks: tuple[DiagnosticChunk, ...]
    total: int = Field(ge=0)
    next_cursor: str | None = None
    unattributed_citations_7d: int = Field(ge=0)
    window_started_at: AwareDatetime
    window_ended_at: AwareDatetime


class DiagnosticCursor(BaseModel):
    """有界正文页的文档、版本和顺序绑定。

    Attributes:
        doc_id: 不能跨文档复用。
        revision: 全部元数据头的修订指纹。
        last_index: 上一页最后的实际块序号。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    doc_id: UUID
    revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    last_index: int = Field(ge=0)
