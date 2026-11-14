"""正式轮次中可公开和回放的 Generative UI v1 数据。"""
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UIModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')


class RetrievalToolArgs(UIModel):
    query: str = Field(min_length=1)
    kb_ids: list[str]
    attempt: int = Field(ge=1)
    reason: Literal['initial', 'expand_scope', 'escalate']


class RetrievalHit(UIModel):
    doc_id: str
    chunk_id: str
    kb_id: str
    filename: str
    page: int | None
    score: float | None = Field(default=None, ge=0, le=1, exclude_if=lambda v: v is None)
    snippet: str = Field(max_length=200)
    used: bool


class RetrievalToolResult(UIModel):
    retrieved: int = Field(ge=0)
    relevant: int = Field(ge=0)
    hits: list[RetrievalHit] = Field(max_length=10)


class ToolError(UIModel):
    code: Literal['retrieval_failed'] = 'retrieval_failed'
    message: str = '检索失败'


class ToolCall(UIModel):
    tool_call_id: str
    tool_name: Literal['retrieve_knowledge'] = 'retrieve_knowledge'
    status: Literal['running', 'complete', 'error']
    args: RetrievalToolArgs
    result: RetrievalToolResult | None = Field(default=None, exclude_if=lambda v: v is None)
    error: ToolError | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode='after')
    def valid_state(self) -> Self:
        if (self.result is not None) != (self.status == 'complete'):
            raise ValueError('仅 complete 工具事件必须携带 result')
        if (self.error is not None) != (self.status == 'error'):
            raise ValueError('仅 error 工具事件必须携带 error')
        return self


class TableData(UIModel):
    model_config = ConfigDict(strict=True)
    caption: str
    columns: list[str] = Field(min_length=1, max_length=10)
    rows: list[list[str | int | float]] = Field(max_length=50)
    source_positions: list[int]

    @model_validator(mode='after')
    def valid_table(self) -> Self:
        import math
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ValueError('表格列数不一致')
            if any(isinstance(v, bool) or isinstance(v, float) and not math.isfinite(v) for v in row):
                raise ValueError('单元格只能是有限数字或字符串')
        if any(p < 1 for p in self.source_positions):
            raise ValueError('引用位置必须为正数')
        return self


class DataBlock(UIModel):
    data_id: str
    type: Literal['table'] = 'table'
    data: TableData
