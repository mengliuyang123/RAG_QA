# -*- coding: utf-8 -*-
"""入库任务冻结契约；公开进度与内部提交、租约资料分离。"""

from pathlib import PurePath
from pathlib import PureWindowsPath
from typing import Literal
from typing import Self
from uuid import UUID

from pydantic import AwareDatetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from app.domain.knowledge_bases import KB_ID_PATTERN


IngestionJobStatus = Literal["queued", "running", "succeeded", "failed"]
IngestionStage = Literal["parsing", "chunking", "embedding", "indexing"]
IngestionFailureCode = Literal[
    "empty_document", "parse_failed", "embedding_failed", "indexing_failed",
    "source_unavailable", "interrupted",
]
FAILURE_MESSAGES: dict[IngestionFailureCode, str] = {
    "empty_document": "文档没有可提取的正文，请检查文件内容。",
    "parse_failed": "文档解析失败，请检查格式后重新上传。",
    "embedding_failed": "文档向量化失败，请稍后重新上传。",
    "indexing_failed": "文档索引未完成，内容尚未发布。",
    "source_unavailable": "任务源文件缺失或不完整，请重新上传。",
    "interrupted": "入库任务中断，内容尚未发布。",
}


class IngestionJobSubmission(BaseModel):
    """服务端保存源文件后提交的任务身份，不接受客户端文件路径。

    Attributes:
        job_id: 服务端生成的不透明任务标识。
        doc_id: 此任务专属的新文档标识，不覆盖旧版本。
        kb_id: 已注册的目标知识库。
        filename: 已清理的单个文件名。
        size_bytes: 持久源文件的实际字节数。
        uploaded_by: 实际登录管理员身份，不能从请求正文指定。
        content_sha256: 恢复执行前校验源文件完整性的摘要。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    job_id: str = Field(pattern=r"^job_[0-9a-f]{32}$")
    doc_id: UUID
    kb_id: str = Field(pattern=KB_ID_PATTERN)
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    uploaded_by: UUID
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """拒绝持久化目录、控制字符或空白名称，恢复不信任历史路径。

        Args:
            value: 已由上传层清理的候选名称。

        Returns:
            校验后的原名称，不静默改名。
        """
        if not value.strip() or value in {".", ".."}:
            raise ValueError("任务文件名无效")
        if PurePath(value).name != value:
            raise ValueError("任务文件名不能包含目录")
        if PureWindowsPath(value).name != value or ":" in value:
            raise ValueError("任务文件名不能包含Windows路径")
        for character in value:
            if ord(character) < 32 or ord(character) == 127:
                raise ValueError("任务文件名不能包含控制字符")
        return value


class DocumentJobInfo(BaseModel):
    """管理文档时关联的任务元数据，不返回摘要、租约或源路径。

    Attributes:
        doc_id: 任务对应的精确文档。
        kb_id: 服务端登记的知识库。
        filename: 服务端清理后的原始文件名。
        uploaded_by: 实际提交任务的管理员身份。
        status: 已通过完整任务一致性校验的状态。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: UUID
    kb_id: str = Field(pattern=KB_ID_PATTERN)
    filename: str
    uploaded_by: UUID
    status: IngestionJobStatus


class IngestionJobFailure(BaseModel):
    """只公开预定义分类与固定说明，绝不序列化上游异常。

    Attributes:
        code: 有限的失败分类。
        message: 与分类绑定的安全说明。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    code: IngestionFailureCode
    message: str

    @model_validator(mode="after")
    def validate_safe_message(self) -> Self:
        """禁止调用方把异常或凭据当作任务说明持久化。

        Returns:
            完整校验后的错误对象。
        """
        if self.message != FAILURE_MESSAGES[self.code]:
            raise ValueError("任务失败说明必须使用安全模板")
        return self


class IngestionJob(BaseModel):
    """供进度查询使用的稳定外壳，不包含文件路径、摘要和租约。

    Attributes:
        job_id: 任务标识。
        doc_id: 文档标识。
        kb_id: 目标知识库。
        filename: 清理后的名称。
        status: 排队、执行或终态。
        stage: 最近实际进入的处理阶段。
        progress: 实际阶段和已完成批次推导的单调进度。
        chunks_count: 仅成功时记录已发布块数。
        error: 仅失败时公开安全说明。
        created_at: 持久入队时间。
        updated_at: 最近状态变化时间，不随心跳伪增进度。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    job_id: str = Field(pattern=r"^job_[0-9a-f]{32}$")
    doc_id: UUID
    kb_id: str = Field(pattern=KB_ID_PATTERN)
    filename: str = Field(min_length=1, max_length=255)
    status: IngestionJobStatus
    stage: IngestionStage
    progress: float = Field(ge=0, le=1, allow_inf_nan=False)
    chunks_count: int | None = Field(default=None, gt=0)
    error: IngestionJobFailure | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def validate_terminal_state(self) -> Self:
        """拒绝损坏的成功标记、未分类失败或反向时间。

        Returns:
            状态一致的任务。
        """
        if self.updated_at < self.created_at:
            raise ValueError("任务更新时间不能早于创建时间")
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("失败任务必须且仅能包含安全错误")
        if self.status == "succeeded":
            if self.stage != "indexing" or self.progress != 1:
                raise ValueError("成功任务必须完成索引阶段")
            if self.chunks_count is None:
                raise ValueError("成功任务必须包含已发布块数")
        elif self.chunks_count is not None or self.progress == 1:
            raise ValueError("未成功的任务不得声称已发布")
        return self


class IngestionJobClaim(BaseModel):
    """内部执行权快照；HTTP仅返回job，不能返回执行资料。

    Attributes:
        job: 当前公开进度。
        submission: 用于核对源文件的原始提交资料。
        lease_token: 此次认领专属的写入栅栏。
        attempt: 持久化执行次数。
        recovered: 是否接管了曾经running的过期任务。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    job: IngestionJob
    submission: IngestionJobSubmission = Field(exclude=True, repr=False)
    lease_token: UUID = Field(exclude=True, repr=False)
    attempt: int = Field(ge=1)
    recovered: bool


class IngestionProgress(BaseModel):
    """流水线实际阶段，不传模拟百分比或任意错误正文。

    Attributes:
        stage: 当前真实执行阶段。
        completed_batches: 已成功写入的向量批次数。
        total_batches: 完成分块后计算的向量总批次数。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    stage: IngestionStage
    completed_batches: int = Field(default=0, ge=0)
    total_batches: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_batches(self) -> Self:
        """只在向量化阶段接受真实批次数。

        Returns:
            计数一致的冻结阶段对象。
        """
        if self.stage == "embedding":
            if self.total_batches <= 0:
                raise ValueError("向量化阶段需要真实总批次数")
            if self.completed_batches > self.total_batches:
                raise ValueError("完成批次不能超过总批次")
        elif self.completed_batches != 0 or self.total_batches != 0:
            raise ValueError("非向量化阶段不接收批次数")
        return self
