# -*- coding: utf-8 -*-
"""问答滑动窗口的公开额度，不冒充日额度、余额或模型用量。"""

from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator


AdmissionStatus = Literal[
    "admitted", "rate_limited", "queue_full", "queue_timeout",
    "cancelled", "error",
]


class AdmissionDecision(BaseModel):
    """准入的实际结果，不包含问题正文或模型推断。

    Attributes:
        status: 入场、运行拒绝或协作取消。
        queued: 是否实际进入过等待队列。
        queue_wait_ms: 从进入队列到结果确定的单调时钟耗时。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: AdmissionStatus
    queued: bool
    queue_wait_ms: float = Field(ge=0, allow_inf_nan=False)


class QuestionQuota(BaseModel):
    """当前滑动窗口中实际还能接收的提问次数。

    Attributes:
        limit: 单个用户的窗口接收上限。
        remaining: 尚未接收的次数。
        window_seconds: 滑动窗口秒数。
        retry_after: 额度耗尽时最早可再次接收的等待秒数，否则为0。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    limit: int = Field(ge=1)
    remaining: int = Field(ge=0)
    window_seconds: int = Field(ge=1)
    retry_after: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_remaining(self) -> Self:
        """拒绝超出上限或仍有余额却要求等待的矛盾状态。

        Returns:
            状态一致的公开额度。
        """
        if self.remaining > self.limit:
            raise ValueError("剩余额度不能超过上限")
        if self.remaining > 0 and self.retry_after != 0:
            raise ValueError("未耗尽额度时不应要求等待")
        return self
