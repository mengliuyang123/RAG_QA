# -*- coding: utf-8 -*-
"""保存可在 Fast 与 Agent 之间无损交接的执行状态。"""

import math
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem
from app.domain.errors import DuplicateRetrievalError
from app.domain.errors import ExecutionDeadlineError
from app.domain.errors import RetrievalBudgetExceededError
from app.domain.query_debug import RetrievalObservation


class RetrievalAttempt(BaseModel):
    """一次实际查询及其范围，区分扩查和重复调用。

    Attributes:
        query: 实际检索文本。
        kb_ids: 实际选择的范围；空表示尚未分库的旧全局知识库。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    query: str = Field(min_length=1)
    kb_ids: tuple[str, ...] = ()

    @field_validator("kb_ids", mode="before")
    @classmethod
    def accept_json_array(cls, value: object) -> object:
        """仅转换数组容器，不容忍非法 ID 类型。

        Args:
            value: 序列化后的范围。

        Returns:
            不可变范围或原始非法值。
        """
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        """拒绝无效查询与重复范围，防止去重键含义不明确。

        Returns:
            可规范化比较的查询记录。
        """
        if not self.query.strip():
            raise ValueError("检索查询不能为空白")
        if len(set(self.kb_ids)) != len(self.kb_ids):
            raise ValueError("检索范围不得重复")
        for kb_id in self.kb_ids:
            if not kb_id.strip() or kb_id != kb_id.strip():
                raise ValueError("检索范围 ID 不得含空白")
        return self

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        """生成不受查询空白、大小写或范围排列影响的比较键。

        Returns:
            规范查询和排序后的范围。
        """
        return " ".join(self.query.split()).casefold(), tuple(sorted(self.kb_ids))


class ExecutionBudget(BaseModel):
    """以不可变快照记录整轮已消费次数和截止时间。

    Attributes:
        deadline: 同一执行进程中单调时钟的截止值，不跨进程恢复执行。
        retrieval_calls: 已经消费的检索次数。
        attempts: 与次数一一对应的实际查询。
        evidence_assessments: 已消费的证据判断次数。
        answer_assessments: 已消费的答案审核次数。
        repair_attempted: 本轮是否已使用唯一修复机会。
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False,
    )

    deadline: float
    retrieval_calls: int = Field(default=0, ge=0, le=3)
    attempts: tuple[RetrievalAttempt, ...] = Field(default=(), max_length=3)
    evidence_assessments: int = Field(default=0, ge=0, le=3)
    answer_assessments: int = Field(default=0, ge=0, le=2)
    repair_attempted: bool = False

    @field_validator("attempts", mode="before")
    @classmethod
    def accept_json_array(cls, value: object) -> object:
        """接收 JSON 快照且保留严格的记录元素校验。

        Args:
            value: 序列化查询列表。

        Returns:
            不可变查询记录。
        """
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_accounting(self) -> Self:
        """次数和去重记录必须能互相证明，不能靠升级恢复额度。

        Returns:
            一致的预算快照。
        """
        if self.retrieval_calls != len(self.attempts):
            raise ValueError("检索次数必须与查询记录一致")
        keys = [attempt.key for attempt in self.attempts]
        if len(keys) != len(set(keys)):
            raise ValueError("预算含重复检索记录")
        return self

    def ensure_time(self, now: float) -> None:
        """检查共享截止时间，但不禁止已用完检索额度后回答。

        Args:
            now: 当前单调时钟值。

        Returns:
            None。

        Raises:
            ExecutionDeadlineError: 截止时间到达或时钟不合法。
        """
        if not math.isfinite(now) or now >= self.deadline:
            raise ExecutionDeadlineError("整轮时间预算已耗尽")

    def record_retrieval(
        self, attempt: RetrievalAttempt, *, now: float,
    ) -> "ExecutionBudget":
        """执行前消费一次额度，返回保留原截止时间的新快照。

        Args:
            attempt: 本次将执行的查询及实际范围。
            now: 本轮单调时钟值。

        Returns:
            消费一次检索额度后的新预算，原实例保持不变。

        Raises:
            ExecutionDeadlineError: 已超时。
            RetrievalBudgetExceededError: 已经执行三次检索。
            DuplicateRetrievalError: 查询及其范围已经执行过。
        """
        self.ensure_time(now)
        if self.retrieval_calls >= 3:
            raise RetrievalBudgetExceededError("整轮检索额度已耗尽")
        if any(previous.key == attempt.key for previous in self.attempts):
            raise DuplicateRetrievalError("不能重复执行相同查询和范围")
        return self.model_copy(update={
            "retrieval_calls": self.retrieval_calls + 1,
            "attempts": (*self.attempts, attempt),
        })


class EvidenceBundle(BaseModel):
    """携带候选证据、裁判结果和预算升级到另一个通道。

    Attributes:
        evidence: 保留原编号的完整候选证据。
        approved_numbers: 本次判断允许进入生成的原始编号。
        assessment: 已有裁判判断；尚未判断时为空。
        budget: 与这些检索结果对应的共享执行预算。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    evidence: tuple[EvidenceItem, ...] = ()
    approved_numbers: tuple[int, ...] = ()
    assessment: EvidenceAssessment | None = None
    budget: ExecutionBudget
    retrieved: tuple[RetrievalObservation, ...] = Field(default=(), max_length=60)

    @field_validator("evidence", "approved_numbers", "retrieved", mode="before")
    @classmethod
    def accept_json_array(cls, value: object) -> object:
        """将序列化数组恢复为不可变集合，不重编号。

        Args:
            value: 候选证据或批准编号数组。

        Returns:
            保持原始元素的 tuple。
        """
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_registry(self) -> Self:
        """防止编号碰撞、来源冲突和不同版本裁判结果混用。

        Returns:
            可安全交接的证据注册表。
        """
        numbers = [item.number for item in self.evidence]
        chunk_ids = [item.chunk_id for item in self.evidence]
        if len(numbers) != len(set(numbers)):
            raise ValueError("证据编号不得重复")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("同一块不得注册多个编号")
        if len(self.approved_numbers) != len(set(self.approved_numbers)):
            raise ValueError("批准编号不得重复")
        if not set(self.approved_numbers).issubset(numbers):
            raise ValueError("批准编号不存在于证据注册表")
        if self.evidence and self.budget.retrieval_calls == 0:
            raise ValueError("未执行检索的预算不得携带检索证据")
        attempts = self.budget.attempts
        if attempts and all(attempt.kb_ids for attempt in attempts):
            searched_kbs = set()
            for attempt in attempts:
                searched_kbs.update(attempt.kb_ids)
            for item in self.evidence:
                if item.kb_id not in searched_kbs:
                    raise ValueError("证据不属于已执行检索的知识库范围")
        if self.approved_numbers and self.assessment is None:
            raise ValueError("没有裁判结果时不得携带批准编号")
        if self.assessment is not None:
            if self.budget.evidence_assessments == 0:
                raise ValueError("已有裁判结果必须消费判断预算")
            if set(self.assessment.relevant_numbers) != set(self.approved_numbers):
                raise ValueError("裁判判断与批准集合不是同一版结果")
        registry = {item.number: item.chunk_id for item in self.evidence}
        observed = set()
        for item in self.retrieved:
            if item.attempt > self.budget.retrieval_calls:
                raise ValueError("召回观察不能超过实际检索次数")
            if registry.get(item.position) != item.chunk_id:
                raise ValueError("召回观察与证据编号不一致")
            key = (item.attempt, item.rank)
            if key in observed:
                raise ValueError("单次召回排名不能重复")
            observed.add(key)
        return self
