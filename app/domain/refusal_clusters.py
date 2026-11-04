# -*- coding: utf-8 -*-
"""具有明确窗口、样本覆盖与快照分页的拒答聚类契约。"""

from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.domain.insights import RefusalItem


class RefusalClusterSample(BaseModel):
    """同一数据库快照中的有限样本与实际总量。

    Attributes:
        items: 按提交时间倒序的最新样本。
        total: 过滤后窗口内完整拒答数量。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    items: tuple[RefusalItem, ...]
    total: int = Field(ge=0)


class RefusalCluster(BaseModel):
    """相似问题的确定性分组，不附带用户身份。

    Attributes:
        cluster_id: 本快照成员集合的稳定摘要标识。
        representative_question: 实际成员中平均相似度最高的原问题。
        count: 本组实际样本数。
        kb_ids: 成员实际检索范围的去重并集。
        sample_turn_ids: 至多五条真实样例轮次。
        last_asked_at: 本组最近正式提交时间。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    cluster_id: str
    representative_question: str
    count: int = Field(ge=1)
    kb_ids: tuple[str, ...]
    sample_turn_ids: tuple[UUID, ...] = Field(max_length=5)
    last_asked_at: AwareDatetime


class RefusalClusterPage(BaseModel):
    """聚类分页与完整覆盖声明，截断不能解释为全量统计。

    Attributes:
        clusters: 当前页分组。
        next_cursor: 绑定条件及快照的下一页游标。
        total_clusters: 当前样本产生的全部组数。
        total_refusals: 时间与知识库筛选后的真实总数。
        clustered_refusals: 实际参与归组的最新样本数。
        truncated: 窗口总数超过样本上限。
        period_days: UTC日历窗口天数。
        window_started_at: 窗口起点，含边界。
        window_ended_at: 首次请求截止，分页固定。
        similarity_threshold: 余弦阈值，不是准确率。
        embedding_model: 当前部署模型名，不代表固定权重版本。
        algorithm: 实际确定性算法版本。
        coverage: 只覆盖active已提交明确拒答。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    clusters: tuple[RefusalCluster, ...]
    next_cursor: str | None = None
    total_clusters: int = Field(ge=0)
    total_refusals: int = Field(ge=0)
    clustered_refusals: int = Field(ge=0, le=500)
    truncated: bool
    period_days: int = Field(ge=1, le=365)
    window_started_at: AwareDatetime
    window_ended_at: AwareDatetime
    similarity_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    embedding_model: str
    algorithm: Literal["greedy_complete_link_cosine_v1"] = "greedy_complete_link_cosine_v1"
    coverage: Literal["active_committed_explicit_refusals"] = "active_committed_explicit_refusals"


class RefusalClusterPosition(BaseModel):
    """只能在短时同一筛选及同一成员快照中使用的游标。

    Attributes:
        as_of: 首次请求截止时间。
        offset: 下一页起点。
        filters: 条件摘要。
        snapshot: 聚类完整结果摘要。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    as_of: AwareDatetime
    offset: int = Field(ge=1, le=499, strict=True)
    filters: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot: str = Field(pattern=r"^[0-9a-f]{64}$")
