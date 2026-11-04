# -*- coding: utf-8 -*-
"""校验固定能力片段，显式计算案例与策略组合，默认烟雾不跑全量。"""

from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError

from evaluation.datasets import DatasetValidationError
from evaluation.lanes.models import LaneEpisode
from evaluation.lanes.models import LanePolicy
from evaluation.models import CorpusDataset


class LaneDataset(BaseModel):
    """带版本的小型人工标签集。

    Attributes:
        version: 标签协议版本。
        smoke_ids: 按原数据顺序选择的烟雾片段ID。
        episodes: 全部固定能力片段，不是自动扩充的合成问题。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1)
    smoke_ids: tuple[str, ...] = Field(min_length=1)
    episodes: tuple[LaneEpisode, ...] = Field(min_length=1)


def load_lane_dataset(path: Path, corpus: CorpusDataset) -> LaneDataset:
    """执行前校验所有ID和来源，禁止模型调用后才发现错标签。

    Args:
        path: 固定JSON标签路径。
        corpus: 已校验来源及正文哈希的固定语料。

    Returns:
        来源属于当前语料、ID唯一的冻结标签集。
    """
    try:
        dataset = LaneDataset.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as error:
        raise DatasetValidationError("分通道标签无法读取或格式无效") from error
    ids = [episode.id for episode in dataset.episodes]
    if len(set(ids)) != len(ids):
        raise DatasetValidationError("分通道案例ID重复")
    if len(set(dataset.smoke_ids)) != len(dataset.smoke_ids):
        raise DatasetValidationError("烟雾案例ID重复")
    if not set(dataset.smoke_ids).issubset(ids):
        raise DatasetValidationError("烟雾案例ID不存在")
    sources = set(corpus.documents)
    for episode in dataset.episodes:
        for turn in episode.turns:
            if not set(turn.expected_sources).issubset(sources):
                raise DatasetValidationError("分通道预期来源不属于固定语料")
    return dataset


def select_runs(
    dataset: LaneDataset, *, smoke: bool = True,
    policies: tuple[LanePolicy, ...] | None = None,
) -> tuple[tuple[LaneEpisode, LanePolicy], ...]:
    """知识案例比较三种策略，非知识决策仅评估自动路由。

    Args:
        dataset: 已校验标签。
        smoke: 默认仅选择明确标记的小样本。
        policies: 只保留这些策略，None 表示不过滤；大样本可只跑 auto 省去对照组。

    Returns:
        顺序固定的案例和策略组合；总轮次需按turns计算。
    """
    runs = []
    for episode in dataset.episodes:
        if smoke and episode.id not in dataset.smoke_ids:
            continue
        candidates: tuple[LanePolicy, ...] = ("auto",)
        if all(turn.requires_retrieval for turn in episode.turns):
            candidates = ("auto", "fast_only", "agent")
        for policy in candidates:
            if policies is None or policy in policies:
                runs.append((episode, policy))
    return tuple(runs)
