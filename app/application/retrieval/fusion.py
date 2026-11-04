# -*- coding: utf-8 -*-
"""不依赖原始分数尺度的检索排名融合算法。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.retrieval import RetrievalResult


@dataclass
class _FusionEntry:
    """保存一个文本块的代表结果、累计分数和最佳单路排名。"""

    result: RetrievalResult
    score: float
    best_rank: int


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievalResult]],
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """按 chunk_id 融合多路排名，并返回确定性排序结果。

    Args:
        rankings: 每一路按相关性降序排列的检索结果。
        rrf_k: 施加给每个排名位置的 RRF 平滑常数。

    Returns:
        按累计 RRF 分数、最佳排名和文本块 ID 稳定排序的结果。
    """
    entries: dict[str, _FusionEntry] = {}
    for ranking in rankings:
        seen_chunk_ids: set[str] = set()
        for rank, result in enumerate(ranking, start=1):
            if result.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(result.chunk_id)

            contribution = 1.0 / (rrf_k + rank)
            entry = entries.get(result.chunk_id)
            if entry is None:
                entries[result.chunk_id] = _FusionEntry(
                    result=result,
                    score=contribution,
                    best_rank=rank,
                )
            else:
                entry.score += contribution
                entry.best_rank = min(entry.best_rank, rank)

    ordered_entries = sorted(
        entries.values(),
        key=lambda entry: (
            -entry.score,
            entry.best_rank,
            entry.result.chunk_id,
        ),
    )
    return [
        entry.result.model_copy(update={"score": entry.score})
        for entry in ordered_entries
    ]
