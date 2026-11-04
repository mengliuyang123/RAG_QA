# -*- coding: utf-8 -*-
"""基于真实向量的确定性完整链接归组，不依赖词频或生成主题标签。"""

import hashlib
import math

from app.application.errors import EmbeddingResponseError
from app.domain.insights import RefusalItem
from app.domain.refusal_clusters import RefusalCluster


def normalize_vector(vector: list[float]) -> tuple[float, ...]:
    """拒绝空、非数、非有限或零向量，再单位化。

    Args:
        vector: 模型返回的单条向量。

    Returns:
        经过校验的单位向量。
    """
    if not vector:
        raise EmbeddingResponseError("聚类向量为空")
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingResponseError("聚类向量包含非数值")
        if not math.isfinite(value):
            raise EmbeddingResponseError("聚类向量包含非有限数值")
    norm = math.hypot(*vector)
    if norm == 0 or not math.isfinite(norm):
        raise EmbeddingResponseError("聚类向量范数无效")
    return tuple(value / norm for value in vector)


def cluster_refusals(
    items: tuple[RefusalItem, ...], vectors: list[tuple[float, ...]],
    threshold: float,
) -> tuple[RefusalCluster, ...]:
    """每个新成员必须与组内全部成员达到阈值，避免链式跨主题合并。

    Args:
        items: 稳定时间/ID倒序样本。
        vectors: 同顺序已单位化的真实语义向量。
        threshold: 余弦阈值，0–1。

    Returns:
        按数量、最新时间与稳定标识排序的组。
    """
    if len(items) != len(vectors):
        raise EmbeddingResponseError("聚类向量数量不一致")
    if not vectors:
        return ()
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise EmbeddingResponseError("聚类向量维度不一致")
    similarities = []
    for left in vectors:
        row = []
        for right in vectors:
            row.append(math.fsum(a * b for a, b in zip(left, right)))
        similarities.append(row)
    groups: list[list[int]] = []
    for index in range(len(items)):
        selected = None
        best_similarity = -1.0
        for group in groups:
            minimum = min(similarities[index][member] for member in group)
            if minimum >= threshold and minimum > best_similarity:
                selected = group
                best_similarity = minimum
        if selected is None:
            groups.append([index])
        else:
            selected.append(index)
    result = []
    for group in groups:
        representative = max(
            group,
            key=lambda index: math.fsum(similarities[index][other] for other in group),
        )
        kb_ids = set()
        identifiers = []
        for member in group:
            kb_ids.update(items[member].kb_ids)
            identifiers.append(str(items[member].turn_id))
        digest = hashlib.sha256("\n".join(sorted(identifiers)).encode()).hexdigest()
        result.append(RefusalCluster(
            cluster_id="cl_" + digest[:24],
            representative_question=items[representative].question,
            count=len(group), kb_ids=tuple(sorted(kb_ids)),
            sample_turn_ids=tuple(items[index].turn_id for index in group[:5]),
            last_asked_at=max(items[index].created_at for index in group),
        ))
    result.sort(key=lambda group: (
        -group.count, -group.last_asked_at.timestamp(), group.cluster_id,
    ))
    return tuple(result)
