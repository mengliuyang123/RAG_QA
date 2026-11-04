# -*- coding: utf-8 -*-
"""证据统计四种口径的不变式与旧记录兼容性。"""

import pytest
from pydantic import ValidationError

from app.domain.answer_metadata import AnswerMetadata
from app.domain.answer_metadata import EvidenceStats


def test_relevant_but_insufficient_evidence_is_representable() -> None:
    """相关但不足以回答的块只出现在 relevant 中，不再表现为全零。"""
    stats = EvidenceStats(used=0, excluded=0, retrieved=1, relevant=1)
    assert stats.retrieved == 1
    assert stats.relevant == 1


def test_legacy_stats_without_new_fields_still_load() -> None:
    """旧记录只有 used 和 excluded，两个新字段保持 None。"""
    stored = AnswerMetadata.model_validate_json(
        '{"lane": "fast", "outcome": "answered", "grounding": "grounded",'
        ' "evidence_stats": {"used": 2, "excluded": 1}}'
    )
    assert stored.evidence_stats.retrieved is None
    assert stored.evidence_stats.relevant is None


@pytest.mark.parametrize(
    "fields",
    [
        {"retrieved": 3},
        {"relevant": 1},
        {"excluded": 1, "retrieved": 3, "relevant": 1},
    ],
)
def test_inconsistent_totals_are_rejected(fields: dict[str, int]) -> None:
    """新口径必须成对出现，且检索数等于相关数加排除数。

    Args:
        fields: 违反不变式的字段组合。
    """
    with pytest.raises(ValidationError):
        EvidenceStats(**fields)
