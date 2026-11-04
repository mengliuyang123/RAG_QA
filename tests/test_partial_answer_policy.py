# -*- coding: utf-8 -*-
"""部分依据标记必须与回答是否完整通过审核一致，且仍要求已校验引用。"""

import pytest
from pydantic import ValidationError

from app.domain.answer_metadata import AnswerMetadata
from app.domain.conversations import AgentResult
from app.domain.conversations import AnswerMode
from app.domain.conversations import TurnCitation


def _result(
    *, partial: bool, grounding: str, citations: bool = True,
) -> AgentResult:
    return AgentResult(
        answer="只覆盖部分问题的回答[1]",
        answer_mode=AnswerMode.RETRIEVAL,
        retrieval_calls=1,
        citations=[TurnCitation(
            position=1, doc_id="d", filename="d.md", page=None,
            chunk_text="只支持一部分事实",
        )] if citations else [],
        partial=partial,
        metadata=AnswerMetadata(
            lane="agent", outcome="answered", grounding=grounding,
        ),
    )


def test_partial_answer_is_published_with_partial_grounding() -> None:
    """未完整通过审核的回答只能标为 partial 发布。"""
    assert _result(partial=True, grounding="partial").metadata.grounding == "partial"


@pytest.mark.parametrize("partial,grounding", [
    (False, "partial"),
    (True, "grounded"),
    (True, "ungrounded"),
])
def test_grounding_must_match_review_result(partial: bool, grounding: str) -> None:
    """完整回答不能降级标记，部分回答也不能冒充完整依据。

    Args:
        partial: 回答是否未完整通过审核。
        grounding: 声明的依据类别。
    """
    with pytest.raises(ValidationError):
        _result(partial=partial, grounding=grounding)


def test_partial_answer_still_requires_citations() -> None:
    """部分依据不是免引用发布的许可。"""
    with pytest.raises(ValidationError):
        _result(partial=True, grounding="partial", citations=False)
