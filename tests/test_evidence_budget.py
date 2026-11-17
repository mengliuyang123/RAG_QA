# -*- coding: utf-8 -*-
"""父块上下文超出输入预算时整条裁剪证据（retrieval-evidence-context 规格）。"""

import pytest

from app.application.query.context_budget import PromptBudgetGuard, fit_evidence_to_budget
from app.application.query.generator import AnswerGenerator
from app.application.query.prompting import PromptBuilder
from app.domain.agent_quality import EvidenceItem
from app.domain.errors import PromptContextTooLargeError


def item(number: int, size: int) -> EvidenceItem:
    return EvidenceItem(
        number=number, chunk_id=f"c{number}", doc_id=f"d{number}", filename=f"{number}.md",
        page=None, text="命中", context="上下文" + "字" * size,
    )


def limit_chars(limit: int):
    def check(items):
        if sum(len(i.context or i.text) for i in items) > limit:
            raise PromptContextTooLargeError("too large")
    return check


def test_within_budget_keeps_everything() -> None:
    evidence = [item(1, 10), item(2, 10)]

    assert fit_evidence_to_budget(evidence, limit_chars(1000)) == evidence


def test_over_budget_drops_lowest_ranked_whole_items() -> None:
    evidence = [item(3, 100), item(1, 100), item(5, 100), item(2, 100)]

    kept = fit_evidence_to_budget(evidence, limit_chars(250))

    assert [i.number for i in kept] == [1, 2]
    assert all(i.context == "上下文" + "字" * 100 for i in kept)


def test_single_item_still_too_large_raises() -> None:
    with pytest.raises(PromptContextTooLargeError):
        fit_evidence_to_budget([item(1, 500), item(2, 500)], limit_chars(100))


class _CharCounter:
    def count_messages(self, messages, **_kwargs) -> int:
        return sum(len(str(m.get("content", ""))) for m in messages)

    def count_text(self, text: str) -> int:
        return len(text)


class _Provider:
    def __init__(self) -> None:
        self.messages = []

    def generate(self, messages):
        self.messages.append(messages)
        return "回答[1]"


def test_generator_trims_evidence_before_calling_model() -> None:
    provider = _Provider()
    guard = PromptBudgetGuard(_CharCounter(), max_input_tokens=1200)
    generator = AnswerGenerator(PromptBuilder(), provider, prompt_guard=guard)

    generator.generate_grounded("问题", [], [item(1, 300), item(2, 300), item(3, 300), item(4, 300)])

    prompt = provider.messages[0][-1]["content"]
    assert "[1]\n" in prompt
    assert "[4]\n" not in prompt
    assert ("字" * 300) in prompt
