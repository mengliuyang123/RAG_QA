# -*- coding: utf-8 -*-
"""检索次数、截止时间和稳定证据编号必须跨通道连续。"""

import pytest
from pydantic import ValidationError

from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem
from app.domain.errors import DuplicateRetrievalError
from app.domain.errors import ExecutionDeadlineError
from app.domain.errors import RetrievalBudgetExceededError
from app.domain.execution import EvidenceBundle
from app.domain.execution import ExecutionBudget
from app.domain.execution import RetrievalAttempt


def test_expansion_and_agent_share_three_calls_without_resetting_time() -> None:
    """扩大范围和补查只能消费剩余次数，第三次充分后仍可回答。"""
    original = ExecutionBudget(deadline=100.0)
    budget = original.record_retrieval(
        RetrievalAttempt(query="A", kb_ids=("ops",)), now=1.0,
    )
    budget = budget.record_retrieval(
        RetrievalAttempt(query="A", kb_ids=("ops", "dev")), now=2.0,
    )
    budget = budget.record_retrieval(
        RetrievalAttempt(query="B 的配置", kb_ids=("ops", "dev")), now=3.0,
    )
    assert original.retrieval_calls == 0
    assert budget.retrieval_calls == 3
    assert budget.deadline == 100.0
    budget.ensure_time(now=4.0)
    with pytest.raises(RetrievalBudgetExceededError):
        budget.record_retrieval(RetrievalAttempt(query="第四次"), now=4.0)


def test_same_query_and_same_scope_cannot_be_retrieved_twice() -> None:
    """大小写空白和范围顺序不同仍算同一次查询。"""
    budget = ExecutionBudget(deadline=100.0).record_retrieval(
        RetrievalAttempt(query="  LangGraph   SQLite ", kb_ids=("ops", "dev")),
        now=1.0,
    )
    with pytest.raises(DuplicateRetrievalError):
        budget.record_retrieval(
            RetrievalAttempt(query="langgraph sqlite", kb_ids=("dev", "ops")),
            now=2.0,
        )
    assert budget.retrieval_calls == 1


def test_expired_deadline_stops_even_with_remaining_calls() -> None:
    """截止时间到达时不能借仍有次数继续调用外部服务。"""
    budget = ExecutionBudget(deadline=5.0)
    with pytest.raises(ExecutionDeadlineError):
        budget.record_retrieval(RetrievalAttempt(query="问题"), now=5.0)
    with pytest.raises(ExecutionDeadlineError):
        budget.ensure_time(now=6.0)


def _item(number: int, chunk: str) -> EvidenceItem:
    return EvidenceItem(
        number=number, chunk_id=chunk, doc_id="doc", filename="a.md",
        page=None, text=f"证据{number}",
    )


def _budget() -> ExecutionBudget:
    return ExecutionBudget(deadline=100.0, evidence_assessments=1).record_retrieval(
        RetrievalAttempt(query="问题"), now=1.0,
    )


def test_evidence_bundle_roundtrip_preserves_sparse_numbers_and_budget() -> None:
    """跨通道序列化不能按过滤后的数组下标重编号。"""
    bundle = EvidenceBundle(
        evidence=(_item(2, "a"), _item(5, "b")),
        approved_numbers=(5,), budget=_budget(),
        assessment=EvidenceAssessment(
            sufficient=True, relevant_numbers=[5],
            missing_information=[], next_query=None,
        ),
    )
    restored = EvidenceBundle.model_validate_json(bundle.model_dump_json())
    assert [item.number for item in restored.evidence] == [2, 5]
    assert restored.approved_numbers == (5,)
    assert restored.budget.retrieval_calls == 1
    assert restored.budget.deadline == 100.0


@pytest.mark.parametrize("evidence,approved", [
    ((_item(1, "a"), _item(1, "b")), (1,)),
    ((_item(1, "a"), _item(2, "a")), (1,)),
    ((_item(1, "a"),), (9,)),
    ((_item(1, "a"),), (1, 1)),
])
def test_bundle_rejects_ambiguous_or_unknown_evidence(
    evidence: tuple[EvidenceItem, ...], approved: tuple[int, ...],
) -> None:
    """重复编号、重复块和虚构批准编号不能进入升级种子。

    Args:
        evidence: 有冲突的候选证据。
        approved: 有冲突或越界的批准编号。
    """
    with pytest.raises(ValidationError):
        EvidenceBundle(evidence=evidence, approved_numbers=approved, budget=_budget())


def test_bundle_rejects_assessment_inconsistent_with_approved_set() -> None:
    """传入的裁判结果与批准集必须是同一版判断。"""
    with pytest.raises(ValidationError):
        EvidenceBundle(
            evidence=(_item(1, "a"), _item(2, "b")),
            approved_numbers=(1,), budget=_budget(),
            assessment=EvidenceAssessment(
                sufficient=True, relevant_numbers=[2],
                missing_information=[], next_query=None,
            ),
        )


def test_budget_cannot_claim_calls_without_attempts() -> None:
    """调用次数和记录不同步时不能恢复出可继续执行的预算。"""
    with pytest.raises(ValidationError):
        ExecutionBudget(deadline=100.0, retrieval_calls=1)


def test_bundle_cannot_approve_evidence_without_assessment() -> None:
    """仅携带候选证据不等于已经获裁判批准。"""
    with pytest.raises(ValidationError):
        EvidenceBundle(
            evidence=(_item(1, "a"),), approved_numbers=(1,),
            budget=_budget(),
        )


def test_recording_retrieval_preserves_shared_quality_budgets() -> None:
    """新增检索不能给证据判断、答案判断和修复恢复额度。"""
    budget = ExecutionBudget(
        deadline=100.0, evidence_assessments=2,
        answer_assessments=1, repair_attempted=True,
    ).record_retrieval(RetrievalAttempt(query="A"), now=1.0)
    restored = ExecutionBudget.model_validate_json(budget.model_dump_json())
    assert restored.evidence_assessments == 2
    assert restored.answer_assessments == 1
    assert restored.repair_attempted is True


def test_seed_assessment_requires_consumed_judgement_budget() -> None:
    """已经判断过的种子不得声称尚未消费判断次数。"""
    with pytest.raises(ValidationError):
        EvidenceBundle(
            evidence=(_item(1, "a"),), approved_numbers=(1,),
            assessment=EvidenceAssessment(
                sufficient=True, relevant_numbers=[1],
                missing_information=[], next_query=None,
            ),
            budget=ExecutionBudget(deadline=100.0).record_retrieval(
                RetrievalAttempt(query="A"), now=1.0,
            ),
        )
