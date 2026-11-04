# -*- coding: utf-8 -*-
"""定义可由 checkpoint 序列化的本轮 Agent 图状态。"""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """保存正式输入、本轮执行记录与质量判定，不跨轮累加。"""

    conversation_id: str
    current_question: str
    history: list[dict[str, str]]
    messages: list[dict[str, object]]
    pending_tool_calls: list[dict[str, object]]
    evidence: list[dict[str, object]]
    retrieved: list[dict[str, object]]
    approved_evidence: list[dict[str, object]]
    attempted_queries: list[str]
    retrieval_calls: int
    repair_attempted: bool
    limit_reached: bool
    draft_answer: str
    answer_mode: str
    final_answer: str
    final_citations: list[dict[str, object]]
    # 证据被判为不足仍作答，或审核未通过而按部分依据发布。
    answer_partial: bool
    # 最近一次引用校验通过但审核未通过的回答，审核预算用尽时按部分依据发布。
    rejected_answer: str
    rejected_citations: list[dict[str, object]]
    evidence_assessments: int
    answer_assessments: int
    evidence_assessment: dict[str, object]
    answer_assessment: dict[str, object]
    next_query: str | None
    quality_issues: list[str]
    deadline: float
    execution_budget: dict[str, object]
    seed_bundle: dict[str, object] | None
    kb_ids: list[str] | None
    retrieval_strategy: str
    top_k: int
    initial_query: str | None
    sub_queries: list[str]
