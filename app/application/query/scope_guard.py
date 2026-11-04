# -*- coding: utf-8 -*-
"""防止通道交接把局部范围误当作全库，或带入不属于当前范围的正文。"""

from app.application.errors import ConversationAgentError
from app.domain.execution import EvidenceBundle
from app.domain.knowledge_bases import validate_kb_ids


def validate_handoff_scope(
    bundle: EvidenceBundle, kb_ids: tuple[str, ...] | None,
) -> None:
    """要求种子的全部来源和旧查询都能由当前服务端范围解释。

    Args:
        bundle: 内部传递的证据和预算。
        kb_ids: 本通道实际使用且模型无法更改的范围。

    Returns:
        None。

    Raises:
        ConversationAgentError: 有范围种子丢失范围，或当前范围排除了旧来源。
        ValueError: 范围类型或ID非法。
    """
    validate_kb_ids(kb_ids)
    if kb_ids is None:
        if any(attempt.kb_ids for attempt in bundle.budget.attempts):
            raise ConversationAgentError("有范围的种子必须显式保留执行范围")
        return
    selected = set(kb_ids)
    for attempt in bundle.budget.attempts:
        if not attempt.kb_ids or not set(attempt.kb_ids).issubset(selected):
            raise ConversationAgentError("已有检索记录超出当前执行范围")
    for item in bundle.evidence:
        if item.kb_id not in selected:
            raise ConversationAgentError("已有证据超出当前执行范围")
