# -*- coding: utf-8 -*-
"""从真实执行记录投影管理员检索明细，不重新检索或判断。"""

from app.application.errors import ConversationAgentError
from app.domain.conversations import AgentResult
from app.domain.query_debug import RetrievalDebugItem


def retrieval_debug(result: AgentResult) -> tuple[RetrievalDebugItem, ...]:
    """保留每次召回排名，将最终批准与实际引用分别表示。

    Args:
        result: 统一编排器已完成验证的结果。

    Returns:
        按检索次数和原排名排序的明细；没有检索则为空。
    """
    bundle = result.execution
    if bundle is None:
        raise ConversationAgentError("调试结果缺少实际执行记录")
    evidence = {item.number: item for item in bundle.evidence}
    cited = {item.position for item in result.citations}
    rows = []
    for observation in bundle.retrieved:
        item = evidence[observation.position]
        rows.append(RetrievalDebugItem(
            **observation.model_dump(), doc_id=item.doc_id,
            filename=item.filename, page=item.page, kb_id=item.kb_id,
            chunk_text=item.text,
            approved=item.number in bundle.approved_numbers,
            cited=item.number in cited,
        ))
    return tuple(rows)
