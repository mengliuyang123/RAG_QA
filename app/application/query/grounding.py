# -*- coding: utf-8 -*-
"""将答案中的稳定引用编号映射到裁判批准的来源。"""

import re

from app.domain.agent_quality import EvidenceItem
from app.domain.conversations import TurnCitation


_CITATION_PATTERN = re.compile(r"\[([0-9]+)\]")


def validate_grounded_citations(
    answer: str, evidence: list[EvidenceItem],
) -> list[TurnCitation]:
    """验证所有引用存在且获批，不能按过滤后的列表下标解释编号。

    Args:
        answer: 尚未发布的草稿。
        evidence: 本轮允许生成使用的批准证据。

    Returns:
        按正文首次出现顺序排列的可信引用。

    Raises:
        ValueError: 引用缺失、未批准或注册表编号冲突。
    """
    registry = {item.number: item for item in evidence}
    if len(registry) != len(evidence):
        raise ValueError("批准证据存在重复编号")
    numbers = []
    for match in _CITATION_PATTERN.finditer(answer):
        number = int(match.group(1))
        if number not in registry:
            raise ValueError("回答引用未批准的证据")
        if number not in numbers:
            numbers.append(number)
    if not numbers:
        raise ValueError("知识回答缺少证据引用")
    citations = []
    for number in numbers:
        item = registry[number]
        citations.append(TurnCitation(
            position=number, doc_id=item.doc_id, filename=item.filename,
            page=item.page, chunk_text=item.text,
            chunk_id=item.chunk_id, kb_id=item.kb_id,
            doc_updated_at=item.doc_updated_at,
        ))
    return citations
