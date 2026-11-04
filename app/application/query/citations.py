# -*- coding: utf-8 -*-
"""把回答中的引用编号映射为可信的检索来源。"""

import logging
import re

from app.domain.answers import Citation
from app.domain.retrieval import RetrievalResult


CITATION_PATTERN = re.compile(r"\[([0-9]+)\]")
logger = logging.getLogger(__name__)


def extract_citations(
    answer: str,
    results: list[RetrievalResult],
) -> list[Citation]:
    """按首次出现顺序提取回答中的有效引用。

    Args:
        answer: 语言模型生成的完整回答。
        results: 已按提示词编号顺序排列的检索结果。

    Returns:
        去重后的可信来源列表。来源元数据只取自检索
        结果，不采信回答自行生成的来源信息。
    """
    citations: list[Citation] = []
    seen_numbers: set[int] = set()
    invalid_numbers: list[int] = []
    seen_invalid_numbers: set[int] = set()

    for match in CITATION_PATTERN.finditer(answer):
        citation_number = int(match.group(1))
        if citation_number in seen_numbers:
            continue
        if citation_number < 1 or citation_number > len(results):
            if citation_number not in seen_invalid_numbers:
                invalid_numbers.append(citation_number)
                seen_invalid_numbers.add(citation_number)
            continue

        seen_numbers.add(citation_number)
        result = results[citation_number - 1]
        citations.append(
            Citation(
                position=citation_number,
                chunk_id=result.chunk_id, kb_id=result.kb_id,
                doc_updated_at=result.doc_updated_at,
                doc_id=result.doc_id,
                filename=result.filename,
                page=result.page,
                chunk_text=result.text,
            )
        )

    if invalid_numbers:
        logger.warning(
            "忽略超出上下文范围的引用编号",
            extra={"invalid_citation_numbers": invalid_numbers},
        )

    return citations
