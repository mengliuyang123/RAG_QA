# -*- coding: utf-8 -*-
"""保证正文编号可以跨 API、流式及历史找到对应来源。"""

from uuid import uuid4

from app.application.query.citations import extract_citations
from app.domain.conversations import AnswerMode
from app.domain.conversations import ConversationQueryResult
from app.domain.conversations import TurnCitation
from app.domain.retrieval import RetrievalResult
from app.interfaces.http.routes.conversations import _query_response


def test_single_turn_citations_keep_original_sparse_positions() -> None:
    """引用顺序与检索顺序不同时仍保留原编号。"""
    results = [RetrievalResult(
        chunk_id=f"doc:{index}", doc_id="doc", text=f"证据{index}",
        filename="a.md", page=None, chunk_index=index, score=0.8,
    ) for index in range(5)]
    citations = extract_citations("先看第五项[5]，再看第二项[2]", results)
    assert [item.position for item in citations] == [5, 2]


def test_conversation_response_keeps_positions_instead_of_list_indices() -> None:
    """前端能将正文 [5] 对应到编号 5 而不是列表第五项。"""
    result = ConversationQueryResult(
        conversation_id=uuid4(), turn_id=uuid4(), answer="答案[5][2]",
        answer_mode=AnswerMode.RETRIEVAL, retrieval_calls=2, latency_ms=1,
        citations=[TurnCitation(
            position=number, doc_id="doc", filename="a.md",
            page=None, chunk_text=f"证据{number}",
        ) for number in [5, 2]],
    )
    payload = _query_response(result).model_dump(mode="json")
    assert [item["position"] for item in payload["citations"]] == [5, 2]
