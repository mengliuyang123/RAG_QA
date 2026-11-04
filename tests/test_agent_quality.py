# -*- coding: utf-8 -*-
"""验证在线质量判断的严格协议与证据边界。"""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.application.conversations.quality import normalize_query
from app.application.conversations.quality import select_evidence
from app.application.errors import ConversationAgentError
from app.domain.agent_quality import AnswerAssessment
from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem
from app.infrastructure.conversations.quality import ModelQualityEvaluator


def _evidence(number: int = 1) -> EvidenceItem:
    return EvidenceItem(
        number=number,
        chunk_id=f"chunk-{number}",
        doc_id="doc",
        filename="notes.md",
        page=None,
        text="A 方案使用 SQLite。",
    )


class _JSONModel:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.messages: list[dict[str, object]] = []

    def invoke(self, messages: list[dict[str, object]]) -> object:
        """替代外部模型调用，返回待检验的原始响应。

        Args:
            messages: 待发送的完整模型消息。

        Returns:
            带 content 的模型消息。
        """
        self.messages = messages
        if isinstance(self.payload, Exception):
            raise self.payload
        return SimpleNamespace(content=self.payload)


def test_evidence_selection_preserves_original_numbers() -> None:
    assessment = EvidenceAssessment(
        sufficient=True, relevant_numbers=[3],
        missing_information=[], next_query=None,
    )
    approved = select_evidence([_evidence(1), _evidence(3)], assessment)
    assert [item.number for item in approved] == [3]
    assert approved[0].chunk_id == "chunk-3"


def test_unknown_evidence_number_is_a_protocol_error() -> None:
    assessment = EvidenceAssessment(
        sufficient=True, relevant_numbers=[9],
        missing_information=[], next_query=None,
    )
    with pytest.raises(ConversationAgentError):
        select_evidence([_evidence()], assessment)


@pytest.mark.parametrize("changes", [
    {"sufficient": "true"},
    {"relevant_numbers": [True]},
    {"relevant_numbers": [1, 1]},
    {"relevant_numbers": []},
    {"missing_information": ["缺少 B"]},
    {"next_query": "B 方案"},
    {"untrusted": "extra"},
])
def test_evidence_verdict_rejects_ambiguous_protocol(changes: dict) -> None:
    payload = {
        "sufficient": True, "relevant_numbers": [1],
        "missing_information": [], "next_query": None,
    }
    payload.update(changes)
    with pytest.raises(ValidationError):
        EvidenceAssessment.model_validate(payload)


@pytest.mark.parametrize("payload", [
    {"supported": True, "complete": True, "issues": ["无依据"]},
    {"supported": False, "complete": True, "issues": []},
    {"supported": "true", "complete": True, "issues": []},
])
def test_answer_verdict_rejects_contradictory_protocol(payload: dict) -> None:
    with pytest.raises(ValidationError):
        AnswerAssessment.model_validate(payload)


def test_query_normalization_catches_rephrased_whitespace() -> None:
    assert normalize_query("  B   SQLITE\n模式  ") == "b sqlite 模式"


def test_model_assesses_evidence_without_binding_tools() -> None:
    model = _JSONModel(json.dumps({
        "sufficient": False, "relevant_numbers": [1],
        "missing_information": ["缺少 B 的存储方案"],
        "next_query": "B 方案存储方式",
    }))
    result = ModelQualityEvaluator(model).assess_evidence(
        "比较 A 与 B", [], [_evidence()],
    )
    assert result.sufficient is False
    assert result.next_query == "B 方案存储方式"
    request = json.loads(model.messages[-1]["content"])
    assert request["question"] == "比较 A 与 B"
    assert request["evidence"][0]["number"] == 1


def test_empty_evidence_does_not_call_model() -> None:
    model = _JSONModel(AssertionError("不应调用模型"))
    result = ModelQualityEvaluator(model).assess_evidence("问题", [], [])
    assert result.sufficient is False
    assert result.relevant_numbers == []
    assert result.next_query is None


def test_answer_assessment_receives_question_history_and_cited_evidence(
) -> None:
    model = _JSONModel(
        '{"supported":false,"complete":true,"issues":["存储结论错误"]}'
    )
    history = [{"user_message": "A 是什么", "assistant_message": "方案 A"}]
    result = ModelQualityEvaluator(model).assess_answer(
        "它用什么存储？", history, "A 使用 PostgreSQL[1]", [_evidence()],
    )
    assert result.supported is False
    request = json.loads(model.messages[-1]["content"])
    assert request["history"] == history
    assert request["answer"] == "A 使用 PostgreSQL[1]"
    assert request["evidence"][0]["text"] == "A 方案使用 SQLite。"


@pytest.mark.parametrize("payload", [
    "not json", "{}", "[]", RuntimeError("private endpoint"),
    '{"sufficient":true,"relevant_numbers":[9],'
    '"missing_information":[],"next_query":null}',
])
def test_invalid_evaluator_output_is_stable_agent_error(payload: object) -> None:
    with pytest.raises(ConversationAgentError) as error:
        ModelQualityEvaluator(_JSONModel(payload)).assess_evidence(
            "问题", [], [_evidence()],
        )
    assert "private" not in str(error.value)


@pytest.mark.parametrize("prefix,suffix", [
    ("```json\n", "\n```"),
    ("<think>先检查证据，不应作为评估 JSON。</think>\n", ""),
    ("<think>检查</think>\n```json\n", "\n```"),
    ("评估结果如下：\n", "\n审核结束。"),
])
def test_quality_accepts_wrapping_without_relaxing_schema(
    prefix: str, suffix: str,
) -> None:
    """明确的展示包装不应导致本来合法的判断失败。

    Args:
        prefix: JSON 前的模型展示包装。
        suffix: JSON 后的模型展示包装。
    """
    payload = (
        '{"sufficient":true,"relevant_numbers":[1],'
        '"missing_information":[],"next_query":null}'
    )
    result = ModelQualityEvaluator(_JSONModel(
        prefix + payload + suffix,
    )).assess_evidence("问题", [], [_evidence()])
    assert result.sufficient is True


def test_quality_preserves_literal_think_tags_inside_json_strings() -> None:
    """JSON 字段可以合法讨论思考标签，而不是把它当成外部包装。"""
    issue = "答案对 <think> 与 </think> 标签的解释缺少来源"
    payload = json.dumps({
        "supported": False, "complete": False, "issues": [issue],
    })
    result = ModelQualityEvaluator(_JSONModel(payload)).assess_answer(
        "问题", [], "回答[1]", [_evidence()],
    )
    assert result.issues == [issue]


@pytest.mark.parametrize("payload", [
    '{"sufficient":false,"sufficient":true,"relevant_numbers":[1],'
    '"missing_information":[],"next_query":null}',
    '<think>未闭合 {"sufficient":true,"relevant_numbers":[1],'
    '"missing_information":[],"next_query":null}',
    '{"supported":true,"complete":true,"issues":[]} '
    '{"supported":false,"complete":true,"issues":["无依据"]}',
])
def test_quality_rejects_ambiguous_wrapping_or_duplicate_keys(
    payload: str,
) -> None:
    """不得从矛盾判断中选择看似可用的一个对象。

    Args:
        payload: 有歧义、残缺或重复键的模型文本。
    """
    with pytest.raises(ConversationAgentError):
        ModelQualityEvaluator(_JSONModel(payload)).assess_answer(
            "问题", [], "回答[1]", [_evidence()],
        )
