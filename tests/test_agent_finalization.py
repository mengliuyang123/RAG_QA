"""证据充分或额度耗尽时必须退出工具循环，再走引用与答案审核。"""
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from app.application.errors import LLMGenerationError
from app.domain.agent_quality import EvidenceAssessment
from tests.test_agent_quality_graph import _seed_agent, _seed, _enough, _passed, _incomplete, _Evaluator, _Retriever


class ToolLoopModel:
    """有工具时永远搜索，无工具才会输出正文；复现实际云模型的选择。"""
    def __init__(self, *, broken=False):
        self.calls = []
        self.broken = broken

    def bind_tools(self, tools, *, strict):
        parent = self
        class Bound:
            def invoke(self, messages):
                return parent.respond(messages, True)
        return Bound()

    def invoke(self, messages):
        return self.respond(messages, False)

    def respond(self, messages, tools):
        self.calls.append((tools, messages))
        if tools or self.broken:
            return AIMessage(content='', tool_calls=[{
                'name':'search_knowledge_base', 'id':f'call-{len(self.calls)}',
                'type':'tool_call', 'args':{'query':f'额外查询{len(self.calls)}'},
            }])
        return AIMessage(content='A 使用 SQLite[5]，资料未提及 B。')


@pytest.mark.parametrize('calls', [1, 3])
def test_sufficient_evidence_closes_tools_and_enters_answer_assessment(calls):
    model = ToolLoopModel()
    retriever = _Retriever({})
    evaluator = _Evaluator([_enough([5])]*3, [_passed()])
    result = _seed_agent(model, retriever, evaluator).run(
        uuid4(), 'langchain 怎么使用', [], seed=_seed(_enough([5]), calls=calls),
    )
    assert result.answer.startswith('A 使用 SQLite')
    assert result.retrieval_calls == calls and retriever.queries == []
    assert [enabled for enabled,_ in model.calls] == [False]
    assert result.execution.budget.answer_assessments == 1
    assert 'UNRELATED_SEED' not in str(model.calls)


def test_exhausted_budget_uses_approved_partial_evidence():
    model = ToolLoopModel()
    retriever = _Retriever({})
    verdict = EvidenceAssessment(sufficient=False, relevant_numbers=[5],
                                 missing_information=['缺少 B'], next_query=None)
    result = _seed_agent(model, retriever, _Evaluator([], [_incomplete()])).run(
        uuid4(), '比较 A/B', [], seed=_seed(verdict, calls=3),
    )
    assert result.partial and result.retrieval_calls == 3
    assert result.execution.budget.answer_assessments == 1
    assert retriever.queries == [] and not model.calls[0][0]


def test_empty_tool_response_after_forced_generation_is_generation_error():
    model = ToolLoopModel(broken=True)
    retriever = _Retriever({})
    with pytest.raises(LLMGenerationError):
        _seed_agent(model, retriever, _Evaluator([])).run(
            uuid4(), 'langchain 怎么使用', [], seed=_seed(_enough([5]), calls=3),
        )
    assert retriever.queries == []
    assert len(model.calls) <= 2
    assert all(not tools for tools,_ in model.calls)


def test_orchestrator_does_not_label_sufficient_generation_failure_as_missing_data(tmp_path):
    from types import SimpleNamespace
    from tests.test_query_orchestrator import _build
    from app.domain.conversations import AgentResult, AnswerMode, KNOWLEDGE_REFUSAL_ANSWER
    from app.domain.routing import KnowledgeScope
    from app.domain.execution import RetrievalAttempt
    orchestrator, _ = _build(tmp_path)
    seed = _seed(_enough([5]))
    seed = seed.model_copy(update={
        'evidence': tuple(e.model_copy(update={'kb_id':'ops'}) for e in seed.evidence),
        'budget':seed.budget.model_copy(update={'attempts':(RetrievalAttempt(query='A',kb_ids=('ops',)),)}),
    })
    failed = AgentResult(answer=KNOWLEDGE_REFUSAL_ANSWER, answer_mode=AnswerMode.RETRIEVAL,
                         retrieval_calls=1, citations=[], execution=seed)
    outcome = SimpleNamespace(result=failed, lane='agent', searched_kb_ids=('ops',),
                              scope_expanded=False, escalated=False)
    orchestrator._knowledge_runner = lambda: SimpleNamespace(run=lambda *args,**kwargs: outcome)
    with pytest.raises(LLMGenerationError):
        orchestrator.run(uuid4(), 'langchain 怎么使用', [], kb_scope=KnowledgeScope(mode='explicit',kb_ids=('ops',)))


def test_incomplete_generation_is_stream_error_and_not_a_saved_refusal(tmp_path):
    from tests.test_conversation_service import _service, _FakeAgent
    service = _service(tmp_path, _FakeAgent(error=LLMGenerationError('internal diagnostic')))
    conversation = service.create()
    events = list(service.stream_query(conversation.id, 'langchain 怎么使用'))
    assert events[-1].event == 'error'
    assert events[-1].data.code == 'generation_error'
    assert events[-1].data.message == '回答生成未能完成，请重试'
    assert not any(e.event in ('token','complete') for e in events)
    assert service.get(conversation.id).turns == []


def test_genuinely_insufficient_partial_evidence_can_still_refuse():
    from tests.test_agent_quality_graph import _Model
    from app.domain.conversations import KNOWLEDGE_REFUSAL_ANSWER
    verdict = EvidenceAssessment(sufficient=False, relevant_numbers=[5],
                                 missing_information=['现有相关资料不含所需教程'], next_query=None)
    model = _Model([AIMessage(content=KNOWLEDGE_REFUSAL_ANSWER)]*2)
    result = _seed_agent(model, _Retriever({}), _Evaluator([])).run(
        uuid4(), 'langchain 怎么使用', [], seed=_seed(verdict, calls=3),
    )
    assert result.answer == KNOWLEDGE_REFUSAL_ANSWER
    assert not result.execution.assessment.sufficient
    assert result.citations == []
