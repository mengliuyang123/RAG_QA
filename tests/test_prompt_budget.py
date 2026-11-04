# -*- coding: utf-8 -*-
"""包含系统提示、证据和工具描述的完整输入不得超出预算。"""

from uuid import uuid4
from pathlib import Path

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.application.query.context_budget import PromptBudgetGuard
from app.application.query.generator import AnswerGenerator
from app.application.query.prompting import PromptBuilder
from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem
from app.domain.errors import PromptContextTooLargeError
from app.domain.execution import EvidenceBundle
from app.domain.execution import ExecutionBudget
from app.domain.execution import RetrievalAttempt
from app.infrastructure.conversations.langgraph_agent import (
    LangGraphConversationAgent,
)
from app.infrastructure.conversations.quality import ModelQualityEvaluator
from app.infrastructure.generation.token_counting import (
    Utf8TokenUpperBoundCounter,
)
from app.interfaces.http.dependencies import get_query_service
from tests.http_auth import create_test_app as create_app
from tests.test_conversation_service import _FakeAgent
from tests.test_conversation_service import _service
from tests.test_query_api import FakeQueryService


class _NoNetworkModel:
    def bind_tools(self, tools: object, *, strict: bool) -> "_NoNetworkModel":
        """保持真实图装配，不替代预算检查。

        Args:
            tools: 工具描述。
            strict: 参数校验要求。

        Returns:
            当前离线模型。
        """
        return self

    def invoke(self, messages: list[dict[str, object]]) -> object:
        """预算失败后不得走到外部调用。

        Args:
            messages: 本不应发送的超长输入。

        Returns:
            不返回，调用说明测试失败。
        """
        raise AssertionError("超长输入仍发起了模型调用")

    def generate(self, messages: list[dict[str, str]]) -> str:
        """复用相同断言验证同步生成端口。

        Args:
            messages: 本不应发送的超长输入。

        Returns:
            不返回，调用说明测试失败。
        """
        raise AssertionError("超长输入仍发起了生成调用")


def _guard() -> PromptBudgetGuard:
    return PromptBudgetGuard(Utf8TokenUpperBoundCounter(), 2000)


def _item() -> EvidenceItem:
    return EvidenceItem(
        number=5, chunk_id="a", doc_id="a", filename="a.md", page=None,
        text="长证据" * 3000,
    )


def test_prompt_guard_counts_tool_schema_and_unicode_payload() -> None:
    """工具参数描述和中文不能逃过只检查用户文本的限额。"""
    guard = _guard()
    guard.ensure([{"role": "user", "content": "问题"}])
    with pytest.raises(PromptContextTooLargeError):
        guard.ensure(
            [{"role": "user", "content": "问题"}],
            extra_payload={"tools": [{"description": "中文" * 1000}]},
        )


def test_numbered_generator_checks_full_prompt_before_provider_call() -> None:
    """证据超长时不能只因当前问题很短就发送给上游。"""
    generator = AnswerGenerator(
        PromptBuilder(), _NoNetworkModel(), prompt_guard=_guard(),
    )
    with pytest.raises(PromptContextTooLargeError):
        generator.generate_grounded("A?", [], [_item()])


def test_quality_input_guard_error_is_not_disguised_as_provider_failure() -> None:
    """预算错误应保留为输入错误，而不是误报裁判服务502。"""
    evaluator = ModelQualityEvaluator(_NoNetworkModel(), prompt_guard=_guard())
    with pytest.raises(PromptContextTooLargeError):
        evaluator.assess_evidence("A?", [], [_item()])


def test_agent_seed_prompt_guard_prevents_model_call() -> None:
    """升级带入的长正文同样受完整输入预算约束。"""
    budget = ExecutionBudget(
        deadline=100.0, evidence_assessments=1,
    ).record_retrieval(RetrievalAttempt(query="A"), now=1.0)
    seed = EvidenceBundle(
        evidence=(_item(),), approved_numbers=(5,), budget=budget,
        assessment=EvidenceAssessment(
            sufficient=True, relevant_numbers=[5],
            missing_information=[], next_query=None,
        ),
    )
    agent = LangGraphConversationAgent(
        model=_NoNetworkModel(), retrieval_service=object(),
        checkpointer=InMemorySaver(), top_k=5, clock=lambda: 2.0,
        prompt_guard=_guard(),
    )
    with pytest.raises(PromptContextTooLargeError):
        agent.run(uuid4(), "A?", [], seed=seed)


@pytest.mark.anyio
async def test_query_http_prompt_budget_error_is_422() -> None:
    """查询入口不能把已注册的输入预算异常笼统包装成500。"""
    service = FakeQueryService()
    service.error = PromptContextTooLargeError("private context")
    app = create_app()
    app.dependency_overrides[get_query_service] = lambda: service
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.post("/api/query", json={"question": "A?"})
        assert response.status_code == 422
        assert "private" not in response.text
    finally:
        app.state.container.close()


def test_stream_prompt_budget_error_has_terminal_event(tmp_path: Path) -> None:
    """流开始后预算错误仍有终止事件，且不保存正式轮次。

    Args:
        tmp_path: 隔离的正式会话存储目录。
    """
    service = _service(
        tmp_path, _FakeAgent(error=PromptContextTooLargeError("private")),
    )
    conversation = service.create()
    events = list(service.stream_query(conversation.id, "A?"))
    assert [event.event for event in events] == ["status", "error"]
    assert events[-1].data.code == "context_too_large"
    assert service.get(conversation.id).turns_count == 0
    assert not service._conversation_locks


@pytest.fixture
def anyio_backend() -> str:
    """不依赖未安装的异步测试后端。

    Returns:
        本项目使用的 asyncio 后端。
    """
    return "asyncio"
