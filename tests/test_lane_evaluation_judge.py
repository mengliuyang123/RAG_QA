# -*- coding: utf-8 -*-
"""独立评分使用真实SDK结构化边界，核对事实依据而非引用编号存在。"""

import json
from pathlib import Path

import httpx
import pytest
from langchain_openai import ChatOpenAI

from app.application.query.context_budget import PromptBudgetGuard
from app.config import Settings
from app.domain.errors import PromptContextTooLargeError
from app.domain.retrieval import RetrievalStrategy
from app.infrastructure.generation.token_counting import Utf8TokenUpperBoundCounter
from evaluation.lanes.models import LaneEpisode
from evaluation.lanes.models import LaneTurn
from evaluation.lanes.runtime import LaneEpisodeRunner
from tests.test_unified_query_api import _app


def test_independent_judge_receives_actual_answer_and_cited_text_with_real_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """如果裁判复用生成模型、没有证据正文或捏造usage，此测试失败。

    Args:
        tmp_path: 所有存储的隔离目录。
        monkeypatch: 仅替换SDK使用的网络传输。
    """
    from app.bootstrap.factories import create_evaluation_judge_provider
    from evaluation.lanes.judge import LaneJudge

    requests = []
    options_seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "judge.example"
        assert request.headers["authorization"] == "Bearer judge-secret"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 1,
            "model": "independent-revision",
            "usage": {"prompt_tokens": 71, "completion_tokens": 19, "total_tokens": 90},
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps({
                    "correctness": 0.4, "supported": False,
                    "complete": False, "reason": "只给了资料字样，未回答存储问题",
                }, ensure_ascii=False),
            }}],
        })

    client = httpx.Client(transport=httpx.MockTransport(handle))

    def model(**options: object) -> ChatOpenAI:
        options_seen.append(options)
        return ChatOpenAI(**options, http_client=client)

    monkeypatch.setattr("app.bootstrap.factories.ChatOpenAI", model)
    settings = Settings(
        _env_file=None, llm_api_key="generator-secret", llm_model="generator",
        llm_base_url="https://generator.example/v1",
        llm_output_token_parameter="max_completion_tokens",
        evaluation_judge_api_key="judge-secret",
        evaluation_judge_base_url="https://judge.example/v1",
        evaluation_judge_model="independent", evaluation_judge_max_output_tokens=113,
        evaluation_judge_timeout_seconds=3,
        evaluation_judge_output_token_parameter="max_tokens",
    )
    provider = create_evaluation_judge_provider(settings)
    app, _, _ = _app(tmp_path)
    root = app.state.container
    try:
        turn = LaneTurn(question="needle", reference="使用SQLite保存状态",
                        expected_lane="fast", expected_sources=("guide.md",))
        episode = LaneEpisode(id="judge", tags=("single_hop",), turns=(turn,))
        observation = LaneEpisodeRunner(
            root.query_orchestrator, root.history_window,
            root.trace_repository, root.cleanup_temporary_execution,
        ).run(episode, "auto", RetrievalStrategy.HYBRID)[0]
        judge = LaneJudge(
            provider, root.trace_service, root.trace_repository,
            PromptBudgetGuard(Utf8TokenUpperBoundCounter(), 16000),
        )
        record = judge.score(turn, observation)
        assert record.source_trace_id == observation.trace.trace_id
        assert record.judgement.correctness == 0.4
        assert record.judgement.supported is False
        assert record.judgement.complete is False
        assert requests[0]["model"] == "independent"
        assert requests[0]["max_tokens"] == 113
        assert requests[0]["response_format"]["json_schema"]["name"] == "LaneJudgement"
        assert options_seen[0]["timeout"] == 3
        assert options_seen[0]["max_retries"] == 0
        payload = json.loads(requests[0]["messages"][-1]["content"])
        assert payload["answer"] == "资料[1]"
        assert payload["reference"] == "使用SQLite保存状态"
        assert payload["citations"][0]["position"] == 1
        assert payload["citations"][0]["text"] == "needle 使用SQLite保存状态。"
        assert record.trace.calls[0].model == "independent"
        assert record.trace.calls[0].purpose == "evaluation_judge"
        assert record.trace.calls[0].usage.input == 71
        assert record.trace.calls[0].usage.output == 19
        assert record.trace.calls[0].cost.status == "unknown"
        assert record.trace.question is None
        assert "使用SQLite保存状态。" not in record.trace.model_dump_json()
        assert observation.trace.calls == ()
        small = LaneJudge(
            provider, root.trace_service, root.trace_repository,
            PromptBudgetGuard(Utf8TokenUpperBoundCounter(), 1),
        )
        with pytest.raises(PromptContextTooLargeError):
            small.score(turn, observation)
        assert len(requests) == 1
    finally:
        provider.close()
        root.close()
    assert client.is_closed


def test_judge_factory_rejects_generator_identity_before_creating_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同源同模型不算独立裁判，必须在构造远程资源前拒绝。

    Args:
        monkeypatch: 检查禁止的客户端构造。
    """
    from app.bootstrap.factories import create_evaluation_judge_provider

    def forbidden(**options: object) -> None:
        raise AssertionError("不应创建客户端")

    monkeypatch.setattr("app.bootstrap.factories.ChatOpenAI", forbidden)
    settings = Settings(
        _env_file=None, llm_model="same", llm_base_url="https://model.example/v1",
        evaluation_judge_model="same", evaluation_judge_api_key="judge",
        evaluation_judge_base_url="https://model.example/v1",
    )
    with pytest.raises(ValueError, match="独立"):
        create_evaluation_judge_provider(settings)
