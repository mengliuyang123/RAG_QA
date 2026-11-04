# -*- coding: utf-8 -*-
"""追踪汇总不得把未报告的用量／费用当零，也不得跨币种相加。"""

from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from app.application.tracing import TraceService
from app.application.tracing import provider_call
from app.domain.trace_queries import TraceFilters
from app.domain.tracing import ModelPrice
from app.domain.tracing import TokenUsage
from tests.test_unified_query_api import _app


def test_trace_totals_preserve_known_usage_unknown_calls_and_price_versions(tmp_path: Path) -> None:
    """少报一条调用、未知补零或混合价格版本都应使本测试失败。

    Args:
        tmp_path: 隔离所有数据。
    """
    app, _, _ = _app(tmp_path)
    try:
        prices = (
            ModelPrice(origin="https://generator.example", model="gen", version="usd-v1", currency="USD", input_per_million=Decimal("1"), output_per_million=Decimal("2")),
            ModelPrice(origin="https://rerank.example", model="rank", version="cny-v2", currency="CNY", input_per_million=Decimal("3"), output_per_million=Decimal("0")),
        )
        service = TraceService(app.state.container.trace_repository, prices=prices)
        with service.start("question", request_id=uuid4(), entry="query") as trace:
            with provider_call("generation", "https://generator.example/v1", "gen") as call:
                call.usage = TokenUsage(input=10, output=2)
            with provider_call("reranking", "https://rerank.example", "rank") as call:
                call.usage = TokenUsage(input=4, output=0)
            with provider_call("embedding", "https://embed.example", "embed"):
                pass
        query = app.state.container.trace_query_service
        detail = query.get(trace.id)
        assert detail.model_calls == 3
        assert detail.llm_calls == 1
        assert detail.tokens.input is None
        assert detail.tokens.output is None
        assert detail.tokens.known_input == 14
        assert detail.tokens.known_output == 2
        assert detail.tokens.unknown_input_calls == 1
        assert detail.tokens.unknown_output_calls == 1
        assert detail.cost.status == "partial"
        assert detail.cost.unknown_calls == 1
        assert {(group.currency, group.pricing_version, group.amount) for group in detail.cost.estimates} == {
            ("USD", "usd-v1", Decimal("0.000014")),
            ("CNY", "cny-v2", Decimal("0.000012")),
        }
        listed = query.list(TraceFilters()).traces[0]
        assert listed.tokens == detail.tokens
        assert listed.cost == detail.cost
    finally:
        app.state.container.close()


def test_no_calls_is_known_zero_not_missing_usage(tmp_path: Path) -> None:
    """没有调用与调用后缺usage是不同状态，前端不能混淆。

    Args:
        tmp_path: 隔离数据库。
    """
    app, _, _ = _app(tmp_path)
    try:
        with app.state.container.trace_service.start("你好", request_id=uuid4(), entry="query") as trace:
            pass
        detail = app.state.container.trace_query_service.get(trace.id)
        assert detail.model_calls == detail.llm_calls == 0
        assert detail.tokens.input == detail.tokens.output == 0
        assert detail.cost.unknown_calls == 0
        assert detail.cost.estimates == ()
    finally:
        app.state.container.close()
