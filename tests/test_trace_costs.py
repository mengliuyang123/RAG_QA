# -*- coding: utf-8 -*-
"""成本必须基于供应商实际usage和指定版本价格，不能用字符估算冒充。"""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.domain.tracing import ModelPrice
from app.domain.tracing import TokenUsage
from app.domain.tracing import estimate_cost


def _price(**overrides: object) -> ModelPrice:
    values = {
        "origin": "https://model.example", "model": "small",
        "version": "test-price-v1", "currency": "USD",
        "input_per_million": Decimal("2"),
        "output_per_million": Decimal("6"),
    }
    values.update(overrides)
    return ModelPrice.model_validate(values)


def test_cost_uses_reported_tokens_and_versioned_prices() -> None:
    """输入1000、输出200按测试费率应为0.0032，不计成免费或实际账单。"""
    cost = estimate_cost(TokenUsage(input=1000, output=200), _price())
    assert cost.status == "estimated"
    assert cost.amount == Decimal("0.0032")
    assert cost.currency == "USD"
    assert cost.pricing_version == "test-price-v1"


@pytest.mark.parametrize("usage,price,reason", [
    (TokenUsage(), _price(), "usage_unavailable"),
    (TokenUsage(input=1000), _price(), "usage_unavailable"),
    (TokenUsage(input=1000, output=200), None, "price_unavailable"),
])
def test_missing_usage_or_price_is_unknown_not_zero(
    usage: TokenUsage, price: ModelPrice | None, reason: str,
) -> None:
    """不能把缺失usage或未知价格当作费用0。

    Args:
        usage: 供应商实际提供的字段。
        price: 可选版本价格。
        reason: 手工预期的未知原因。
    """
    cost = estimate_cost(usage, price)
    assert cost.status == "unknown"
    assert cost.amount is None
    assert cost.reason == reason


def test_cached_input_cost_requires_actual_cache_count() -> None:
    """有单独缓存费率时，缺缓存计数不能按全量或全免费猜测。"""
    price = _price(cached_input_per_million=Decimal("0.5"))
    unknown = estimate_cost(TokenUsage(input=1000, output=200), price)
    assert unknown.status == "unknown"
    assert unknown.reason == "cache_usage_unavailable"
    reported = estimate_cost(
        TokenUsage(input=1000, output=200, cached_input=600), price,
    )
    assert reported.amount == Decimal("0.0023")


def test_unknown_cost_still_keeps_selected_pricing_version() -> None:
    """没有usage不等于没有选过价格，仍应能审计当时配置的版本。"""
    cost = estimate_cost(TokenUsage(), _price())
    assert cost.status == "unknown"
    assert cost.amount is None
    assert cost.pricing_version == "test-price-v1"
    assert cost.currency == "USD"


def test_zero_is_only_reported_when_provider_actually_reported_zero() -> None:
    """零值是一个有效测量，但不能作为缺省值。"""
    cost = estimate_cost(TokenUsage(input=0, output=0), _price())
    assert cost.status == "estimated"
    assert cost.amount == Decimal("0")


@pytest.mark.parametrize("origin", [
    "unknown", "https://model.example/v1", "https://model.example?key=private",
    "https://user:password@model.example",
])
def test_price_origin_must_be_exact_nonsecret_service_identity(origin: str) -> None:
    """价格不能依赖未知来源、端点路径或凭据。

    Args:
        origin: 无效的费率来源标识。
    """
    with pytest.raises(ValidationError):
        _price(origin=origin)
