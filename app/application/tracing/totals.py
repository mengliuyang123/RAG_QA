# -*- coding: utf-8 -*-
"""汇总实际调用记录，未知不补零，币种／费率版本不混算。"""

from decimal import Decimal

from app.application.errors import TraceStorageError
from app.domain.trace_queries import CostGroup
from app.domain.trace_queries import CostTotals
from app.domain.trace_queries import TokenTotals
from app.domain.tracing import ModelCallTrace
from app.domain.tracing import TokenUsage
from app.domain.tracing import CostEstimate


class TokenAccumulator:
    """以固定大小计数器流式聚合，不保存每次调用对象。"""

    def __init__(self) -> None:
        """初始化明确的空集合零计数。

        Returns:
            None。
        """
        self._known_input = 0
        self._known_output = 0
        self._unknown_input = 0
        self._unknown_output = 0

    def add(self, usage: TokenUsage) -> None:
        """纳入一份实际usage，缺失不补零。

        Args:
            usage: 供应商已报告的计数。

        Returns:
            None。
        """
        if usage.input is None:
            self._unknown_input += 1
        else:
            self._known_input += usage.input
        if usage.output is None:
            self._unknown_output += 1
        else:
            self._known_output += usage.output

    def result(self) -> TokenTotals:
        """返回不可变小计。

        Returns:
            真实已知小计、未知次数及完整总量。
        """
        return TokenTotals(
            input=self._known_input if self._unknown_input == 0 else None,
            output=self._known_output if self._unknown_output == 0 else None,
            known_input=self._known_input, known_output=self._known_output,
            unknown_input_calls=self._unknown_input,
            unknown_output_calls=self._unknown_output,
        )


class CostAccumulator:
    """只按币种与费率版本聚合已知估计，保留未知次数。"""

    def __init__(self) -> None:
        """初始化空价格版本分组。

        Returns:
            None。
        """
        self._groups: dict[tuple[str, str], tuple[Decimal, int]] = {}
        self._unknown = 0

    def add(self, cost: CostEstimate) -> None:
        """纳入单次已验证费用估计，不做汇率换算。

        Args:
            cost: 明确版本的估计或未知状态。

        Returns:
            None。
        """
        if cost.status == "unknown":
            self._unknown += 1
            return
        if cost.currency is None or cost.pricing_version is None or cost.amount is None:
            raise TraceStorageError("费用估计缺少版本或币种")
        key = (cost.currency, cost.pricing_version)
        amount, count = self._groups.get(key, (Decimal(0), 0))
        self._groups[key] = (amount + cost.amount, count + 1)

    def result(self) -> CostTotals:
        """返回分版本小计，不把已知小计冒充完整总额。

        Returns:
            已知小计和未知调用数。
        """
        estimates = []
        for key in sorted(self._groups):
            amount, count = self._groups[key]
            estimates.append(CostGroup(
                currency=key[0], pricing_version=key[1], amount=amount, calls=count,
            ))
        status = "estimated"
        if self._unknown:
            status = "partial" if estimates else "unknown"
        return CostTotals(
            status=status, estimates=tuple(estimates), unknown_calls=self._unknown,
        )


def token_totals(calls: tuple[ModelCallTrace, ...]) -> TokenTotals:
    """累加供应商已报告用量，同时保留缺失计数。

    Args:
        calls: 实际调用记录，不含预算估算。

    Returns:
        完整总量或未知，以及明确的小计和缺失次数。
    """
    accumulator = TokenAccumulator()
    for call in calls:
        accumulator.add(call.usage)
    return accumulator.result()


def cost_totals(calls: tuple[ModelCallTrace, ...]) -> CostTotals:
    """按明确费率版本与币种分别求和，不做汇率换算或猜价。

    Args:
        calls: 实际调用及其已验证费用估计。

    Returns:
        已知估计的小计，以及未知调用数。
    """
    accumulator = CostAccumulator()
    for call in calls:
        accumulator.add(call.cost)
    return accumulator.result()
