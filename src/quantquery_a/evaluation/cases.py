"""Frozen synthetic cases; no production database is read."""

from __future__ import annotations

from datetime import date, timedelta
import math

from quantquery_a.agents.contracts import (
    AnalyzeRequest,
    MacdSpec,
    RunStatus,
    TaskType,
)

from .runner import EvalCase


def _bars(count: int) -> list[dict[str, object]]:
    start = date(2025, 1, 1)
    return [
        {
            "trade_date": start + timedelta(days=index),
            "open": 100.2 + math.sin(index / 4) * 5 + index * 0.03,
            "close": 100.0 + math.sin(index / 4) * 5 + index * 0.03,
        }
        for index in range(count)
    ]


def build_default_cases() -> tuple[EvalCase, ...]:
    return (
        EvalCase(
            name="market_fast_path",
            request=AnalyzeRequest(
                query="查询 DEMO.SH 的收盘价行情",
                symbols=["DEMO.SH"],
                bars={"DEMO.SH": _bars(20)},
            ),
            expected_task=TaskType.MARKET_QUERY,
            expected_status=RunStatus.COMPLETED,
        ),
        EvalCase(
            name="audited_backtest",
            request=AnalyzeRequest(
                query="回测 DEMO.SH 的 MACD 策略并检查最大回撤",
                symbols=["DEMO.SH"],
                bars={"DEMO.SH": _bars(90)},
                macd=MacdSpec(fast=2, slow=5, signal=2),
            ),
            expected_task=TaskType.BACKTEST,
            expected_status=RunStatus.COMPLETED,
        ),
        EvalCase(
            name="ambiguous_query",
            request=AnalyzeRequest(query="帮我看看这个"),
            expected_task=TaskType.MARKET_QUERY,
            expected_status=RunStatus.NEEDS_CLARIFICATION,
            require_exact_task=False,
        ),
        EvalCase(
            name="risk_blocks_short_window",
            request=AnalyzeRequest(
                query="回测 DEMO.SH",
                symbols=["DEMO.SH"],
                bars={"DEMO.SH": _bars(5)},
                macd=MacdSpec(fast=2, slow=5, signal=2),
            ),
            expected_task=TaskType.BACKTEST,
            expected_status=RunStatus.BLOCKED,
        ),
    )
