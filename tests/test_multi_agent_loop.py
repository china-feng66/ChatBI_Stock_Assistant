from __future__ import annotations

from datetime import date, timedelta
import math

from quantquery_a.agents.contracts import (
    AnalyzeRequest,
    MacdSpec,
    RiskFinding,
    RiskReview,
    RunStatus,
    TaskType,
)
from quantquery_a.agents.risk_critic import RiskCritic
from quantquery_a.agents.supervisor import QuantSupervisor
from quantquery_a.observability.trace import TraceRecorder


def _bars(count: int = 90):
    start = date(2025, 1, 1)
    return [
        {
            "trade_date": start + timedelta(days=index),
            "open": 100.2 + math.sin(index / 4) * 5 + index * 0.03,
            "close": 100.0 + math.sin(index / 4) * 5 + index * 0.03,
        }
        for index in range(count)
    ]


def _request(count: int = 90) -> AnalyzeRequest:
    return AnalyzeRequest(
        query="回测 DEMO.SH 的 MACD 策略并检查风险",
        task_hint=TaskType.BACKTEST,
        symbols=["demo.sh"],
        bars={"DEMO.SH": _bars(count)},
        macd=MacdSpec(fast=2, slow=5, signal=2),
    )


def test_planned_path_runs_data_quant_and_mandatory_risk_gate() -> None:
    recorder = TraceRecorder()
    report = QuantSupervisor(trace_recorder=recorder).analyze(_request())

    assert report.status is RunStatus.COMPLETED
    assert report.risk_review is not None
    assert report.risk_review.approved is True
    assert [step.role for step in report.steps[:4]] == [
        "router",
        "data",
        "quant",
        "risk",
    ]
    assert report.analyses[0].trades
    for trade in report.analyses[0].trades:
        assert trade.execution_date > trade.signal_date
        assert trade.shares % report.frame.costs.lot_size == 0

    spans = {event.span for event in recorder.events_for(report.trace_id)}
    assert {"router", "data", "quant", "risk"} <= spans


def test_fast_path_only_validates_read_only_evidence() -> None:
    request = AnalyzeRequest(
        query="查询 DEMO.SH 行情",
        task_hint=TaskType.MARKET_QUERY,
        symbols=["DEMO.SH"],
        bars={"DEMO.SH": _bars(10)},
    )

    report = QuantSupervisor().analyze(request)

    assert report.status is RunStatus.COMPLETED
    assert report.route.path.value == "fast"
    assert report.evidence[0].row_count == 10
    assert report.analyses == []
    assert report.risk_review is None


def test_risk_gate_blocks_insufficient_observations() -> None:
    report = QuantSupervisor().analyze(_request(count=5))

    assert report.status is RunStatus.BLOCKED
    assert report.risk_review is not None
    assert report.risk_review.approved is False
    assert "insufficient_observations" in {
        finding.code for finding in report.risk_review.findings
    }


class _RetryOnceCritic(RiskCritic):
    def __init__(self) -> None:
        self.calls = 0

    def review(self, frame, data, analyses):
        self.calls += 1
        if self.calls == 1:
            return RiskReview(
                approved=False,
                checked_symbols=list(frame.symbols),
                findings=[
                    RiskFinding(
                        code="transient_incomplete_result",
                        severity="error",
                        message="retry once",
                        retryable=True,
                    )
                ],
            )
        return super().review(frame, data, analyses)


def test_supervisor_bounds_directed_retry() -> None:
    critic = _RetryOnceCritic()
    report = QuantSupervisor(risk_critic=critic, max_attempts=2).analyze(_request())

    assert report.status is RunStatus.COMPLETED
    assert report.attempts == 2
    assert critic.calls == 2
