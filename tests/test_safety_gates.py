from __future__ import annotations

from datetime import date, timedelta
import math

import pytest
from pydantic import ValidationError

from quantquery_a.agents.contracts import AnalyzeRequest, MacdSpec, RunStatus, TaskType
from quantquery_a.agents.data_agent import DataAgent, DataBundle
from quantquery_a.agents.quant_agent import QuantAgent
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


def _backtest_request() -> AnalyzeRequest:
    return AnalyzeRequest(
        query="回测 DEMO.SH 的 MACD 策略",
        task_hint=TaskType.BACKTEST,
        symbols=["DEMO.SH"],
        bars={"DEMO.SH": _bars()},
        macd=MacdSpec(fast=2, slow=5, signal=2),
    )


def test_unsupported_request_is_blocked_before_data_access() -> None:
    report = QuantSupervisor().analyze(
        AnalyzeRequest(
            query="请替我实盘下单并保证盈利",
            symbols=["DEMO.SH"],
            bars={"DEMO.SH": _bars(20)},
        )
    )

    assert report.route.task_type is TaskType.UNSUPPORTED
    assert report.status is RunStatus.BLOCKED
    assert report.evidence == []
    assert report.analyses == []


def test_request_rejects_untrusted_symbols_and_unrequested_data() -> None:
    with pytest.raises(ValidationError):
        AnalyzeRequest(
            query="查询行情",
            symbols=["AUTHORIZATION-BEARER-SECRET"],
        )

    with pytest.raises(ValidationError):
        AnalyzeRequest(
            query="查询行情",
            symbols=["DEMO.SH"],
            bars={"OTHER.SH": _bars(2)},
        )


def test_empty_requested_date_range_is_blocked_without_an_api_error() -> None:
    report = QuantSupervisor().analyze(
        AnalyzeRequest(
            query="查询 DEMO.SH 行情",
            task_hint=TaskType.MARKET_QUERY,
            symbols=["DEMO.SH"],
            bars={"DEMO.SH": _bars(20)},
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 31),
        )
    )

    assert report.status is RunStatus.BLOCKED
    assert report.evidence == []


def test_non_trading_calendar_boundaries_preserve_observed_evidence() -> None:
    report = QuantSupervisor().analyze(
        AnalyzeRequest(
            query="Query DEMO.SH market data",
            task_hint=TaskType.MARKET_QUERY,
            symbols=["DEMO.SH"],
            bars={"DEMO.SH": _bars(4)[1:3]},
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 4),
        )
    )

    assert report.status is RunStatus.COMPLETED
    assert report.evidence[0].start_date == date(2025, 1, 2)
    assert report.evidence[0].end_date == date(2025, 1, 3)


def test_risk_critic_detects_metric_and_fingerprint_tampering() -> None:
    request = _backtest_request()
    supervisor = QuantSupervisor()
    route = supervisor.router.route(request)
    frame = supervisor._build_frame(request, route)
    data = DataAgent().run(frame, request.bars)
    analyses = QuantAgent().run(frame, data)

    tampered_metrics = dict(analyses[0].metrics)
    tampered_metrics["total_cost"] += 1.0
    tampered_analysis = analyses[0].model_copy(update={"metrics": tampered_metrics})
    metric_review = RiskCritic().review(frame, data, (tampered_analysis,))
    assert metric_review.approved is False
    assert "invalid_total_cost" in {finding.code for finding in metric_review.findings}
    assert "deterministic_metric_mismatch" in {
        finding.code for finding in metric_review.findings
    }

    tampered_performance = dict(analyses[0].metrics)
    tampered_performance["sharpe_ratio"] += 1.0
    performance_analysis = analyses[0].model_copy(
        update={"metrics": tampered_performance}
    )
    performance_review = RiskCritic().review(frame, data, (performance_analysis,))
    assert performance_review.approved is False
    assert "deterministic_metric_mismatch" in {
        finding.code for finding in performance_review.findings
    }

    bad_evidence = data.evidence[0].model_copy(update={"fingerprint": "0" * 64})
    bad_bundle = DataBundle(frames=data.frames, evidence=(bad_evidence,))
    fingerprint_review = RiskCritic().review(frame, bad_bundle, analyses)
    assert fingerprint_review.approved is False
    assert "invalid_evidence_fingerprint" in {
        finding.code for finding in fingerprint_review.findings
    }


def test_in_memory_trace_is_bounded() -> None:
    recorder = TraceRecorder(max_events=2)
    trace_id = recorder.new_trace_id()
    for index in range(3):
        recorder.emit(
            trace_id,
            span="test",
            event=f"event_{index}",
            status="ok",
        )

    assert [event.event for event in recorder.events_for(trace_id)] == [
        "event_1",
        "event_2",
    ]
