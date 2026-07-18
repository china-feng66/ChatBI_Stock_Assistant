from __future__ import annotations

from datetime import date, timedelta
import math
import time

from fastapi.testclient import TestClient

from quantquery_a.agents.contracts import AnalyzeRequest, TaskType
from quantquery_a.api import create_app
from quantquery_a.workbench.contracts import (
    ResearchMode,
    RunCreateRequest,
    RouterWeights,
)
from quantquery_a.workbench.llm import FakeQwenClient
from quantquery_a.workbench.market import (
    BoundedMarketGateway,
    MarketCache,
    MarketProviderError,
    ProviderBatch,
)
from quantquery_a.workbench.routing import HybridIntentRouter, INTENT_PROTOTYPES
from quantquery_a.workbench.service import WorkbenchService


def _bars(count: int, offset: float = 0.0) -> list[dict[str, object]]:
    start = date(2025, 1, 1)
    return [
        {
            "trade_date": (start + timedelta(days=index)).isoformat(),
            "open": 100.2
            + offset
            + math.sin(index / 4) * 5
            + index * 0.03,
            "close": 100.0
            + offset
            + math.sin(index / 4) * 5
            + index * 0.03,
        }
        for index in range(count)
    ]


def test_hybrid_router_reaches_frozen_twenty_case_target() -> None:
    router = HybridIntentRouter(
        weights=RouterWeights(llm=0.45, tfidf=0.30, keyword=0.25)
    )
    correct = 0
    for text, expected in INTENT_PROTOTYPES:
        probabilities = {task.value: 0.01 for task in TaskType}
        probabilities[expected.value] = 0.94
        decision = router.route(
            AnalyzeRequest(query=text, symbols=["DEMO.SH"]), probabilities
        )
        correct += decision.task_type is expected
    assert correct / len(INTENT_PROTOTYPES) >= 0.9


def test_five_agent_macd_run_records_usage_and_risk(tmp_path) -> None:
    service = WorkbenchService(
        runtime_dir=tmp_path / "workbench-runtime",
        llm=FakeQwenClient(),
    )
    snapshot = service.run_sync(
        RunCreateRequest(
            query="回测 DEMO.SH 的 MACD 策略",
            symbols=["DEMO.SH"],
            mode=ResearchMode.MACD,
            bars={"DEMO.SH": _bars(90)},
            macd={"fast": 2, "slow": 5, "signal": 2},
        )
    )
    assert snapshot.status.value == "completed"
    assert snapshot.llm_calls == 5
    assert 0 < snapshot.token_total <= 15_000
    assert snapshot.risk and snapshot.risk["approved"] is True
    assert service.store.metrics()["llm_call_count"] == 5


def test_factor_portfolio_uses_next_bar_and_passes_gate(tmp_path) -> None:
    service = WorkbenchService(
        runtime_dir=tmp_path / "factor-runtime",
        llm=FakeQwenClient(),
    )
    symbols = ["AAA.SH", "BBB.SH", "CCC.SH"]
    snapshot = service.run_sync(
        RunCreateRequest(
            query="构建价格成交量三因子组合",
            symbols=symbols,
            mode=ResearchMode.FACTOR,
            bars={symbol: _bars(80, index * 10) for index, symbol in enumerate(symbols)},
        )
    )
    assert snapshot.status.value == "completed"
    trades = snapshot.analyses[0]["trades"]
    assert all(trade["execution_date"] > trade["signal_date"] for trade in trades)
    assert "holdout_return" in snapshot.analyses[0]["metrics"]


class _DailyProvider:
    name = "fake-daily"

    def __init__(self) -> None:
        self.fail = False

    def fetch_daily(self, symbol, *, start_date, end_date, limit):
        del start_date, end_date
        if self.fail:
            raise MarketProviderError("offline")
        rows = _bars(limit)
        return ProviderBatch(
            provider=self.name,
            symbol=symbol,
            rows=rows,
            adjustment_status="qfq",
            as_of=str(rows[-1]["trade_date"]),
        )


def test_market_gateway_labels_cache_fallback_stale(tmp_path) -> None:
    provider = _DailyProvider()
    gateway = BoundedMarketGateway(
        cache=MarketCache(tmp_path / "market.sqlite3"),
        daily_providers=[provider],
    )
    _, first_evidence = gateway.daily_bars(
        ["DEMO.SH"], start_date=None, end_date=None, limit=40
    )
    provider.fail = True
    _, cached_evidence = gateway.daily_bars(
        ["DEMO.SH"], start_date=None, end_date=None, limit=40
    )
    assert first_evidence[0]["stale"] is False
    assert cached_evidence[0]["stale"] is True
    assert cached_evidence[0]["fingerprint"] == first_evidence[0]["fingerprint"]


def test_workbench_api_runs_and_exposes_metrics(tmp_path) -> None:
    service = WorkbenchService(
        runtime_dir=tmp_path / "api-runtime",
        llm=FakeQwenClient(),
    )
    with TestClient(create_app(workbench_service=service)) as client:
        response = client.post(
            "/api/v1/runs",
            json={
                "query": "查询 DEMO.SH 最新行情",
                "symbols": ["DEMO.SH"],
                "mode": "market",
                "bars": {"DEMO.SH": _bars(20)},
            },
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        payload = response.json()
        deadline = time.monotonic() + 15
        while payload["status"] in {"queued", "running"}:
            assert time.monotonic() < deadline
            time.sleep(0.05)
            payload = client.get(f"/api/v1/runs/{run_id}").json()
        assert payload["status"] == "completed"
        assert payload["llm_calls"] == 5
        metrics = client.get("/api/v1/metrics/summary").json()
        assert metrics["runs_started"] >= 1
        assert metrics["total_tokens"] > 0

