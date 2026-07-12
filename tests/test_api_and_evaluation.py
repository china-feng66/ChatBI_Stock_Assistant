from __future__ import annotations

from datetime import date, timedelta
import math

from fastapi.testclient import TestClient

from quantquery_a.api import create_app


def _bars(count: int = 90) -> list[dict[str, object]]:
    start = date(2025, 1, 1)
    return [
        {
            "trade_date": (start + timedelta(days=index)).isoformat(),
            "open": 100.2 + math.sin(index / 4) * 5 + index * 0.03,
            "close": 100.0 + math.sin(index / 4) * 5 + index * 0.03,
        }
        for index in range(count)
    ]


def test_analyze_api_exposes_audited_loop() -> None:
    client = TestClient(create_app())

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["loop"] == "data-quant-risk"

    response = client.post(
        "/analyze",
        json={
            "query": "回测 DEMO.SH 的 MACD 策略",
            "task_hint": "backtest",
            "symbols": ["DEMO.SH"],
            "bars": {"DEMO.SH": _bars()},
            "macd": {"fast": 2, "slow": 5, "signal": 2},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["risk_review"]["approved"] is True
    assert [step["role"] for step in payload["steps"][:4]] == [
        "router",
        "data",
        "quant",
        "risk",
    ]


def test_analyze_api_blocks_an_empty_filtered_window_without_500() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/analyze",
        json={
            "query": "?? DEMO.SH ??",
            "task_hint": "market_query",
            "symbols": ["DEMO.SH"],
            "bars": {"DEMO.SH": _bars(2)},
            "start_date": "2026-01-01",
            "end_date": "2026-01-31",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "blocked"


def test_eval_endpoint_runs_frozen_synthetic_cases() -> None:
    client = TestClient(create_app())
    response = client.post("/eval/run")

    assert response.status_code == 200
    payload = response.json()
    assert payload["case_count"] == 4
    assert payload["passed_count"] == 4
    assert payload["pass_rate"] == 1.0
    assert payload["llm_judge_status"].startswith("not_configured")
