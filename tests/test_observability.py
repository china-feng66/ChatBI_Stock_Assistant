from __future__ import annotations

from datetime import date, timedelta
import math

from fastapi.testclient import TestClient

from quantquery_a.api import create_app
from quantquery_a.workbench.benchmark import RoutingBenchmark
from quantquery_a.workbench.contracts import ResearchMode, RunCreateRequest
from quantquery_a.workbench.llm import FakeQwenClient
from quantquery_a.workbench.observability import ObservabilityService
from quantquery_a.workbench.service import WorkbenchService


def _bars(count: int) -> list[dict[str, object]]:
    start = date(2025, 1, 1)
    return [
        {
            "trade_date": (start + timedelta(days=index)).isoformat(),
            "open": 100.2 + math.sin(index / 4) * 5 + index * 0.03,
            "close": 100.0 + math.sin(index / 4) * 5 + index * 0.03,
        }
        for index in range(count)
    ]


def _completed_service(tmp_path) -> tuple[WorkbenchService, str]:
    service = WorkbenchService(
        runtime_dir=tmp_path / "observable-runtime",
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
    return service, snapshot.run_id


def test_run_observability_projects_agents_edges_and_gates(tmp_path) -> None:
    service, run_id = _completed_service(tmp_path)
    observer = ObservabilityService(service.database_path)
    detail = observer.run_detail(service.get_run(run_id))

    assert detail["status"] == "completed"
    assert detail["token_total"] > 0
    assert detail["gates"]["t_plus_one_violation_count"] == 0
    assert detail["gates"]["evidence_completeness_rate"] == 1.0
    assert {item["agent"] for item in detail["agent_metrics"]} == {
        "planner",
        "data",
        "quant",
        "risk",
        "reporter",
    }
    edges = {
        (item["source"], item["target"])
        for item in detail["communication_edges"]
    }
    assert ("orchestrator", "planner") in edges
    assert ("planner", "data") in edges
    assert ("data", "quant") in edges
    assert ("quant", "risk") in edges
    assert ("risk", "reporter") in edges

    summary = observer.system_summary()
    assert summary["technical_terminal_rate"] == 1.0
    assert summary["evidence_completeness_rate"] == 1.0
    assert summary["token_over_budget_count"] == 0
    assert summary["llm_latency_p95_ms"] >= 0


def test_observability_and_routing_benchmark_api(tmp_path) -> None:
    service, run_id = _completed_service(tmp_path)
    with TestClient(create_app(workbench_service=service)) as client:
        detail = client.get(f"/api/v1/runs/{run_id}/observability")
        assert detail.status_code == 200
        assert len(detail.json()["communication_edges"]) >= 5

        metrics = client.get("/api/v1/metrics/observability")
        assert metrics.status_code == 200
        assert len(metrics.json()["agent_metrics"]) == 5

        benchmark = client.post("/api/v1/evals/routing")
        assert benchmark.status_code == 200
        payload = benchmark.json()
        assert payload["case_count"] >= 30
        assert payload["top1_accuracy"] >= 0.9
        assert payload["clarification_recall"] >= 0.9


def test_routing_benchmark_persists_latest_result(tmp_path) -> None:
    database = tmp_path / "routing.sqlite3"
    benchmark = RoutingBenchmark(database)
    summary = benchmark.run()
    latest = benchmark.latest()

    assert summary.error_count == 0
    assert summary.top1_accuracy >= 0.9
    assert summary.top3_recall >= summary.top1_accuracy
    assert summary.clarification_recall == 1.0
    assert summary.unsupported_block_recall == 1.0
    assert latest and latest.benchmark_id == summary.benchmark_id
