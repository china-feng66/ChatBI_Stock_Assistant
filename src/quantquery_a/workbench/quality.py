"""Frozen evaluation and manually promoted router-weight candidates."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from pydantic import Field

from quantquery_a.agents.contracts import AnalyzeRequest, TaskType

from .contracts import (
    RouterWeights,
    RunCreateRequest,
    WeightCandidate,
    WorkbenchModel,
)
from .llm import FakeQwenClient
from .market import BoundedMarketGateway, MarketCache, ProviderBatch
from .routing import HybridIntentRouter, INTENT_PROTOTYPES
from .service import WorkbenchService


TOKEN_BUDGET = 15_000


class WorkbenchEvalCaseResult(WorkbenchModel):
    name: str
    passed: bool
    expected_status: str
    actual_status: str
    timing_gate_passed: bool = True
    risk_gate_passed: bool = True
    token_gate_passed: bool = True
    repair_count: int = Field(default=0, ge=0, le=1)


class WorkbenchEvalSummary(WorkbenchModel):
    evaluation_id: str
    mode: str
    case_count: int
    passed_count: int
    route_accuracy: float
    technical_terminal_rate: float
    evaluation_task_success_rate: float
    research_completion_rate: float
    deterministic_gate_rate: float
    token_over_budget_count: int
    llm_judge_status: str
    cases: list[WorkbenchEvalCaseResult]
    created_at: datetime


class QualityManager:
    """Keeps quality artifacts separate from the runtime's execution tables."""

    def __init__(self, database_path: str | Path, runtime_dir: str | Path) -> None:
        self.database_path = Path(database_path)
        self.runtime_dir = Path(runtime_dir)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    evaluation_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS router_weight_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    promoted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                """
            )

    def evaluate(self, *, live: bool = False) -> WorkbenchEvalSummary:
        evaluation_id = uuid4().hex
        evaluation_root = self.runtime_dir / "evaluations" / evaluation_id
        llm = None if live else FakeQwenClient()
        cases: list[WorkbenchEvalCaseResult] = []

        inline_service = WorkbenchService(
            runtime_dir=evaluation_root / "inline",
            llm=llm,
        )
        bars = _bars(90)
        factor_bars = {
            symbol: _bars(90, offset=index * 8.0)
            for index, symbol in enumerate(("AAA.SH", "BBB.SH", "CCC.SH"))
        }
        requests = (
            (
                "market",
                RunCreateRequest(
                    query="查询 DEMO.SH 最新行情",
                    symbols=["DEMO.SH"],
                    mode="market",
                    bars={"DEMO.SH": bars[:30]},
                ),
                "completed",
            ),
            (
                "macd",
                RunCreateRequest(
                    query="回测 DEMO.SH 的 MACD 策略",
                    symbols=["DEMO.SH"],
                    mode="macd",
                    bars={"DEMO.SH": bars},
                    macd={"fast": 2, "slow": 5, "signal": 2},
                ),
                "completed",
            ),
            (
                "bollinger",
                RunCreateRequest(
                    query="研究 DEMO.SH 的布林带策略",
                    symbols=["DEMO.SH"],
                    mode="bollinger",
                    bars={"DEMO.SH": bars},
                ),
                "completed",
            ),
            (
                "factor",
                RunCreateRequest(
                    query="构建价格成交量三因子组合",
                    symbols=list(factor_bars),
                    mode="factor",
                    bars=factor_bars,
                ),
                "completed",
            ),
            (
                "comparison",
                RunCreateRequest(
                    query="比较 AAA.SH 和 BBB.SH",
                    symbols=["AAA.SH", "BBB.SH"],
                    mode="compare",
                    bars={key: factor_bars[key] for key in ("AAA.SH", "BBB.SH")},
                ),
                "completed",
            ),
        )
        for name, request, expected in requests:
            snapshot = inline_service.run_sync(request)
            cases.append(_case_result(name, expected, snapshot))

        unavailable = BoundedMarketGateway(
            cache=MarketCache(evaluation_root / "unavailable.sqlite3"),
            daily_providers=[],
        )
        unavailable_service = WorkbenchService(
            runtime_dir=evaluation_root / "unavailable",
            llm=llm,
            market_gateway=unavailable,
        )
        snapshot = unavailable_service.run_sync(
            RunCreateRequest(
                query="查询没有缓存的数据",
                symbols=["MISS.SH"],
                mode="market",
            )
        )
        cases.append(_case_result("data_source_failure", "blocked", snapshot))

        blocked_provider = _SequenceProvider(["unadjusted", "unadjusted"])
        blocked_service = _provider_service(
            evaluation_root / "risk-block", blocked_provider, llm
        )
        snapshot = blocked_service.run_sync(_provider_request())
        cases.append(_case_result("risk_block", "blocked", snapshot))

        repair_provider = _SequenceProvider(["unadjusted", "qfq"])
        repair_service = _provider_service(
            evaluation_root / "repair", repair_provider, llm
        )
        snapshot = repair_service.run_sync(_provider_request())
        result = _case_result("allowlisted_repair", "completed", snapshot)
        result.passed = result.passed and snapshot.repair_count == 1
        cases.append(result)

        route_accuracy = _route_accuracy(RouterWeights(llm=0.45, tfidf=0.30, keyword=0.25))
        terminal = {"completed", "blocked", "needs_clarification"}
        passed_count = sum(item.passed for item in cases)
        executable = [item for item in cases if item.expected_status == "completed"]
        approved = [
            item
            for item in executable
            if item.actual_status == "completed" and item.risk_gate_passed
        ]
        summary = WorkbenchEvalSummary(
            evaluation_id=evaluation_id,
            mode="live" if live else "mock",
            case_count=len(cases),
            passed_count=passed_count,
            route_accuracy=route_accuracy,
            technical_terminal_rate=sum(item.actual_status in terminal for item in cases)
            / len(cases),
            evaluation_task_success_rate=passed_count / len(cases),
            research_completion_rate=len(approved) / (len(executable) or 1),
            deterministic_gate_rate=sum(
                item.timing_gate_passed and item.risk_gate_passed for item in cases
            )
            / len(cases),
            token_over_budget_count=sum(not item.token_gate_passed for item in cases),
            llm_judge_status=(
                "not_run: deterministic gates are authoritative; judge is limited to "
                "relevance, clarity and evidence consistency"
            ),
            cases=cases,
            created_at=datetime.now(timezone.utc),
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evaluation_results VALUES (?, ?, ?, ?)",
                (
                    summary.evaluation_id,
                    summary.mode,
                    summary.model_dump_json(),
                    summary.created_at.isoformat(),
                ),
            )
        return summary

    def generate_candidate(self) -> WeightCandidate:
        baseline = RouterWeights(llm=0.45, tfidf=0.30, keyword=0.25)
        baseline_accuracy = _route_accuracy(baseline)
        alternatives = (
            RouterWeights(llm=0.50, tfidf=0.25, keyword=0.25),
            RouterWeights(llm=0.40, tfidf=0.35, keyword=0.25),
            RouterWeights(llm=0.40, tfidf=0.30, keyword=0.30),
        )
        selected = max(alternatives, key=_route_accuracy)
        candidate = WeightCandidate(
            candidate_id=uuid4().hex,
            weights=selected,
            baseline_accuracy=baseline_accuracy,
            candidate_accuracy=_route_accuracy(selected),
            promoted=False,
            created_at=datetime.now(timezone.utc),
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO router_weight_candidates VALUES (?, ?, 0, ?)",
                (
                    candidate.candidate_id,
                    candidate.model_dump_json(),
                    candidate.created_at.isoformat(),
                ),
            )
        return candidate

    def candidates(self) -> list[WeightCandidate]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json, promoted FROM router_weight_candidates "
                "ORDER BY created_at DESC"
            ).fetchall()
        values = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["promoted"] = bool(row["promoted"])
            values.append(WeightCandidate.model_validate(payload))
        return values

    def promote(self, candidate_id: str, workbench: WorkbenchService) -> WeightCandidate:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM router_weight_candidates WHERE candidate_id=?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise KeyError(candidate_id)
            payload = json.loads(row["payload_json"])
            candidate = WeightCandidate.model_validate({**payload, "promoted": True})
            if candidate.candidate_accuracy < candidate.baseline_accuracy:
                raise ValueError("candidate failed the frozen routing evaluation")
            connection.execute("UPDATE router_weight_candidates SET promoted=0")
            connection.execute(
                "UPDATE router_weight_candidates SET promoted=1 WHERE candidate_id=?",
                (candidate_id,),
            )
        workbench._workflow.router = HybridIntentRouter(weights=candidate.weights)
        return candidate

    def apply_active(self, workbench: WorkbenchService) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM router_weight_candidates "
                "WHERE promoted=1 ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if row:
            candidate = WeightCandidate.model_validate_json(row["payload_json"])
            workbench._workflow.router = HybridIntentRouter(weights=candidate.weights)


def _bars(count: int, offset: float = 0.0) -> list[dict[str, Any]]:
    start = date(2025, 1, 1)
    return [
        {
            "trade_date": (start + timedelta(days=index)).isoformat(),
            "open": 100.2 + offset + math.sin(index / 4) * 5 + index * 0.03,
            "close": 100.0 + offset + math.sin(index / 4) * 5 + index * 0.03,
        }
        for index in range(count)
    ]


def _case_result(name: str, expected: str, snapshot: Any) -> WorkbenchEvalCaseResult:
    timing = all(
        trade["execution_date"] > trade["signal_date"]
        for analysis in snapshot.analyses
        for trade in analysis.get("trades", [])
    )
    if expected == "completed":
        risk = bool(snapshot.risk and snapshot.risk.get("approved"))
    elif expected == "blocked" and snapshot.risk:
        risk = not snapshot.risk.get("approved", False)
    else:
        risk = True
    token = snapshot.token_total <= TOKEN_BUDGET
    return WorkbenchEvalCaseResult(
        name=name,
        passed=snapshot.status.value == expected and timing and risk and token,
        expected_status=expected,
        actual_status=snapshot.status.value,
        timing_gate_passed=timing,
        risk_gate_passed=risk,
        token_gate_passed=token,
        repair_count=snapshot.repair_count,
    )


def _route_accuracy(weights: RouterWeights) -> float:
    router = HybridIntentRouter(weights=weights)
    correct = 0
    for query, expected in INTENT_PROTOTYPES:
        probabilities = {task.value: 0.01 for task in TaskType}
        probabilities[expected.value] = 0.94
        decision = router.route(
            AnalyzeRequest(query=query, symbols=["DEMO.SH"]), probabilities
        )
        correct += decision.task_type is expected
    ambiguous = router.route(
        AnalyzeRequest(query="帮我看看", symbols=["DEMO.SH"]),
        {task.value: 1.0 for task in TaskType},
    )
    correct += ambiguous.path.value == "clarification"
    return correct / (len(INTENT_PROTOTYPES) + 1)


class _SequenceProvider:
    name = "frozen-eval-provider"

    def __init__(self, adjustments: list[str]) -> None:
        self.adjustments = adjustments
        self.calls = 0

    def fetch_daily(self, symbol: str, *, start_date: Any, end_date: Any, limit: int) -> ProviderBatch:
        del start_date, end_date
        index = min(self.calls, len(self.adjustments) - 1)
        adjustment = self.adjustments[index]
        self.calls += 1
        rows = _bars(min(limit, 90))
        return ProviderBatch(
            provider=self.name,
            symbol=symbol,
            rows=rows,
            adjustment_status=adjustment,
            as_of=str(rows[-1]["trade_date"]),
        )


def _provider_service(
    runtime_dir: Path,
    provider: _SequenceProvider,
    llm: Any,
) -> WorkbenchService:
    gateway = BoundedMarketGateway(
        cache=MarketCache(runtime_dir / "market.sqlite3"),
        daily_providers=[provider],
    )
    return WorkbenchService(runtime_dir=runtime_dir, llm=llm, market_gateway=gateway)


def _provider_request() -> RunCreateRequest:
    return RunCreateRequest(
        query="回测 DEMO.SH 的 MACD 策略",
        symbols=["DEMO.SH"],
        mode="macd",
        macd={"fast": 2, "slow": 5, "signal": 2},
    )
