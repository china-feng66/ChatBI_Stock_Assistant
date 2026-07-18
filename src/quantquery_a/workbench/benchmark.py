"""Labeled routing benchmark for Fake Qwen and live Qwen evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from pydantic import Field

from quantquery_a.agents.contracts import AnalyzeRequest, RoutePath, TaskType

from .contracts import WorkbenchModel
from .llm import FakeQwenClient, QwenClient
from .observability import _percentile
from .routing import HybridIntentRouter


class RoutingCase(WorkbenchModel):
    name: str
    query: str
    expected_task: TaskType | None
    symbols: list[str] = Field(default_factory=lambda: ["DEMO.SH"])
    expect_clarification: bool = False


class RoutingCaseResult(WorkbenchModel):
    name: str
    query: str
    expected_task: str | None
    actual_task: str
    route_path: str
    top1_correct: bool
    top3_correct: bool
    clarification_expected: bool
    clarification_actual: bool
    confidence: float
    margin: float
    tokens: int
    latency_ms: float
    error_type: str | None = None


class RoutingBenchmarkSummary(WorkbenchModel):
    benchmark_id: str
    mode: str
    model: str
    case_count: int
    evaluated_count: int
    error_count: int
    top1_accuracy: float
    top3_recall: float
    clarification_precision: float
    clarification_recall: float
    clarification_f1: float
    unsupported_block_recall: float
    total_tokens: int
    average_tokens: float
    latency_p50_ms: float
    latency_p95_ms: float
    confusion_matrix: dict[str, dict[str, int]]
    cases: list[RoutingCaseResult]
    created_at: datetime


def _labeled(
    prefix: str,
    task: TaskType,
    queries: tuple[str, ...],
    *,
    symbols: list[str] | None = None,
) -> list[RoutingCase]:
    return [
        RoutingCase(
            name=f"{prefix}_{index + 1}",
            query=query,
            expected_task=task,
            symbols=symbols if symbols is not None else ["DEMO.SH"],
        )
        for index, query in enumerate(queries)
    ]


ROUTING_CASES: tuple[RoutingCase, ...] = tuple(
    _labeled(
        "market",
        TaskType.MARKET_QUERY,
        (
            "查询 600519.SH 最新收盘价",
            "今天 000001.SZ 的行情怎么样",
            "拉取 DEMO.SH 最近二十个交易日行情",
            "给我看一下 600036.SH 的实时快照",
            "查询 000858.SZ 最新价格和成交量",
        ),
    )
    + _labeled(
        "strategy",
        TaskType.STRATEGY_ANALYSIS,
        (
            "分析 600519.SH 的 MACD 信号",
            "用布林带研究 DEMO.SH 买卖区间",
            "分析 000001.SZ 当前技术信号",
            "研究 600036.SH 的入场和离场条件",
        ),
    )
    + _labeled(
        "backtest",
        TaskType.BACKTEST,
        (
            "回测 DEMO.SH 的 MACD 策略",
            "计算交易成本后的历史回测表现",
            "回测历史表现并使用买入持有作为基准",
            "评估 600519.SH 过去一年的策略收益",
            "验证收盘信号次日开盘成交的回测",
        ),
    )
    + _labeled(
        "risk",
        TaskType.RISK_DIAGNOSIS,
        (
            "检查策略最大回撤是否异常",
            "诊断策略是否错误使用未来数据",
            "复核手续费和滑点是否被遗漏",
            "检查复权状态和数据来源风险",
        ),
    )
    + _labeled(
        "comparison",
        TaskType.STOCK_COMPARISON,
        (
            "比较 600519.SH 和 000858.SZ 的表现",
            "给自选池股票按收益和风险排序",
            "构建价格成交量三因子组合",
            "用动量低波动和流动性因子选择前五只",
            "哪个更好",
        ),
        symbols=["AAA.SH", "BBB.SH", "CCC.SH"],
    )
    + _labeled(
        "knowledge",
        TaskType.KNOWLEDGE_EXPLAIN,
        (
            "什么是夏普比率",
            "什么是收益波动的统计含义",
            "解释信号确认与执行时间错开的原理",
            "说明数据标准化和 Z-score 的原理",
        ),
        symbols=[],
    )
    + _labeled(
        "unsupported",
        TaskType.UNSUPPORTED,
        (
            "替我下单买入 600519.SH",
            "连接实盘账户自动交易",
            "保证这个策略每个月盈利",
        ),
    )
    + [
        RoutingCase(
            name=f"ambiguous_{index + 1}",
            query=query,
            expected_task=None,
            expect_clarification=True,
        )
        for index, query in enumerate(
            (
                "帮我看看 DEMO.SH",
                "分析一下这个",
                "这个策略怎么样",
                "最近有什么变化",
            )
        )
    ]
)


class RoutingBenchmark:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS routing_benchmark_results (
                benchmark_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                model TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
                )"""
            )

    def run(
        self,
        *,
        live: bool = False,
        client: QwenClient | None = None,
        cases: tuple[RoutingCase, ...] = ROUTING_CASES,
    ) -> RoutingBenchmarkSummary:
        llm = client or FakeQwenClient()
        router = HybridIntentRouter()
        results = [self._evaluate_case(case, llm, router) for case in cases]
        summary = _summarize(results, llm.model, live)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO routing_benchmark_results VALUES (?, ?, ?, ?, ?)",
                (
                    summary.benchmark_id,
                    summary.mode,
                    summary.model,
                    summary.model_dump_json(),
                    summary.created_at.isoformat(),
                ),
            )
        return summary

    @staticmethod
    def _evaluate_case(
        case: RoutingCase,
        llm: QwenClient,
        router: HybridIntentRouter,
    ) -> RoutingCaseResult:
        try:
            response = llm.complete(
                agent="planner",
                system_prompt=(
                    "Classify the user's quantitative-research intent. Return a JSON "
                    "object with probabilities for every label: market_query, "
                    "strategy_analysis, backtest, risk_diagnosis, stock_comparison, "
                    "knowledge_explain, unsupported. Probabilities must sum to one. "
                    "Ambiguous requests should have close top scores."
                ),
                payload={"query": case.query, "symbols": case.symbols, "mode": "auto"},
                max_output_tokens=500,
            )
            decision = router.route(
                AnalyzeRequest(query=case.query, symbols=case.symbols),
                _probabilities(response.content),
            )
            ranked = _ranked_tasks(decision)
            clarification = decision.path is RoutePath.CLARIFICATION
            return RoutingCaseResult(
                name=case.name,
                query=case.query,
                expected_task=(case.expected_task.value if case.expected_task else None),
                actual_task=decision.task_type.value,
                route_path=decision.path.value,
                top1_correct=(
                    decision.task_type is case.expected_task
                    if case.expected_task is not None
                    else clarification
                ),
                top3_correct=(
                    case.expected_task in ranked[:3]
                    if case.expected_task is not None
                    else clarification
                ),
                clarification_expected=case.expect_clarification,
                clarification_actual=clarification,
                confidence=decision.confidence,
                margin=decision.margin,
                tokens=response.usage.total_tokens,
                latency_ms=response.usage.latency_ms,
            )
        except Exception as exc:
            return RoutingCaseResult(
                name=case.name,
                query=case.query,
                expected_task=(case.expected_task.value if case.expected_task else None),
                actual_task="error",
                route_path="error",
                top1_correct=False,
                top3_correct=False,
                clarification_expected=case.expect_clarification,
                clarification_actual=False,
                confidence=0.0,
                margin=0.0,
                tokens=0,
                latency_ms=0.0,
                error_type=type(exc).__name__,
            )

    def latest(self) -> RoutingBenchmarkSummary | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM routing_benchmark_results "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return RoutingBenchmarkSummary.model_validate_json(row[0]) if row else None


def _probabilities(content: dict[str, Any]) -> dict[str, float]:
    raw = content.get("probabilities")
    if isinstance(raw, dict):
        return {str(key): float(value) for key, value in raw.items()}
    task = str(content.get("task_type", "market_query"))
    return {item.value: 1.0 if item.value == task else 0.0 for item in TaskType}


def _ranked_tasks(decision: Any) -> list[TaskType]:
    combined = {task: 0.0 for task in TaskType}
    weights = {"llm": 0.45, "tfidf": 0.30, "keyword": 0.25}
    for source in decision.source_scores:
        for task in TaskType:
            combined[task] += weights.get(source.source, 0.0) * float(
                source.probabilities.get(task.value, 0.0)
            )
    return sorted(combined, key=combined.get, reverse=True)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _summarize(
    results: list[RoutingCaseResult], model: str, live: bool
) -> RoutingBenchmarkSummary:
    labeled = [item for item in results if item.expected_task is not None]
    expected_clarification = [item for item in results if item.clarification_expected]
    actual_clarification = [item for item in results if item.clarification_actual]
    true_clarification = [
        item
        for item in results
        if item.clarification_expected and item.clarification_actual
    ]
    unsupported = [item for item in results if item.expected_task == "unsupported"]
    precision = _ratio(len(true_clarification), len(actual_clarification))
    recall = _ratio(len(true_clarification), len(expected_clarification))
    confusion: dict[str, dict[str, int]] = {}
    for item in labeled:
        expected = str(item.expected_task)
        confusion.setdefault(expected, {})
        confusion[expected][item.actual_task] = (
            confusion[expected].get(item.actual_task, 0) + 1
        )
    latencies = [item.latency_ms for item in results if item.error_type is None]
    tokens = [item.tokens for item in results]
    return RoutingBenchmarkSummary(
        benchmark_id=uuid4().hex,
        mode="live" if live else "mock",
        model=model,
        case_count=len(results),
        evaluated_count=sum(item.error_type is None for item in results),
        error_count=sum(item.error_type is not None for item in results),
        top1_accuracy=_ratio(sum(item.top1_correct for item in labeled), len(labeled)),
        top3_recall=_ratio(sum(item.top3_correct for item in labeled), len(labeled)),
        clarification_precision=precision,
        clarification_recall=recall,
        clarification_f1=(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        unsupported_block_recall=_ratio(
            sum(item.actual_task == "unsupported" for item in unsupported),
            len(unsupported),
        ),
        total_tokens=sum(tokens),
        average_tokens=sum(tokens) / (len(tokens) or 1),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        confusion_matrix=confusion,
        cases=results,
        created_at=datetime.now(timezone.utc),
    )
