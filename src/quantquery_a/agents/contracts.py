"""Typed contracts shared by the router and all agent roles."""

from __future__ import annotations

from datetime import date
from enum import Enum
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.]{0,15}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TaskType(str, Enum):
    MARKET_QUERY = "market_query"
    STRATEGY_ANALYSIS = "strategy_analysis"
    BACKTEST = "backtest"
    RISK_DIAGNOSIS = "risk_diagnosis"
    STOCK_COMPARISON = "stock_comparison"
    KNOWLEDGE_EXPLAIN = "knowledge_explain"
    UNSUPPORTED = "unsupported"


class RoutePath(str, Enum):
    FAST = "fast"
    PLANNED = "planned"
    CLARIFICATION = "clarification"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    NEEDS_CLARIFICATION = "needs_clarification"
    ERROR = "error"


class MarketBar(StrictModel):
    trade_date: date
    open: float = Field(gt=0)
    close: float = Field(gt=0)


class MacdSpec(StrictModel):
    fast: int = Field(default=12, ge=1, le=500)
    slow: int = Field(default=26, ge=2, le=1_000)
    signal: int = Field(default=9, ge=1, le=500)

    @model_validator(mode="after")
    def validate_periods(self) -> "MacdSpec":
        if self.fast >= self.slow:
            raise ValueError("fast must be smaller than slow")
        return self


class CostSpec(StrictModel):
    initial_cash: float = Field(default=100_000.0, gt=0)
    lot_size: int = Field(default=100, ge=1, le=1_000_000)
    commission_rate: float = Field(default=0.0, ge=0, le=1)
    min_commission: float = Field(default=0.0, ge=0)
    sell_tax_rate: float = Field(default=0.0, ge=0, le=1)
    slippage_bps: float = Field(default=0.0, ge=0, le=10_000)
    annualization_days: int = Field(default=252, ge=1, le=366)


class AnalyzeRequest(StrictModel):
    query: str = Field(min_length=1, max_length=1_000)
    symbols: list[str] = Field(default_factory=list, max_length=8)
    bars: dict[str, list[MarketBar]] = Field(default_factory=dict)
    task_hint: TaskType | None = None
    start_date: date | None = None
    end_date: date | None = None
    frequency: Literal["1d"] = "1d"
    strategy: Literal["macd"] = "macd"
    macd: MacdSpec = Field(default_factory=MacdSpec)
    costs: CostSpec = Field(default_factory=CostSpec)
    benchmark: Literal["buy_and_hold"] = "buy_and_hold"

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, symbols: list[str]) -> list[str]:
        normalized: list[str] = []
        for symbol in symbols:
            value = symbol.strip().upper()
            if not value:
                raise ValueError("symbols cannot contain an empty value")
            if not _SYMBOL_RE.fullmatch(value):
                raise ValueError("symbols must use a safe market identifier format")
            if value not in normalized:
                normalized.append(value)
        return normalized

    @field_validator("bars", mode="before")
    @classmethod
    def normalize_bar_keys(cls, bars: object) -> object:
        if not isinstance(bars, dict):
            return bars
        return {str(key).strip().upper(): value for key, value in bars.items()}

    @model_validator(mode="after")
    def validate_request(self) -> "AnalyzeRequest":
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date cannot be after end_date")
        if any(len(symbol_bars) > 5_000 for symbol_bars in self.bars.values()):
            raise ValueError("each symbol is limited to 5000 bars")
        if sum(len(symbol_bars) for symbol_bars in self.bars.values()) > 20_000:
            raise ValueError("request is limited to 20000 total bars")
        unknown_symbols = set(self.bars).difference(self.symbols)
        if unknown_symbols:
            raise ValueError(
                "bars cannot contain symbols outside the requested universe"
            )
        return self


class IntentSourceScore(StrictModel):
    source: str
    probabilities: dict[str, float]

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, values: dict[str, float]) -> dict[str, float]:
        if not values:
            raise ValueError("probabilities cannot be empty")
        if any(value < 0 or value > 1 for value in values.values()):
            raise ValueError("probabilities must be between 0 and 1")
        return values


class RouteDecision(StrictModel):
    task_type: TaskType
    path: RoutePath
    confidence: float = Field(ge=0, le=1)
    margin: float = Field(ge=0, le=1)
    source_scores: list[IntentSourceScore]
    missing_fields: list[str] = Field(default_factory=list)


class QueryFrame(StrictModel):
    query_hash: str
    task_type: TaskType
    route_path: RoutePath
    symbols: list[str]
    start_date: date | None
    end_date: date | None
    frequency: Literal["1d"]
    strategy: Literal["macd"]
    macd: MacdSpec
    signal_time: Literal["close"] = "close"
    execution_time: Literal["next_open"] = "next_open"
    costs: CostSpec
    benchmark: Literal["buy_and_hold"]
    metrics: list[str] = Field(
        default_factory=lambda: [
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "benchmark_return",
            "excess_return",
            "turnover",
            "total_cost",
        ]
    )
    evidence_requirements: list[str] = Field(
        default_factory=lambda: [
            "market_data_fingerprint",
            "signal_execution_timing",
            "cost_assumptions",
            "benchmark",
        ]
    )
    confidence: float = Field(ge=0, le=1)
    missing_fields: list[str] = Field(default_factory=list)


class EvidenceRecord(StrictModel):
    evidence_id: str
    source_type: Literal["inline_bars", "sqlite"]
    symbol: str
    row_count: int = Field(ge=0)
    start_date: date | None
    end_date: date | None
    fingerprint: str

    latest_close: float = Field(gt=0)


class TradeView(StrictModel):
    signal_date: date
    execution_date: date
    side: Literal["BUY", "SELL"]
    shares: int = Field(gt=0)
    fill_price: float = Field(gt=0)
    gross_value: float = Field(gt=0)
    commission: float = Field(ge=0)
    tax: float = Field(ge=0)


class SymbolAnalysis(StrictModel):
    symbol: str
    evidence_id: str
    observation_count: int = Field(ge=0)
    metrics: dict[str, float]
    trades: list[TradeView]
    assumptions: list[str]


class RiskFinding(StrictModel):
    code: str
    severity: Literal["error", "warning"]
    message: str
    retryable: bool = False


class RiskReview(StrictModel):
    approved: bool
    findings: list[RiskFinding]
    checked_symbols: list[str]

    @property
    def retryable(self) -> bool:
        return any(finding.retryable for finding in self.findings)


class AgentStep(StrictModel):
    role: Literal["router", "data", "quant", "risk", "supervisor"]
    status: Literal["ok", "blocked", "error", "skipped"]
    attempt: int = Field(default=1, ge=1)
    detail: str


class AnalysisReport(StrictModel):
    trace_id: str
    status: RunStatus
    route: RouteDecision
    frame: QueryFrame
    evidence: list[EvidenceRecord]
    analyses: list[SymbolAnalysis]
    risk_review: RiskReview | None
    steps: list[AgentStep]
    attempts: int = Field(ge=0)
    answer: str
