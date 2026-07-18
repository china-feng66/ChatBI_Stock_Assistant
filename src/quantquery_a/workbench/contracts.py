"""Typed public contracts for the realtime research workbench."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quantquery_a.agents.contracts import CostSpec, MacdSpec, MarketBar, TaskType


class WorkbenchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class WorkbenchStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    NEEDS_CLARIFICATION = "needs_clarification"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    ERROR = "error"


class ResearchMode(str, Enum):
    AUTO = "auto"
    MARKET = "market"
    MACD = "macd"
    BOLLINGER = "bollinger"
    FACTOR = "factor"
    COMPARE = "compare"


class RunCreateRequest(WorkbenchModel):
    query: str = Field(min_length=1, max_length=1_000)
    symbols: list[str] = Field(min_length=1, max_length=20)
    mode: ResearchMode = ResearchMode.AUTO
    start_date: date | None = None
    end_date: date | None = None
    bars: dict[str, list[MarketBar]] = Field(default_factory=dict)
    macd: MacdSpec = Field(default_factory=MacdSpec)
    costs: CostSpec = Field(default_factory=CostSpec)

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            symbol = value.strip().upper()
            if not symbol or len(symbol) > 16:
                raise ValueError("invalid symbol")
            if symbol not in normalized:
                normalized.append(symbol)
        return normalized

    @field_validator("bars", mode="before")
    @classmethod
    def normalize_bar_keys(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        return {str(key).strip().upper(): rows for key, rows in value.items()}

    @model_validator(mode="after")
    def validate_request(self) -> "RunCreateRequest":
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date cannot be after end_date")
        if set(self.bars).difference(self.symbols):
            raise ValueError("bars contain symbols outside the watchlist")
        if sum(len(rows) for rows in self.bars.values()) > 5_000:
            raise ValueError("bounded workbench requests allow at most 5000 bars")
        return self


class AgentUsage(WorkbenchModel):
    agent: Literal["planner", "data", "quant", "risk", "reporter"]
    model: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    success: bool
    error_type: str | None = None


class AgentEvent(WorkbenchModel):
    sequence: int = Field(ge=1)
    run_id: str
    event_type: str
    agent: str | None = None
    status: str
    timestamp: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class RunSnapshot(WorkbenchModel):
    run_id: str
    status: WorkbenchStatus
    request: RunCreateRequest
    created_at: datetime
    updated_at: datetime
    route: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    analyses: list[dict[str, Any]] = Field(default_factory=list)
    risk: dict[str, Any] | None = None
    report: str | None = None
    token_total: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    repair_count: int = Field(default=0, ge=0, le=1)
    error: str | None = None


class WatchlistUpdate(WorkbenchModel):
    symbols: list[str] = Field(min_length=1, max_length=20)

    @field_validator("symbols")
    @classmethod
    def normalize(cls, values: list[str]) -> list[str]:
        return RunCreateRequest(query="watchlist", symbols=values).symbols


class RouterWeights(WorkbenchModel):
    llm: float = Field(ge=0, le=1)
    tfidf: float = Field(ge=0, le=1)
    keyword: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_total(self) -> "RouterWeights":
        if abs(self.llm + self.tfidf + self.keyword - 1.0) > 1e-6:
            raise ValueError("router weights must sum to one")
        return self


class WeightCandidate(WorkbenchModel):
    candidate_id: str
    weights: RouterWeights
    baseline_accuracy: float = Field(ge=0, le=1)
    candidate_accuracy: float = Field(ge=0, le=1)
    promoted: bool = False
    created_at: datetime


MODE_TASK_HINT: dict[ResearchMode, TaskType | None] = {
    ResearchMode.AUTO: None,
    ResearchMode.MARKET: TaskType.MARKET_QUERY,
    ResearchMode.MACD: TaskType.BACKTEST,
    ResearchMode.BOLLINGER: TaskType.STRATEGY_ANALYSIS,
    ResearchMode.FACTOR: TaskType.STOCK_COMPARISON,
    ResearchMode.COMPARE: TaskType.STOCK_COMPARISON,
}

