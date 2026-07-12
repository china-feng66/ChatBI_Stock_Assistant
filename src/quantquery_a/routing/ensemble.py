"""Calibratable multi-source intent routing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from quantquery_a.agents.contracts import (
    AnalyzeRequest,
    IntentSourceScore,
    RouteDecision,
    RoutePath,
    TaskType,
)


class IntentScorer(Protocol):
    name: str

    def score(self, query: str) -> Mapping[TaskType, float]: ...


def _normalize(values: Mapping[TaskType, float]) -> dict[TaskType, float]:
    cleaned = {task: max(float(values.get(task, 0.0)), 0.0) for task in TaskType}
    total = sum(cleaned.values())
    if total <= 0:
        uniform = 1.0 / len(TaskType)
        return {task: uniform for task in TaskType}
    return {task: value / total for task, value in cleaned.items()}


class RuleIntentScorer:
    """Transparent first-stage scorer; weights are evaluation starting points."""

    name = "keyword_rules"

    _keywords: dict[TaskType, tuple[tuple[str, float], ...]] = {
        TaskType.MARKET_QUERY: (
            ("查询", 1.5),
            ("收盘价", 2.0),
            ("行情", 2.0),
            ("股价", 1.5),
            ("market", 1.5),
        ),
        TaskType.STRATEGY_ANALYSIS: (
            ("macd", 2.0),
            ("策略", 1.5),
            ("信号", 1.5),
            ("买卖点", 2.0),
        ),
        TaskType.BACKTEST: (
            ("回测", 6.0),
            ("收益", 1.0),
            ("backtest", 3.0),
        ),
        TaskType.RISK_DIAGNOSIS: (
            ("风险", 2.0),
            ("回撤", 2.0),
            ("前视", 2.0),
            ("夏普", 1.0),
        ),
        TaskType.STOCK_COMPARISON: (
            ("比较", 3.0),
            ("对比", 3.0),
            ("哪个", 1.0),
        ),
        TaskType.KNOWLEDGE_EXPLAIN: (
            ("解释", 2.0),
            ("什么是", 2.0),
            ("原理", 2.0),
            ("含义", 1.5),
        ),
        TaskType.UNSUPPORTED: (
            ("下单", 2.0),
            ("保证盈利", 3.0),
            ("实盘交易", 2.0),
        ),
    }

    def score(self, query: str) -> Mapping[TaskType, float]:
        lowered = query.lower()
        values = {task: 0.01 for task in TaskType}
        for task, keywords in self._keywords.items():
            for keyword, weight in keywords:
                if keyword in lowered:
                    values[task] += weight
        return _normalize(values)


@dataclass(frozen=True)
class StaticIntentScorer:
    """Test/adapter scorer used to plug in vector or LLM probabilities."""

    name: str
    probabilities: Mapping[TaskType, float]

    def score(self, query: str) -> Mapping[TaskType, float]:
        del query
        return _normalize(self.probabilities)


class EnsembleIntentRouter:
    def __init__(
        self,
        scorers: Sequence[IntentScorer] | None = None,
        *,
        weights: Mapping[str, float] | None = None,
        clarification_threshold: float = 0.45,
        margin_threshold: float = 0.10,
        fast_path_threshold: float = 0.65,
    ) -> None:
        self.scorers = tuple(scorers or (RuleIntentScorer(),))
        if not self.scorers:
            raise ValueError("at least one intent scorer is required")
        self.weights = dict(weights or {})
        self.clarification_threshold = clarification_threshold
        self.margin_threshold = margin_threshold
        self.fast_path_threshold = fast_path_threshold

    def route(self, request: AnalyzeRequest) -> RouteDecision:
        if request.task_hint is not None:
            probabilities = {task.value: 0.0 for task in TaskType}
            probabilities[request.task_hint.value] = 1.0
            source_scores = [
                IntentSourceScore(source="explicit_hint", probabilities=probabilities)
            ]
            task = request.task_hint
            confidence = 1.0
            margin = 1.0
        else:
            scored: list[tuple[IntentScorer, dict[TaskType, float]]] = [
                (scorer, _normalize(scorer.score(request.query)))
                for scorer in self.scorers
            ]
            raw_weights = [
                max(self.weights.get(scorer.name, 1.0), 0.0) for scorer, _ in scored
            ]
            weight_total = sum(raw_weights)
            if weight_total <= 0:
                raise ValueError("intent scorer weights must contain a positive value")
            normalized_weights = [weight / weight_total for weight in raw_weights]

            combined = {task_type: 0.0 for task_type in TaskType}
            source_scores = []
            for (scorer, probabilities), weight in zip(
                scored,
                normalized_weights,
                strict=True,
            ):
                for task_type, probability in probabilities.items():
                    combined[task_type] += weight * probability
                source_scores.append(
                    IntentSourceScore(
                        source=scorer.name,
                        probabilities={
                            task_type.value: probability
                            for task_type, probability in probabilities.items()
                        },
                    )
                )

            ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
            task, confidence = ranked[0]
            margin = confidence - ranked[1][1]

        missing_fields: list[str] = []
        tasks_requiring_symbols = {
            TaskType.MARKET_QUERY,
            TaskType.STRATEGY_ANALYSIS,
            TaskType.BACKTEST,
            TaskType.RISK_DIAGNOSIS,
            TaskType.STOCK_COMPARISON,
        }
        if task in tasks_requiring_symbols and not request.symbols:
            missing_fields.append("symbols")

        if (
            confidence < self.clarification_threshold
            or margin < self.margin_threshold
            or missing_fields
        ):
            path = RoutePath.CLARIFICATION
        elif (
            task in {TaskType.MARKET_QUERY, TaskType.KNOWLEDGE_EXPLAIN}
            and confidence >= self.fast_path_threshold
        ):
            path = RoutePath.FAST
        else:
            path = RoutePath.PLANNED

        return RouteDecision(
            task_type=task,
            path=path,
            confidence=confidence,
            margin=margin,
            source_scores=source_scores,
            missing_fields=missing_fields,
        )
