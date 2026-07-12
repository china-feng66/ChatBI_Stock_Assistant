from __future__ import annotations

import json

from quantquery_a.agents.contracts import AnalyzeRequest, RoutePath, TaskType
from quantquery_a.observability.trace import TraceRecorder
from quantquery_a.routing.ensemble import (
    EnsembleIntentRouter,
    RuleIntentScorer,
    StaticIntentScorer,
)


def test_three_source_router_fuses_probabilities() -> None:
    semantic = StaticIntentScorer(
        "semantic_similarity",
        {TaskType.BACKTEST: 0.8, TaskType.STRATEGY_ANALYSIS: 0.2},
    )
    llm = StaticIntentScorer(
        "llm_classifier",
        {TaskType.BACKTEST: 0.9, TaskType.RISK_DIAGNOSIS: 0.1},
    )
    router = EnsembleIntentRouter(
        [RuleIntentScorer(), semantic, llm],
        weights={
            "keyword_rules": 0.4,
            "semantic_similarity": 0.25,
            "llm_classifier": 0.35,
        },
    )

    decision = router.route(AnalyzeRequest(query="回测 MACD 策略", symbols=["DEMO.SH"]))

    assert decision.task_type is TaskType.BACKTEST
    assert decision.path is RoutePath.PLANNED
    assert {score.source for score in decision.source_scores} == {
        "keyword_rules",
        "semantic_similarity",
        "llm_classifier",
    }
    assert 0 < decision.confidence <= 1


def test_router_requests_clarification_for_unknown_intent() -> None:
    decision = EnsembleIntentRouter().route(
        AnalyzeRequest(query="帮我看看这个", symbols=["DEMO.SH"])
    )

    assert decision.path is RoutePath.CLARIFICATION
    assert decision.confidence < 0.45


def test_trace_redacts_sensitive_attributes(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(path)
    trace_id = recorder.new_trace_id()

    recorder.emit(
        trace_id,
        span="router",
        event="completed",
        status="ok",
        attributes={"api_key": "do-not-write", "nested": {"token": "secret"}},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["attributes"]["api_key"] == "[REDACTED]"
    assert payload["attributes"]["nested"]["token"] == "[REDACTED]"
    assert "do-not-write" not in path.read_text(encoding="utf-8")
