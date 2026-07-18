"""Read-only observability projections over persisted workbench state."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from .contracts import RunSnapshot


AGENT_ORDER = ("planner", "data", "quant", "risk", "reporter")
TERMINAL_STATUSES = {
    "completed",
    "blocked",
    "needs_clarification",
    "cancelled",
    "error",
}
EVIDENCE_FIELDS = {
    "provider",
    "as_of",
    "adjustment_status",
    "freshness",
    "fingerprint",
}


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _milliseconds(start: str | None, end: str | None) -> float:
    if not start or not end:
        return 0.0
    return max(
        (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
        * 1_000,
        0.0,
    )


class ObservabilityService:
    """Builds task and system dashboards without storing prompts or secrets."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def run_detail(self, snapshot: RunSnapshot) -> dict[str, Any]:
        with self._connect() as connection:
            event_rows = connection.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY sequence",
                (snapshot.run_id,),
            ).fetchall()
            usage_rows = connection.execute(
                "SELECT * FROM llm_usage WHERE run_id=? ORDER BY id",
                (snapshot.run_id,),
            ).fetchall()
        events = [
            {
                "sequence": row["sequence"],
                "type": row["event_type"],
                "agent": row["agent"],
                "status": row["status"],
                "timestamp": row["timestamp"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in event_rows
        ]
        usages = [dict(row) for row in usage_rows]
        agent_metrics = self._agent_metrics(events, usages)
        edges = self._communication_edges(events)
        gates = self._gates(snapshot)
        return {
            "run_id": snapshot.run_id,
            "status": snapshot.status.value,
            "created_at": snapshot.created_at.isoformat(),
            "updated_at": snapshot.updated_at.isoformat(),
            "duration_ms": _milliseconds(
                snapshot.created_at.isoformat(), snapshot.updated_at.isoformat()
            ),
            "route": snapshot.route,
            "plan": snapshot.plan,
            "token_total": snapshot.token_total,
            "token_budget": 15_000,
            "token_utilization": snapshot.token_total / 15_000,
            "llm_calls": snapshot.llm_calls,
            "llm_call_budget": 6,
            "repair_count": snapshot.repair_count,
            "agent_metrics": agent_metrics,
            "communication_edges": edges,
            "events": events,
            "gates": gates,
        }

    def system_summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            run_rows = connection.execute(
                "SELECT run_id, status, state_json, created_at, updated_at FROM runs"
            ).fetchall()
            usage_rows = connection.execute("SELECT * FROM llm_usage").fetchall()
        states = [json.loads(row["state_json"]) for row in run_rows]
        durations = [
            _milliseconds(row["created_at"], row["updated_at"]) for row in run_rows
        ]
        usages = [dict(row) for row in usage_rows]
        llm_latencies = [float(row["latency_ms"]) for row in usage_rows]
        completed = [row for row in run_rows if row["status"] == "completed"]
        terminal = [row for row in run_rows if row["status"] in TERMINAL_STATUSES]
        repaired = [state for state in states if int(state.get("repair_count", 0)) > 0]
        repaired_completed = [
            row
            for row, state in zip(run_rows, states, strict=True)
            if int(state.get("repair_count", 0)) > 0 and row["status"] == "completed"
        ]
        evidence = [
            item for state in states for item in state.get("evidence", [])
        ]
        complete_evidence = [
            item for item in evidence if EVIDENCE_FIELDS.issubset(item)
        ]
        route_confidences = [
            float(state["route"]["confidence"])
            for state in states
            if state.get("route") and state["route"].get("confidence") is not None
        ]
        agent_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for usage in usages:
            agent_groups[str(usage["agent"])].append(usage)
        agent_metrics = []
        for agent in AGENT_ORDER:
            rows = agent_groups.get(agent, [])
            latencies = [float(row["latency_ms"]) for row in rows]
            tokens = [int(row["total_tokens"]) for row in rows]
            agent_metrics.append(
                {
                    "agent": agent,
                    "calls": len(rows),
                    "success_rate": sum(int(row["success"]) for row in rows)
                    / (len(rows) or 1),
                    "error_rate": sum(not int(row["success"]) for row in rows)
                    / (len(rows) or 1),
                    "total_tokens": sum(tokens),
                    "average_tokens": sum(tokens) / (len(tokens) or 1),
                    "latency_p50_ms": _percentile(latencies, 0.50),
                    "latency_p95_ms": _percentile(latencies, 0.95),
                }
            )
        completed_tokens = [
            int(state.get("token_total", 0))
            for row, state in zip(run_rows, states, strict=True)
            if row["status"] == "completed"
        ]
        return {
            "runs_started": len(run_rows),
            "technical_terminal_rate": len(terminal) / (len(run_rows) or 1),
            "research_completion_rate": len(completed) / (len(run_rows) or 1),
            "clarification_rate": sum(
                row["status"] == "needs_clarification" for row in run_rows
            )
            / (len(run_rows) or 1),
            "repair_attempt_count": len(repaired),
            "repair_success_rate": len(repaired_completed) / (len(repaired) or 1),
            "average_route_confidence": sum(route_confidences)
            / (len(route_confidences) or 1),
            "evidence_batch_count": len(evidence),
            "evidence_completeness_rate": len(complete_evidence)
            / (len(evidence) or 1),
            "stale_evidence_rate": sum(bool(item.get("stale")) for item in evidence)
            / (len(evidence) or 1),
            "tokens_per_completed_run": sum(completed_tokens)
            / (len(completed_tokens) or 1),
            "task_latency_p50_ms": _percentile(durations, 0.50),
            "task_latency_p95_ms": _percentile(durations, 0.95),
            "llm_latency_p50_ms": _percentile(llm_latencies, 0.50),
            "llm_latency_p95_ms": _percentile(llm_latencies, 0.95),
            "llm_error_rate": sum(not int(row["success"]) for row in usage_rows)
            / (len(usage_rows) or 1),
            "token_over_budget_count": sum(
                int(state.get("token_total", 0)) > 15_000 for state in states
            ),
            "agent_metrics": agent_metrics,
        }

    @staticmethod
    def _agent_metrics(
        events: list[dict[str, Any]], usages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for usage in usages:
            by_agent[str(usage["agent"])].append(usage)
        event_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            if event.get("agent"):
                event_by_agent[str(event["agent"])].append(event)
        output = []
        for agent in AGENT_ORDER:
            rows = by_agent.get(agent, [])
            agent_events = event_by_agent.get(agent, [])
            latencies = [float(row["latency_ms"]) for row in rows]
            output.append(
                {
                    "agent": agent,
                    "calls": len(rows),
                    "status": agent_events[-1]["status"] if agent_events else "standby",
                    "prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows),
                    "completion_tokens": sum(
                        int(row["completion_tokens"]) for row in rows
                    ),
                    "total_tokens": sum(int(row["total_tokens"]) for row in rows),
                    "average_latency_ms": sum(latencies) / (len(latencies) or 1),
                    "latency_p95_ms": _percentile(latencies, 0.95),
                    "error_count": sum(not int(row["success"]) for row in rows),
                    "phase_duration_ms": _milliseconds(
                        agent_events[0]["timestamp"] if agent_events else None,
                        agent_events[-1]["timestamp"] if agent_events else None,
                    ),
                    "event_count": len(agent_events),
                }
            )
        return output

    @staticmethod
    def _communication_edges(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        starts = [
            event
            for event in events
            if event["type"] in {"agent.started", "repair.started"}
            and event.get("agent")
        ]
        edge_counts: Counter[tuple[str, str]] = Counter()
        edge_sequences: dict[tuple[str, str], list[int]] = defaultdict(list)
        previous = "orchestrator"
        for event in starts:
            current = str(event["agent"])
            edge = (previous, current)
            edge_counts[edge] += 1
            edge_sequences[edge].append(int(event["sequence"]))
            previous = current
        if starts:
            edge_counts[(previous, "terminal")] += 1
            edge_sequences[(previous, "terminal")].append(
                int(events[-1]["sequence"]) if events else 0
            )
        return [
            {
                "source": source,
                "target": target,
                "count": count,
                "sequences": edge_sequences[(source, target)],
                "contract": "typed_research_state",
            }
            for (source, target), count in edge_counts.items()
        ]

    @staticmethod
    def _gates(snapshot: RunSnapshot) -> dict[str, Any]:
        evidence = snapshot.evidence
        trades = [
            trade
            for analysis in snapshot.analyses
            for trade in analysis.get("trades", [])
        ]
        timing_violations = sum(
            trade.get("execution_date", "") <= trade.get("signal_date", "")
            for trade in trades
        )
        return {
            "risk_approved": bool(snapshot.risk and snapshot.risk.get("approved")),
            "risk_finding_count": len((snapshot.risk or {}).get("findings", [])),
            "evidence_count": len(evidence),
            "evidence_completeness_rate": sum(
                EVIDENCE_FIELDS.issubset(item) for item in evidence
            )
            / (len(evidence) or 1),
            "stale_evidence_count": sum(bool(item.get("stale")) for item in evidence),
            "unknown_adjustment_count": sum(
                item.get("adjustment_status") in {None, "unknown", "unadjusted"}
                for item in evidence
            ),
            "trade_count": len(trades),
            "t_plus_one_violation_count": timing_violations,
            "token_budget_passed": snapshot.token_total <= 15_000,
            "repair_scope_passed": snapshot.repair_count <= 1,
        }
