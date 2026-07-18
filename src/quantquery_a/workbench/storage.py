"""Small SQLite persistence layer for local workbench state and metrics."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any

from .contracts import AgentUsage, RunCreateRequest, WorkbenchStatus


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkbenchStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            request_json TEXT NOT NULL,
            state_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            run_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            agent TEXT,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (run_id, sequence)
        );
        CREATE TABLE IF NOT EXISTS llm_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            agent TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            latency_ms REAL NOT NULL,
            success INTEGER NOT NULL,
            error_type TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT PRIMARY KEY,
            position INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS market_bars (
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            trade_time TEXT NOT NULL,
            provider TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            adjustment_status TEXT NOT NULL,
            as_of TEXT NOT NULL,
            PRIMARY KEY (symbol, interval, trade_time, provider)
        );
        CREATE TABLE IF NOT EXISTS router_weight_candidates (
            candidate_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            promoted INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """
        with self._lock, self._connect() as connection:
            connection.executescript(schema)

    def create_run(self, run_id: str, request: RunCreateRequest) -> None:
        now = _utcnow()
        state = {
            "run_id": run_id,
            "status": WorkbenchStatus.QUEUED.value,
            "token_total": 0,
            "llm_calls": 0,
            "repair_count": 0,
        }
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (
                    run_id,
                    WorkbenchStatus.QUEUED.value,
                    request.model_dump_json(),
                    json.dumps(state, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def update_run(
        self,
        run_id: str,
        *,
        status: WorkbenchStatus,
        state: dict[str, Any],
        error: str | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status=?, state_json=?, updated_at=?, error=? WHERE run_id=?",
                (
                    status.value,
                    json.dumps(state, ensure_ascii=False, default=str),
                    _utcnow(),
                    error,
                    run_id,
                ),
            )

    def get_run_row(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_run_rows(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        status: str,
        agent: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        with self._lock, self._connect() as connection:
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM events WHERE run_id=?",
                    (run_id,),
                ).fetchone()[0]
            )
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    sequence,
                    event_type,
                    agent,
                    status,
                    _utcnow(),
                    json.dumps(payload or {}, ensure_ascii=False, default=str),
                ),
            )
        return sequence

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE run_id=? AND sequence>? ORDER BY sequence",
                (run_id, after),
            ).fetchall()
        return [
            {
                **dict(row),
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def record_usage(self, run_id: str, usage: AgentUsage) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO llm_usage
                (run_id, agent, model, prompt_tokens, completion_tokens,
                 total_tokens, latency_ms, success, error_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    usage.agent,
                    usage.model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                    usage.latency_ms,
                    int(usage.success),
                    usage.error_type,
                    _utcnow(),
                ),
            )

    def set_watchlist(self, symbols: list[str]) -> None:
        now = _utcnow()
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM watchlist")
            connection.executemany(
                "INSERT INTO watchlist(symbol, position, updated_at) VALUES (?, ?, ?)",
                [(symbol, index, now) for index, symbol in enumerate(symbols)],
            )

    def watchlist(self) -> list[str]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT symbol FROM watchlist ORDER BY position"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def metrics(self) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            status_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM runs GROUP BY status"
            ).fetchall()
            usage = connection.execute(
                """SELECT COALESCE(SUM(total_tokens), 0),
                COALESCE(AVG(latency_ms), 0), COUNT(*) FROM llm_usage"""
            ).fetchone()
            agent_rows = connection.execute(
                """SELECT agent, COUNT(*) AS calls, SUM(success) AS successes,
                SUM(total_tokens) AS tokens, AVG(latency_ms) AS latency
                FROM llm_usage GROUP BY agent"""
            ).fetchall()
        statuses = {row["status"]: row["count"] for row in status_rows}
        started = sum(statuses.values())
        terminal = sum(
            statuses.get(status, 0)
            for status in (
                "completed",
                "blocked",
                "needs_clarification",
                "cancelled",
                "error",
            )
        )
        return {
            "runs_started": started,
            "status_counts": statuses,
            "technical_terminal_rate": terminal / (started or 1),
            "total_tokens": int(usage[0]),
            "average_llm_latency_ms": float(usage[1]),
            "llm_call_count": int(usage[2]),
            "agents": [
                {
                    "agent": row["agent"],
                    "calls": row["calls"],
                    "success_rate": row["successes"] / (row["calls"] or 1),
                    "tokens": row["tokens"],
                    "average_latency_ms": row["latency"],
                }
                for row in agent_rows
            ],
        }

