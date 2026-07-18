"""Local run manager joining LangGraph, SQLite persistence, and FastAPI."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver

from .contracts import RunCreateRequest, RunSnapshot, WorkbenchStatus
from .graph import FiveAgentWorkflow, ResearchState
from .llm import QwenClient, build_qwen_client
from .market import BoundedMarketGateway, MarketStreamHub, build_market_gateway
from .storage import WorkbenchStore


class RunNotFoundError(KeyError):
    pass


class WorkbenchService:
    def __init__(
        self,
        *,
        runtime_dir: str | Path = ".runtime",
        llm: QwenClient | None = None,
        market_gateway: BoundedMarketGateway | None = None,
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.runtime_dir / "workbench.sqlite3"
        self.store = WorkbenchStore(self.database_path)
        self.llm = llm or build_qwen_client()
        self.market_gateway = market_gateway or build_market_gateway(self.database_path)
        self.stream_hub = MarketStreamHub(self.market_gateway)
        self._checkpoint_connection = sqlite3.connect(
            self.database_path, check_same_thread=False
        )
        self._checkpointer = SqliteSaver(self._checkpoint_connection)
        self._workflow = FiveAgentWorkflow(
            llm=self.llm,
            store=self.store,
            market_gateway=self.market_gateway,
        )
        self._graph = self._workflow.build(self._checkpointer)
        self._semaphore = asyncio.Semaphore(2)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancelled: set[str] = set()

    async def start(self) -> None:
        self.stream_hub.set_symbols(self.store.watchlist())
        if self.market_gateway.realtime_provider:
            await self.stream_hub.start()

    async def stop(self) -> None:
        await self.stream_hub.stop()
        for task in tuple(self._tasks.values()):
            if not task.done():
                task.cancel()
        for task in tuple(self._tasks.values()):
            with suppress(asyncio.CancelledError):
                await task

    async def create_run(self, request: RunCreateRequest) -> RunSnapshot:
        run_id = uuid4().hex
        self.store.create_run(run_id, request)
        self.store.append_event(run_id, event_type="run.queued", status="queued")
        self._tasks[run_id] = asyncio.create_task(self._execute(run_id, request))
        return self.get_run(run_id)

    async def _execute(self, run_id: str, request: RunCreateRequest) -> None:
        async with self._semaphore:
            if run_id in self._cancelled:
                self._mark_cancelled(run_id)
                return
            existing_row = self.store.get_run_row(run_id)
            existing_state = (
                json.loads(existing_row["state_json"]) if existing_row else {}
            )
            prior_calls, prior_tokens = self._usage_totals(run_id)
            state: ResearchState = {
                "run_id": run_id,
                "request": request.model_dump(mode="json"),
                "status": WorkbenchStatus.RUNNING.value,
                "token_total": prior_tokens,
                "llm_calls": prior_calls,
                "repair_count": int(existing_state.get("repair_count", 0)),
                "error": None,
            }
            self.store.update_run(
                run_id, status=WorkbenchStatus.RUNNING, state=dict(state)
            )
            self.store.append_event(
                run_id, event_type="run.started", status="running"
            )
            try:
                result = await asyncio.to_thread(
                    self._graph.invoke,
                    state,
                    {"configurable": {"thread_id": run_id}},
                )
                if run_id in self._cancelled:
                    self._mark_cancelled(run_id)
                    return
                result = self._reconcile_usage(run_id, dict(result))
                status = WorkbenchStatus(result.get("status", "error"))
                self.store.update_run(run_id, status=status, state=result)
            except Exception as exc:
                state["status"] = WorkbenchStatus.ERROR.value
                state["error"] = type(exc).__name__
                reconciled = self._reconcile_usage(run_id, dict(state))
                self.store.append_event(
                    run_id,
                    event_type="run.failed",
                    status="error",
                    payload={"error_type": type(exc).__name__},
                )
                self.store.update_run(
                    run_id,
                    status=WorkbenchStatus.ERROR,
                    state=reconciled,
                    error=type(exc).__name__,
                )

    def run_sync(self, request: RunCreateRequest) -> RunSnapshot:
        run_id = uuid4().hex
        self.store.create_run(run_id, request)
        initial: ResearchState = {
            "run_id": run_id,
            "request": request.model_dump(mode="json"),
            "status": WorkbenchStatus.RUNNING.value,
            "token_total": 0,
            "llm_calls": 0,
            "repair_count": 0,
            "error": None,
        }
        result = self._graph.invoke(
            initial, {"configurable": {"thread_id": run_id}}
        )
        reconciled = self._reconcile_usage(run_id, dict(result))
        status = WorkbenchStatus(reconciled.get("status", "error"))
        self.store.update_run(run_id, status=status, state=reconciled)
        return self.get_run(run_id)

    def _usage_totals(self, run_id: str) -> tuple[int, int]:
        with sqlite3.connect(self.database_path, timeout=10.0) as connection:
            row = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(total_tokens), 0) "
                "FROM llm_usage WHERE run_id=?",
                (run_id,),
            ).fetchone()
        return int(row[0]), int(row[1])

    def _reconcile_usage(
        self, run_id: str, state: dict[str, object]
    ) -> dict[str, object]:
        """Use the append-only usage ledger as the authoritative token counter."""

        calls, tokens = self._usage_totals(run_id)
        state["llm_calls"] = calls
        state["token_total"] = tokens
        return state

    def get_run(self, run_id: str) -> RunSnapshot:
        row = self.store.get_run_row(run_id)
        if not row:
            raise RunNotFoundError(run_id)
        request = RunCreateRequest.model_validate_json(row["request_json"])
        state = self._reconcile_usage(run_id, json.loads(row["state_json"]))
        return RunSnapshot(
            run_id=run_id,
            status=WorkbenchStatus(row["status"]),
            request=request,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            route=state.get("route"),
            plan=state.get("plan"),
            evidence=state.get("evidence", []),
            analyses=state.get("analyses", []),
            risk=state.get("risk"),
            report=state.get("report"),
            token_total=state.get("token_total", 0),
            llm_calls=state.get("llm_calls", 0),
            repair_count=state.get("repair_count", 0),
            error=row.get("error") or state.get("error"),
        )

    def list_runs(self, limit: int = 50) -> list[RunSnapshot]:
        return [self.get_run(row["run_id"]) for row in self.store.list_run_rows(limit)]

    def cancel(self, run_id: str) -> RunSnapshot:
        if not self.store.get_run_row(run_id):
            raise RunNotFoundError(run_id)
        self._cancelled.add(run_id)
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()
        self._mark_cancelled(run_id)
        return self.get_run(run_id)

    async def resume(self, run_id: str) -> RunSnapshot:
        snapshot = self.get_run(run_id)
        if snapshot.status not in {
            WorkbenchStatus.ERROR,
            WorkbenchStatus.INTERRUPTED,
            WorkbenchStatus.CANCELLED,
            WorkbenchStatus.BLOCKED,
        }:
            return snapshot
        self._cancelled.discard(run_id)
        self.store.update_run(
            run_id,
            status=WorkbenchStatus.QUEUED,
            state={
                "run_id": run_id,
                "status": "queued",
                "token_total": snapshot.token_total,
                "llm_calls": snapshot.llm_calls,
                "repair_count": snapshot.repair_count,
            },
        )
        self._tasks[run_id] = asyncio.create_task(
            self._execute(run_id, snapshot.request)
        )
        return self.get_run(run_id)

    def set_watchlist(self, symbols: list[str]) -> list[str]:
        self.store.set_watchlist(symbols[:20])
        self.stream_hub.set_symbols(symbols[:20])
        return self.store.watchlist()

    def _mark_cancelled(self, run_id: str) -> None:
        row = self.store.get_run_row(run_id)
        if not row:
            return
        state = self._reconcile_usage(run_id, json.loads(row["state_json"]))
        state["status"] = WorkbenchStatus.CANCELLED.value
        self.store.append_event(
            run_id, event_type="run.cancelled", status="cancelled"
        )
        self.store.update_run(
            run_id, status=WorkbenchStatus.CANCELLED, state=state
        )
