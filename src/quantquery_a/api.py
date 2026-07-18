"""FastAPI surface for legacy analysis and the realtime workbench."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from quantquery_a.agents.contracts import AnalysisReport, AnalyzeRequest
from quantquery_a.agents.supervisor import QuantSupervisor, build_default_supervisor
from quantquery_a.evaluation import EvalSummary, EvaluationRunner, build_default_cases
from quantquery_a.query import ReadOnlyQueryService
from quantquery_a.workbench.benchmark import (
    RoutingBenchmark,
    RoutingBenchmarkSummary,
)
from quantquery_a.workbench.contracts import (
    RunCreateRequest,
    RunSnapshot,
    WatchlistUpdate,
    WeightCandidate,
)
from quantquery_a.workbench.market import MarketProviderError
from quantquery_a.workbench.observability import ObservabilityService
from quantquery_a.workbench.quality import QualityManager, WorkbenchEvalSummary
from quantquery_a.workbench.service import RunNotFoundError, WorkbenchService


API_VERSION = "0.4.0"
TERMINAL_STATUSES = {
    "completed",
    "blocked",
    "needs_clarification",
    "cancelled",
    "error",
}


def create_app(
    supervisor: QuantSupervisor | None = None,
    workbench_service: WorkbenchService | None = None,
) -> FastAPI:
    active_supervisor = supervisor or _supervisor_from_environment()
    active_workbench = workbench_service or WorkbenchService(
        runtime_dir=os.environ.get("QUANTQUERY_RUNTIME_DIR", ".runtime")
    )
    quality = QualityManager(
        active_workbench.database_path,
        active_workbench.runtime_dir,
    )
    quality.apply_active(active_workbench)
    observability = ObservabilityService(active_workbench.database_path)
    routing_benchmark = RoutingBenchmark(active_workbench.database_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await active_workbench.start()
        try:
            yield
        finally:
            await active_workbench.stop()

    app = FastAPI(
        title="QuantQuery-A Realtime Multi-Agent Research API",
        version=API_VERSION,
        description=(
            "Legacy audited Data -> Quant -> Risk endpoints plus a durable "
            "five-agent LangGraph workbench."
        ),
        lifespan=lifespan,
    )
    app.state.workbench = active_workbench
    app.state.quality = quality
    app.state.observability = observability
    app.state.routing_benchmark = routing_benchmark
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "version": API_VERSION,
            "loop": "data-quant-risk",
            "workbench_loop": "planner-data-quant-risk-reporter",
            "llm_required": os.environ.get("QUANTQUERY_LLM_MODE", "fake") == "live",
            "observability": "run-events-agent-metrics-communication-edges",
        }

    @app.post("/analyze", response_model=AnalysisReport)
    def analyze(request: AnalyzeRequest) -> AnalysisReport:
        try:
            return active_supervisor.analyze(request)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="analysis could not be completed",
            ) from exc

    @app.post("/eval/run", response_model=EvalSummary)
    def run_evaluation() -> EvalSummary:
        try:
            return EvaluationRunner(QuantSupervisor()).run(build_default_cases())
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="evaluation could not be completed",
            ) from exc

    @app.post(
        "/api/v1/runs",
        response_model=RunSnapshot,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_run(request: RunCreateRequest) -> RunSnapshot:
        return await active_workbench.create_run(request)

    @app.get("/api/v1/runs", response_model=list[RunSnapshot])
    def list_runs(limit: int = 50) -> list[RunSnapshot]:
        return active_workbench.list_runs(max(1, min(limit, 100)))

    @app.get("/api/v1/runs/{run_id}", response_model=RunSnapshot)
    def get_run(run_id: str) -> RunSnapshot:
        try:
            return active_workbench.get_run(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/api/v1/runs/{run_id}/observability")
    def get_run_observability(run_id: str) -> dict[str, object]:
        try:
            snapshot = active_workbench.get_run(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return observability.run_detail(snapshot)

    @app.post("/api/v1/runs/{run_id}/cancel", response_model=RunSnapshot)
    def cancel_run(run_id: str) -> RunSnapshot:
        try:
            return active_workbench.cancel(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.post("/api/v1/runs/{run_id}/resume", response_model=RunSnapshot)
    async def resume_run(run_id: str) -> RunSnapshot:
        try:
            return await active_workbench.resume(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.websocket("/api/v1/runs/{run_id}/events")
    async def run_events(websocket: WebSocket, run_id: str) -> None:
        try:
            active_workbench.get_run(run_id)
        except RunNotFoundError:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        sequence = 0
        try:
            while True:
                events = active_workbench.store.events(run_id, sequence)
                for event in events:
                    sequence = max(sequence, int(event["sequence"]))
                    await websocket.send_json(
                        {
                            "sequence": event["sequence"],
                            "run_id": run_id,
                            "type": event["event_type"],
                            "agent": event["agent"],
                            "status": event["status"],
                            "timestamp": event["timestamp"],
                            "payload": event["payload"],
                        }
                    )
                snapshot = active_workbench.get_run(run_id)
                if snapshot.status.value in TERMINAL_STATUSES and not events:
                    await websocket.send_json(
                        {"type": "stream.closed", "status": snapshot.status.value}
                    )
                    break
                await asyncio.sleep(0.2)
        except WebSocketDisconnect:
            return

    @app.get("/api/v1/watchlist", response_model=list[str])
    def get_watchlist() -> list[str]:
        return active_workbench.store.watchlist()

    @app.post("/api/v1/watchlist", response_model=list[str])
    def set_watchlist(update: WatchlistUpdate) -> list[str]:
        return active_workbench.set_watchlist(update.symbols)

    @app.get("/api/v1/market/{symbol}/daily")
    def daily_market(symbol: str, limit: int = 250) -> dict[str, object]:
        normalized = symbol.strip().upper()
        try:
            bars, evidence = active_workbench.market_gateway.daily_bars(
                [normalized],
                start_date=None,
                end_date=None,
                limit=max(2, min(limit, 250)),
            )
            return {
                "symbol": normalized,
                "bars": bars[normalized],
                "evidence": evidence[0],
            }
        except MarketProviderError as exc:
            raise HTTPException(status_code=503, detail="market data unavailable") from exc

    @app.get("/api/v1/market/{symbol}/minute")
    def minute_market(symbol: str, limit: int = 240) -> dict[str, object]:
        normalized = symbol.strip().upper()
        try:
            rows = active_workbench.market_gateway.minutes(
                normalized, max(1, min(limit, 1_200))
            )
            return {"symbol": normalized, "bars": rows}
        except MarketProviderError as exc:
            raise HTTPException(status_code=503, detail="minute data unavailable") from exc

    @app.websocket("/api/v1/market/stream")
    async def market_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = await active_workbench.stream_hub.subscribe()
        try:
            while True:
                message = await queue.get()
                await websocket.send_json(message)
        except WebSocketDisconnect:
            return
        finally:
            active_workbench.stream_hub.unsubscribe(queue)

    @app.get("/api/v1/metrics/summary")
    def metrics_summary() -> dict[str, object]:
        return active_workbench.store.metrics()

    @app.get("/api/v1/metrics/observability")
    def observability_summary() -> dict[str, object]:
        return observability.system_summary()

    @app.post("/api/v1/evals/run", response_model=WorkbenchEvalSummary)
    def workbench_evaluation(live: bool = False) -> WorkbenchEvalSummary:
        if live and active_workbench.llm.model == "fake-qwen":
            raise HTTPException(
                status_code=409,
                detail="restart the server with QUANTQUERY_LLM_MODE=live",
            )
        try:
            return quality.evaluate(live=live)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="workbench evaluation could not be completed",
            ) from exc

    @app.post(
        "/api/v1/evals/routing",
        response_model=RoutingBenchmarkSummary,
    )
    def routing_evaluation(live: bool = False) -> RoutingBenchmarkSummary:
        if live and active_workbench.llm.model == "fake-qwen":
            raise HTTPException(
                status_code=409,
                detail="restart the server with QUANTQUERY_LLM_MODE=live",
            )
        return routing_benchmark.run(
            live=live,
            client=active_workbench.llm if live else None,
        )

    @app.get(
        "/api/v1/evals/routing/latest",
        response_model=RoutingBenchmarkSummary | None,
    )
    def latest_routing_evaluation() -> RoutingBenchmarkSummary | None:
        return routing_benchmark.latest()

    @app.get(
        "/api/v1/router/weight-candidates",
        response_model=list[WeightCandidate],
    )
    def weight_candidates() -> list[WeightCandidate]:
        return quality.candidates()

    @app.post(
        "/api/v1/router/weight-candidates",
        response_model=WeightCandidate,
        status_code=status.HTTP_201_CREATED,
    )
    def generate_weight_candidate() -> WeightCandidate:
        return quality.generate_candidate()

    @app.post(
        "/api/v1/router/weight-candidates/{candidate_id}/promote",
        response_model=WeightCandidate,
    )
    def promote_weight_candidate(candidate_id: str) -> WeightCandidate:
        try:
            return quality.promote(candidate_id, active_workbench)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="candidate not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    _mount_frontend_if_built(app)
    return app


def _supervisor_from_environment() -> QuantSupervisor:
    db_path = os.environ.get("QUANTQUERY_DB_PATH")
    trace_path = os.environ.get("QUANTQUERY_TRACE_PATH")
    query_service = ReadOnlyQueryService(db_path, max_rows=5_000) if db_path else None
    return build_default_supervisor(
        query_service=query_service,
        trace_path=trace_path,
    )


def _mount_frontend_if_built(app: FastAPI) -> None:
    configured = os.environ.get("QUANTQUERY_FRONTEND_DIST")
    directory = Path(configured) if configured else Path("frontend/dist")
    if directory.is_dir() and (directory / "index.html").exists():
        app.mount("/", StaticFiles(directory=directory, html=True), name="frontend")


app = create_app()
