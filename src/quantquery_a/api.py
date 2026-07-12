"""FastAPI surface for analysis and deterministic evaluation."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from quantquery_a.agents.contracts import AnalysisReport, AnalyzeRequest
from quantquery_a.agents.supervisor import QuantSupervisor, build_default_supervisor
from quantquery_a.evaluation import EvalSummary, EvaluationRunner, build_default_cases
from quantquery_a.query import ReadOnlyQueryService


API_VERSION = "0.2.0"


def create_app(supervisor: QuantSupervisor | None = None) -> FastAPI:
    active_supervisor = supervisor or _supervisor_from_environment()
    app = FastAPI(
        title="QuantQuery-A Multi-Agent Research API",
        version=API_VERSION,
        description=(
            "Auditable Data -> Quant -> Risk loop with typed routing, evidence "
            "gates and deterministic evaluation."
        ),
    )

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "version": API_VERSION,
            "loop": "data-quant-risk",
            "llm_required": False,
        }

    @app.post("/analyze", response_model=AnalysisReport)
    def analyze(request: AnalyzeRequest) -> AnalysisReport:
        try:
            return active_supervisor.analyze(request)
        except Exception as exc:  # stable public boundary; details remain in local logs
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

    return app


def _supervisor_from_environment() -> QuantSupervisor:
    db_path = os.environ.get("QUANTQUERY_DB_PATH")
    trace_path = os.environ.get("QUANTQUERY_TRACE_PATH")
    query_service = ReadOnlyQueryService(db_path, max_rows=5_000) if db_path else None
    return build_default_supervisor(
        query_service=query_service,
        trace_path=trace_path,
    )


app = create_app()
