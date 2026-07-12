"""Supervisor for the bounded Data -> Quant -> Risk research loop."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from quantquery_a.observability.trace import TraceRecorder
from quantquery_a.query import ReadOnlyQueryService
from quantquery_a.routing.ensemble import EnsembleIntentRouter

from .contracts import (
    AgentStep,
    AnalysisReport,
    AnalyzeRequest,
    QueryFrame,
    RiskReview,
    RoutePath,
    RunStatus,
    SymbolAnalysis,
    TaskType,
)
from .data_agent import DataAgent, DataAgentError, DataBundle
from .quant_agent import QuantAgent
from .risk_critic import RiskCritic


class QuantSupervisor:
    def __init__(
        self,
        *,
        router: EnsembleIntentRouter | None = None,
        data_agent: DataAgent | None = None,
        quant_agent: QuantAgent | None = None,
        risk_critic: RiskCritic | None = None,
        trace_recorder: TraceRecorder | None = None,
        max_attempts: int = 2,
    ) -> None:
        if not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        self.router = router or EnsembleIntentRouter()
        self.data_agent = data_agent or DataAgent()
        self.quant_agent = quant_agent or QuantAgent()
        self.risk_critic = risk_critic or RiskCritic()
        self.trace_recorder = trace_recorder or TraceRecorder()
        self.max_attempts = max_attempts

    def analyze(self, request: AnalyzeRequest) -> AnalysisReport:
        trace_id = self.trace_recorder.new_trace_id()
        route = self.router.route(request)
        frame = self._build_frame(request, route)
        steps = [
            AgentStep(
                role="router",
                status="ok",
                detail=f"{route.task_type.value}:{route.path.value}",
            )
        ]
        self.trace_recorder.emit(
            trace_id,
            span="router",
            event="route_selected",
            status="ok",
            attributes={
                "query_hash": frame.query_hash,
                "task_type": route.task_type.value,
                "path": route.path.value,
                "confidence": route.confidence,
                "symbol_count": len(frame.symbols),
            },
        )

        if route.path is RoutePath.CLARIFICATION:
            missing = ", ".join(route.missing_fields) or "任务目标或约束"
            steps.append(
                AgentStep(
                    role="supervisor",
                    status="blocked",
                    detail="clarification_required",
                )
            )
            return AnalysisReport(
                trace_id=trace_id,
                status=RunStatus.NEEDS_CLARIFICATION,
                route=route,
                frame=frame,
                evidence=[],
                analyses=[],
                risk_review=None,
                steps=steps,
                attempts=0,
                answer=f"需要补充或澄清：{missing}。",
            )

        if route.task_type is TaskType.UNSUPPORTED:
            steps.append(
                AgentStep(
                    role="supervisor", status="blocked", detail="unsupported_request"
                )
            )
            return self._blocked_report(
                trace_id,
                route,
                frame,
                steps,
                answer="该请求超出研究原型的安全范围，系统不会执行或生成交易结论。",
            )
        if route.task_type is TaskType.KNOWLEDGE_EXPLAIN:
            steps.append(
                AgentStep(
                    role="supervisor",
                    status="blocked",
                    detail="knowledge_base_not_configured",
                )
            )
            return AnalysisReport(
                trace_id=trace_id,
                status=RunStatus.BLOCKED,
                route=route,
                frame=frame,
                evidence=[],
                analyses=[],
                risk_review=None,
                steps=steps,
                attempts=0,
                answer="MVP 尚未配置研究方法知识库，因此不生成无依据解释。",
            )

        try:
            data = self.data_agent.run(frame, request.bars)
        except DataAgentError:
            self.trace_recorder.emit(
                trace_id,
                span="data",
                event="evidence_failed",
                status="blocked",
                attributes={"symbol_count": len(frame.symbols)},
            )
            steps.append(
                AgentStep(role="data", status="blocked", detail="evidence_unavailable")
            )
            return self._blocked_report(
                trace_id,
                route,
                frame,
                steps,
                answer="没有可用的受控行情证据，研究任务已停止。",
            )

        steps.append(
            AgentStep(
                role="data",
                status="ok",
                detail=f"evidence_count={len(data.evidence)}",
            )
        )
        self.trace_recorder.emit(
            trace_id,
            span="data",
            event="evidence_ready",
            status="ok",
            attributes={
                "evidence": [
                    {
                        "evidence_id": item.evidence_id,
                        "row_count": item.row_count,
                        "fingerprint": item.fingerprint[:12],
                    }
                    for item in data.evidence
                ]
            },
        )

        if route.path is RoutePath.FAST:
            steps.append(
                AgentStep(role="supervisor", status="ok", detail="fast_path_complete")
            )
            summary = "; ".join(
                f"{item.symbol} latest_close={item.latest_close:.4f}, {item.row_count} rows ({item.start_date}..{item.end_date})"
                for item in data.evidence
            )
            return AnalysisReport(
                trace_id=trace_id,
                status=RunStatus.COMPLETED,
                route=route,
                frame=frame,
                evidence=list(data.evidence),
                analyses=[],
                risk_review=None,
                steps=steps,
                attempts=1,
                answer=f"只读行情证据已验证：{summary}。",
            )

        analyses: tuple[SymbolAnalysis, ...] = ()
        risk_review: RiskReview | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                analyses = self.quant_agent.run(frame, data)
            except (ValueError, KeyError):
                self.trace_recorder.emit(
                    trace_id,
                    span="quant",
                    event="analysis_failed",
                    status="error",
                    attempt=attempt,
                    attributes={"symbol_count": len(frame.symbols)},
                )
                steps.append(
                    AgentStep(
                        role="quant",
                        status="error",
                        attempt=attempt,
                        detail="deterministic_analysis_failed",
                    )
                )
                return self._blocked_report(
                    trace_id,
                    route,
                    frame,
                    steps,
                    data=data,
                    answer="确定性量化计算失败，系统未输出策略结论。",
                    attempts=attempt,
                )

            steps.append(
                AgentStep(
                    role="quant",
                    status="ok",
                    attempt=attempt,
                    detail=f"analysis_count={len(analyses)}",
                )
            )
            self.trace_recorder.emit(
                trace_id,
                span="quant",
                event="analysis_ready",
                status="ok",
                attempt=attempt,
                attributes={
                    "analysis_count": len(analyses),
                    "trade_count_total": sum(
                        len(analysis.trades) for analysis in analyses
                    ),
                },
            )

            risk_review = self.risk_critic.review(frame, data, analyses)
            risk_status = "ok" if risk_review.approved else "blocked"
            steps.append(
                AgentStep(
                    role="risk",
                    status=risk_status,
                    attempt=attempt,
                    detail=f"finding_count={len(risk_review.findings)}",
                )
            )
            self.trace_recorder.emit(
                trace_id,
                span="risk",
                event="audit_complete",
                status=risk_status,
                attempt=attempt,
                attributes={
                    "approved": risk_review.approved,
                    "finding_codes": [finding.code for finding in risk_review.findings],
                },
            )
            if risk_review.approved:
                steps.append(
                    AgentStep(
                        role="supervisor",
                        status="ok",
                        attempt=attempt,
                        detail="evidence_gate_passed",
                    )
                )
                return AnalysisReport(
                    trace_id=trace_id,
                    status=RunStatus.COMPLETED,
                    route=route,
                    frame=frame,
                    evidence=list(data.evidence),
                    analyses=list(analyses),
                    risk_review=risk_review,
                    steps=steps,
                    attempts=attempt,
                    answer=self._summarize(analyses),
                )
            if not risk_review.retryable:
                break

        steps.append(
            AgentStep(
                role="supervisor",
                status="blocked",
                attempt=max(1, len([step for step in steps if step.role == "quant"])),
                detail="evidence_gate_failed",
            )
        )
        return AnalysisReport(
            trace_id=trace_id,
            status=RunStatus.BLOCKED,
            route=route,
            frame=frame,
            evidence=list(data.evidence),
            analyses=list(analyses),
            risk_review=risk_review,
            steps=steps,
            attempts=len([step for step in steps if step.role == "quant"]),
            answer="研究结果未通过风险与证据门禁，因此不输出策略结论。",
        )

    @staticmethod
    def _build_frame(request: AnalyzeRequest, route) -> QueryFrame:
        return QueryFrame(
            query_hash=sha256(request.query.encode("utf-8")).hexdigest()[:16],
            task_type=route.task_type,
            route_path=route.path,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            frequency=request.frequency,
            strategy=request.strategy,
            macd=request.macd,
            costs=request.costs,
            benchmark=request.benchmark,
            confidence=route.confidence,
            missing_fields=route.missing_fields,
        )

    @staticmethod
    def _summarize(analyses: tuple[SymbolAnalysis, ...]) -> str:
        parts = []
        for analysis in analyses:
            metrics = analysis.metrics
            parts.append(
                f"{analysis.symbol}: total_return={metrics['total_return']:.4f}, "
                f"sharpe={metrics['sharpe_ratio']:.4f}, "
                f"max_drawdown={metrics['max_drawdown']:.4f}, "
                f"trades={len(analysis.trades)}"
            )
        return "风险门禁通过；" + "; ".join(parts) + "。结果仅用于研究演示。"

    @staticmethod
    def _blocked_report(
        trace_id,
        route,
        frame,
        steps,
        *,
        data: DataBundle | None = None,
        answer: str,
        attempts: int = 0,
    ) -> AnalysisReport:
        return AnalysisReport(
            trace_id=trace_id,
            status=RunStatus.BLOCKED,
            route=route,
            frame=frame,
            evidence=list(data.evidence) if data else [],
            analyses=[],
            risk_review=None,
            steps=steps,
            attempts=attempts,
            answer=answer,
        )


def build_default_supervisor(
    *,
    query_service: ReadOnlyQueryService | None = None,
    trace_path: str | Path | None = None,
) -> QuantSupervisor:
    return QuantSupervisor(
        data_agent=DataAgent(query_service=query_service),
        trace_recorder=TraceRecorder(trace_path),
    )
