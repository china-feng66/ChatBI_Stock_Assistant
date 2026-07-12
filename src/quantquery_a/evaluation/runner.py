"""Layered evaluation: deterministic gates before any optional LLM judge."""

from __future__ import annotations

from pydantic import Field

from quantquery_a.agents.contracts import (
    AnalyzeRequest,
    RunStatus,
    StrictModel,
    TaskType,
)
from quantquery_a.agents.supervisor import QuantSupervisor


class EvalCase(StrictModel):
    name: str
    request: AnalyzeRequest
    expected_task: TaskType
    expected_status: RunStatus
    require_exact_task: bool = True


class EvalCaseResult(StrictModel):
    name: str
    passed: bool
    route_correct: bool
    status_correct: bool
    route_applicable: bool
    timing_gate_passed: bool
    risk_gate_passed: bool
    deterministic_gates_applicable: bool
    actual_task: TaskType
    actual_status: RunStatus
    trace_id: str


class EvalSummary(StrictModel):
    case_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    route_accuracy: float = Field(ge=0, le=1)
    status_accuracy: float = Field(ge=0, le=1)
    deterministic_gate_rate: float = Field(ge=0, le=1)
    llm_judge_status: str
    cases: list[EvalCaseResult]


class EvaluationRunner:
    def __init__(self, supervisor: QuantSupervisor) -> None:
        self.supervisor = supervisor

    def run(self, cases: tuple[EvalCase, ...]) -> EvalSummary:
        results: list[EvalCaseResult] = []
        for case in cases:
            report = self.supervisor.analyze(case.request)
            route_correct = (
                report.route.task_type is case.expected_task
                if case.require_exact_task
                else True
            )
            status_correct = report.status is case.expected_status
            deterministic_gates_applicable = bool(report.analyses)
            timing_gate_passed = all(
                trade.execution_date > trade.signal_date
                for analysis in report.analyses
                for trade in analysis.trades
            )
            if report.status is RunStatus.COMPLETED and report.analyses:
                risk_gate_passed = bool(
                    report.risk_review and report.risk_review.approved
                )
            elif report.status is RunStatus.BLOCKED and report.analyses:
                risk_gate_passed = bool(
                    report.risk_review and not report.risk_review.approved
                )
            else:
                risk_gate_passed = True
            passed = (
                route_correct
                and status_correct
                and timing_gate_passed
                and risk_gate_passed
            )
            results.append(
                EvalCaseResult(
                    name=case.name,
                    passed=passed,
                    route_correct=route_correct,
                    status_correct=status_correct,
                    route_applicable=case.require_exact_task,
                    timing_gate_passed=timing_gate_passed,
                    risk_gate_passed=risk_gate_passed,
                    deterministic_gates_applicable=deterministic_gates_applicable,
                    actual_task=report.route.task_type,
                    actual_status=report.status,
                    trace_id=report.trace_id,
                )
            )

        count = len(results)
        divisor = count or 1
        passed_count = sum(result.passed for result in results)
        route_results = [result for result in results if result.route_applicable]
        gate_results = [
            result for result in results if result.deterministic_gates_applicable
        ]
        return EvalSummary(
            case_count=count,
            passed_count=passed_count,
            pass_rate=passed_count / divisor,
            route_accuracy=sum(result.route_correct for result in route_results)
            / (len(route_results) or 1),
            status_accuracy=sum(result.status_correct for result in results) / divisor,
            deterministic_gate_rate=sum(
                result.timing_gate_passed and result.risk_gate_passed
                for result in gate_results
            )
            / (len(gate_results) or 1),
            llm_judge_status=(
                "not_configured: deterministic truth gates run first; "
                "an optional judge may score explanation quality later"
            ),
            cases=results,
        )
