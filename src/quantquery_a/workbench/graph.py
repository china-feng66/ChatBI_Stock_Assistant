"""LangGraph orchestration for five typed Qwen-backed agent roles."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from quantquery_a.agents.contracts import AnalyzeRequest, RoutePath, TaskType

from .contracts import (
    AgentUsage,
    MODE_TASK_HINT,
    ResearchMode,
    RunCreateRequest,
    WorkbenchStatus,
)
from .llm import QwenClient
from .research import deterministic_risk_audit, run_research_tool
from .routing import HybridIntentRouter
from .storage import WorkbenchStore


TOKEN_BUDGET = 15_000
MAX_LLM_CALLS = 6


class MarketGatewayProtocol(Protocol):
    def daily_bars(
        self,
        symbols: list[str],
        *,
        start_date: str | None,
        end_date: str | None,
        limit: int,
    ) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]: ...


class ResearchState(TypedDict, total=False):
    run_id: str
    request: dict[str, Any]
    status: str
    route: dict[str, Any]
    plan: dict[str, Any]
    bars: dict[str, list[dict[str, Any]]]
    evidence: list[dict[str, Any]]
    analyses: list[dict[str, Any]]
    tool_result: dict[str, Any]
    risk: dict[str, Any]
    report: str
    token_total: int
    llm_calls: int
    repair_count: int
    error: str | None


class FiveAgentWorkflow:
    def __init__(
        self,
        *,
        llm: QwenClient,
        store: WorkbenchStore,
        market_gateway: MarketGatewayProtocol | None = None,
        router: HybridIntentRouter | None = None,
    ) -> None:
        self.llm = llm
        self.store = store
        self.market_gateway = market_gateway
        self.router = router or HybridIntentRouter()

    def build(self, checkpointer=None):
        graph = StateGraph(ResearchState)
        graph.add_node("planner", self._planner)
        graph.add_node("data", self._data)
        graph.add_node("quant", self._quant)
        graph.add_node("risk", self._risk)
        graph.add_node("repair", self._repair)
        graph.add_node("reporter", self._reporter)
        graph.add_edge(START, "planner")
        graph.add_conditional_edges(
            "planner",
            self._after_planner,
            {"data": "data", "reporter": "reporter"},
        )
        graph.add_conditional_edges(
            "data",
            self._after_data,
            {"quant": "quant", "reporter": "reporter"},
        )
        graph.add_edge("quant", "risk")
        graph.add_conditional_edges(
            "risk",
            self._after_risk,
            {"repair": "repair", "reporter": "reporter"},
        )
        graph.add_edge("repair", "reporter")
        graph.add_edge("reporter", END)
        return graph.compile(checkpointer=checkpointer)

    def _emit(
        self,
        state: ResearchState,
        event_type: str,
        status: str,
        *,
        agent: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.store.append_event(
            state["run_id"],
            event_type=event_type,
            status=status,
            agent=agent,
            payload=payload,
        )

    def _call(
        self,
        state: ResearchState,
        *,
        agent: str,
        prompt: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, int]]:
        calls = int(state.get("llm_calls", 0))
        used = int(state.get("token_total", 0))
        if calls >= MAX_LLM_CALLS or used >= TOKEN_BUDGET:
            raise RuntimeError("token budget exhausted")
        remaining = TOKEN_BUDGET - used
        response = self.llm.complete(
            agent=agent,
            system_prompt=prompt,
            payload=payload,
            max_output_tokens=max(64, min(1_500, remaining // 2)),
        )
        usage = AgentUsage(
            agent=agent,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            latency_ms=response.usage.latency_ms,
            success=True,
        )
        self.store.record_usage(state["run_id"], usage)
        self._emit(
            state,
            "agent.completed",
            "ok",
            agent=agent,
            payload={
                "tokens": usage.total_tokens,
                "latency_ms": usage.latency_ms,
            },
        )
        total = used + usage.total_tokens
        return response.content, {"token_total": total, "llm_calls": calls + 1}

    def _planner(self, state: ResearchState) -> dict[str, Any]:
        request = RunCreateRequest.model_validate(state["request"])
        self._emit(state, "agent.started", "running", agent="planner")
        try:
            content, counters = self._call(
                state,
                agent="planner",
                prompt=(
                    "Classify the research intent and return JSON probabilities and a short "
                    "allowlisted plan. Never request trading or arbitrary code execution."
                ),
                payload={
                    "query": request.query,
                    "symbols": request.symbols,
                    "mode": request.mode.value,
                },
            )
            analyze = AnalyzeRequest(
                query=request.query,
                symbols=request.symbols,
                task_hint=MODE_TASK_HINT[request.mode],
            )
            route = self.router.route(analyze, content.get("probabilities", {}))
            status = WorkbenchStatus.RUNNING.value
            if route.path is RoutePath.CLARIFICATION:
                status = WorkbenchStatus.NEEDS_CLARIFICATION.value
            elif route.task_type is TaskType.UNSUPPORTED:
                status = WorkbenchStatus.BLOCKED.value
            mode = request.mode
            if mode is ResearchMode.AUTO:
                mode = _mode_from_task(route.task_type, request.query)
            return {
                **counters,
                "status": status,
                "route": route.model_dump(mode="json"),
                "plan": {
                    "mode": mode.value,
                    "steps": content.get("steps", []),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            }
        except Exception as exc:
            self._emit(state, "agent.failed", "error", agent="planner")
            return {"status": WorkbenchStatus.ERROR.value, "error": type(exc).__name__}

    @staticmethod
    def _after_planner(state: ResearchState) -> str:
        return "data" if state.get("status") == "running" else "reporter"

    def _data(self, state: ResearchState) -> dict[str, Any]:
        request = RunCreateRequest.model_validate(state["request"])
        self._emit(state, "agent.started", "running", agent="data")
        try:
            content, counters = self._call(
                state,
                agent="data",
                prompt=(
                    "Choose only configured market-data tools. Return the required fields and "
                    "freshness policy. Never invent observations."
                ),
                payload={
                    "query": request.query,
                    "symbols": request.symbols,
                    "mode": state.get("plan", {}).get("mode"),
                    "repair": False,
                },
            )
            if request.bars:
                bars = {
                    symbol: [row.model_dump(mode="json") for row in rows]
                    for symbol, rows in request.bars.items()
                }
                evidence = _inline_evidence(bars)
            elif self.market_gateway:
                bars, evidence = self.market_gateway.daily_bars(
                    request.symbols,
                    start_date=(request.start_date.isoformat() if request.start_date else None),
                    end_date=(request.end_date.isoformat() if request.end_date else None),
                    limit=250,
                )
            else:
                raise RuntimeError("no approved market provider is configured")
            if set(bars) != set(request.symbols):
                raise RuntimeError("market evidence does not cover the requested universe")
            return {
                **counters,
                "bars": bars,
                "evidence": evidence,
                "data_policy": content,
            }
        except Exception as exc:
            self._emit(
                state,
                "tool.failed",
                "blocked",
                agent="data",
                payload={"error_type": type(exc).__name__},
            )
            return {
                "status": WorkbenchStatus.BLOCKED.value,
                "error": type(exc).__name__,
                "evidence": [],
            }

    @staticmethod
    def _after_data(state: ResearchState) -> str:
        return "quant" if state.get("status") == "running" else "reporter"

    def _quant(self, state: ResearchState) -> dict[str, Any]:
        request = RunCreateRequest.model_validate(state["request"])
        mode = ResearchMode(state.get("plan", {}).get("mode", "auto"))
        self._emit(state, "agent.started", "running", agent="quant")
        try:
            content, counters = self._call(
                state,
                agent="quant",
                prompt=(
                    "Select exactly one allowlisted research tool. Numerical work must be done "
                    "by deterministic code, never by generated Python."
                ),
                payload={
                    "query": request.query,
                    "mode": mode.value,
                    "symbols": request.symbols,
                    "evidence_ids": [item.get("evidence_id") for item in state["evidence"]],
                },
            )
            result = run_research_tool(request, mode=mode, bars=state["bars"])
            self._emit(
                state,
                "tool.completed",
                "ok",
                agent="quant",
                payload={"tool": result.get("tool"), "selection": content.get("tool")},
            )
            return {
                **counters,
                "tool_result": result,
                "analyses": result.get("analyses", []),
            }
        except Exception as exc:
            self._emit(state, "tool.failed", "blocked", agent="quant")
            return {"status": WorkbenchStatus.BLOCKED.value, "error": type(exc).__name__}

    def _risk(self, state: ResearchState) -> dict[str, Any]:
        mode = ResearchMode(state.get("plan", {}).get("mode", "auto"))
        self._emit(state, "agent.started", "running", agent="risk")
        try:
            _, counters = self._call(
                state,
                agent="risk",
                prompt=(
                    "Review evidence lineage and assumptions. Return qualitative findings only; "
                    "the deterministic gate has final authority."
                ),
                payload={
                    "mode": mode.value,
                    "evidence": state.get("evidence", []),
                    "metrics": [item.get("metrics", {}) for item in state.get("analyses", [])],
                },
            )
            risk = deterministic_risk_audit(
                mode=mode,
                result=state.get("tool_result", {}),
                evidence=state.get("evidence", []),
            )
            self._emit(
                state,
                "risk.completed",
                "ok" if risk["approved"] else "blocked",
                agent="risk",
                payload={
                    "approved": risk["approved"],
                    "finding_codes": [item.get("code") for item in risk["findings"]],
                },
            )
            return {**counters, "risk": risk}
        except Exception as exc:
            return {"status": WorkbenchStatus.ERROR.value, "error": type(exc).__name__}

    @staticmethod
    def _after_risk(state: ResearchState) -> str:
        risk = state.get("risk", {})
        if (
            not risk.get("approved", False)
            and risk.get("retryable", False)
            and int(state.get("repair_count", 0)) == 0
            and int(state.get("llm_calls", 0)) < MAX_LLM_CALLS - 1
        ):
            return "repair"
        return "reporter"

    def _repair(self, state: ResearchState) -> dict[str, Any]:
        request = RunCreateRequest.model_validate(state["request"])
        self._emit(state, "repair.started", "running", agent="data")
        try:
            _, counters = self._call(
                state,
                agent="data",
                prompt=(
                    "Perform one allowlisted repair only: refetch data, normalize tool arguments, "
                    "or fill a missing field. Never change strategy or risk assumptions."
                ),
                payload={
                    "query": request.query,
                    "repair": True,
                    "finding_codes": [
                        item.get("code") for item in state.get("risk", {}).get("findings", [])
                    ],
                },
            )
            # A bounded refetch is the only automatic data repair in v0.3.
            if self.market_gateway and not request.bars:
                bars, evidence = self.market_gateway.daily_bars(
                    request.symbols,
                    start_date=(request.start_date.isoformat() if request.start_date else None),
                    end_date=(request.end_date.isoformat() if request.end_date else None),
                    limit=250,
                )
                mode = ResearchMode(state.get("plan", {}).get("mode", "auto"))
                result = run_research_tool(request, mode=mode, bars=bars)
                risk = deterministic_risk_audit(
                    mode=mode, result=result, evidence=evidence
                )
                return {
                    **counters,
                    "repair_count": 1,
                    "bars": bars,
                    "evidence": evidence,
                    "tool_result": result,
                    "analyses": result.get("analyses", []),
                    "risk": risk,
                }
            return {**counters, "repair_count": 1}
        except Exception as exc:
            return {
                "repair_count": 1,
                "status": WorkbenchStatus.BLOCKED.value,
                "error": type(exc).__name__,
            }

    def _reporter(self, state: ResearchState) -> dict[str, Any]:
        self._emit(state, "agent.started", "running", agent="reporter")
        deterministic_summary = _deterministic_summary(state)
        try:
            if int(state.get("llm_calls", 0)) >= MAX_LLM_CALLS:
                report = deterministic_summary
                counters = {}
            else:
                content, counters = self._call(
                    state,
                    agent="reporter",
                    prompt=(
                        "Write a concise Chinese research report using only supplied evidence IDs, "
                        "verified metrics and risk findings. Include a non-investment disclaimer."
                    ),
                    payload={
                        "deterministic_summary": deterministic_summary,
                        "evidence_ids": [
                            item.get("evidence_id") for item in state.get("evidence", [])
                        ],
                        "risk": state.get("risk"),
                    },
                )
                report = str(content.get("summary", deterministic_summary))
                disclaimer = content.get("disclaimer")
                if disclaimer:
                    report = f"{report}\n\n{disclaimer}"
            current = state.get("status", WorkbenchStatus.RUNNING.value)
            risk = state.get("risk")
            if current in {
                WorkbenchStatus.ERROR.value,
                WorkbenchStatus.NEEDS_CLARIFICATION.value,
                WorkbenchStatus.BLOCKED.value,
            }:
                final_status = current
            elif risk and not risk.get("approved", False):
                final_status = WorkbenchStatus.BLOCKED.value
            else:
                final_status = WorkbenchStatus.COMPLETED.value
            self._emit(state, "run.completed", final_status, agent="reporter")
            return {**counters, "report": report, "status": final_status}
        except Exception as exc:
            return {
                "report": deterministic_summary,
                "status": WorkbenchStatus.ERROR.value,
                "error": type(exc).__name__,
            }


def _mode_from_task(task: TaskType, query: str) -> ResearchMode:
    lowered = query.lower()
    if "布林" in lowered or "bollinger" in lowered:
        return ResearchMode.BOLLINGER
    if "因子" in lowered or "组合" in lowered:
        return ResearchMode.FACTOR
    if task is TaskType.MARKET_QUERY:
        return ResearchMode.MARKET
    if task is TaskType.STOCK_COMPARISON:
        return ResearchMode.COMPARE
    return ResearchMode.MACD


def _inline_evidence(
    bars: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    import hashlib

    evidence = []
    for symbol, rows in bars.items():
        canonical = json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str)
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        evidence.append(
            {
                "evidence_id": f"inline:{symbol}:{fingerprint[:12]}",
                "symbol": symbol,
                "provider": "inline_fixture",
                "row_count": len(rows),
                "as_of": str(rows[-1]["trade_date"]),
                "freshness": "frozen",
                "stale": False,
                "adjustment_status": "synthetic_adjusted",
                "fingerprint": fingerprint,
            }
        )
    return evidence


def _deterministic_summary(state: ResearchState) -> str:
    if state.get("status") == WorkbenchStatus.NEEDS_CLARIFICATION.value:
        return "任务意图或必要参数不明确，请补充标的或研究目标。"
    if state.get("status") == WorkbenchStatus.ERROR.value:
        return "Agent 工作流发生受控错误，未生成研究结论。"
    risk = state.get("risk")
    if risk and not risk.get("approved", False):
        codes = ", ".join(item.get("code", "unknown") for item in risk["findings"])
        return f"研究结果未通过确定性风险门禁：{codes}。"
    analyses = state.get("analyses", [])
    if not analyses:
        return "没有可验证的分析结果，任务已阻断。"
    parts = []
    for item in analyses:
        metrics = item.get("metrics", {})
        if metrics:
            parts.append(
                f"{item.get('symbol')}: total_return={metrics.get('total_return', 0):.4f}, "
                f"max_drawdown={metrics.get('max_drawdown', 0):.4f}"
            )
        else:
            parts.append(
                f"{item.get('symbol')}: latest_close={item.get('latest_close', 'n/a')}"
            )
    return "；".join(parts) + "。结果仅用于研究演示，不构成投资建议。"

