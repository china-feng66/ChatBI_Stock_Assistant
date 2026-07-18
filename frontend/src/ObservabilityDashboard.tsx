import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, RunSnapshot } from "./api";
import { Chart } from "./components/Chart";
import {
  AgentMetric,
  observabilityApi,
  RoutingBenchmark,
  RunObservability
} from "./observabilityApi";

const AGENTS = ["planner", "data", "quant", "risk", "reporter"];

function pct(value = 0) { return `${(value * 100).toFixed(1)}%`; }
function ms(value = 0) { return value >= 1_000 ? `${(value / 1_000).toFixed(2)} s` : `${value.toFixed(0)} ms`; }
function compact(value = 0) { return value >= 1_000 ? `${(value / 1_000).toFixed(1)}k` : value.toFixed(0); }

function Metric({ label, value, note, tone = "neutral" }: { label: string; value: string; note: string; tone?: string }) {
  return <div className={`obs-metric obs-metric--${tone}`}><span>{label}</span><strong>{value}</strong><small>{note}</small></div>;
}

function incomingEdges(run: RunObservability | undefined, target: string) {
  return run?.communication_edges.filter((edge) => edge.target === target) ?? [];
}

function AgentNode({ metric, active }: { metric?: AgentMetric; active: boolean }) {
  return <div className={`agent-node ${active ? "agent-node--active" : ""}`}><div className="agent-node__signal" /><span>{metric?.agent.toUpperCase()}</span><strong>{compact(metric?.total_tokens)} tk</strong><small>{ms(metric?.latency_p95_ms)} p95 · {metric?.calls ?? 0} calls</small></div>;
}

function CommunicationMap({ run, systemAgents }: { run?: RunObservability; systemAgents: AgentMetric[] }) {
  return <div className="comm-map"><div className="comm-source">ORCHESTRATOR<small>typed state</small></div>{AGENTS.map((agent) => {
    const incoming = incomingEdges(run, agent);
    const count = incoming.reduce((total, edge) => total + edge.count, 0);
    const sources = incoming.map((edge) => edge.source.toUpperCase()).join("+");
    return <div className="comm-step" key={agent}><div className={`comm-edge ${count ? "comm-edge--active" : ""}`}><span>{count ? `${sources} ${count}` : "·"}</span></div><AgentNode metric={(run?.agent_metrics ?? systemAgents).find((item) => item.agent === agent)} active={count > 0} /></div>;
  })}<div className={`comm-terminal ${run?.status === "completed" ? "comm-terminal--pass" : ""}`}>{run?.status ?? "TERMINAL"}<small>{incomingEdges(run, "terminal").map((edge) => `${edge.source} ${edge.count}`).join(" · ")}</small></div></div>;
}

function GatePanel({ run }: { run?: RunObservability }) {
  const gates = run?.gates;
  const items = [
    ["Evidence", gates ? gates.evidence_completeness_rate === 1 : false, gates ? pct(gates.evidence_completeness_rate) : "—"],
    ["T+1", gates ? gates.t_plus_one_violation_count === 0 : false, gates ? `${gates.t_plus_one_violation_count} violations` : "—"],
    ["Risk", gates?.risk_approved ?? false, gates ? (gates.risk_approved ? "approved" : `${gates.risk_finding_count} findings`) : "—"],
    ["Token", gates?.token_budget_passed ?? false, run ? `${run.token_total}/${run.token_budget}` : "—"],
    ["Repair", gates?.repair_scope_passed ?? false, run ? `${run.repair_count}/1` : "—"]
  ];
  return <div className="gate-grid">{items.map(([name, passed, detail]) => <div key={String(name)} className={passed ? "gate-pass" : "gate-hold"}><i>{passed ? "✓" : "!"}</i><span>{name}</span><small>{detail}</small></div>)}</div>;
}

function RunTimeline({ run }: { run?: RunObservability }) {
  if (!run) return <div className="obs-empty">选择一个任务查看事件时间线。</div>;
  return <div className="event-stream">{run.events.map((event) => <div className="event-line" key={event.sequence}><code>{String(event.sequence).padStart(2, "0")}</code><time>{new Date(event.timestamp).toLocaleTimeString("zh-CN", { hour12: false })}</time><strong>{event.agent?.toUpperCase() ?? "SYSTEM"}</strong><span>{event.type}</span><em>{event.status}</em></div>)}</div>;
}

function BenchmarkPanel({ benchmark, running, onMock, onLive, error }: { benchmark?: RoutingBenchmark | null; running: boolean; onMock: () => void; onLive: () => void; error?: Error | null }) {
  const failures = benchmark?.cases.filter((item) => !item.top1_correct).slice(0, 8) ?? [];
  return <section className="obs-panel benchmark-panel"><div className="obs-panel__head"><div><span>LABELED ROUTING BENCHMARK</span><h2>路由与澄清评测</h2></div><div className="obs-actions"><button onClick={onMock} disabled={running}>Mock 评测</button><button className="live-button" onClick={onLive} disabled={running}>当前 Qwen 评测</button></div></div>{error && <div className="obs-error">{error.message}</div>}{benchmark ? <><div className="benchmark-stats"><Metric label="Top-1" value={pct(benchmark.top1_accuracy)} note={`${benchmark.evaluated_count}/${benchmark.case_count} evaluated`} tone={benchmark.top1_accuracy >= .9 ? "good" : "warn"} /><Metric label="Top-3 recall" value={pct(benchmark.top3_recall)} note="候选召回" tone="good" /><Metric label="澄清召回" value={pct(benchmark.clarification_recall)} note={`precision ${pct(benchmark.clarification_precision)}`} tone={benchmark.clarification_recall >= .9 ? "good" : "warn"} /><Metric label="安全阻断" value={pct(benchmark.unsupported_block_recall)} note="unsupported recall" tone={benchmark.unsupported_block_recall === 1 ? "good" : "warn"} /><Metric label="p95 延迟" value={ms(benchmark.latency_p95_ms)} note={benchmark.model} /><Metric label="Token" value={compact(benchmark.total_tokens)} note={`${compact(benchmark.average_tokens)} / case`} /></div><div className="benchmark-meta"><code>{benchmark.benchmark_id.slice(0, 12)}</code><span>{benchmark.mode.toUpperCase()}</span><span>{new Date(benchmark.created_at).toLocaleString("zh-CN")}</span><span>{benchmark.error_count} errors</span></div>{failures.length > 0 && <div className="failure-table"><div className="failure-table__title">需要复盘的案例</div>{failures.map((item) => <div key={item.name}><code>{item.name}</code><span>{item.query}</span><small>{item.expected_task ?? "clarification"} → {item.actual_task}/{item.route_path}</small></div>)}</div>}</> : <div className="obs-empty">运行标注集后生成 Top-1、Top-3、澄清和安全阻断指标。</div>}</section>;
}

export function ObservabilityDashboard() {
  const queryClient = useQueryClient();
  const [runId, setRunId] = useState<string>();
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.listRuns, refetchInterval: 4_000 });
  const summary = useQuery({ queryKey: ["observability-summary"], queryFn: observabilityApi.summary, refetchInterval: 3_000 });
  const latest = useQuery({ queryKey: ["routing-latest"], queryFn: observabilityApi.latestRouting });
  const selected = runId ?? runs.data?.[0]?.run_id;
  const run = useQuery({ queryKey: ["run-observability", selected], queryFn: () => observabilityApi.run(selected!), enabled: Boolean(selected), refetchInterval: 2_000 });
  const benchmark = useMutation({ mutationFn: observabilityApi.runRouting, onSuccess: (value) => { queryClient.setQueryData(["routing-latest"], value); } });
  const system = summary.data;
  const agentOption = useMemo(() => ({
    grid: { left: 42, right: 18, top: 18, bottom: 30 },
    xAxis: { type: "category", data: (system?.agent_metrics ?? []).map((item) => item.agent.toUpperCase()), axisLabel: { color: "#8fa29e" } },
    yAxis: [{ type: "value", name: "token", splitLine: { lineStyle: { color: "#263b37" } } }, { type: "value", name: "ms" }],
    tooltip: { trigger: "axis" },
    series: [{ name: "tokens", type: "bar", data: (system?.agent_metrics ?? []).map((item) => item.total_tokens), itemStyle: { color: "#55d6ad" } }, { name: "latency p95", type: "line", yAxisIndex: 1, data: (system?.agent_metrics ?? []).map((item) => item.latency_p95_ms), lineStyle: { color: "#5f8cff" }, itemStyle: { color: "#5f8cff" } }]
  }), [system]);

  return <div className="observability-shell"><header className="obs-header"><div><a href="#">Q/A</a><span>AGENT OBSERVABILITY CONTROL ROOM</span></div><div className="obs-header__right"><i /> LOCAL TELEMETRY <button onClick={() => { summary.refetch(); run.refetch(); latest.refetch(); }}>刷新</button></div></header><main className="obs-main"><section className="obs-title"><div><span>CONTROL PLANE / v0.4</span><h1>看见 Agent 如何协作，<br />也看见它哪里失效。</h1></div><div className="run-picker"><label>观察任务<select value={selected ?? ""} onChange={(event) => setRunId(event.target.value)}>{(runs.data ?? []).map((item: RunSnapshot) => <option key={item.run_id} value={item.run_id}>{item.run_id.slice(0, 10)} · {item.status} · {item.request.query.slice(0, 24)}</option>)}</select></label><small>只展示类型化状态、事件和指标；不记录 prompt 与密钥。</small></div></section><section className="obs-kpis"><Metric label="技术终态率" value={pct(system?.technical_terminal_rate)} note={`${system?.runs_started ?? 0} runs`} tone="good" /><Metric label="研究完成率" value={pct(system?.research_completion_rate)} note="Risk approved" /><Metric label="证据完整率" value={pct(system?.evidence_completeness_rate)} note={`${system?.evidence_batch_count ?? 0} batches`} tone={(system?.evidence_completeness_rate ?? 0) === 1 ? "good" : "warn"} /><Metric label="任务 p95" value={ms(system?.task_latency_p95_ms)} note="end-to-end" /><Metric label="LLM p95" value={ms(system?.llm_latency_p95_ms)} note="per call" /><Metric label="Token / 完成" value={compact(system?.tokens_per_completed_run)} note={`${system?.token_over_budget_count ?? 0} over budget`} /></section><section className="obs-panel communication-panel"><div className="obs-panel__head"><div><span>TYPED STATE TRANSFER</span><h2>Agent 通信拓扑</h2></div><div className="run-badge"><strong>{run.data?.status ?? "NO RUN"}</strong><span>{run.data ? `${ms(run.data.duration_ms)} · ${run.data.llm_calls}/${run.data.llm_call_budget} calls` : "等待任务"}</span></div></div><CommunicationMap run={run.data} systemAgents={system?.agent_metrics ?? []} /><GatePanel run={run.data} /></section><div className="obs-grid"><section className="obs-panel"><div className="obs-panel__head"><div><span>AGENT COST / LATENCY</span><h2>角色性能分布</h2></div></div><Chart option={agentOption} ariaLabel="各 Agent token 和延迟" className="obs-chart" /><div className="agent-ledger">{(system?.agent_metrics ?? []).map((agent) => <div key={agent.agent}><strong>{agent.agent}</strong><span>{agent.calls} calls</span><span>{compact(agent.total_tokens)} tk</span><span>{ms(agent.latency_p95_ms)} p95</span><em>{pct(agent.error_rate)}</em></div>)}</div></section><section className="obs-panel"><div className="obs-panel__head"><div><span>EVENT STREAM</span><h2>任务事件时间线</h2></div><code>{selected?.slice(0, 10) ?? "—"}</code></div><RunTimeline run={run.data} /></section></div><BenchmarkPanel benchmark={latest.data} running={benchmark.isPending} onMock={() => benchmark.mutate(false)} onLive={() => benchmark.mutate(true)} error={benchmark.error} /></main></div>;
}
