import { FormEvent, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, AgentEvent, EvaluationSummary, ResearchMode, RunSnapshot, openRunEvents } from "./api";
import { Chart } from "./components/Chart";

type Page = "workbench" | "market" | "lab" | "monitor";
const AGENTS = ["planner", "data", "quant", "risk", "reporter"] as const;
const TERMINAL = new Set(["completed", "blocked", "needs_clarification", "cancelled", "error"]);

const modeNames: Record<ResearchMode, string> = {
  auto: "自动路由",
  market: "行情快照",
  macd: "MACD 回测",
  bollinger: "布林带",
  factor: "三因子组合",
  compare: "标的比较"
};

export function StatusPill({ status }: { status: string }) {
  return <span className={`status status--${status}`}>{status.replaceAll("_", " ")}</span>;
}

function fmt(value: number | undefined, digits = 1) {
  return value === undefined || Number.isNaN(value) ? "—" : value.toFixed(digits);
}

function pct(value: number | undefined) {
  return value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}

function safeArray(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? (value as Array<Record<string, unknown>>) : [];
}

function download(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

function exportRun(run: RunSnapshot, format: "json" | "csv" | "html") {
  if (format === "json") {
    download(`${run.run_id}.json`, JSON.stringify(run, null, 2), "application/json");
    return;
  }
  if (format === "csv") {
    const rows = run.analyses.map((item) => ({
      symbol: String(item.symbol ?? ""),
      latest_close: String(item.latest_close ?? ""),
      total_return: String((item.metrics as Record<string, unknown> | undefined)?.total_return ?? ""),
      max_drawdown: String((item.metrics as Record<string, unknown> | undefined)?.max_drawdown ?? "")
    }));
    const header = "symbol,latest_close,total_return,max_drawdown";
    download(`${run.run_id}.csv`, [header, ...rows.map((row) => Object.values(row).join(","))].join("\n"), "text/csv");
    return;
  }
  const report = (run.report ?? "无报告").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
  download(`${run.run_id}.html`, `<!doctype html><meta charset="utf-8"><title>QuantQuery-A</title><h1>研究报告</h1><pre>${report}</pre>`, "text/html");
}

export default function App() {
  const queryClient = useQueryClient();
  const [page, setPage] = useState<Page>("workbench");
  const [activeRunId, setActiveRunId] = useState<string>();
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationSummary>();

  const runs = useQuery({ queryKey: ["runs"], queryFn: api.listRuns, refetchInterval: 4_000 });
  const metrics = useQuery({ queryKey: ["metrics"], queryFn: api.metrics, refetchInterval: 5_000 });
  const watchlist = useQuery({ queryKey: ["watchlist"], queryFn: api.getWatchlist });
  const activeRun = useQuery({
    queryKey: ["run", activeRunId],
    queryFn: () => api.getRun(activeRunId!),
    enabled: Boolean(activeRunId),
    refetchInterval: (query) => TERMINAL.has(query.state.data?.status ?? "") ? false : 800
  });

  useEffect(() => {
    if (!activeRunId) return;
    setEvents([]);
    const socket = openRunEvents(activeRunId, (event) => {
      setEvents((current) => event.type === "stream.closed" ? current : [...current, event]);
      if (event.type === "stream.closed") {
        queryClient.invalidateQueries({ queryKey: ["run", activeRunId] });
        queryClient.invalidateQueries({ queryKey: ["runs"] });
        queryClient.invalidateQueries({ queryKey: ["metrics"] });
      }
    });
    return () => socket.close();
  }, [activeRunId, queryClient]);

  const selectedRun = activeRun.data ?? runs.data?.find((item) => item.run_id === activeRunId) ?? runs.data?.[0];

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => setPage("workbench")} aria-label="返回工作台">
          <span className="brand-mark">Q/A</span>
          <span><strong>QuantQuery-A</strong><small>实时多 Agent 量化研究台</small></span>
        </button>
        <nav aria-label="主要页面">
          {(["workbench", "market", "lab", "monitor"] as Page[]).map((item) => (
            <button key={item} className={page === item ? "nav-active" : ""} onClick={() => setPage(item)}>
              {{ workbench: "工作台", market: "行情", lab: "实验室", monitor: "监控" }[item]}
            </button>
          ))}
        </nav>
        <div className="system-state"><i /> LOCAL / SAFE MODE</div>
      </header>

      <main>
        {page === "workbench" && (
          <WorkbenchPage run={selectedRun} events={events} onCreated={setActiveRunId} />
        )}
        {page === "market" && (
          <MarketPage symbols={watchlist.data ?? []} onSaved={() => watchlist.refetch()} />
        )}
        {page === "lab" && <LabPage run={selectedRun} onCreated={(id) => { setActiveRunId(id); setPage("workbench"); }} />}
        {page === "monitor" && (
          <MonitorPage
            metrics={metrics.data}
            runs={runs.data ?? []}
            evaluation={evaluation}
            onEvaluation={setEvaluation}
          />
        )}
      </main>
    </div>
  );
}

function WorkbenchPage({ run, events, onCreated }: { run?: RunSnapshot; events: AgentEvent[]; onCreated: (id: string) => void }) {
  const [query, setQuery] = useState("回测 600519.SH 的 MACD 策略并复核风险");
  const [symbols, setSymbols] = useState("600519.SH");
  const [mode, setMode] = useState<ResearchMode>("macd");
  const create = useMutation({ mutationFn: api.createRun, onSuccess: (value) => onCreated(value.run_id) });
  const cancel = useMutation({ mutationFn: api.cancelRun });
  const resume = useMutation({ mutationFn: api.resumeRun });

  function submit(event: FormEvent) {
    event.preventDefault();
    create.mutate({
      query,
      symbols: symbols.split(/[,，\s]+/).map((item) => item.trim().toUpperCase()).filter(Boolean),
      mode
    });
  }

  return (
    <div className="page-grid workbench-grid">
      <section className="task-column">
        <div className="section-kicker">RESEARCH ORDER</div>
        <h1>把研究问题交给<br /><em>可审计的执行链</em></h1>
        <p className="lead">每一个判断都要落到数据指纹、确定性工具和风险门禁。Agent 负责计划与解释，代码负责算数。</p>
        <form className="research-form" onSubmit={submit}>
          <label>研究任务<textarea value={query} onChange={(event) => setQuery(event.target.value)} rows={4} /></label>
          <div className="form-row">
            <label>标的池<input value={symbols} onChange={(event) => setSymbols(event.target.value)} placeholder="600519.SH, 000001.SZ" /></label>
            <label>研究模式<select value={mode} onChange={(event) => setMode(event.target.value as ResearchMode)}>{Object.entries(modeNames).map(([value, name]) => <option key={value} value={value}>{name}</option>)}</select></label>
          </div>
          <button className="primary" disabled={create.isPending}>{create.isPending ? "正在编排…" : "启动研究任务 →"}</button>
          {create.error && <p className="form-error">任务未启动：{create.error.message}</p>}
        </form>
        <div className="boundary-note"><strong>边界</strong><span>不下单 · 不执行生成代码 · 不宣称发现 Alpha</span></div>
      </section>

      <section className="execution-column">
        <div className="panel-heading"><div><span>AGENT EXECUTION TAPE</span><h2>执行磁带</h2></div>{run && <StatusPill status={run.status} />}</div>
        <ExecutionTape run={run} events={events} />
        <ResultPanel run={run} onCancel={() => run && cancel.mutate(run.run_id)} onResume={() => run && resume.mutate(run.run_id)} />
      </section>
    </div>
  );
}

function ExecutionTape({ run, events }: { run?: RunSnapshot; events: AgentEvent[] }) {
  const latest = new Map<string, AgentEvent>();
  events.forEach((event) => { if (event.agent) latest.set(event.agent, event); });
  return (
    <ol className="agent-tape">
      {AGENTS.map((agent, index) => {
        const event = latest.get(agent);
        const active = event?.status === "running" || (run?.status === "running" && index === latest.size);
        const done = event?.type === "agent.completed" || event?.type === "risk.completed";
        return <li key={agent} className={`${active ? "is-active" : ""} ${done ? "is-done" : ""}`}><div className="tape-index">0{index + 1}</div><div><strong>{agent.toUpperCase()}</strong><span>{{ planner: "意图融合与计划", data: "取数与证据登记", quant: "白名单工具计算", risk: "数值门禁与复核", reporter: "证据约束报告" }[agent]}</span></div><small>{event ? event.status : index === 0 && run ? "waiting" : "standby"}</small></li>;
      })}
    </ol>
  );
}

function ResultPanel({ run, onCancel, onResume }: { run?: RunSnapshot; onCancel: () => void; onResume: () => void }) {
  const analysis = run?.analyses?.[0];
  const metrics = (analysis?.metrics ?? {}) as Record<string, number>;
  const equity = safeArray(analysis?.equity_curve);
  const chartOption = useMemo(() => ({
    animationDuration: 500,
    grid: { left: 48, right: 18, top: 24, bottom: 34 },
    xAxis: { type: "category", data: equity.map((row) => String(row.trade_date)), axisLabel: { color: "#6c7775", hideOverlap: true } },
    yAxis: { type: "value", scale: true, splitLine: { lineStyle: { color: "#dbe2df" } } },
    tooltip: { trigger: "axis" },
    series: [{ type: "line", data: equity.map((row) => Number(row.equity)), showSymbol: false, smooth: 0.15, lineStyle: { color: "#245dd8", width: 2 }, areaStyle: { color: "rgba(36,93,216,.08)" } }]
  }), [equity]);
  if (!run) return <div className="empty-result"><span>等待第一条研究任务</span><p>执行事件、证据和风险结论会在这里逐步展开。</p></div>;
  return (
    <div className="result-panel">
      <div className="run-meta"><code>{run.run_id.slice(0, 12)}</code><span>{run.llm_calls} calls</span><span>{run.token_total.toLocaleString()} tokens</span><span>repair {run.repair_count}/1</span></div>
      {analysis && <div className="metric-strip"><div><span>全期收益</span><strong>{pct(metrics.total_return)}</strong></div><div><span>最大回撤</span><strong>{pct(metrics.max_drawdown)}</strong></div><div><span>后 30%</span><strong>{pct(metrics.holdout_return)}</strong></div><div><span>风险门禁</span><strong>{run.risk?.approved ? "PASS" : "HOLD"}</strong></div></div>}
      {equity.length > 1 && <Chart option={chartOption} ariaLabel="策略净值曲线" />}
      {run.report && <article className="report"><h3>报告摘要</h3><p>{run.report}</p></article>}
      <div className="result-actions"><button onClick={() => exportRun(run, "html")}>HTML</button><button onClick={() => exportRun(run, "csv")}>CSV</button><button onClick={() => exportRun(run, "json")}>JSON</button>{["blocked", "cancelled", "error"].includes(run.status) && <button onClick={onResume}>重新运行</button>}{["queued", "running"].includes(run.status) && <button className="danger" onClick={onCancel}>取消</button>}</div>
    </div>
  );
}

function MarketPage({ symbols, onSaved }: { symbols: string[]; onSaved: () => void }) {
  const [text, setText] = useState(symbols.join(", ") || "600519.SH, 000001.SZ");
  const [selected, setSelected] = useState(symbols[0] ?? "600519.SH");
  const save = useMutation({ mutationFn: api.setWatchlist, onSuccess: onSaved });
  const daily = useMutation({ mutationFn: api.getDaily });
  const bars = daily.data?.bars ?? [];
  const option = useMemo(() => ({
    grid: { left: 50, right: 24, top: 26, bottom: 40 },
    xAxis: { type: "category", data: bars.map((row) => String(row.trade_date)), axisLabel: { hideOverlap: true } },
    yAxis: { scale: true, splitLine: { lineStyle: { color: "#dbe2df" } } },
    tooltip: { trigger: "axis" },
    series: [{ type: "candlestick", data: bars.map((row) => [Number(row.open), Number(row.close), Number(row.low ?? row.close), Number(row.high ?? row.close)]), itemStyle: { color: "#c84f3c", color0: "#24856b", borderColor: "#c84f3c", borderColor0: "#24856b" } }]
  }), [bars]);
  return <div className="standard-page"><div className="page-title"><div><span>MARKET GATEWAY</span><h1>自选行情与数据血缘</h1></div><p>最多 20 个标的。实时快照按 60 秒轮询，非交易时段可回放缓存分钟线。</p></div><section className="market-layout"><aside className="watchlist"><h2>自选池</h2><textarea value={text} onChange={(event) => setText(event.target.value)} rows={5} /><button className="primary" onClick={() => save.mutate(text.split(/[,，\s]+/).filter(Boolean))}>保存自选池</button>{(symbols.length ? symbols : ["600519.SH"]).map((symbol) => <button key={symbol} className={selected === symbol ? "symbol-active" : ""} onClick={() => { setSelected(symbol); daily.mutate(symbol); }}>{symbol}<span>读取日线 →</span></button>)}</aside><div className="market-chart"><div className="panel-heading"><div><span>QFQ DAILY / {selected}</span><h2>前复权日线</h2></div>{daily.isPending && <StatusPill status="running" />}</div>{bars.length ? <Chart option={option} ariaLabel={`${selected} 日线 K 线图`} className="chart chart--large" /> : <div className="empty-result"><span>选择标的读取行情</span><p>未配置数据源或无缓存时，后端会明确阻断，不生成模拟行情。</p></div>}{daily.data && <div className="evidence-row"><code>{String(daily.data.evidence.fingerprint ?? "").slice(0, 16)}</code><span>{String(daily.data.evidence.provider ?? "unknown")}</span><span>{String(daily.data.evidence.adjustment_status ?? "unknown")}</span></div>}</div></section></div>;
}

function LabPage({ run, onCreated }: { run?: RunSnapshot; onCreated: (id: string) => void }) {
  const create = useMutation({ mutationFn: api.createRun, onSuccess: (value) => onCreated(value.run_id) });
  const recipes: Array<{ mode: ResearchMode; title: string; note: string; query: string; symbols: string[] }> = [
    { mode: "macd", title: "MACD / T+1", note: "收盘信号，下一交易日开盘成交", query: "回测 600519.SH 的 MACD 策略", symbols: ["600519.SH"] },
    { mode: "bollinger", title: "Bollinger / 20·2", note: "均值回归目标与独立风险门禁", query: "研究 600519.SH 的布林带策略", symbols: ["600519.SH"] },
    { mode: "factor", title: "Price × Volume", note: "动量、低波动、流动性等权合成", query: "构建价格成交量三因子组合", symbols: ["600519.SH", "000001.SZ", "000858.SZ"] }
  ];
  return <div className="standard-page"><div className="page-title"><div><span>EXPERIMENT NOTEBOOK</span><h1>策略与因子实验室</h1></div><p>固定交易时点、成本、复权与时间切分。结果是工程验证，不是收益承诺。</p></div><div className="recipe-grid">{recipes.map((recipe, index) => <article key={recipe.mode}><span>0{index + 1}</span><h2>{recipe.title}</h2><p>{recipe.note}</p><button onClick={() => create.mutate({ query: recipe.query, symbols: recipe.symbols, mode: recipe.mode })}>用当前数据启动 →</button></article>)}</div>{run && <section className="recent-run"><div><span>RECENT RUN</span><h2>{run.request.query}</h2></div><StatusPill status={run.status} /></section>}</div>;
}

function MonitorPage({ metrics, runs, evaluation, onEvaluation }: { metrics?: Awaited<ReturnType<typeof api.metrics>>; runs: RunSnapshot[]; evaluation?: EvaluationSummary; onEvaluation: (value: EvaluationSummary) => void }) {
  const evaluate = useMutation({ mutationFn: api.runEvaluation, onSuccess: onEvaluation });
  const agents = metrics?.agents ?? [];
  return <div className="standard-page"><div className="page-title"><div><span>CONTROL ROOM</span><h1>完成率、token 与评测闭环</h1></div><button className="primary compact" onClick={() => evaluate.mutate()} disabled={evaluate.isPending}>{evaluate.isPending ? "正在运行 8 案例…" : "运行冻结评测"}</button></div><div className="monitor-stats"><div><span>技术终态率</span><strong>{pct(metrics?.technical_terminal_rate)}</strong></div><div><span>累计 token</span><strong>{metrics?.total_tokens.toLocaleString() ?? "—"}</strong></div><div><span>平均延迟</span><strong>{fmt(metrics?.average_llm_latency_ms, 0)} ms</strong></div><div><span>评测成功率</span><strong>{evaluation ? `${evaluation.passed_count}/${evaluation.case_count}` : "未运行"}</strong></div></div><div className="monitor-grid"><section><div className="panel-heading"><div><span>AGENT LEDGER</span><h2>角色调用账本</h2></div></div><table><thead><tr><th>Agent</th><th>调用</th><th>成功率</th><th>Token</th><th>延迟</th></tr></thead><tbody>{agents.map((agent) => <tr key={agent.agent}><td>{agent.agent}</td><td>{agent.calls}</td><td>{pct(agent.success_rate)}</td><td>{agent.tokens}</td><td>{fmt(agent.average_latency_ms)} ms</td></tr>)}</tbody></table>{!agents.length && <div className="table-empty">运行任务后生成角色指标。</div>}</section><section><div className="panel-heading"><div><span>FROZEN EVALUATION</span><h2>门禁检查</h2></div></div>{evaluation ? <dl className="gate-list"><div><dt>20 条路由集</dt><dd>{pct(evaluation.route_accuracy)}</dd></div><div><dt>8 条任务成功率</dt><dd>{pct(evaluation.evaluation_task_success_rate)}</dd></div><div><dt>确定性门禁</dt><dd>{pct(evaluation.deterministic_gate_rate)}</dd></div><div><dt>Token 超限</dt><dd>{evaluation.token_over_budget_count}</dd></div></dl> : <div className="empty-result"><span>尚未运行冻结评测</span><p>LLM Judge 只评表达质量，不能覆盖数值门禁。</p></div>}</section></div><section className="run-history"><div className="panel-heading"><div><span>RUN ARCHIVE</span><h2>任务历史</h2></div></div>{runs.slice(0, 8).map((run) => <div className="history-row" key={run.run_id}><code>{run.run_id.slice(0, 10)}</code><span>{run.request.query}</span><small>{run.token_total} tk</small><StatusPill status={run.status} /></div>)}</section></div>;
}
