export type RunStatus = "queued" | "running" | "completed" | "blocked" | "needs_clarification" | "cancelled" | "interrupted" | "error";
export type ResearchMode = "auto" | "market" | "macd" | "bollinger" | "factor" | "compare";

export interface RunRequest { query: string; symbols: string[]; mode: ResearchMode }
export interface RunSnapshot {
  run_id: string; status: RunStatus; request: RunRequest; created_at: string; updated_at: string;
  route?: { task_type?: string; confidence?: number; margin?: number };
  plan?: { mode?: string; steps?: string[] };
  evidence: Array<Record<string, unknown>>; analyses: Array<Record<string, unknown>>;
  risk?: { approved?: boolean; findings?: Array<Record<string, unknown>> };
  report?: string; token_total: number; llm_calls: number; repair_count: number; error?: string;
}
export interface AgentEvent { sequence?: number; type: string; agent?: string; status: string; timestamp?: string; payload?: Record<string, unknown> }
export interface MetricsSummary {
  runs_started: number; status_counts: Record<string, number>; technical_terminal_rate: number;
  total_tokens: number; average_llm_latency_ms: number; llm_call_count: number;
  agents: Array<{ agent: string; calls: number; success_rate: number; tokens: number; average_latency_ms: number }>;
}
export interface EvaluationSummary {
  evaluation_id: string; passed_count: number; case_count: number; route_accuracy: number;
  technical_terminal_rate: number; evaluation_task_success_rate: number; research_completion_rate: number;
  deterministic_gate_rate: number; token_over_budget_count: number;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
  if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

export const api = {
  createRun: (payload: RunRequest) => request<RunSnapshot>("/api/v1/runs", { method: "POST", body: JSON.stringify(payload) }),
  getRun: (runId: string) => request<RunSnapshot>(`/api/v1/runs/${runId}`),
  listRuns: () => request<RunSnapshot[]>("/api/v1/runs?limit=50"),
  cancelRun: (runId: string) => request<RunSnapshot>(`/api/v1/runs/${runId}/cancel`, { method: "POST" }),
  resumeRun: (runId: string) => request<RunSnapshot>(`/api/v1/runs/${runId}/resume`, { method: "POST" }),
  metrics: () => request<MetricsSummary>("/api/v1/metrics/summary"),
  runEvaluation: () => request<EvaluationSummary>("/api/v1/evals/run", { method: "POST" }),
  getWatchlist: () => request<string[]>("/api/v1/watchlist"),
  setWatchlist: (symbols: string[]) => request<string[]>("/api/v1/watchlist", { method: "POST", body: JSON.stringify({ symbols }) }),
  getDaily: (symbol: string) => request<{ symbol: string; bars: Array<Record<string, unknown>>; evidence: Record<string, unknown> }>(`/api/v1/market/${encodeURIComponent(symbol)}/daily`)
};

export function openRunEvents(runId: string, onEvent: (event: AgentEvent) => void): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}/api/v1/runs/${runId}/events`);
  socket.onmessage = (message) => onEvent(JSON.parse(message.data) as AgentEvent);
  return socket;
}
