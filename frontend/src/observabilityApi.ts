export interface AgentMetric {
  agent: string;
  calls: number;
  status?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens: number;
  average_tokens?: number;
  average_latency_ms?: number;
  latency_p50_ms?: number;
  latency_p95_ms: number;
  phase_duration_ms?: number;
  error_count?: number;
  error_rate?: number;
  success_rate?: number;
  event_count?: number;
}

export interface CommunicationEdge {
  source: string;
  target: string;
  count: number;
  sequences: number[];
  contract: string;
}

export interface ObservableEvent {
  sequence: number;
  type: string;
  agent?: string;
  status: string;
  timestamp: string;
  payload: Record<string, unknown>;
}

export interface RunObservability {
  run_id: string;
  status: string;
  duration_ms: number;
  token_total: number;
  token_budget: number;
  token_utilization: number;
  llm_calls: number;
  llm_call_budget: number;
  repair_count: number;
  route?: { task_type?: string; confidence?: number; margin?: number };
  agent_metrics: AgentMetric[];
  communication_edges: CommunicationEdge[];
  events: ObservableEvent[];
  gates: {
    risk_approved: boolean;
    risk_finding_count: number;
    evidence_count: number;
    evidence_completeness_rate: number;
    stale_evidence_count: number;
    unknown_adjustment_count: number;
    trade_count: number;
    t_plus_one_violation_count: number;
    token_budget_passed: boolean;
    repair_scope_passed: boolean;
  };
}

export interface ObservableSummary {
  runs_started: number;
  technical_terminal_rate: number;
  research_completion_rate: number;
  clarification_rate: number;
  repair_attempt_count: number;
  repair_success_rate: number;
  average_route_confidence: number;
  evidence_batch_count: number;
  evidence_completeness_rate: number;
  stale_evidence_rate: number;
  tokens_per_completed_run: number;
  task_latency_p50_ms: number;
  task_latency_p95_ms: number;
  llm_latency_p50_ms: number;
  llm_latency_p95_ms: number;
  llm_error_rate: number;
  token_over_budget_count: number;
  agent_metrics: AgentMetric[];
}

export interface RoutingCaseResult {
  name: string;
  query: string;
  expected_task?: string;
  actual_task: string;
  route_path: string;
  top1_correct: boolean;
  top3_correct: boolean;
  clarification_expected: boolean;
  clarification_actual: boolean;
  confidence: number;
  margin: number;
  tokens: number;
  latency_ms: number;
  error_type?: string;
}

export interface RoutingBenchmark {
  benchmark_id: string;
  mode: string;
  model: string;
  case_count: number;
  evaluated_count: number;
  error_count: number;
  top1_accuracy: number;
  top3_recall: number;
  clarification_precision: number;
  clarification_recall: number;
  clarification_f1: number;
  unsupported_block_recall: number;
  total_tokens: number;
  average_tokens: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  cases: RoutingCaseResult[];
  created_at: string;
}

async function get<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const observabilityApi = {
  summary: () => get<ObservableSummary>("/api/v1/metrics/observability"),
  run: (runId: string) => get<RunObservability>(`/api/v1/runs/${runId}/observability`),
  latestRouting: () => get<RoutingBenchmark | null>("/api/v1/evals/routing/latest"),
  runRouting: (live: boolean) =>
    get<RoutingBenchmark>(`/api/v1/evals/routing?live=${live}`, { method: "POST" })
};
