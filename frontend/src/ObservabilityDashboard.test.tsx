import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { ObservabilityDashboard } from "./ObservabilityDashboard";

vi.mock("./components/Chart", () => ({
  Chart: ({ ariaLabel }: { ariaLabel: string }) => <div role="img" aria-label={ariaLabel} />
}));

afterEach(() => vi.unstubAllGlobals());

test("observability dashboard renders a valid empty state", async () => {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/api/v1/runs?")) return new Response(JSON.stringify([]));
    if (url.includes("/api/v1/evals/routing/latest")) return new Response("null");
    if (url.includes("/api/v1/metrics/observability")) {
      return new Response(JSON.stringify({ runs_started: 0, agent_metrics: [] }));
    }
    return new Response("not found", { status: 404 });
  }));

  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <ObservabilityDashboard />
    </QueryClientProvider>
  );

  expect(await screen.findByText("Agent 通信拓扑")).toBeInTheDocument();
  expect(screen.getByText("选择一个任务查看事件时间线。")).toBeInTheDocument();
  expect(screen.getByRole("img", { name: "各 Agent token 和延迟" })).toBeInTheDocument();
});
