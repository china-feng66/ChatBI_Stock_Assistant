import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { StatusPill } from "./App";

test("status pill renders deterministic terminal state", () => {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <StatusPill status="needs_clarification" />
    </QueryClientProvider>
  );
  expect(screen.getByText("needs clarification")).toBeInTheDocument();
});
