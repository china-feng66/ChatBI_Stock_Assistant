"""Qwen adapter with structured results and strict token accounting."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Protocol

from openai import OpenAI


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class LLMResponse:
    content: dict[str, Any]
    model: str
    usage: LLMUsage


class QwenClient(Protocol):
    model: str

    def complete(
        self,
        *,
        agent: str,
        system_prompt: str,
        payload: dict[str, Any],
        max_output_tokens: int,
    ) -> LLMResponse: ...


class DashScopeQwenClient:
    """OpenAI-compatible DashScope client. Secrets are read only from env."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured")
        self.model = model or os.environ.get("QWEN_MODEL", "qwen-plus")
        endpoint = base_url or os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self._client = OpenAI(api_key=key, base_url=endpoint)

    def complete(
        self,
        *,
        agent: str,
        system_prompt: str,
        payload: dict[str, Any],
        max_output_tokens: int,
    ) -> LLMResponse:
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            max_tokens=max(64, max_output_tokens),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                },
            ],
        )
        elapsed = (time.perf_counter() - started) * 1_000
        message = response.choices[0].message.content or "{}"
        parsed = json.loads(message)
        usage = response.usage
        return LLMResponse(
            content=parsed if isinstance(parsed, dict) else {"result": parsed},
            model=self.model,
            usage=LLMUsage(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(
                    getattr(usage, "completion_tokens", 0) or 0
                ),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                latency_ms=elapsed,
            ),
        )


class FakeQwenClient:
    """Deterministic local replacement used by tests and offline demos."""

    model = "fake-qwen"

    def complete(
        self,
        *,
        agent: str,
        system_prompt: str,
        payload: dict[str, Any],
        max_output_tokens: int,
    ) -> LLMResponse:
        del system_prompt, max_output_tokens
        started = time.perf_counter()
        query = str(payload.get("query", "")).lower()
        if agent == "planner":
            task = "market_query"
            if "因子" in query or "组合" in query:
                task = "stock_comparison"
            elif "回测" in query or "backtest" in query:
                task = "backtest"
            elif "布林" in query or "bollinger" in query:
                task = "strategy_analysis"
            elif "风险" in query or "回撤" in query:
                task = "risk_diagnosis"
            elif "解释" in query or "什么是" in query:
                task = "knowledge_explain"
            content = {
                "task_type": task,
                "probabilities": {task: 0.9, "market_query": 0.1},
                "steps": ["collect_evidence", "run_tool", "risk_review"],
            }
        elif agent == "data":
            content = {
                "provider_preference": "bounded_inline_or_configured_provider",
                "required_fields": ["trade_date", "open", "close"],
                "repair": bool(payload.get("repair")),
            }
        elif agent == "quant":
            content = {
                "tool": payload.get("mode", "auto"),
                "reason": "selected from the allowlisted research tools",
            }
        elif agent == "risk":
            content = {
                "qualitative_findings": [],
                "decision_basis": "deterministic gate has final authority",
            }
        else:
            content = {
                "summary": payload.get(
                    "deterministic_summary",
                    "Evidence-linked research task finished.",
                ),
                "disclaimer": "Research demonstration only; not investment advice.",
            }
        input_size = len(json.dumps(payload, ensure_ascii=False, default=str))
        output_size = len(json.dumps(content, ensure_ascii=False))
        prompt_tokens = max(1, input_size // 4)
        completion_tokens = max(1, output_size // 4)
        return LLMResponse(
            content=content,
            model=self.model,
            usage=LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=(time.perf_counter() - started) * 1_000,
            ),
        )


def build_qwen_client() -> QwenClient:
    mode = os.environ.get("QUANTQUERY_LLM_MODE", "fake").strip().lower()
    if mode == "live":
        return DashScopeQwenClient()
    return FakeQwenClient()

