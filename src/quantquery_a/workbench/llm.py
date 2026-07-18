"""Qwen adapter with structured results and strict token accounting."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Protocol

from openai import OpenAI


TASK_LABELS = (
    "market_query",
    "strategy_analysis",
    "backtest",
    "risk_diagnosis",
    "stock_comparison",
    "knowledge_explain",
    "unsupported",
)


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
        parsed = json.loads(response.choices[0].message.content or "{}")
        usage = response.usage
        return LLMResponse(
            content=_normalize_qwen_content(parsed),
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


def _normalize_qwen_content(parsed: Any) -> dict[str, Any]:
    """Accept both nested and top-level probability JSON from Qwen."""

    if not isinstance(parsed, dict):
        return {"result": parsed}
    if isinstance(parsed.get("probabilities"), dict):
        return parsed
    probabilities: dict[str, float] = {}
    for label in TASK_LABELS:
        value = parsed.get(label)
        if isinstance(value, (int, float)):
            probabilities[label] = float(value)
    if probabilities:
        remaining = {key: value for key, value in parsed.items() if key not in TASK_LABELS}
        remaining["probabilities"] = probabilities
        remaining.setdefault(
            "task_type", max(probabilities, key=probabilities.get)
        )
        return remaining
    return parsed


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
            task = _fake_task(query)
            probabilities = {
                item: (0.9 if item == task else 0.1 / 6) for item in TASK_LABELS
            }
            if _is_ambiguous(query):
                probabilities = {
                    "market_query": 0.24,
                    "strategy_analysis": 0.22,
                    "backtest": 0.14,
                    "risk_diagnosis": 0.10,
                    "stock_comparison": 0.15,
                    "knowledge_explain": 0.10,
                    "unsupported": 0.05,
                }
            content = {
                "task_type": task,
                "probabilities": probabilities,
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


def _fake_task(query: str) -> str:
    if any(word in query for word in ("下单", "实盘", "保证", "auto trade")):
        return "unsupported"
    if any(word in query for word in ("因子", "组合", "比较", "排序", "哪个更好", "top5")):
        return "stock_comparison"
    if any(word in query for word in ("回测", "历史模拟", "过去一年", "backtest")):
        return "backtest"
    if any(word in query for word in ("风险", "回撤", "前视", "滑点", "复权", "遗漏")):
        return "risk_diagnosis"
    if any(word in query for word in ("什么是", "解释", "为什么", "原理", "含义")):
        return "knowledge_explain"
    if any(word in query for word in ("macd", "布林", "信号", "入场", "离场")):
        return "strategy_analysis"
    return "market_query"


def _is_ambiguous(query: str) -> bool:
    return query.strip() in {
        "帮我看看 demo.sh",
        "分析一下这个",
        "这个策略怎么样",
        "最近有什么变化",
    }


def build_qwen_client() -> QwenClient:
    mode = os.environ.get("QUANTQUERY_LLM_MODE", "fake").strip().lower()
    if mode == "live":
        return DashScopeQwenClient()
    return FakeQwenClient()
