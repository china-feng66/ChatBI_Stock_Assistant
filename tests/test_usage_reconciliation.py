from __future__ import annotations

import asyncio
import json

from quantquery_a.workbench.contracts import RunCreateRequest
from quantquery_a.workbench.llm import FakeQwenClient
from quantquery_a.workbench.market import BoundedMarketGateway, MarketCache
from quantquery_a.workbench.service import WorkbenchService


def _blocked_service(tmp_path) -> WorkbenchService:
    gateway = BoundedMarketGateway(
        cache=MarketCache(tmp_path / "market.sqlite3"),
        daily_providers=[],
    )
    return WorkbenchService(
        runtime_dir=tmp_path / "runtime",
        llm=FakeQwenClient(),
        market_gateway=gateway,
    )


def _request() -> RunCreateRequest:
    return RunCreateRequest(
        query="查询无缓存行情",
        symbols=["MISS.SH"],
        mode="market",
    )


def test_failed_tool_path_keeps_successful_llm_usage_in_run_totals(tmp_path) -> None:
    service = _blocked_service(tmp_path)
    snapshot = service.run_sync(_request())
    row = service.store.get_run_row(snapshot.run_id)
    persisted = json.loads(row["state_json"])
    metrics = service.store.metrics()

    assert snapshot.status.value == "blocked"
    assert snapshot.llm_calls == 3
    assert snapshot.token_total == metrics["total_tokens"]
    assert persisted["llm_calls"] == snapshot.llm_calls
    assert persisted["token_total"] == snapshot.token_total


def test_resume_keeps_prior_calls_inside_the_six_call_budget(tmp_path) -> None:
    service = _blocked_service(tmp_path)
    first = service.run_sync(_request())

    async def resume_and_wait():
        await service.resume(first.run_id)
        await service._tasks[first.run_id]

    asyncio.run(resume_and_wait())
    resumed = service.get_run(first.run_id)
    calls, tokens = service._usage_totals(first.run_id)

    assert resumed.status.value == "blocked"
    assert resumed.llm_calls == calls == 6
    assert resumed.token_total == tokens
    assert resumed.llm_calls <= 6
