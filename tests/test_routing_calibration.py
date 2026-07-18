from __future__ import annotations

from quantquery_a.agents.contracts import AnalyzeRequest, RoutePath, TaskType
from quantquery_a.workbench.routing import HybridIntentRouter


def _route(query: str, probabilities: dict[str, float], symbols=None):
    return HybridIntentRouter().route(
        AnalyzeRequest(
            query=query,
            symbols=symbols if symbols is not None else ["DEMO.SH"],
        ),
        probabilities,
    )


def test_finance_rules_correct_known_live_qwen_confusions() -> None:
    risk = _route(
        "诊断策略是否错误使用未来数据",
        {"risk_diagnosis": 0.60, "strategy_analysis": 0.15, "backtest": 0.10},
    )
    factor = _route(
        "构建价格成交量三因子组合",
        {"strategy_analysis": 0.65, "backtest": 0.20, "stock_comparison": 0.04},
        ["AAA.SH", "BBB.SH", "CCC.SH"],
    )
    ranking = _route(
        "给自选池股票按收益和风险排序",
        {
            "strategy_analysis": 0.25,
            "risk_diagnosis": 0.25,
            "stock_comparison": 0.25,
        },
        ["AAA.SH", "BBB.SH", "CCC.SH"],
    )

    assert risk.task_type is TaskType.RISK_DIAGNOSIS
    assert factor.task_type is TaskType.STOCK_COMPARISON
    assert ranking.task_type is TaskType.STOCK_COMPARISON


def test_underspecified_request_always_clarifies() -> None:
    decision = _route(
        "这个策略怎么样",
        {"strategy_analysis": 0.90, "market_query": 0.02},
    )

    assert decision.path is RoutePath.CLARIFICATION


def test_unsafe_trading_and_guarantee_requests_are_deterministically_blocked() -> None:
    for query in ("连接实盘账户自动交易", "保证这个策略每个月盈利"):
        decision = _route(
            query,
            {"strategy_analysis": 0.80, "unsupported": 0.02},
        )
        assert decision.task_type is TaskType.UNSUPPORTED
        assert decision.path is RoutePath.PLANNED
        assert decision.confidence == 1.0
        assert any(item.source == "safety_guard" for item in decision.source_scores)


def test_router_keeps_fixed_three_way_weights() -> None:
    weights = HybridIntentRouter().weights
    assert (weights.llm, weights.tfidf, weights.keyword) == (0.45, 0.30, 0.25)
