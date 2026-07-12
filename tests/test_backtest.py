import pandas as pd
import pytest

from quantquery_a.backtest import BacktestConfig, run_long_only_backtest


def bars():
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
            ),
            "open": [10.0, 11.0, 12.0, 13.0],
            "close": [10.5, 11.5, 12.5, 13.5],
            "target_position": [1.0, 1.0, 0.0, 0.0],
        }
    )


def test_next_bar_execution_and_lot_size():
    result = run_long_only_backtest(
        bars(),
        config=BacktestConfig(initial_cash=10_000, lot_size=100),
    )
    buy, sell = result.trades
    assert buy.signal_date == pd.Timestamp("2026-01-02")
    assert buy.execution_date == pd.Timestamp("2026-01-05")
    assert buy.fill_price == pytest.approx(11.0)
    assert buy.shares == 900
    assert sell.execution_date == pd.Timestamp("2026-01-07")
    assert sell.fill_price == pytest.approx(13.0)
    assert result.equity_curve.iloc[-1]["equity"] == pytest.approx(11_800.0)
    for trade in result.trades:
        assert trade.execution_date > trade.signal_date
        assert trade.shares % 100 == 0
        assert trade.cash_after >= 0


def test_costs_and_slippage_reduce_equity():
    zero_cost = run_long_only_backtest(
        bars(), config=BacktestConfig(initial_cash=100_000)
    )
    with_costs = run_long_only_backtest(
        bars(),
        config=BacktestConfig(
            initial_cash=100_000,
            commission_rate=0.0003,
            min_commission=5.0,
            sell_tax_rate=0.0005,
            slippage_bps=5.0,
        ),
    )
    assert with_costs.metrics["total_cost"] > 0
    assert with_costs.equity_curve.iloc[-1]["equity"] < zero_cost.equity_curve.iloc[-1]["equity"]


def test_insufficient_cash_produces_no_trade():
    result = run_long_only_backtest(
        bars(), config=BacktestConfig(initial_cash=500, lot_size=100)
    )
    assert result.trades == ()
    assert result.equity_curve.iloc[-1]["equity"] == pytest.approx(500.0)


def test_metrics_and_invalid_targets():
    result = run_long_only_backtest(
        bars(), config=BacktestConfig(initial_cash=10_000)
    )
    assert {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "benchmark_return",
        "excess_return",
        "turnover",
        "total_cost",
    }.issubset(result.metrics)

    invalid = bars()
    invalid.loc[0, "target_position"] = 0.5
    with pytest.raises(ValueError, match="only 0 or 1"):
        run_long_only_backtest(invalid)
