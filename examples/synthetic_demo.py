import pandas as pd

from quantquery_a import BacktestConfig, generate_macd_targets, run_long_only_backtest


dates = pd.bdate_range("2025-01-02", periods=120)
bars = pd.DataFrame(
    {
        "trade_date": dates,
        "open": [100 + index * 0.08 for index in range(len(dates))],
        "close": [100 + index * 0.08 + ((index % 12) - 6) * 0.25 for index in range(len(dates))],
    }
)
targets = generate_macd_targets(bars)
input_frame = bars.merge(
    targets[["trade_date", "target_position"]],
    on="trade_date",
    how="left",
)
result = run_long_only_backtest(
    input_frame,
    config=BacktestConfig(
        initial_cash=100_000,
        lot_size=100,
        commission_rate=0.0003,
        min_commission=5.0,
        sell_tax_rate=0.0005,
        slippage_bps=5.0,
    ),
)

print("metrics")
for name, value in result.metrics.items():
    print(f"{name}: {value:.6f}")
print(f"trades: {len(result.trades)}")
