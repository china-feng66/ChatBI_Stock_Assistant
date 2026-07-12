import pandas as pd

from quantquery_a.strategy import generate_macd_targets


def test_macd_targets_are_ordered_and_binary():
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-01-01", periods=80, freq="D")[::-1],
            "close": [100 + index * 0.2 + ((index % 10) - 5) for index in range(80)],
        }
    )
    result = generate_macd_targets(frame)
    assert result["trade_date"].is_monotonic_increasing
    assert set(result["target_position"].unique()).issubset({0, 1})
    assert {"macd", "signal_line", "histogram"}.issubset(result.columns)
