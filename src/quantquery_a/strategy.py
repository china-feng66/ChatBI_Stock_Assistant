"""Transparent strategy signal generation."""

from __future__ import annotations

import pandas as pd


def generate_macd_targets(
    bars: pd.DataFrame,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Generate persistent 0/1 targets from MACD line crossings.

    The output at row T only uses closes through T. Execution timing belongs to
    the backtest engine, which delays this target until T+1 open.
    """

    if not 0 < fast < slow:
        raise ValueError("expected 0 < fast < slow")
    if signal <= 0:
        raise ValueError("signal must be positive")
    if "trade_date" not in bars or "close" not in bars:
        raise ValueError("bars must contain trade_date and close")

    frame = bars.loc[:, ["trade_date", "close"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise")
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    if frame["trade_date"].duplicated().any():
        raise ValueError("trade_date values must be unique")
    if (frame["close"] <= 0).any():
        raise ValueError("close prices must be positive")

    fast_ema = frame["close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = frame["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
    frame["macd"] = fast_ema - slow_ema
    frame["signal_line"] = frame["macd"].ewm(
        span=signal,
        adjust=False,
        min_periods=signal,
    ).mean()
    frame["histogram"] = frame["macd"] - frame["signal_line"]

    cross_up = (frame["macd"] > frame["signal_line"]) & (
        frame["macd"].shift(1) <= frame["signal_line"].shift(1)
    )
    cross_down = (frame["macd"] < frame["signal_line"]) & (
        frame["macd"].shift(1) >= frame["signal_line"].shift(1)
    )
    targets = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    targets.loc[cross_up] = 1
    targets.loc[cross_down] = 0
    frame["target_position"] = targets.ffill().fillna(0).astype(int)
    return frame
