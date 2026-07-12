"""Deterministic long-only daily-bar backtest."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    lot_size: int = 100
    commission_rate: float = 0.0
    min_commission: float = 0.0
    sell_tax_rate: float = 0.0
    slippage_bps: float = 0.0
    annualization_days: int = 252

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        for name in (
            "commission_rate",
            "min_commission",
            "sell_tax_rate",
            "slippage_bps",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.annualization_days <= 0:
            raise ValueError("annualization_days must be positive")


@dataclass(frozen=True)
class Trade:
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    side: str
    shares: int
    fill_price: float
    gross_value: float
    commission: float
    tax: float
    cash_after: float
    shares_after: int


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: tuple[Trade, ...]
    metrics: dict[str, float]
    assumptions: tuple[str, ...]


def _commission(gross_value: float, config: BacktestConfig) -> float:
    if gross_value <= 0:
        return 0.0
    return max(gross_value * config.commission_rate, config.min_commission)


def _maximum_affordable_shares(
    cash: float,
    fill_price: float,
    config: BacktestConfig,
) -> int:
    lots = int(cash // (fill_price * config.lot_size))
    while lots > 0:
        shares = lots * config.lot_size
        gross = shares * fill_price
        if gross + _commission(gross, config) <= cash + 1e-9:
            return shares
        lots -= 1
    return 0


def _validate_bars(bars: pd.DataFrame, target_column: str) -> pd.DataFrame:
    required = {"trade_date", "open", "close", target_column}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars is missing required columns: {sorted(missing)}")
    if bars.empty:
        raise ValueError("bars cannot be empty")

    frame = bars.loc[:, ["trade_date", "open", "close", target_column]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise")
    frame["open"] = pd.to_numeric(frame["open"], errors="raise")
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    frame[target_column] = pd.to_numeric(frame[target_column], errors="raise")
    frame = frame.sort_values("trade_date").reset_index(drop=True)

    if frame["trade_date"].duplicated().any():
        raise ValueError("trade_date values must be unique")
    if (frame[["open", "close"]] <= 0).any().any():
        raise ValueError("open and close prices must be positive")
    if not frame[target_column].isin([0, 1]).all():
        raise ValueError(f"{target_column} must contain only 0 or 1")
    return frame


def _calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: tuple[Trade, ...],
    initial_cash: float,
    annualization_days: int,
    gross_turnover: float,
) -> dict[str, float]:
    equity = equity_curve["equity"].astype(float)
    daily_returns = equity.pct_change().fillna(0.0)
    periods = max(len(equity) - 1, 1)
    total_return = float(equity.iloc[-1] / initial_cash - 1.0)
    annualized_return = float(
        (equity.iloc[-1] / initial_cash) ** (annualization_days / periods) - 1.0
    )
    return_std = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0
    annualized_volatility = return_std * math.sqrt(annualization_days)
    sharpe = (
        float(daily_returns.mean() / return_std * math.sqrt(annualization_days))
        if return_std > 0
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1.0
    benchmark_return = float(
        equity_curve["close"].iloc[-1] / equity_curve["close"].iloc[0] - 1.0
    )
    average_equity = float(equity.mean())

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": sharpe,
        "max_drawdown": float(-drawdown.min()),
        "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return,
        "turnover": gross_turnover / average_equity if average_equity > 0 else 0.0,
        "trade_count": float(len(trades)),
        "total_cost": float(sum(trade.commission + trade.tax for trade in trades)),
    }


def run_long_only_backtest(
    bars: pd.DataFrame,
    *,
    target_column: str = "target_position",
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Execute row T's target no earlier than row T+1's open."""

    config = config or BacktestConfig()
    frame = _validate_bars(bars, target_column)
    cash = float(config.initial_cash)
    shares = 0
    trades: list[Trade] = []
    curve_rows: list[dict[str, Any]] = []
    gross_turnover = 0.0
    slippage = config.slippage_bps / 10_000.0

    for index, row in frame.iterrows():
        execution_date = pd.Timestamp(row["trade_date"])
        target = int(frame.loc[index - 1, target_column]) if index > 0 else 0
        signal_date = (
            pd.Timestamp(frame.loc[index - 1, "trade_date"])
            if index > 0
            else execution_date
        )

        if target == 1 and shares == 0:
            fill_price = float(row["open"]) * (1.0 + slippage)
            quantity = _maximum_affordable_shares(cash, fill_price, config)
            if quantity > 0:
                gross = quantity * fill_price
                commission = _commission(gross, config)
                cash -= gross + commission
                shares = quantity
                gross_turnover += gross
                trades.append(
                    Trade(
                        signal_date,
                        execution_date,
                        "BUY",
                        quantity,
                        fill_price,
                        gross,
                        commission,
                        0.0,
                        cash,
                        shares,
                    )
                )
        elif target == 0 and shares > 0:
            fill_price = float(row["open"]) * (1.0 - slippage)
            quantity = shares
            gross = quantity * fill_price
            commission = _commission(gross, config)
            tax = gross * config.sell_tax_rate
            cash += gross - commission - tax
            shares = 0
            gross_turnover += gross
            trades.append(
                Trade(
                    signal_date,
                    execution_date,
                    "SELL",
                    quantity,
                    fill_price,
                    gross,
                    commission,
                    tax,
                    cash,
                    shares,
                )
            )

        close = float(row["close"])
        market_value = shares * close
        equity = cash + market_value
        curve_rows.append(
            {
                "trade_date": execution_date,
                "open": float(row["open"]),
                "close": close,
                "target_position": target,
                "cash": cash,
                "shares": shares,
                "market_value": market_value,
                "equity": equity,
            }
        )

    equity_curve = pd.DataFrame(curve_rows)
    trade_tuple = tuple(trades)
    metrics = _calculate_metrics(
        equity_curve,
        trade_tuple,
        config.initial_cash,
        config.annualization_days,
        gross_turnover,
    )
    assumptions = (
        "Signals are observed after T close and executed no earlier than T+1 open.",
        "The engine is long-only and uses a full-cash/full-position target.",
        "Open-price slippage and configurable transaction costs are applied.",
        f"Buy quantities are rounded down to lots of {config.lot_size} shares.",
        "The final position is marked to the last close and is not forced to sell.",
    )
    return BacktestResult(equity_curve, trade_tuple, metrics, assumptions)
