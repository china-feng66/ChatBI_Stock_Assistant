"""Allowlisted deterministic research tools used by the LLM agents."""

from __future__ import annotations

from datetime import date
import math
from typing import Any

import pandas as pd

from quantquery_a.agents.contracts import AnalyzeRequest, TaskType
from quantquery_a.agents.supervisor import QuantSupervisor
from quantquery_a.backtest import BacktestConfig, run_long_only_backtest

from .contracts import ResearchMode, RunCreateRequest


ALLOWLISTED_TOOLS = {
    "market",
    "macd",
    "bollinger",
    "factor",
    "compare",
}


def generate_bollinger_targets(
    prices: pd.DataFrame,
    *,
    window: int = 20,
    deviations: float = 2.0,
) -> pd.DataFrame:
    if window < 2 or deviations <= 0:
        raise ValueError("invalid Bollinger parameters")
    frame = prices.loc[:, ["trade_date", "close"]].copy()
    frame["middle"] = frame["close"].rolling(window).mean()
    standard = frame["close"].rolling(window).std(ddof=0)
    frame["upper"] = frame["middle"] + deviations * standard
    frame["lower"] = frame["middle"] - deviations * standard
    holding = 0
    targets: list[int] = []
    for row in frame.itertuples(index=False):
        if pd.notna(row.lower) and row.close < row.lower:
            holding = 1
        elif pd.notna(row.middle) and row.close > row.middle:
            holding = 0
        targets.append(holding)
    frame["target_position"] = targets
    return frame


def _normalized_frames(
    bars: dict[str, list[dict[str, Any]]],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol, rows in bars.items():
        frame = pd.DataFrame(rows)
        required = {"trade_date", "open", "close"}
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError(f"{symbol} is missing required market fields")
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise")
        for column in ("open", "close", "high", "low", "volume", "amount"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.sort_values("trade_date").drop_duplicates("trade_date")
        if len(frame) < 2 or (frame[["open", "close"]] <= 0).any().any():
            raise ValueError(f"{symbol} has invalid observations")
        frames[symbol] = frame.reset_index(drop=True)
    return frames


def _trades_payload(result) -> list[dict[str, Any]]:
    return [
        {
            "signal_date": trade.signal_date.date().isoformat(),
            "execution_date": trade.execution_date.date().isoformat(),
            "side": trade.side,
            "shares": trade.shares,
            "fill_price": trade.fill_price,
            "gross_value": trade.gross_value,
            "commission": trade.commission,
            "tax": trade.tax,
        }
        for trade in result.trades
    ]


def run_research_tool(
    request: RunCreateRequest,
    *,
    mode: ResearchMode,
    bars: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if mode.value not in ALLOWLISTED_TOOLS and mode is not ResearchMode.AUTO:
        raise ValueError("tool is not allowlisted")
    if mode in {ResearchMode.AUTO, ResearchMode.MACD}:
        return _run_macd(request, bars)
    if mode is ResearchMode.BOLLINGER:
        return _run_bollinger(request, bars)
    if mode is ResearchMode.FACTOR:
        return _run_factor_portfolio(request, bars)
    if mode is ResearchMode.COMPARE:
        return _run_comparison(bars)
    return _run_market(bars)


def _run_market(bars: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    frames = _normalized_frames(bars)
    analyses = []
    for symbol, frame in frames.items():
        last = frame.iloc[-1]
        analyses.append(
            {
                "symbol": symbol,
                "latest_close": float(last["close"]),
                "latest_date": last["trade_date"].date().isoformat(),
                "observation_count": len(frame),
            }
        )
    return {"tool": "market", "analyses": analyses}


def _run_comparison(bars: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    frames = _normalized_frames(bars)
    analyses = []
    for symbol, frame in frames.items():
        total_return = float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1)
        volatility = float(frame["close"].pct_change().std(ddof=0) * math.sqrt(252))
        analyses.append(
            {
                "symbol": symbol,
                "total_return": total_return,
                "annualized_volatility": volatility,
                "observation_count": len(frame),
            }
        )
    analyses.sort(key=lambda item: item["total_return"], reverse=True)
    return {"tool": "compare", "analyses": analyses}


def _run_macd(
    request: RunCreateRequest,
    bars: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    analyze = AnalyzeRequest(
        query=request.query,
        symbols=request.symbols,
        bars={
            symbol: [
                {key: row[key] for key in ("trade_date", "open", "close")}
                for row in rows
            ]
            for symbol, rows in bars.items()
        },
        task_hint=TaskType.BACKTEST,
        start_date=request.start_date,
        end_date=request.end_date,
        macd=request.macd,
        costs=request.costs,
    )
    report = QuantSupervisor().analyze(analyze)
    return {
        "tool": "macd",
        "status": report.status.value,
        "analyses": [item.model_dump(mode="json") for item in report.analyses],
        "deterministic_risk": (
            report.risk_review.model_dump(mode="json")
            if report.risk_review
            else None
        ),
        "answer": report.answer,
    }


def _run_bollinger(
    request: RunCreateRequest,
    bars: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    frames = _normalized_frames(bars)
    analyses = []
    config = BacktestConfig(**request.costs.model_dump())
    for symbol, frame in frames.items():
        targets = generate_bollinger_targets(frame)
        input_frame = frame.merge(
            targets[["trade_date", "target_position"]],
            on="trade_date",
            validate="one_to_one",
        )
        result = run_long_only_backtest(input_frame, config=config)
        analyses.append(
            {
                "symbol": symbol,
                "observation_count": len(frame),
                "metrics": dict(result.metrics),
                "trades": _trades_payload(result),
                "assumptions": list(result.assumptions)
                + ["Bollinger(20,2) mean-reversion target"],
            }
        )
    return {"tool": "bollinger", "analyses": analyses}


def _zscore(series: pd.Series) -> pd.Series:
    deviation = float(series.std(ddof=0))
    if deviation == 0 or not math.isfinite(deviation):
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / deviation


def _run_factor_portfolio(
    request: RunCreateRequest,
    bars: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    frames = _normalized_frames(bars)
    if len(frames) < 3:
        raise ValueError("factor research requires at least three symbols")
    common_dates = sorted(
        set.intersection(
            *(set(frame["trade_date"].dt.date) for frame in frames.values())
        )
    )
    if len(common_dates) < 30:
        raise ValueError("factor research requires at least 30 common observations")
    by_symbol = {
        symbol: frame.set_index(frame["trade_date"].dt.date)
        for symbol, frame in frames.items()
    }
    week_last: list[date] = []
    for _, dates in pd.Series(common_dates).groupby(
        [pd.Timestamp(value).isocalendar().year * 100 + pd.Timestamp(value).isocalendar().week for value in common_dates]
    ):
        week_last.append(max(dates.tolist()))
    signal_dates = [value for value in week_last if common_dates.index(value) >= 20]
    scores_by_date: dict[date, dict[str, float]] = {}
    selected_by_date: dict[date, list[str]] = {}
    factor_rows: list[dict[str, Any]] = []
    for signal_date in signal_dates:
        position = common_dates.index(signal_date)
        window_dates = common_dates[position - 20 : position + 1]
        raw: dict[str, dict[str, float]] = {}
        for symbol, indexed in by_symbol.items():
            window = indexed.loc[window_dates]
            close = window["close"].astype(float)
            returns = close.pct_change().dropna()
            amount = (
                window["amount"].astype(float)
                if "amount" in window
                else close * (
                    window["volume"].astype(float)
                    if "volume" in window
                    else 1_000_000.0
                )
            )
            raw[symbol] = {
                "momentum": float(close.iloc[-1] / close.iloc[0] - 1),
                "low_volatility": -float(returns.std(ddof=0) * math.sqrt(252)),
                "liquidity": float(math.log(max(float(amount.mean()), 1.0))),
            }
        raw_frame = pd.DataFrame.from_dict(raw, orient="index")
        for column in raw_frame.columns:
            lower, upper = raw_frame[column].quantile([0.05, 0.95])
            raw_frame[column] = raw_frame[column].clip(lower, upper)
            raw_frame[column] = _zscore(raw_frame[column])
        composite = raw_frame.mean(axis=1)
        top = composite.sort_values(ascending=False).head(min(5, len(composite)))
        scores_by_date[signal_date] = composite.to_dict()
        selected_by_date[signal_date] = list(top.index)
        factor_rows.extend(
            {
                "signal_date": signal_date.isoformat(),
                "symbol": symbol,
                "score": float(score),
                "selected": symbol in top.index,
            }
            for symbol, score in composite.items()
        )

    initial_cash = request.costs.initial_cash
    cash = initial_cash
    holdings = {symbol: 0 for symbol in frames}
    equity_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    pending = {
        common_dates[common_dates.index(signal_date) + 1]: signal_date
        for signal_date in signal_dates
        if common_dates.index(signal_date) + 1 < len(common_dates)
    }
    for current_date in common_dates:
        if current_date in pending:
            signal_date = pending[current_date]
            targets = selected_by_date[signal_date]
            open_prices = {
                symbol: float(by_symbol[symbol].loc[current_date, "open"])
                for symbol in frames
            }
            portfolio_open = cash + sum(
                holdings[symbol] * open_prices[symbol] for symbol in frames
            )
            target_value = portfolio_open / max(len(targets), 1)
            for symbol in frames:
                target_shares = (
                    int(target_value / open_prices[symbol] / request.costs.lot_size)
                    * request.costs.lot_size
                    if symbol in targets
                    else 0
                )
                delta = target_shares - holdings[symbol]
                if delta == 0:
                    continue
                side = "BUY" if delta > 0 else "SELL"
                shares = abs(delta)
                slippage = request.costs.slippage_bps / 10_000.0
                fill = open_prices[symbol] * (
                    1 + slippage if side == "BUY" else 1 - slippage
                )
                gross = shares * fill
                commission = max(
                    gross * request.costs.commission_rate,
                    request.costs.min_commission,
                )
                tax = gross * request.costs.sell_tax_rate if side == "SELL" else 0.0
                if side == "BUY" and gross + commission > cash:
                    affordable = int(
                        max(cash - request.costs.min_commission, 0)
                        / fill
                        / request.costs.lot_size
                    ) * request.costs.lot_size
                    shares = min(shares, affordable)
                    if shares <= 0:
                        continue
                    gross = shares * fill
                    commission = max(
                        gross * request.costs.commission_rate,
                        request.costs.min_commission,
                    )
                cash += gross - commission - tax if side == "SELL" else -gross - commission
                holdings[symbol] += shares if side == "BUY" else -shares
                trades.append(
                    {
                        "signal_date": signal_date.isoformat(),
                        "execution_date": current_date.isoformat(),
                        "symbol": symbol,
                        "side": side,
                        "shares": shares,
                        "fill_price": fill,
                        "commission": commission,
                        "tax": tax,
                    }
                )
        equity = cash + sum(
            holdings[symbol] * float(by_symbol[symbol].loc[current_date, "close"])
            for symbol in frames
        )
        equity_rows.append({"trade_date": current_date.isoformat(), "equity": equity})
    equity = pd.Series([row["equity"] for row in equity_rows], dtype=float)
    returns = equity.pct_change().fillna(0.0)
    drawdown = equity / equity.cummax() - 1.0
    split = max(int(len(equity) * 0.7), 1)
    holdout = equity.iloc[split:]
    metrics = {
        "total_return": float(equity.iloc[-1] / initial_cash - 1),
        "annualized_volatility": float(returns.std(ddof=0) * math.sqrt(252)),
        "sharpe_ratio": float(
            returns.mean() / returns.std(ddof=0) * math.sqrt(252)
            if returns.std(ddof=0) > 0
            else 0.0
        ),
        "max_drawdown": float(drawdown.min()),
        "holdout_return": float(
            holdout.iloc[-1] / holdout.iloc[0] - 1 if len(holdout) > 1 else 0.0
        ),
        "total_cost": float(
            sum(trade["commission"] + trade["tax"] for trade in trades)
        ),
    }
    return {
        "tool": "factor",
        "analyses": [
            {
                "symbol": "PRICE_VOLUME_TOP5",
                "observation_count": len(common_dates),
                "metrics": metrics,
                "trades": trades,
                "equity_curve": equity_rows,
                "factor_scores": factor_rows,
                "assumptions": [
                    "20-day momentum, low-volatility and liquidity",
                    "weekly close signal and next-open execution",
                    "top five equal target value, long-only",
                    "70/30 chronological performance split",
                ],
            }
        ],
    }


def deterministic_risk_audit(
    *,
    mode: ResearchMode,
    result: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    if not evidence:
        findings.append(
            {"code": "missing_evidence", "severity": "error", "retryable": True}
        )
    if mode not in {ResearchMode.MARKET, ResearchMode.COMPARE}:
        unadjusted = [
            item.get("symbol")
            for item in evidence
            if item.get("adjustment_status") == "unadjusted"
        ]
        if unadjusted:
            findings.append(
                {
                    "code": "unadjusted_research_data",
                    "severity": "error",
                    "retryable": True,
                    "symbols": unadjusted,
                }
            )
    analyses = result.get("analyses", [])
    if not analyses:
        findings.append(
            {"code": "missing_analysis", "severity": "error", "retryable": True}
        )
    for analysis in analyses:
        metrics = analysis.get("metrics", {})
        if any(not math.isfinite(float(value)) for value in metrics.values()):
            findings.append(
                {"code": "non_finite_metric", "severity": "error", "retryable": False}
            )
        for trade in analysis.get("trades", []):
            if trade["execution_date"] <= trade["signal_date"]:
                findings.append(
                    {
                        "code": "invalid_execution_timing",
                        "severity": "error",
                        "retryable": False,
                    }
                )
                break
    embedded = result.get("deterministic_risk")
    if embedded and not embedded.get("approved", False):
        findings.extend(embedded.get("findings", []))
    approved = not any(item.get("severity") == "error" for item in findings)
    return {
        "approved": approved,
        "findings": findings,
        "checked_tool": result.get("tool"),
        "retryable": any(item.get("retryable") for item in findings),
    }

