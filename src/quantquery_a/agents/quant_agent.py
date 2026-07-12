"""Deterministic strategy and backtest agent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from quantquery_a.backtest import BacktestConfig, run_long_only_backtest
from quantquery_a.strategy import generate_macd_targets

from .contracts import QueryFrame, SymbolAnalysis, TradeView
from .data_agent import DataBundle


class QuantAgent:
    def __init__(self, *, max_parallelism: int = 4) -> None:
        if max_parallelism <= 0:
            raise ValueError("max_parallelism must be positive")
        self.max_parallelism = max_parallelism

    def run(self, frame: QueryFrame, data: DataBundle) -> tuple[SymbolAnalysis, ...]:
        workers = min(self.max_parallelism, len(frame.symbols))
        evidence_by_symbol = {item.symbol: item for item in data.evidence}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                symbol: executor.submit(
                    self._analyze_symbol,
                    symbol,
                    frame,
                    data.frames[symbol],
                    evidence_by_symbol[symbol].evidence_id,
                )
                for symbol in frame.symbols
            }
            results = {symbol: future.result() for symbol, future in futures.items()}
        return tuple(results[symbol] for symbol in frame.symbols)

    @staticmethod
    def _analyze_symbol(
        symbol: str,
        frame: QueryFrame,
        prices: pd.DataFrame,
        evidence_id: str,
    ) -> SymbolAnalysis:
        signals = generate_macd_targets(
            prices.loc[:, ["trade_date", "close"]],
            fast=frame.macd.fast,
            slow=frame.macd.slow,
            signal=frame.macd.signal,
        )
        backtest_bars = prices.merge(
            signals.loc[:, ["trade_date", "target_position"]],
            on="trade_date",
            how="inner",
            validate="one_to_one",
        )
        result = run_long_only_backtest(
            backtest_bars,
            config=BacktestConfig(**frame.costs.model_dump()),
        )
        trades = [
            TradeView(
                signal_date=trade.signal_date.date(),
                execution_date=trade.execution_date.date(),
                side=trade.side,
                shares=trade.shares,
                fill_price=trade.fill_price,
                gross_value=trade.gross_value,
                commission=trade.commission,
                tax=trade.tax,
            )
            for trade in result.trades
        ]
        return SymbolAnalysis(
            symbol=symbol,
            evidence_id=evidence_id,
            observation_count=len(prices),
            metrics=dict(result.metrics),
            trades=trades,
            assumptions=list(result.assumptions),
        )
