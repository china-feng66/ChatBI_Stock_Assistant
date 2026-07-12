"""Mandatory deterministic audit gate for strategy and backtest results."""

from __future__ import annotations

import math
from hashlib import sha256

import pandas as pd

from quantquery_a.backtest import BacktestConfig, run_long_only_backtest
from quantquery_a.strategy import generate_macd_targets

from .contracts import QueryFrame, RiskFinding, RiskReview, SymbolAnalysis
from .data_agent import DataBundle


class RiskCritic:
    _KNOWN_REQUIREMENTS = {
        "market_data_fingerprint",
        "signal_execution_timing",
        "cost_assumptions",
        "benchmark",
    }

    def review(
        self,
        frame: QueryFrame,
        data: DataBundle,
        analyses: tuple[SymbolAnalysis, ...],
    ) -> RiskReview:
        findings: list[RiskFinding] = []
        evidence_by_symbol = {item.symbol: item for item in data.evidence}
        analyses_by_symbol = {analysis.symbol: analysis for analysis in analyses}
        minimum_observations = max(frame.macd.slow + frame.macd.signal, 10)
        if len(evidence_by_symbol) != len(data.evidence) or set(
            evidence_by_symbol
        ) != set(frame.symbols):
            findings.append(
                RiskFinding(
                    code="invalid_evidence_set",
                    severity="error",
                    message="evidence symbols do not exactly match the requested universe",
                )
            )
        if len(analyses_by_symbol) != len(analyses) or set(analyses_by_symbol) != set(
            frame.symbols
        ):
            findings.append(
                RiskFinding(
                    code="invalid_analysis_set",
                    severity="error",
                    message="analysis symbols do not exactly match the requested universe",
                )
            )
        if set(data.frames) != set(frame.symbols):
            findings.append(
                RiskFinding(
                    code="invalid_data_set",
                    severity="error",
                    message="data symbols do not exactly match the requested universe",
                )
            )

        unknown_requirements = set(frame.evidence_requirements).difference(
            self._KNOWN_REQUIREMENTS
        )
        if unknown_requirements:
            findings.append(
                RiskFinding(
                    code="unknown_evidence_requirement",
                    severity="error",
                    message=f"unsupported evidence requirements: {sorted(unknown_requirements)}",
                )
            )

        for symbol in frame.symbols:
            evidence = evidence_by_symbol.get(symbol)
            analysis = analyses_by_symbol.get(symbol)
            prices = data.frames.get(symbol)
            if evidence is None or prices is None:
                findings.append(
                    RiskFinding(
                        code="missing_evidence",
                        severity="error",
                        message=f"{symbol} has no approved market evidence",
                    )
                )
                continue
            if analysis is None:
                findings.append(
                    RiskFinding(
                        code="missing_analysis",
                        severity="error",
                        message=f"{symbol} has no analysis result",
                        retryable=True,
                    )
                )
                continue

            if analysis.evidence_id != evidence.evidence_id:
                findings.append(
                    RiskFinding(
                        code="evidence_binding_mismatch",
                        severity="error",
                        message=f"{symbol} result is linked to the wrong evidence",
                    )
                )
            if (
                analysis.observation_count != evidence.row_count
                or len(prices) != evidence.row_count
            ):
                findings.append(
                    RiskFinding(
                        code="observation_count_mismatch",
                        severity="error",
                        message=f"{symbol} observation count does not match its evidence",
                    )
                )
            canonical = "\n".join(
                f"{row.trade_date.date().isoformat()}|{float(row.open):.12g}|{float(row.close):.12g}"
                for row in prices.itertuples(index=False)
            )
            expected_fingerprint = sha256(canonical.encode("utf-8")).hexdigest()
            if evidence.fingerprint != expected_fingerprint:
                findings.append(
                    RiskFinding(
                        code="invalid_evidence_fingerprint",
                        severity="error",
                        message=f"{symbol} evidence fingerprint is invalid",
                    )
                )
            if analysis.observation_count < minimum_observations:
                findings.append(
                    RiskFinding(
                        code="insufficient_observations",
                        severity="error",
                        message=(
                            f"{symbol} requires at least {minimum_observations} observations "
                            "for the configured MACD audit"
                        ),
                    )
                )

            missing_metrics = sorted(set(frame.metrics).difference(analysis.metrics))
            if missing_metrics:
                findings.append(
                    RiskFinding(
                        code="missing_metrics",
                        severity="error",
                        message=f"{symbol} is missing required metrics: {missing_metrics}",
                    )
                )
            if any(not math.isfinite(value) for value in analysis.metrics.values()):
                findings.append(
                    RiskFinding(
                        code="non_finite_metric",
                        severity="error",
                        message=f"{symbol} contains a non-finite metric",
                    )
                )
            findings.extend(
                self._deterministic_replay_findings(frame, prices, analysis)
            )
            if "benchmark_return" not in analysis.metrics:
                findings.append(
                    RiskFinding(
                        code="missing_benchmark",
                        severity="error",
                        message=f"{symbol} does not include a benchmark result",
                    )
                )
            if "total_cost" not in analysis.metrics:
                findings.append(
                    RiskFinding(
                        code="missing_cost_metric",
                        severity="error",
                        message=f"{symbol} does not include transaction costs",
                    )
                )
            if "benchmark_return" in analysis.metrics:
                expected_benchmark = (
                    float(prices["close"].iloc[-1]) / float(prices["close"].iloc[0])
                    - 1.0
                )
                if not math.isclose(
                    analysis.metrics["benchmark_return"],
                    expected_benchmark,
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ):
                    findings.append(
                        RiskFinding(
                            code="invalid_benchmark",
                            severity="error",
                            message=f"{symbol} benchmark return is inconsistent",
                        )
                    )
            if "total_cost" in analysis.metrics:
                expected_total_cost = sum(
                    trade.commission + trade.tax for trade in analysis.trades
                )
                if not math.isclose(
                    analysis.metrics["total_cost"],
                    expected_total_cost,
                    rel_tol=1e-10,
                    abs_tol=1e-6,
                ):
                    findings.append(
                        RiskFinding(
                            code="invalid_total_cost",
                            severity="error",
                            message=f"{symbol} total cost is inconsistent",
                        )
                    )
            if not math.isclose(
                evidence.latest_close,
                float(prices["close"].iloc[-1]),
                rel_tol=1e-10,
                abs_tol=1e-10,
            ):
                findings.append(
                    RiskFinding(
                        code="invalid_latest_close",
                        severity="error",
                        message=f"{symbol} latest close is inconsistent",
                    )
                )

            dates = [timestamp.date() for timestamp in prices["trade_date"]]
            next_date = {
                dates[index]: dates[index + 1] for index in range(len(dates) - 1)
            }
            open_by_date = {
                row.trade_date.date(): float(row.open)
                for row in prices.itertuples(index=False)
            }
            slippage = frame.costs.slippage_bps / 10_000.0
            for trade in analysis.trades:
                expected_date = next_date.get(trade.signal_date)
                if expected_date != trade.execution_date:
                    findings.append(
                        RiskFinding(
                            code="invalid_execution_timing",
                            severity="error",
                            message=f"{symbol} trade is not executed at the next bar",
                        )
                    )
                execution_open = open_by_date.get(trade.execution_date)
                if execution_open is None:
                    findings.append(
                        RiskFinding(
                            code="missing_execution_open",
                            severity="error",
                            message=f"{symbol} trade has no matching execution open",
                        )
                    )
                    continue
                expected_fill = execution_open * (
                    1.0 + slippage if trade.side == "BUY" else 1.0 - slippage
                )
                if not math.isclose(
                    trade.fill_price, expected_fill, rel_tol=1e-10, abs_tol=1e-8
                ):
                    findings.append(
                        RiskFinding(
                            code="invalid_fill_price",
                            severity="error",
                            message=f"{symbol} fill price does not match open plus slippage",
                        )
                    )
                if trade.shares % frame.costs.lot_size != 0:
                    findings.append(
                        RiskFinding(
                            code="invalid_lot_size",
                            severity="error",
                            message=f"{symbol} contains a non-lot-sized trade",
                        )
                    )
                expected_gross = trade.shares * trade.fill_price
                if not math.isclose(
                    trade.gross_value, expected_gross, rel_tol=1e-10, abs_tol=1e-6
                ):
                    findings.append(
                        RiskFinding(
                            code="invalid_gross_value",
                            severity="error",
                            message=f"{symbol} trade gross value is inconsistent",
                        )
                    )
                expected_commission = max(
                    trade.gross_value * frame.costs.commission_rate,
                    frame.costs.min_commission,
                )
                expected_tax = (
                    trade.gross_value * frame.costs.sell_tax_rate
                    if trade.side == "SELL"
                    else 0.0
                )
                if not math.isclose(
                    trade.commission,
                    expected_commission,
                    rel_tol=1e-10,
                    abs_tol=1e-6,
                ):
                    findings.append(
                        RiskFinding(
                            code="invalid_commission",
                            severity="error",
                            message=f"{symbol} commission is inconsistent with assumptions",
                        )
                    )
                if not math.isclose(
                    trade.tax, expected_tax, rel_tol=1e-10, abs_tol=1e-6
                ):
                    findings.append(
                        RiskFinding(
                            code="invalid_tax",
                            severity="error",
                            message=f"{symbol} tax is inconsistent with assumptions",
                        )
                    )

            if not analysis.trades:
                findings.append(
                    RiskFinding(
                        code="no_trades",
                        severity="warning",
                        message=f"{symbol} produced no trades for this window",
                    )
                )

        approved = not any(finding.severity == "error" for finding in findings)
        return RiskReview(
            approved=approved,
            findings=findings,
            checked_symbols=list(frame.symbols),
        )

    @staticmethod
    def _deterministic_replay_findings(
        frame: QueryFrame,
        prices: pd.DataFrame,
        analysis: SymbolAnalysis,
    ) -> list[RiskFinding]:
        """Re-run the approved deterministic engine and compare its full result."""

        try:
            signals = generate_macd_targets(
                prices.loc[:, ["trade_date", "close"]],
                fast=frame.macd.fast,
                slow=frame.macd.slow,
                signal=frame.macd.signal,
            )
            replay_bars = prices.merge(
                signals.loc[:, ["trade_date", "target_position"]],
                on="trade_date",
                how="inner",
                validate="one_to_one",
            )
            replay = run_long_only_backtest(
                replay_bars,
                config=BacktestConfig(**frame.costs.model_dump()),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return [
                RiskFinding(
                    code="deterministic_replay_failed",
                    severity="error",
                    message=(
                        f"{analysis.symbol} deterministic replay failed: "
                        f"{type(exc).__name__}"
                    ),
                )
            ]

        findings: list[RiskFinding] = []
        for metric, expected_value in replay.metrics.items():
            actual_value = analysis.metrics.get(metric)
            if actual_value is None or not math.isclose(
                actual_value,
                expected_value,
                rel_tol=1e-10,
                abs_tol=1e-10,
            ):
                findings.append(
                    RiskFinding(
                        code="deterministic_metric_mismatch",
                        severity="error",
                        message=(
                            f"{analysis.symbol} {metric} differs from "
                            "deterministic replay"
                        ),
                    )
                )

        if len(analysis.trades) != len(replay.trades):
            findings.append(
                RiskFinding(
                    code="deterministic_trade_mismatch",
                    severity="error",
                    message=(
                        f"{analysis.symbol} trade count differs from "
                        "deterministic replay"
                    ),
                )
            )
            return findings

        for actual_trade, expected_trade in zip(
            analysis.trades, replay.trades, strict=True
        ):
            structural_match = (
                actual_trade.signal_date == expected_trade.signal_date.date()
                and actual_trade.execution_date == expected_trade.execution_date.date()
                and actual_trade.side == expected_trade.side
                and actual_trade.shares == expected_trade.shares
            )
            numeric_match = all(
                math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8)
                for actual, expected in (
                    (actual_trade.fill_price, expected_trade.fill_price),
                    (actual_trade.gross_value, expected_trade.gross_value),
                    (actual_trade.commission, expected_trade.commission),
                    (actual_trade.tax, expected_trade.tax),
                )
            )
            if not structural_match or not numeric_match:
                findings.append(
                    RiskFinding(
                        code="deterministic_trade_mismatch",
                        severity="error",
                        message=(
                            f"{analysis.symbol} trade differs from deterministic replay"
                        ),
                    )
                )
                break
        return findings
