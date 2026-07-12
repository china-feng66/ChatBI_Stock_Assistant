"""Read-only market-data agent with a narrow, deterministic interface."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence

import pandas as pd

from quantquery_a.query import (
    QueryExecutionError,
    QueryPolicyError,
    ReadOnlyQueryService,
)

from .contracts import EvidenceRecord, MarketBar, QueryFrame


class DataAgentError(RuntimeError):
    """The requested evidence could not be produced safely."""


@dataclass(frozen=True)
class DataBundle:
    frames: dict[str, pd.DataFrame]
    evidence: tuple[EvidenceRecord, ...]


class DataAgent:
    def __init__(
        self,
        query_service: ReadOnlyQueryService | None = None,
        *,
        max_parallelism: int = 4,
    ) -> None:
        if max_parallelism <= 0:
            raise ValueError("max_parallelism must be positive")
        self.query_service = query_service
        self.max_parallelism = max_parallelism

    def run(
        self,
        frame: QueryFrame,
        inline_bars: Mapping[str, Sequence[MarketBar]],
    ) -> DataBundle:
        if not frame.symbols:
            raise DataAgentError("symbols are required")

        workers = min(self.max_parallelism, len(frame.symbols))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                symbol: executor.submit(
                    self._load_symbol,
                    symbol,
                    frame,
                    inline_bars.get(symbol, ()),
                )
                for symbol in frame.symbols
            }
            loaded = {symbol: future.result() for symbol, future in futures.items()}

        frames = {symbol: loaded[symbol][0] for symbol in frame.symbols}
        evidence = tuple(loaded[symbol][1] for symbol in frame.symbols)
        return DataBundle(frames=frames, evidence=evidence)

    def _load_symbol(
        self,
        symbol: str,
        frame: QueryFrame,
        bars: Sequence[MarketBar],
    ) -> tuple[pd.DataFrame, EvidenceRecord]:
        if bars and self.query_service is None:
            price_frame = pd.DataFrame([bar.model_dump() for bar in bars])
            source_type = "inline_bars"
        elif self.query_service is not None:
            sql = "SELECT trade_date, open, close FROM stock_history WHERE ts_code = ?"
            params: list[object] = [symbol]
            if frame.start_date:
                sql += " AND trade_date >= ?"
                params.append(frame.start_date.isoformat())
            if frame.end_date:
                sql += " AND trade_date <= ?"
                params.append(frame.end_date.isoformat())
            sql += " ORDER BY trade_date"
            try:
                result = self.query_service.execute(sql, params)
            except (FileNotFoundError, QueryExecutionError, QueryPolicyError) as exc:
                raise DataAgentError(
                    "the approved market data source could not be queried"
                ) from exc
            if result.truncated:
                raise DataAgentError("market query exceeded the approved row limit")
            price_frame = pd.DataFrame(result.rows, columns=result.columns)
            source_type = "sqlite"
        else:
            raise DataAgentError("no approved data source is configured")

        if price_frame.empty:
            raise DataAgentError("no market observations were returned")

        required = {"trade_date", "open", "close"}
        if not required.issubset(price_frame.columns):
            raise DataAgentError("market data is missing required fields")

        price_frame = price_frame.loc[:, ["trade_date", "open", "close"]].copy()
        price_frame["trade_date"] = pd.to_datetime(
            price_frame["trade_date"], errors="raise"
        )
        price_frame["open"] = pd.to_numeric(price_frame["open"], errors="raise")
        price_frame["close"] = pd.to_numeric(price_frame["close"], errors="raise")
        if frame.start_date:
            price_frame = price_frame[
                price_frame["trade_date"].dt.date >= frame.start_date
            ]
        if frame.end_date:
            price_frame = price_frame[
                price_frame["trade_date"].dt.date <= frame.end_date
            ]
        price_frame = price_frame.sort_values("trade_date").reset_index(drop=True)
        if len(price_frame) < 2:
            raise DataAgentError("at least two observations are required")
        if price_frame["trade_date"].duplicated().any():
            raise DataAgentError("trade dates must be unique")
        if (price_frame[["open", "close"]] <= 0).any().any():
            raise DataAgentError("prices must be positive")

        canonical = "\n".join(
            f"{row.trade_date.date().isoformat()}|{float(row.open):.12g}|{float(row.close):.12g}"
            for row in price_frame.itertuples(index=False)
        )
        fingerprint = sha256(canonical.encode("utf-8")).hexdigest()
        evidence = EvidenceRecord(
            evidence_id=f"market:{symbol}:{fingerprint[:12]}",
            source_type=source_type,
            symbol=symbol,
            row_count=len(price_frame),
            start_date=price_frame["trade_date"].iloc[0].date(),
            end_date=price_frame["trade_date"].iloc[-1].date(),
            fingerprint=fingerprint,
            latest_close=float(price_frame["close"].iloc[-1]),
        )
        return price_frame, evidence
