"""Bounded market providers, cache, circuit breaker, and stream polling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Protocol

import pandas as pd


class MarketProviderError(RuntimeError):
    """A configured market provider could not return approved evidence."""


@dataclass(frozen=True)
class ProviderBatch:
    provider: str
    symbol: str
    rows: list[dict[str, Any]]
    adjustment_status: str
    as_of: str


class DailyProvider(Protocol):
    name: str

    def fetch_daily(
        self,
        symbol: str,
        *,
        start_date: str | None,
        end_date: str | None,
        limit: int,
    ) -> ProviderBatch: ...


class RealtimeProvider(Protocol):
    name: str

    def fetch_quotes(self, symbols: list[str]) -> list[dict[str, Any]]: ...

    def fetch_minutes(self, symbol: str, limit: int = 240) -> list[dict[str, Any]]: ...


class TushareDailyProvider:
    name = "tushare"

    def __init__(self, token: str | None = None) -> None:
        self._token = token or os.environ.get("TUSHARE_TOKEN")
        if not self._token:
            raise MarketProviderError("TUSHARE_TOKEN is not configured")

    def fetch_daily(
        self,
        symbol: str,
        *,
        start_date: str | None,
        end_date: str | None,
        limit: int,
    ) -> ProviderBatch:
        try:
            import tushare as ts
        except ImportError as exc:
            raise MarketProviderError("tushare optional dependency is missing") from exc
        ts.set_token(self._token)
        try:
            frame = ts.pro_bar(
                ts_code=symbol,
                start_date=(start_date or "").replace("-", "") or None,
                end_date=(end_date or "").replace("-", "") or None,
                adj="qfq",
                freq="D",
            )
        except Exception as exc:
            raise MarketProviderError("Tushare daily request failed") from exc
        if frame is None or frame.empty:
            raise MarketProviderError("Tushare returned no daily observations")
        frame = frame.sort_values("trade_date").tail(limit)
        rows = _canonical_daily_rows(frame)
        return ProviderBatch(
            provider=self.name,
            symbol=symbol,
            rows=rows,
            adjustment_status="qfq",
            as_of=rows[-1]["trade_date"],
        )


class AkShareProvider:
    name = "akshare_eastmoney"

    @staticmethod
    def _ak():
        try:
            import akshare as ak
        except ImportError as exc:
            raise MarketProviderError("akshare optional dependency is missing") from exc
        return ak

    def fetch_daily(
        self,
        symbol: str,
        *,
        start_date: str | None,
        end_date: str | None,
        limit: int,
    ) -> ProviderBatch:
        ak = self._ak()
        code = symbol.split(".")[0]
        try:
            frame = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=(start_date or "19700101").replace("-", ""),
                end_date=(end_date or "20991231").replace("-", ""),
                adjust="qfq",
            )
        except Exception as exc:
            raise MarketProviderError("AkShare daily request failed") from exc
        if frame is None or frame.empty:
            raise MarketProviderError("AkShare returned no daily observations")
        frame = _rename_chinese_market_columns(frame).tail(limit)
        rows = _canonical_daily_rows(frame)
        return ProviderBatch(
            provider=self.name,
            symbol=symbol,
            rows=rows,
            adjustment_status="qfq",
            as_of=rows[-1]["trade_date"],
        )

    def fetch_quotes(self, symbols: list[str]) -> list[dict[str, Any]]:
        ak = self._ak()
        quotes: list[dict[str, Any]] = []
        for symbol in symbols[:20]:
            code = symbol.split(".")[0]
            try:
                frame = ak.stock_bid_ask_em(symbol=code)
            except Exception as exc:
                raise MarketProviderError("AkShare realtime quote failed") from exc
            values = {
                str(row["item"]): row["value"]
                for _, row in frame.iterrows()
                if "item" in frame and "value" in frame
            }
            latest = _first_number(
                values,
                ("最新", "最新价", "现价", "price"),
            )
            if latest is None:
                raise MarketProviderError("quote response has no latest price")
            quotes.append(
                {
                    "symbol": symbol,
                    "price": latest,
                    "provider": self.name,
                    "as_of": datetime.now(timezone.utc).isoformat(),
                    "stale": False,
                }
            )
        return quotes

    def fetch_minutes(self, symbol: str, limit: int = 240) -> list[dict[str, Any]]:
        ak = self._ak()
        code = symbol.split(".")[0]
        end = datetime.now().replace(microsecond=0)
        start = end - timedelta(days=5)
        try:
            frame = ak.stock_zh_a_hist_min_em(
                symbol=code,
                start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
                end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
                period="1",
                adjust="qfq",
            )
        except Exception as exc:
            raise MarketProviderError("AkShare minute request failed") from exc
        if frame is None or frame.empty:
            raise MarketProviderError("AkShare returned no minute observations")
        frame = _rename_chinese_market_columns(frame).tail(limit)
        time_column = "trade_time" if "trade_time" in frame else "trade_date"
        rows = []
        for row in frame.itertuples(index=False):
            payload = row._asdict()
            rows.append(
                {
                    "trade_time": str(payload[time_column]),
                    "open": float(payload["open"]),
                    "high": float(payload.get("high", payload["open"])),
                    "low": float(payload.get("low", payload["open"])),
                    "close": float(payload["close"]),
                    "volume": float(payload.get("volume", 0) or 0),
                    "amount": float(payload.get("amount", 0) or 0),
                }
            )
        return rows


class MarketCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS market_cache (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                trade_time TEXT NOT NULL,
                provider TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                adjustment_status TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                as_of TEXT NOT NULL,
                PRIMARY KEY(symbol, interval, trade_time, provider))"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def save(self, batch: ProviderBatch, interval: str = "1d") -> str:
        canonical = json.dumps(batch.rows, sort_keys=True, ensure_ascii=False)
        fingerprint = sha256(canonical.encode("utf-8")).hexdigest()
        records = []
        for row in batch.rows:
            trade_time = str(row.get("trade_time") or row.get("trade_date"))
            records.append(
                (
                    batch.symbol,
                    interval,
                    trade_time,
                    batch.provider,
                    json.dumps(row, ensure_ascii=False, default=str),
                    batch.adjustment_status,
                    fingerprint,
                    batch.as_of,
                )
            )
        with self._lock, self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO market_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                records,
            )
        return fingerprint

    def load(
        self,
        symbol: str,
        *,
        interval: str = "1d",
        limit: int = 250,
    ) -> ProviderBatch | None:
        with self._lock, self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """SELECT * FROM market_cache WHERE symbol=? AND interval=?
                ORDER BY trade_time DESC LIMIT ?""",
                (symbol, interval, limit),
            ).fetchall()
        if not rows:
            return None
        ordered = list(reversed(rows))
        return ProviderBatch(
            provider=str(ordered[-1]["provider"]),
            symbol=symbol,
            rows=[json.loads(row["payload_json"]) for row in ordered],
            adjustment_status=str(ordered[-1]["adjustment_status"]),
            as_of=str(ordered[-1]["as_of"]),
        )


class BoundedMarketGateway:
    def __init__(
        self,
        *,
        cache: MarketCache,
        daily_providers: list[DailyProvider] | None = None,
        realtime_provider: RealtimeProvider | None = None,
    ) -> None:
        self.cache = cache
        self.daily_providers = list(daily_providers or [])
        self.realtime_provider = realtime_provider
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, datetime] = {}

    def daily_bars(
        self,
        symbols: list[str],
        *,
        start_date: str | None,
        end_date: str | None,
        limit: int,
    ) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
        bars: dict[str, list[dict[str, Any]]] = {}
        evidence: list[dict[str, Any]] = []
        for symbol in symbols[:20]:
            batch, stale = self._daily_symbol(
                symbol,
                start_date=start_date,
                end_date=end_date,
                limit=min(limit, 250),
            )
            canonical = json.dumps(batch.rows, sort_keys=True, ensure_ascii=False)
            fingerprint = sha256(canonical.encode("utf-8")).hexdigest()
            bars[symbol] = batch.rows
            evidence.append(
                {
                    "evidence_id": f"{batch.provider}:{symbol}:{fingerprint[:12]}",
                    "symbol": symbol,
                    "provider": batch.provider,
                    "row_count": len(batch.rows),
                    "as_of": batch.as_of,
                    "freshness": "stale_cache" if stale else "provider_latest",
                    "stale": stale,
                    "adjustment_status": batch.adjustment_status,
                    "fingerprint": fingerprint,
                }
            )
        return bars, evidence

    def _daily_symbol(
        self,
        symbol: str,
        *,
        start_date: str | None,
        end_date: str | None,
        limit: int,
    ) -> tuple[ProviderBatch, bool]:
        for provider in self.daily_providers:
            if self._circuit_open(provider.name):
                continue
            try:
                batch = provider.fetch_daily(
                    symbol,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit,
                )
                self.cache.save(batch)
                self._failures[provider.name] = 0
                return batch, False
            except MarketProviderError:
                self._record_failure(provider.name)
        cached = self.cache.load(symbol, limit=limit)
        if cached:
            return cached, True
        raise MarketProviderError("all daily providers failed and no cache exists")

    def quotes(self, symbols: list[str]) -> list[dict[str, Any]]:
        if not self.realtime_provider:
            raise MarketProviderError("realtime provider is not configured")
        name = self.realtime_provider.name
        if self._circuit_open(name):
            raise MarketProviderError("realtime provider circuit is open")
        try:
            quotes = self.realtime_provider.fetch_quotes(symbols[:20])
            self._failures[name] = 0
            return quotes
        except MarketProviderError:
            self._record_failure(name)
            raise

    def minutes(self, symbol: str, limit: int = 240) -> list[dict[str, Any]]:
        if not self.realtime_provider:
            raise MarketProviderError("realtime provider is not configured")
        return self.realtime_provider.fetch_minutes(symbol, min(limit, 1_200))

    def _record_failure(self, name: str) -> None:
        failures = self._failures.get(name, 0) + 1
        self._failures[name] = failures
        if failures >= 3:
            self._open_until[name] = datetime.now(timezone.utc) + timedelta(minutes=5)

    def _circuit_open(self, name: str) -> bool:
        until = self._open_until.get(name)
        if until and until > datetime.now(timezone.utc):
            return True
        if until:
            self._open_until.pop(name, None)
            self._failures[name] = 0
        return False


class MarketStreamHub:
    def __init__(self, gateway: BoundedMarketGateway, poll_seconds: int = 60) -> None:
        self.gateway = gateway
        self.poll_seconds = poll_seconds
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._symbols: list[str] = []
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    def set_symbols(self, symbols: list[str]) -> None:
        self._symbols = symbols[:20]

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2)
            except TimeoutError:
                self._task.cancel()

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    async def poll_once(self) -> list[dict[str, Any]]:
        if not self._symbols:
            return []
        quotes = await asyncio.to_thread(self.gateway.quotes, self._symbols)
        for quote in quotes:
            await self._publish({"type": "market.quote", "data": quote})
        return quotes

    async def replay(self, symbol: str, delay_seconds: float = 0.05) -> None:
        rows = await asyncio.to_thread(self.gateway.minutes, symbol, 240)
        for row in rows:
            await self._publish(
                {
                    "type": "market.replay",
                    "data": {"symbol": symbol, **row, "replay": True},
                }
            )
            await asyncio.sleep(delay_seconds)

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.poll_once()
            except MarketProviderError as exc:
                await self._publish(
                    {"type": "market.error", "data": {"error_type": type(exc).__name__}}
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                continue

    async def _publish(self, message: dict[str, Any]) -> None:
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(message)


def build_market_gateway(runtime_db: str | Path) -> BoundedMarketGateway:
    providers: list[DailyProvider] = []
    if os.environ.get("TUSHARE_TOKEN"):
        try:
            providers.append(TushareDailyProvider())
        except MarketProviderError:
            pass
    try:
        akshare = AkShareProvider()
        if _optional_module("akshare"):
            providers.append(akshare)
            realtime: RealtimeProvider | None = akshare
        else:
            realtime = None
    except MarketProviderError:
        realtime = None
    return BoundedMarketGateway(
        cache=MarketCache(runtime_db),
        daily_providers=providers,
        realtime_provider=realtime,
    )


def _optional_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _rename_chinese_market_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "日期": "trade_date",
            "时间": "trade_time",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "trade_time": "trade_time",
        }
    )


def _canonical_daily_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"trade_date", "open", "close"}
    if not required.issubset(frame.columns):
        raise MarketProviderError("daily response is missing required fields")
    rows = []
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        trade_date = pd.Timestamp(payload["trade_date"]).date().isoformat()
        item = {
            "trade_date": trade_date,
            "open": float(payload["open"]),
            "close": float(payload["close"]),
        }
        for name in ("high", "low", "volume", "amount"):
            value = payload.get(name)
            if value is not None and pd.notna(value):
                item[name] = float(value)
        rows.append(item)
    if not rows:
        raise MarketProviderError("daily response has no valid rows")
    return rows


def _first_number(values: dict[str, Any], candidates: tuple[str, ...]) -> float | None:
    for candidate in candidates:
        value = values.get(candidate)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None

