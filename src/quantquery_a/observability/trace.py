"""Privacy-conscious JSONL tracing for agent and tool spans."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import re
from threading import Lock
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


_SENSITIVE_KEY = re.compile(r"(?:api[_-]?key|token|secret|password|credential)", re.I)


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    timestamp: datetime
    span: str
    event: str
    status: str
    attempt: int = Field(default=1, ge=1)
    attributes: dict[str, Any]


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, datetime):
        return value.isoformat()
    return value


class TraceRecorder:
    def __init__(
        self,
        jsonl_path: str | Path | None = None,
        *,
        max_events: int = 10_000,
    ) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._events: deque[TraceEvent] = deque(maxlen=max_events)
        self._lock = Lock()

    def new_trace_id(self) -> str:
        return uuid4().hex

    def emit(
        self,
        trace_id: str,
        *,
        span: str,
        event: str,
        status: str,
        attempt: int = 1,
        attributes: dict[str, Any] | None = None,
    ) -> TraceEvent:
        trace_event = TraceEvent(
            trace_id=trace_id,
            timestamp=datetime.now(timezone.utc),
            span=span,
            event=event,
            status=status,
            attempt=attempt,
            attributes=_redact(attributes or {}),
        )
        with self._lock:
            self._events.append(trace_event)
            if self.jsonl_path:
                self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self.jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(trace_event.model_dump_json() + "\n")
        return trace_event

    def events_for(self, trace_id: str) -> tuple[TraceEvent, ...]:
        with self._lock:
            return tuple(event for event in self._events if event.trace_id == trace_id)
