"""Bounded, read-only SQLite access for analytical SQL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Sequence


class QueryPolicyError(ValueError):
    """The SQL violates the service's read-only policy."""


class QueryExecutionError(RuntimeError):
    """SQLite could not execute a permitted query."""


class QueryTimeoutError(QueryExecutionError):
    """The query exceeded the configured execution budget."""


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    truncated: bool
    elapsed_ms: float

    @property
    def row_count(self) -> int:
        return len(self.rows)


_COMMENT_RE = re.compile(r"/\*.*?\*/|--[^\r\n]*", re.DOTALL)
_QUOTED_VALUE_RE = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"")
_FORBIDDEN_RE = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|REPLACE|CREATE|ALTER|DROP|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|ANALYZE|TRIGGER)\b",
    re.IGNORECASE,
)


def _split_statements(sql: str) -> list[str]:
    statements: list[str] = []
    buffer: list[str] = []
    quote: str | None = None
    index = 0

    while index < len(sql):
        char = sql[index]
        if quote:
            buffer.append(char)
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    buffer.append(sql[index + 1])
                    index += 1
                else:
                    quote = None
        elif char in ("'", '"'):
            quote = char
            buffer.append(char)
        elif char == ";":
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
        else:
            buffer.append(char)
        index += 1

    if quote:
        raise QueryPolicyError("SQL contains an unterminated quoted value")

    statement = "".join(buffer).strip()
    if statement:
        statements.append(statement)
    return statements


def validate_read_only_sql(sql: str) -> str:
    if not isinstance(sql, str) or not sql.strip():
        raise QueryPolicyError("SQL must be a non-empty string")

    without_comments = _COMMENT_RE.sub(" ", sql).strip()
    statements = _split_statements(without_comments)
    if len(statements) != 1:
        raise QueryPolicyError("exactly one SQL statement is allowed")

    statement = statements[0]
    if not re.match(r"^(?:SELECT|WITH)\b", statement, re.IGNORECASE):
        raise QueryPolicyError("only SELECT queries and SELECT-based CTEs are allowed")

    keyword_scan = _QUOTED_VALUE_RE.sub("''", statement)
    forbidden = _FORBIDDEN_RE.search(keyword_scan)
    if forbidden:
        raise QueryPolicyError(f"forbidden SQL keyword: {forbidden.group(0).upper()}")
    return statement


class ReadOnlyQueryService:
    """Execute a single bounded SELECT against an existing SQLite database."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        max_rows: int = 500,
        timeout_seconds: float = 2.0,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        if not 1 <= max_rows <= 10_000:
            raise ValueError("max_rows must be between 1 and 10000")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

    def execute(self, sql: str, params: Sequence[Any] = ()) -> QueryResult:
        if not self.db_path.is_file():
            raise FileNotFoundError(f"SQLite database does not exist: {self.db_path}")

        statement = validate_read_only_sql(sql)
        uri = f"{self.db_path.as_uri()}?mode=ro"
        started = time.perf_counter()
        deadline = started + self.timeout_seconds
        connection: sqlite3.Connection | None = None
        cursor: sqlite3.Cursor | None = None

        try:
            connection = sqlite3.connect(uri, uri=True, timeout=1.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            connection.enable_load_extension(False)
            connection.set_progress_handler(
                lambda: 1 if time.perf_counter() > deadline else 0,
                1_000,
            )
            cursor = connection.execute(statement, tuple(params))
            if cursor.description is None:
                raise QueryPolicyError("query did not produce a result set")

            fetched = cursor.fetchmany(self.max_rows + 1)
            truncated = len(fetched) > self.max_rows
            selected = fetched[: self.max_rows]
            columns = tuple(description[0] for description in cursor.description)
            rows = tuple(dict(row) for row in selected)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise QueryTimeoutError("SQLite query exceeded the time budget") from exc
            raise QueryExecutionError(f"SQLite query failed: {exc}") from exc
        except sqlite3.DatabaseError as exc:
            raise QueryExecutionError(f"SQLite query failed: {exc}") from exc
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()

        elapsed_ms = (time.perf_counter() - started) * 1_000
        return QueryResult(
            columns=columns,
            rows=rows,
            truncated=truncated,
            elapsed_ms=elapsed_ms,
        )
