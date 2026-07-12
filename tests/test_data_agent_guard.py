from __future__ import annotations

from pathlib import Path
import sqlite3

from quantquery_a.agents.contracts import AnalyzeRequest, RunStatus, TaskType
from quantquery_a.agents.data_agent import DataAgent
from quantquery_a.agents.supervisor import QuantSupervisor
from quantquery_a.query import ReadOnlyQueryService


def test_data_agent_rejects_truncated_sqlite_evidence() -> None:
    db_path = Path(__file__).with_name("_data_agent_guard.db")
    db_path.unlink(missing_ok=True)
    try:
        connection = sqlite3.connect(db_path)
        try:
            connection.execute(
                "CREATE TABLE stock_history "
                "(ts_code TEXT, trade_date TEXT, open REAL, close REAL)"
            )
            connection.executemany(
                "INSERT INTO stock_history VALUES (?, ?, ?, ?)",
                [
                    ("DEMO.SH", "2025-01-01", 10.0, 10.1),
                    ("DEMO.SH", "2025-01-02", 10.1, 10.2),
                    ("DEMO.SH", "2025-01-03", 10.2, 10.3),
                ],
            )
            connection.commit()
        finally:
            connection.close()

        service = ReadOnlyQueryService(db_path, max_rows=2)
        supervisor = QuantSupervisor(data_agent=DataAgent(service))
        report = supervisor.analyze(
            AnalyzeRequest(
                query="查询 DEMO.SH 行情",
                task_hint=TaskType.MARKET_QUERY,
                symbols=["DEMO.SH"],
            )
        )

        assert report.status is RunStatus.BLOCKED
        assert report.evidence == []
        assert "停止" in report.answer
    finally:
        db_path.unlink(missing_ok=True)


def test_missing_sqlite_source_is_reported_as_blocked(tmp_path: Path) -> None:
    service = ReadOnlyQueryService(tmp_path / "missing.db")
    supervisor = QuantSupervisor(data_agent=DataAgent(service))

    report = supervisor.analyze(
        AnalyzeRequest(
            query="Query DEMO.SH market data",
            task_hint=TaskType.MARKET_QUERY,
            symbols=["DEMO.SH"],
        )
    )

    assert report.status is RunStatus.BLOCKED
    assert report.evidence == []
