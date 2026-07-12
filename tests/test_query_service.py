from pathlib import Path
import sqlite3

import pytest

from quantquery_a.query import QueryPolicyError, ReadOnlyQueryService


def test_read_only_query_policy_and_resource_release():
    db_path = Path(__file__).with_name("_query_service_test.db")
    db_path.unlink(missing_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE stock_history (ts_code TEXT, trade_date TEXT, close REAL)"
        )
        connection.executemany(
            "INSERT INTO stock_history VALUES (?, ?, ?)",
            [
                ("DEMO.SH", "2026-01-02", 10.0),
                ("DEMO.SH", "2026-01-05", 10.5),
                ("DEMO.SH", "2026-01-06", 11.0),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    try:
        service = ReadOnlyQueryService(db_path, max_rows=2)
        result = service.execute(
            "SELECT trade_date, close FROM stock_history ORDER BY trade_date"
        )
        assert result.row_count == 2
        assert result.truncated is True

        cte_result = service.execute(
            """
            WITH selected AS (
                SELECT close FROM stock_history WHERE ts_code = ?
            )
            SELECT AVG(close) AS average_close FROM selected
            """,
            ("DEMO.SH",),
        )
        assert cte_result.rows[0]["average_close"] == pytest.approx(10.5)

        for unsafe_sql in (
            "UPDATE stock_history SET close = 0",
            "DELETE FROM stock_history",
            "DROP TABLE stock_history",
            "PRAGMA table_info(stock_history)",
            "ATTACH DATABASE 'other.db' AS other",
            "SELECT 1; SELECT 2",
        ):
            with pytest.raises(QueryPolicyError):
                service.execute(unsafe_sql)

        check = sqlite3.connect(db_path)
        try:
            count = check.execute("SELECT COUNT(*) FROM stock_history").fetchone()[0]
        finally:
            check.close()
        assert count == 3
    finally:
        db_path.unlink(missing_ok=True)
