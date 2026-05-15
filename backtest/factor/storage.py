"""DuckDB storage for factors_daily."""

from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

from backtest.data.tushare_client import _find_project_root


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "duckdb"
FACTORS_DB_PATH = DATA_DIR / "factors.duckdb"

FACTORS_SCHEMA = """
CREATE TABLE IF NOT EXISTS factors_daily (
    date      DATE,
    symbol    VARCHAR,
    factor_id VARCHAR,
    value     DOUBLE,
    ann_date  VARCHAR,
    f_ann_date VARCHAR,
    PRIMARY KEY (date, symbol, factor_id)
)
"""

FACTOR_COLUMNS = ["date", "symbol", "factor_id", "value", "ann_date", "f_ann_date"]


class FactorStorage:
    """DuckDB storage for factor values in a long-table layout.

    Schema: (date, symbol, factor_id, value, ann_date, f_ann_date)
    - ann_date / f_ann_date are optional provenance for financial factors
    - Non-financial factors leave them NULL
    """

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = FACTORS_DB_PATH
        self.conn = duckdb.connect(str(FACTORS_DB_PATH))
        self._init_tables()

    def _init_tables(self):
        self.conn.execute(FACTORS_SCHEMA)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        self.conn.close()

    @contextmanager
    def _registered(self, name: str, df: pd.DataFrame):
        """Register df as a DuckDB view; unregister on exit."""
        self.conn.register(name, df)
        try:
            yield name
        finally:
            self.conn.unregister(name)

    # -- write ----------------------------------------------------------------

    def insert_factors(self, df: pd.DataFrame):
        """UPSERT factor rows into factors_daily.

        Expected columns: date, symbol, factor_id, value.
        Optional: ann_date, f_ann_date (for financial factor provenance).
        """
        if df.empty:
            return

        # Ensure required columns exist
        required = {"date", "symbol", "factor_id", "value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Build column list from FACTOR_COLUMNS that exist in df
        cols = [c for c in FACTOR_COLUMNS if c in df.columns]
        cols_sql = ", ".join(f'"{c}"' for c in cols)
        excluded_sql = ", ".join(f'"{c}" = excluded."{c}"' for c in cols if c not in ("date", "symbol", "factor_id"))

        with self._registered("_upsert_factors", df) as view:
            self.conn.execute(f"""
                INSERT INTO factors_daily ({cols_sql})
                SELECT {cols_sql} FROM {view}
                ON CONFLICT (date, symbol, factor_id) DO UPDATE SET
                    {excluded_sql}
            """)

    # -- read -----------------------------------------------------------------

    def get_factor(
        self,
        factor_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return time-series for a single factor.

        Returns DataFrame with columns [date, symbol, value].
        """
        conditions = ["factor_id = ?"]
        params = [factor_id]

        if start:
            conditions.append("date >= strptime(?, '%Y%m%d')::DATE")
            params.append(start)
        if end:
            conditions.append("date <= strptime(?, '%Y%m%d')::DATE")
            params.append(end)

        where_clause = " AND ".join(conditions)

        sql = f"""
            SELECT date, symbol, value
            FROM factors_daily
            WHERE {where_clause}
            ORDER BY date, symbol
        """
        return self.conn.execute(sql, params).fetchdf()

    def get_factor_panel(
        self,
        factor_ids: list[str],
        date: str,
    ) -> pd.DataFrame:
        """Return a wide cross-section for multiple factors on a single date.

        Returns DataFrame with columns [date, symbol, f_001, f_002, ...].
        """
        placeholders = ", ".join("?" for _ in factor_ids)
        sql = f"""
            SELECT date, symbol, factor_id, value
            FROM factors_daily
            WHERE date = strptime(?, '%Y%m%d')::DATE
              AND factor_id IN ({placeholders})
            ORDER BY symbol, factor_id
        """
        params = [date] + factor_ids
        df = self.conn.execute(sql, params).fetchdf()
        if df.empty:
            return df
        return df.pivot(index=["date", "symbol"], columns="factor_id", values="value").reset_index()

    # -- stats ----------------------------------------------------------------

    def get_max_date(self, factor_id: str) -> str | None:
        """Return max date for a factor as YYYYMMDD, or None."""
        result = self.conn.execute(
            "SELECT MAX(date) FROM factors_daily WHERE factor_id = ?",
            [factor_id],
        ).fetchone()
        if result[0]:
            return result[0].strftime("%Y%m%d")
        return None

    def get_existing_factor_ids(self) -> set[str]:
        """Return set of factor_ids that already have data."""
        rows = self.conn.execute(
            "SELECT DISTINCT factor_id FROM factors_daily"
        ).fetchall()
        return {r[0] for r in rows}

    def get_factor_stats(self, factor_id: str) -> dict:
        """Return basic stats for a factor."""
        row = self.conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(date), MAX(date) "
            "FROM factors_daily WHERE factor_id = ?",
            [factor_id],
        ).fetchone()
        return {
            "total_rows": row[0],
            "total_symbols": row[1],
            "min_date": row[2].strftime("%Y%m%d") if row[2] else None,
            "max_date": row[3].strftime("%Y%m%d") if row[3] else None,
        }
