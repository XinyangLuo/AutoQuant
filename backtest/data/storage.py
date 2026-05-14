"""DuckDB storage for market daily data."""

from pathlib import Path

import duckdb
import pandas as pd

from backtest.data.tushare_client import _find_project_root


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "duckdb"
DB_PATH = DATA_DIR / "market.duckdb"

# ---------------------------------------------------------------------------
# Schema: market_daily wide table
# ---------------------------------------------------------------------------
# All columns listed here; new columns are added via ALTER TABLE ADD COLUMN
# in _init_table() for backward-compatible migrations.
# insert_daily() is column-dynamic: it INSERTs/UPDATEs only the columns
# present in the DataFrame, leaving other columns untouched.
# ---------------------------------------------------------------------------

DAILY_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "volume",
    "amount",
    "adj_factor",
    "is_st",
    "list_date",
    "limit_up",
    "limit_down",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
]

_COL_DEFS = ",\n    ".join(f"{c:12s} DOUBLE" for c in DAILY_COLUMNS[2:])
DAILY_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS market_daily (
    date        DATE,
    symbol      VARCHAR,
    {_COL_DEFS},
    PRIMARY KEY (date, symbol)
)
"""


class MarketStorage:
    """DuckDB storage for daily market data. Table is created once and never dropped."""

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = DB_PATH
        self.conn = duckdb.connect(str(DB_PATH))
        self._init_table()

    def _init_table(self):
        self.conn.execute(DAILY_SCHEMA)
        # Migration: add columns introduced after initial table creation
        for col in DAILY_COLUMNS:
            if col in ("date", "symbol"):
                continue
            # DuckDB does not have ADD COLUMN IF NOT EXISTS in older versions,
            # so we wrap each attempt in a try/except.
            try:
                self.conn.execute(f'ALTER TABLE market_daily ADD COLUMN "{col}" DOUBLE')
            except Exception:
                pass

    # -- context manager --------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # -- queries ----------------------------------------------------------

    def get_max_date(self) -> str | None:
        """Return max date in DB as YYYYMMDD, or None if empty."""
        result = self.conn.execute("SELECT MAX(date) FROM market_daily").fetchone()
        if result[0]:
            return result[0].strftime("%Y%m%d")
        return None

    def get_stats(self) -> dict:
        total_rows = self.conn.execute("SELECT COUNT(*) FROM market_daily").fetchone()[0]
        total_symbols = self.conn.execute("SELECT COUNT(DISTINCT symbol) FROM market_daily").fetchone()[0]
        date_range = self.conn.execute("SELECT MIN(date), MAX(date) FROM market_daily").fetchone()
        return {
            "total_rows": total_rows,
            "total_symbols": total_symbols,
            "min_date": date_range[0],
            "max_date": date_range[1],
        }

    # -- writes -----------------------------------------------------------

    def insert_daily(self, df: pd.DataFrame):
        """
        Insert/UPSERT daily data.
        Only the columns present in *df* are written; other table columns are
        left untouched. This allows backfill scripts to update a subset of
        columns without re-fetching unrelated data.
        """
        if df.empty:
            return

        # Use only columns that exist in both DataFrame and table schema
        cols = [c for c in df.columns if c in DAILY_COLUMNS]
        if not cols:
            return

        cols_sql = ", ".join(f'"{c}"' for c in cols)
        updates = [f'"{c}" = excluded."{c}"' for c in cols if c not in ("date", "symbol")]
        update_sql = ", ".join(updates) if updates else "open = excluded.open"  # no-op fallback

        self.conn.register("daily_df", df)
        try:
            self.conn.execute(f"""
                INSERT INTO market_daily ({cols_sql})
                SELECT {cols_sql} FROM daily_df
                ON CONFLICT (date, symbol) DO UPDATE SET
                    {update_sql}
            """)
        finally:
            self.conn.unregister("daily_df")

    def close(self):
        self.conn.close()
