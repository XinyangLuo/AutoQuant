"""DuckDB storage for market daily data."""

from pathlib import Path

import duckdb
import pandas as pd

from backtest.data.tushare_client import _find_project_root


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "duckdb"
DB_PATH = DATA_DIR / "market.duckdb"

DAILY_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_daily (
    date        DATE,
    symbol      VARCHAR,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    pre_close   DOUBLE,
    change      DOUBLE,
    pct_chg     DOUBLE,
    volume      BIGINT,
    amount      DOUBLE,
    adj_factor  DOUBLE,
    is_st       BOOLEAN,
    list_date   DATE,
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
        self.conn.execute("ALTER TABLE market_daily ADD COLUMN IF NOT EXISTS is_st BOOLEAN")
        self.conn.execute("ALTER TABLE market_daily ADD COLUMN IF NOT EXISTS list_date DATE")

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
        """Insert daily data using UPSERT. Existing (date, symbol) rows are overwritten."""
        if df.empty:
            return

        self.conn.register("daily_df", df)
        try:
            self.conn.execute("""
                INSERT INTO market_daily
                SELECT date, symbol, open, high, low, close,
                       pre_close, change, pct_chg, volume, amount, adj_factor,
                       is_st, list_date
                FROM daily_df
                ON CONFLICT (date, symbol) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    pre_close = excluded.pre_close,
                    change = excluded.change,
                    pct_chg = excluded.pct_chg,
                    volume = excluded.volume,
                    amount = excluded.amount,
                    adj_factor = excluded.adj_factor,
                    is_st = excluded.is_st,
                    list_date = excluded.list_date
            """)
        finally:
            self.conn.unregister("daily_df")

    def close(self):
        self.conn.close()
