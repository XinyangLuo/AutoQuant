"""DuckDB storage for cyq_chips (筹码分布) — nested LIST format.

Schema
------

Each (date, symbol) carries two equal-length DOUBLE arrays:

::

    CREATE TABLE cyq_chips (
        date      DATE,
        symbol    VARCHAR,
        n_bins    INTEGER,         -- len(prices) == len(percents)
        prices    DOUBLE[],        -- price levels (ascending)
        percents  DOUBLE[],        -- chip percentage at each level
        PRIMARY KEY (date, symbol)
    );

Why LIST instead of wide / long
--------------------------------

- **Wide** (one column per bin): impossible — bins vary 59~175 per stock.
- **Long** (date, symbol, price, percent): ~125 M rows/year. Reading one
  distribution needs a multi-row scan + client-side reassembly.
- **LIST** (one row per (date, symbol)): ~1.25 M rows/year. A point lookup
  returns the full distribution in a single row. DuckDB's LIST functions
  (list_dot_product, list_max, list_position, unnest) allow SQL-level
  aggregation for simple statistics.

Storage budget
--------------

- Raw: ~1.6 KB/row  =>  ~2 GB/year (5 000 symbols × 250 days).
- DuckDB compression: ~0.5–1 GB/year.
- 10-year backfill: ~5–10 GB, entirely reasonable for a single DuckDB file.

Pre-aggregated factor layer
---------------------------

Raw distributions are rarely consumed directly. A companion
``CyqFactorBuilder`` (``backtest/factor/builtin/cyq/``) computes statistical
summaries (concentration, skewness, peak distance, upper/lower ratio, …)
and writes them as ordinary DOUBLE columns into ``FactorStorage``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

from backtest.data.tushare_client import _find_project_root


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "duckdb"
CYQ_DB_PATH = DATA_DIR / "cyq_chips.duckdb"

CYQ_TABLE = "cyq_chips"

_CYQ_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {CYQ_TABLE} (
    date      DATE,
    symbol    VARCHAR,
    n_bins    INTEGER,
    prices    DOUBLE[],
    percents  DOUBLE[],
    PRIMARY KEY (date, symbol)
)
"""

# ---------------------------------------------------------------------------
# Helper: long-format DataFrame → DuckDB LIST row
# ---------------------------------------------------------------------------

def _pack_cyq(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format [date, symbol, price, percent] → one row per group.

    Returns DataFrame with columns [date, symbol, n_bins, prices, percents].
    ``prices`` and ``percents`` are Python lists (DuckDB registers them as
    DOUBLE[] when the DataFrame is passed via ``conn.register``).
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "symbol", "n_bins", "prices", "percents"])

    required = {"date", "symbol", "price", "percent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure date is native datetime/date
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    def _group_to_lists(grp: pd.DataFrame):
        grp = grp.sort_values("price")
        prices = grp["price"].tolist()
        percents = grp["percent"].tolist()
        return pd.Series({
            "n_bins": len(prices),
            "prices": prices,
            "percents": percents,
        })

    packed = (
        df.groupby(["date", "symbol"], sort=False)
        .apply(_group_to_lists, include_groups=False)
        .reset_index()
    )
    # Drop groups that ended up with zero bins (all NaN prices/percent)
    packed = packed[packed["n_bins"] > 0].reset_index(drop=True)
    return packed


# ---------------------------------------------------------------------------
# Storage class
# ---------------------------------------------------------------------------

class CyqStorage:
    """DuckDB-backed chip-distribution store using LIST columns.

    Public surface is (date, symbol)-centric: insert from long-format Tushare
    response, read back as arrays or unnested DataFrames.
    """

    def __init__(self, db_path: Path | str | None = None, *, read_only: bool = False):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else CYQ_DB_PATH
        self.conn = duckdb.connect(str(self.db_path), read_only=read_only)
        if not read_only:
            self.conn.execute("PRAGMA memory_limit='6GB'")
            try:
                self.conn.execute("PRAGMA temp_directory='/tmp/duckdb_spill'")
            except duckdb.NotImplementedException:
                pass
            self.conn.execute(_CYQ_SCHEMA)

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

    def insert_cyq(self, df: pd.DataFrame):
        """UPSERT chip-distribution rows.

        Parameters
        ----------
        df : pd.DataFrame
            Long-format with columns ``[date, symbol, price, percent]``.
            One row per price bin. ``percent`` is the raw percentage value
            (e.g. 0.35 for 0.35%%), **not** normalised to sum=1.
        """
        if df.empty:
            return
        packed = _pack_cyq(df)
        if packed.empty:
            return

        with self._registered("_cyq_packed", packed):
            self.conn.execute(f"""
                INSERT INTO {CYQ_TABLE} (date, symbol, n_bins, prices, percents)
                SELECT date, symbol, n_bins, prices, percents
                FROM _cyq_packed
                ON CONFLICT (date, symbol) DO UPDATE SET
                    n_bins   = excluded.n_bins,
                    prices   = excluded.prices,
                    percents = excluded.percents
            """)

    # -- read -----------------------------------------------------------------

    def get_cyq(
        self,
        date: str,
        symbol: str,
    ) -> pd.DataFrame | None:
        """Return a single stock's distribution on *date* as a tidy DataFrame.

        Columns: ``[price, percent]``.  Returns ``None`` if no data.
        """
        sql = f"""
            SELECT unnest(prices)   AS price,
                   unnest(percents) AS percent
            FROM {CYQ_TABLE}
            WHERE date = strptime(?, '%Y%m%d')::DATE
              AND symbol = ?
        """
        df = self.conn.execute(sql, [date, symbol]).fetchdf()
        if df.empty:
            return None
        return df

    def get_cyq_panel(
        self,
        date: str,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return cross-section ``[date, symbol, n_bins, prices, percents]``.

        ``prices`` and ``percents`` are Python lists (DuckDB LIST type).
        """
        params: list = [date]
        symbol_filter = ""
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            symbol_filter = f"AND symbol IN ({placeholders})"
            params.extend(symbols)

        sql = f"""
            SELECT date, symbol, n_bins, prices, percents
            FROM {CYQ_TABLE}
            WHERE date = strptime(?, '%Y%m%d')::DATE
              {symbol_filter}
            ORDER BY symbol
        """
        return self.conn.execute(sql, params).fetchdf()

    def get_cyq_history(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return time-series for a single symbol.

        Columns: ``[date, n_bins, prices, percents]``.
        """
        conditions = ["symbol = ?"]
        params: list = [symbol]
        if start:
            conditions.append("date >= strptime(?, '%Y%m%d')::DATE")
            params.append(start)
        if end:
            conditions.append("date <= strptime(?, '%Y%m%d')::DATE")
            params.append(end)

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT date, n_bins, prices, percents
            FROM {CYQ_TABLE}
            WHERE {where_clause}
            ORDER BY date
        """
        return self.conn.execute(sql, params).fetchdf()

    # -- SQL-level aggregation helpers ----------------------------------------

    def get_weighted_prices(
        self,
        date: str,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return chip-weighted average price per (date, symbol).

        Columns: ``[date, symbol, weighted_price]``.
        """
        params: list = [date]
        symbol_filter = ""
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            symbol_filter = f"AND symbol IN ({placeholders})"
            params.extend(symbols)

        sql = f"""
            SELECT date, symbol,
                   list_dot_product(prices, percents)
                       / NULLIF(list_sum(percents), 0.0)
                       AS weighted_price
            FROM {CYQ_TABLE}
            WHERE date = strptime(?, '%Y%m%d')::DATE
              {symbol_filter}
              AND len(percents) > 0
            ORDER BY symbol
        """
        return self.conn.execute(sql, params).fetchdf()

    def get_peak_prices(
        self,
        date: str,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return the price bin with the highest percentage.

        Columns: ``[date, symbol, peak_price]``.
        """
        params: list = [date]
        symbol_filter = ""
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            symbol_filter = f"AND symbol IN ({placeholders})"
            params.extend(symbols)

        sql = f"""
            SELECT date, symbol,
                   prices[list_position(percents, list_max(percents))]
                       AS peak_price
            FROM {CYQ_TABLE}
            WHERE date = strptime(?, '%Y%m%d')::DATE
              {symbol_filter}
              AND len(percents) > 0
            ORDER BY symbol
        """
        return self.conn.execute(sql, params).fetchdf()

    # -- introspection --------------------------------------------------------

    def get_max_date(self) -> str | None:
        """Latest date with data, as YYYYMMDD, or None."""
        result = self.conn.execute(f"SELECT MAX(date) FROM {CYQ_TABLE}").fetchone()
        if result[0]:
            return result[0].strftime("%Y%m%d")
        return None

    def get_stats(self) -> dict:
        """Row count, symbol coverage, date range."""
        row = self.conn.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(date), MAX(date) "
            f"FROM {CYQ_TABLE}"
        ).fetchone()
        return {
            "total_rows": row[0],
            "total_symbols": row[1],
            "min_date": row[2].strftime("%Y%m%d") if row[2] else None,
            "max_date": row[3].strftime("%Y%m%d") if row[3] else None,
        }

    def get_symbols_for_date(self, date: str) -> set[str]:
        """All symbols present on *date*."""
        rows = self.conn.execute(
            f"SELECT DISTINCT symbol FROM {CYQ_TABLE} "
            f"WHERE date = strptime(?, '%Y%m%d')::DATE",
            [date],
        ).fetchall()
        return {r[0] for r in rows}


__all__ = ["CyqStorage", "CYQ_DB_PATH", "CYQ_TABLE", "_pack_cyq"]
