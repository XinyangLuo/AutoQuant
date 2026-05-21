"""DuckDB storage for factor values — wide format, single table per DB.

Two physical DBs share the same schema:

- ``factors.duckdb`` (work area, :class:`FactorStorage`) — research churn.
  ``backfill`` / ``compute`` / ``evaluation`` write here.
- ``factor_library.duckdb`` (stable library, :class:`FactorLibrary`) — only
  ``admit()`` writes here. Evaluation's cross-factor correlation check reads
  from here so admission compares against *stabilised* factors, never the
  temporary research churn.

Schema
------

::

    CREATE TABLE factors_daily (
        date    DATE,
        symbol  VARCHAR,
        <factor_id_1>  DOUBLE,
        <factor_id_2>  DOUBLE,
        ...
        PRIMARY KEY (date, symbol)
    );

Each factor is a column. Adding a factor = ``ALTER TABLE ADD COLUMN
f_xxx DOUBLE`` (O(metadata)). Deleting = ``ALTER TABLE DROP COLUMN f_xxx``
(also O(metadata), no row rewrite). Total rows are bounded by date × symbol
(~25M for 10y × 5000 symbols) and don't grow with the number of factors.

Each factor stores **exactly one** version of its values. The neutralization
pipeline that produced those values is recorded as a ``variant`` label in
``registry.json`` (see :mod:`backtest.factor.variants`), **not** as a
schema dimension here. Re-computing a factor under a different variant
overwrites the column.

PIT provenance (``ann_date`` / ``f_ann_date``) is **not** stored here —
PIT isolation is enforced upstream in :func:`backtest.factor.compute.compute_factor`
when it pulls financial data via ``get_fina_snapshot(date)``. Audit trails
live in ``market.duckdb``'s financial tables, not the factor cache.
"""

from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

from backtest.data.tushare_client import _find_project_root


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "duckdb"
FACTORS_WORK_DB_PATH = DATA_DIR / "factors.duckdb"
FACTOR_LIBRARY_DB_PATH = DATA_DIR / "factor_library.duckdb"

# Backwards-compatible alias for callers and tests that still reference the
# old single-DB name. Always points to the work area.
FACTORS_DB_PATH = FACTORS_WORK_DB_PATH

FACTORS_TABLE = "factors_daily"


def _quote_ident(name: str) -> str:
    if '"' in name:
        raise ValueError(f"Invalid identifier (contains quote): {name!r}")
    return f'"{name}"'


class FactorStorage:
    """DuckDB-backed wide-format factor store.

    Defaults to the work-area DB (``factors.duckdb``). Callers can pass a
    custom ``db_path`` — :class:`FactorLibrary` uses this to back the stable
    library DB at ``factor_library.duckdb``.

    Public surface treats factors as columns: :meth:`get_factor` selects one
    column, :meth:`get_factor_panel` selects a list of columns,
    :meth:`insert_factors` UPSERTs by ``(date, symbol)`` and creates the
    column on first sight.
    """

    def __init__(self, db_path: Path | str | None = None):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path is not None else FACTORS_WORK_DB_PATH
        self.conn = duckdb.connect(str(self.db_path))
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {FACTORS_TABLE} (
                date    DATE,
                symbol  VARCHAR,
                PRIMARY KEY (date, symbol)
            )
        """)
        self._cols_cache: set[str] | None = None

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

    # -- introspection --------------------------------------------------------

    def _existing_columns(self) -> set[str]:
        if self._cols_cache is None:
            rows = self.conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ?",
                [FACTORS_TABLE],
            ).fetchall()
            self._cols_cache = {r[0] for r in rows}
        return self._cols_cache

    def _ensure_factor_column(self, factor_id: str) -> None:
        if factor_id in self._existing_columns():
            return
        self.conn.execute(
            f"ALTER TABLE {FACTORS_TABLE} "
            f"ADD COLUMN {_quote_ident(factor_id)} DOUBLE"
        )
        if self._cols_cache is not None:
            self._cols_cache.add(factor_id)

    # -- write ----------------------------------------------------------------

    def insert_factors(self, df: pd.DataFrame):
        """UPSERT factor values.

        Expected columns: ``date``, ``symbol``, ``factor_id``, ``value``.
        Rows are grouped by ``factor_id``, each group is merged into the
        corresponding column via ``INSERT ... ON CONFLICT DO UPDATE``.
        Columns are auto-created on first write.
        """
        if df.empty:
            return

        required = {"date", "symbol", "factor_id", "value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for factor_id, grp in df.groupby("factor_id", sort=False):
            self._ensure_factor_column(factor_id)
            sub = grp[["date", "symbol", "value"]].rename(columns={"value": factor_id})
            col_q = _quote_ident(factor_id)
            with self._registered("_upsert_factors", sub) as view:
                self.conn.execute(f"""
                    INSERT INTO {FACTORS_TABLE} (date, symbol, {col_q})
                    SELECT date, symbol, {col_q} FROM {view}
                    ON CONFLICT (date, symbol) DO UPDATE SET
                        {col_q} = excluded.{col_q}
                """)

    # -- read -----------------------------------------------------------------

    def get_factor(
        self,
        factor_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return ``[date, symbol, value]`` for one factor's non-null cells."""
        if factor_id not in self._existing_columns():
            return pd.DataFrame(columns=["date", "symbol", "value"])

        col_q = _quote_ident(factor_id)
        conditions: list[str] = [f"{col_q} IS NOT NULL"]
        params: list = []
        if start:
            conditions.append("date >= strptime(?, '%Y%m%d')::DATE")
            params.append(start)
        if end:
            conditions.append("date <= strptime(?, '%Y%m%d')::DATE")
            params.append(end)

        sql = f"""
            SELECT date, symbol, {col_q} AS value
            FROM {FACTORS_TABLE}
            WHERE {" AND ".join(conditions)}
            ORDER BY date, symbol
        """
        return self.conn.execute(sql, params).fetchdf()

    def get_factor_panel(
        self,
        factor_ids: list[str],
        date: str,
    ) -> pd.DataFrame:
        """Return a wide cross-section ``[date, symbol, f_001, f_002, ...]``.

        Missing factors (not yet a column) are returned as all-NaN so that
        callers can request a stable column set.
        """
        present = self._existing_columns()
        selected: list[str] = []
        for fid in factor_ids:
            if fid in present:
                selected.append(_quote_ident(fid))
            else:
                selected.append(f"CAST(NULL AS DOUBLE) AS {_quote_ident(fid)}")
        cols_sql = ", ".join(selected)
        sql = f"""
            SELECT date, symbol, {cols_sql}
            FROM {FACTORS_TABLE}
            WHERE date = strptime(?, '%Y%m%d')::DATE
            ORDER BY symbol
        """
        return self.conn.execute(sql, [date]).fetchdf()

    def get_factors_long(
        self,
        factor_ids: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        exclude: str | None = None,
    ) -> pd.DataFrame:
        """Return long-form ``[date, symbol, factor_id, value]`` rows.

        Used by evaluation's cross-factor correlation check. NULL cells are
        dropped per column (sparse-history factors don't bloat the output).
        """
        present = self._existing_columns() - {"date", "symbol"}
        if factor_ids is not None:
            cols = [fid for fid in factor_ids if fid in present]
        else:
            cols = sorted(present)
        if exclude:
            cols = [c for c in cols if c != exclude]
        if not cols:
            return pd.DataFrame(columns=["date", "symbol", "factor_id", "value"])

        conditions: list[str] = []
        params: list = []
        if start:
            conditions.append("date >= strptime(?, '%Y%m%d')::DATE")
            params.append(start)
        if end:
            conditions.append("date <= strptime(?, '%Y%m%d')::DATE")
            params.append(end)
        where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""

        cols_sql = ", ".join(_quote_ident(c) for c in cols)
        wide = self.conn.execute(
            f"SELECT date, symbol, {cols_sql} FROM {FACTORS_TABLE}{where_clause}",
            params,
        ).fetchdf()
        if wide.empty:
            return pd.DataFrame(columns=["date", "symbol", "factor_id", "value"])

        long = wide.melt(
            id_vars=["date", "symbol"],
            var_name="factor_id",
            value_name="value",
        ).dropna(subset=["value"])
        return long.sort_values(["factor_id", "date", "symbol"]).reset_index(drop=True)

    # -- stats ----------------------------------------------------------------

    def get_max_date(self, factor_id: str) -> str | None:
        """Latest date with a non-null value for ``factor_id`` (YYYYMMDD)."""
        if factor_id not in self._existing_columns():
            return None
        col_q = _quote_ident(factor_id)
        result = self.conn.execute(
            f"SELECT MAX(date) FROM {FACTORS_TABLE} "
            f"WHERE {col_q} IS NOT NULL"
        ).fetchone()
        if result[0]:
            return result[0].strftime("%Y%m%d")
        return None

    def get_existing_factor_ids(self) -> set[str]:
        """All factor columns currently in the table."""
        return self._existing_columns() - {"date", "symbol"}

    def delete_factor(self, factor_id: str) -> int:
        """Drop the factor's column. Returns 1 if dropped, 0 if absent.

        DROP COLUMN is O(metadata) in DuckDB — no row rewrite happens.
        """
        if factor_id not in self._existing_columns():
            return 0
        self.conn.execute(
            f"ALTER TABLE {FACTORS_TABLE} "
            f"DROP COLUMN {_quote_ident(factor_id)}"
        )
        if self._cols_cache is not None:
            self._cols_cache.discard(factor_id)
        return 1

    def delete_factors(self, factor_ids: list[str]) -> dict[str, int]:
        """Drop multiple factor columns. Returns ``{factor_id: 0|1}``."""
        return {fid: self.delete_factor(fid) for fid in factor_ids}

    def get_factor_stats(self, factor_id: str) -> dict:
        """Basic stats for one factor — null-aware row count, date range, n_symbols."""
        if factor_id not in self._existing_columns():
            return {"total_rows": 0, "total_symbols": 0,
                    "min_date": None, "max_date": None}
        col_q = _quote_ident(factor_id)
        row = self.conn.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(date), MAX(date) "
            f"FROM {FACTORS_TABLE} WHERE {col_q} IS NOT NULL"
        ).fetchone()
        return {
            "total_rows": row[0],
            "total_symbols": row[1],
            "min_date": row[2].strftime("%Y%m%d") if row[2] else None,
            "max_date": row[3].strftime("%Y%m%d") if row[3] else None,
        }


class FactorLibrary(FactorStorage):
    """Stable factor library — read-mostly, writes only via ``admit()``.

    Same schema as :class:`FactorStorage` but pointed at
    ``factor_library.duckdb``. :meth:`delete_factor` is disabled because the
    library is append-only.
    """

    def __init__(self, db_path: Path | str | None = None):
        super().__init__(db_path=Path(db_path) if db_path is not None
                                 else FACTOR_LIBRARY_DB_PATH)

    def delete_factor(self, factor_id: str) -> int:  # noqa: D401
        raise NotImplementedError(
            "FactorLibrary is append-only. Use the DuckDB CLI directly if "
            "you really need to remove an admitted factor."
        )

    def delete_factors(self, factor_ids: list[str]) -> dict[str, int]:
        raise NotImplementedError(
            "FactorLibrary is append-only. Use the DuckDB CLI directly if "
            "you really need to remove admitted factors."
        )

    def promote_from_work(
        self,
        factor_id: str,
        work_storage: FactorStorage,
    ) -> int:
        """Copy ``factor_id``'s values from the work DB into the library.

        Caller clears the work DB afterwards (typically in the ``admit``
        orchestrator). Returns the number of rows written.
        """
        df = work_storage.get_factor(factor_id)
        if df.empty:
            return 0
        df = df.copy()
        df["factor_id"] = factor_id
        self.insert_factors(df)
        return len(df)


__all__ = [
    "FactorStorage",
    "FactorLibrary",
    "FACTORS_WORK_DB_PATH",
    "FACTOR_LIBRARY_DB_PATH",
    "FACTORS_DB_PATH",
    "FACTORS_TABLE",
]
