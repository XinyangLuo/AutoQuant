"""DuckDB storage for factor values.

Two physical DBs:

- ``factors.duckdb`` (work area) — backing store for ``FactorStorage``.
  Used by ``backfill`` / ``compute`` / ``evaluation``. New factors land here
  during research. Data here is temporary and cleared after admission.

- ``factor_library.duckdb`` (stable library) — backing store for
  ``FactorLibrary``. Only ``admit()`` writes here. The correlation-with-
  existing check in evaluation reads from here so that admission compares
  against *stabilised* factors, never the temporary research churn.

Schema: ``(date, symbol, factor_id, variant, value, ann_date, f_ann_date)``,
PK = ``(date, symbol, factor_id, variant)``.

``variant`` 是中性化变体标识(如 ``"raw"`` / ``"swl1_capq5"``,见
:mod:`backtest.factor.variants`)。同一因子的不同变体并列存放,可横向比较;
也可独立 ``admit`` / ``reject``。
"""

from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

from backtest.data.tushare_client import _find_project_root
from backtest.factor.variants import RAW_VARIANT, canonicalize_variant


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "duckdb"
FACTORS_WORK_DB_PATH = DATA_DIR / "factors.duckdb"
FACTOR_LIBRARY_DB_PATH = DATA_DIR / "factor_library.duckdb"

# Backwards-compatible alias for callers and tests that still reference the
# old single-DB name. Always points to the work area.
FACTORS_DB_PATH = FACTORS_WORK_DB_PATH

FACTORS_SCHEMA = """
CREATE TABLE IF NOT EXISTS factors_daily (
    date       DATE,
    symbol     VARCHAR,
    factor_id  VARCHAR,
    variant    VARCHAR,
    value      DOUBLE,
    ann_date   VARCHAR,
    f_ann_date VARCHAR,
    PRIMARY KEY (date, symbol, factor_id, variant)
)
"""

FACTOR_COLUMNS = ["date", "symbol", "factor_id", "variant", "value", "ann_date", "f_ann_date"]


class FactorStorage:
    """DuckDB storage for factor values in a long-table layout.

    Defaults to the work-area DB (``factors.duckdb``). Callers can pass a
    custom ``db_path`` to point at a different file — :class:`FactorLibrary`
    uses this to back the stable library DB.

    Schema: ``(date, symbol, factor_id, variant, value, ann_date, f_ann_date)``.
    ``variant`` is the neutralization-variant label (default ``"raw"``).
    ``ann_date`` / ``f_ann_date`` are optional provenance for financial
    factors; non-financial factors leave them NULL.
    """

    def __init__(self, db_path: Path | str | None = None):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path is not None else FACTORS_WORK_DB_PATH
        self.conn = duckdb.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self):
        # 1. Ensure table exists (idempotent CREATE)
        self.conn.execute(FACTORS_SCHEMA)
        # 2. Detect legacy schema (pre-variant). If found and empty → recreate;
        #    if non-empty → raise (caller must run a dedicated migration script).
        cols = {
            r[0] for r in self.conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'factors_daily'"
            ).fetchall()
        }
        if "variant" not in cols:
            row_count = self.conn.execute(
                "SELECT COUNT(*) FROM factors_daily"
            ).fetchone()[0]
            if row_count > 0:
                raise RuntimeError(
                    f"factors_daily at {self.db_path} is on the legacy schema "
                    f"(no `variant` column) and contains {row_count:,} rows. "
                    f"Run a one-off migration to ALTER ADD COLUMN variant DEFAULT 'raw' "
                    f"+ rebuild the PRIMARY KEY before reopening."
                )
            # Empty legacy schema → safe to drop + recreate
            self.conn.execute("DROP TABLE factors_daily")
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

    def insert_factors(self, df: pd.DataFrame, *, default_variant: str = RAW_VARIANT):
        """UPSERT factor rows into factors_daily.

        Expected columns: date, symbol, factor_id, value.
        Optional: variant (defaults to ``default_variant``), ann_date, f_ann_date.
        """
        if df.empty:
            return

        required = {"date", "symbol", "factor_id", "value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        if "variant" not in df.columns:
            df["variant"] = default_variant
        else:
            df["variant"] = df["variant"].apply(canonicalize_variant)

        cols = [c for c in FACTOR_COLUMNS if c in df.columns]
        cols_sql = ", ".join(f'"{c}"' for c in cols)
        pk_cols = ("date", "symbol", "factor_id", "variant")
        excluded_sql = ", ".join(
            f'"{c}" = excluded."{c}"' for c in cols if c not in pk_cols
        )

        with self._registered("_upsert_factors", df) as view:
            self.conn.execute(f"""
                INSERT INTO factors_daily ({cols_sql})
                SELECT {cols_sql} FROM {view}
                ON CONFLICT (date, symbol, factor_id, variant) DO UPDATE SET
                    {excluded_sql}
            """)

    # -- read -----------------------------------------------------------------

    def get_factor(
        self,
        factor_id: str,
        start: str | None = None,
        end: str | None = None,
        *,
        variant: str = RAW_VARIANT,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return time-series for a single factor variant.

        Default columns are ``[date, symbol, value]``. Pass ``columns`` to
        request a different subset — e.g. ``FACTOR_COLUMNS`` to also pull
        ``factor_id``, ``variant``, ``ann_date``, ``f_ann_date`` (used by
        :meth:`FactorLibrary.promote_from_work` to preserve full provenance).

        ``variant`` defaults to ``"raw"``. To read all variants of a factor
        in one query, use :meth:`get_factors_long` with ``variant=None``.
        """
        cols = columns or ["date", "symbol", "value"]
        cols_sql = ", ".join(cols)

        variant = canonicalize_variant(variant)
        conditions = ["factor_id = ?", "variant = ?"]
        params: list = [factor_id, variant]

        if start:
            conditions.append("date >= strptime(?, '%Y%m%d')::DATE")
            params.append(start)
        if end:
            conditions.append("date <= strptime(?, '%Y%m%d')::DATE")
            params.append(end)

        sql = f"""
            SELECT {cols_sql}
            FROM factors_daily
            WHERE {" AND ".join(conditions)}
            ORDER BY date, symbol
        """
        return self.conn.execute(sql, params).fetchdf()

    def get_factor_panel(
        self,
        factor_ids: list[str],
        date: str,
        *,
        variant: str = RAW_VARIANT,
    ) -> pd.DataFrame:
        """Return a wide cross-section for multiple factors on a single date,
        for a single variant.

        Returns DataFrame with columns [date, symbol, f_001, f_002, ...].
        """
        variant = canonicalize_variant(variant)
        placeholders = ", ".join("?" for _ in factor_ids)
        sql = f"""
            SELECT date, symbol, factor_id, value
            FROM factors_daily
            WHERE date = strptime(?, '%Y%m%d')::DATE
              AND variant = ?
              AND factor_id IN ({placeholders})
            ORDER BY symbol, factor_id
        """
        params = [date, variant] + factor_ids
        df = self.conn.execute(sql, params).fetchdf()
        if df.empty:
            return df
        return df.pivot(index=["date", "symbol"], columns="factor_id", values="value").reset_index()

    def get_factors_long(
        self,
        factor_ids: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        exclude: str | None = None,
        *,
        variant: str | None = RAW_VARIANT,
    ) -> pd.DataFrame:
        """Return long-form factor values across multiple factors and dates.

        Used by evaluation's correlation check to avoid an N+1 query as the
        library grows.

        ``variant`` filter:
          - default ``"raw"`` — single variant
          - ``None`` — no filter (return all variants)

        Returns DataFrame with columns [date, symbol, factor_id, variant, value].
        """
        conditions: list[str] = []
        params: list = []

        if factor_ids:
            placeholders = ", ".join("?" for _ in factor_ids)
            conditions.append(f"factor_id IN ({placeholders})")
            params.extend(factor_ids)
        if exclude:
            conditions.append("factor_id != ?")
            params.append(exclude)
        if start:
            conditions.append("date >= strptime(?, '%Y%m%d')::DATE")
            params.append(start)
        if end:
            conditions.append("date <= strptime(?, '%Y%m%d')::DATE")
            params.append(end)
        if variant is not None:
            conditions.append("variant = ?")
            params.append(canonicalize_variant(variant))

        where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
            SELECT date, symbol, factor_id, variant, value
            FROM factors_daily
            {where_clause}
            ORDER BY factor_id, variant, date, symbol
        """
        return self.conn.execute(sql, params).fetchdf()

    # -- stats ----------------------------------------------------------------

    def get_max_date(self, factor_id: str, *, variant: str = RAW_VARIANT) -> str | None:
        """Return max date for a factor variant as YYYYMMDD, or None."""
        variant = canonicalize_variant(variant)
        result = self.conn.execute(
            "SELECT MAX(date) FROM factors_daily "
            "WHERE factor_id = ? AND variant = ?",
            [factor_id, variant],
        ).fetchone()
        if result[0]:
            return result[0].strftime("%Y%m%d")
        return None

    def get_existing_factor_ids(self) -> set[str]:
        """Return set of factor_ids that have any data (any variant)."""
        rows = self.conn.execute(
            "SELECT DISTINCT factor_id FROM factors_daily"
        ).fetchall()
        return {r[0] for r in rows}

    def get_existing_variants(self, factor_id: str) -> set[str]:
        """Return set of variants present for a given factor_id."""
        rows = self.conn.execute(
            "SELECT DISTINCT variant FROM factors_daily WHERE factor_id = ?",
            [factor_id],
        ).fetchall()
        return {r[0] for r in rows}

    def delete_factor(
        self,
        factor_id: str,
        *,
        variant: str | None = None,
    ) -> int:
        """Delete rows for ``factor_id``. Returns number of rows deleted.

        ``variant=None`` (default) deletes **all** variants of the factor.
        Pass an explicit variant string to scope the delete.
        """
        if variant is None:
            result = self.conn.execute(
                "DELETE FROM factors_daily WHERE factor_id = ?",
                [factor_id],
            ).fetchone()
        else:
            result = self.conn.execute(
                "DELETE FROM factors_daily WHERE factor_id = ? AND variant = ?",
                [factor_id, canonicalize_variant(variant)],
            ).fetchone()
        return result[0] if result else 0

    def delete_factors(self, factor_ids: list[str]) -> dict[str, int]:
        """Delete multiple factors (all variants) in one round-trip;
        return ``{fid: rows}``.

        DuckDB's ``DELETE`` doesn't return a per-key breakdown, so we run one
        ``GROUP BY`` to count first, then a single bulk ``DELETE``.
        """
        if not factor_ids:
            return {}
        placeholders = ", ".join("?" for _ in factor_ids)
        params = list(factor_ids)
        rows = self.conn.execute(
            f"SELECT factor_id, COUNT(*) FROM factors_daily "
            f"WHERE factor_id IN ({placeholders}) GROUP BY factor_id",
            params,
        ).fetchall()
        counts = {fid: cnt for fid, cnt in rows}
        self.conn.execute(
            f"DELETE FROM factors_daily WHERE factor_id IN ({placeholders})",
            params,
        )
        return {fid: counts.get(fid, 0) for fid in factor_ids}

    def get_factor_stats(self, factor_id: str, *, variant: str | None = None) -> dict:
        """Return basic stats for a factor.

        ``variant=None`` aggregates across all variants of the factor.
        """
        if variant is None:
            row = self.conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(date), MAX(date), "
                "COUNT(DISTINCT variant) "
                "FROM factors_daily WHERE factor_id = ?",
                [factor_id],
            ).fetchone()
            return {
                "total_rows": row[0],
                "total_symbols": row[1],
                "min_date": row[2].strftime("%Y%m%d") if row[2] else None,
                "max_date": row[3].strftime("%Y%m%d") if row[3] else None,
                "n_variants": row[4],
            }
        variant_canon = canonicalize_variant(variant)
        row = self.conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(date), MAX(date) "
            "FROM factors_daily WHERE factor_id = ? AND variant = ?",
            [factor_id, variant_canon],
        ).fetchone()
        return {
            "total_rows": row[0],
            "total_symbols": row[1],
            "min_date": row[2].strftime("%Y%m%d") if row[2] else None,
            "max_date": row[3].strftime("%Y%m%d") if row[3] else None,
            "variant": variant_canon,
        }


class FactorLibrary(FactorStorage):
    """Stable factor library — read-mostly, writes only via ``admit()``.

    Same schema as :class:`FactorStorage` but pointed at
    ``factor_library.duckdb``. Disables :meth:`delete_factor` because the
    library is append-only: rejected factors never enter, and rejected-after-
    admission would itself be a deliberate action requiring a dedicated path
    (not yet implemented).
    """

    def __init__(self, db_path: Path | str | None = None):
        super().__init__(db_path=Path(db_path) if db_path is not None
                                 else FACTOR_LIBRARY_DB_PATH)

    def delete_factor(self, factor_id: str, *, variant: str | None = None) -> int:  # noqa: D401 — override
        """Disabled on the library DB.

        Admitted factors are intended to be stable. Deleting one is a
        deliberate de-admission and currently has no API surface — do it
        manually via DuckDB if absolutely necessary.
        """
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
        *,
        variant: str = RAW_VARIANT,
    ) -> int:
        """Copy a factor variant's data from the work DB into the library.

        Reads the full long rows (including ``ann_date`` / ``f_ann_date``)
        from ``work_storage`` and upserts them into this library. Caller
        is responsible for clearing the work DB afterwards (typically in
        the ``admit`` orchestrator).

        Returns the number of rows written.
        """
        df = work_storage.get_factor(factor_id, variant=variant, columns=FACTOR_COLUMNS)
        if df.empty:
            return 0
        self.insert_factors(df)
        return len(df)


__all__ = [
    "FactorStorage",
    "FactorLibrary",
    "FACTORS_WORK_DB_PATH",
    "FACTOR_LIBRARY_DB_PATH",
    "FACTORS_DB_PATH",
    "FACTORS_SCHEMA",
    "FACTOR_COLUMNS",
]
