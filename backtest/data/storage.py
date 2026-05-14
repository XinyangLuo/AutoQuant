"""DuckDB storage for market daily data, financial indicators, and dividends."""

from contextlib import contextmanager
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
# in _init_tables() for backward-compatible migrations.
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

_DAILY_COL_DEFS = ",\n    ".join(f"{c:12s} DOUBLE" for c in DAILY_COLUMNS[2:])
DAILY_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS market_daily (
    date        DATE,
    symbol      VARCHAR,
    {_DAILY_COL_DEFS},
    PRIMARY KEY (date, symbol)
)
"""

# ---------------------------------------------------------------------------
# Schema: fina_indicator_quarterly wide table
# ---------------------------------------------------------------------------
# All numeric columns from Tushare pro.fina_indicator. Keys + update_flag are
# VARCHAR; everything else is DOUBLE. (symbol, end_date) is the natural PK
# after deduplication.
# ---------------------------------------------------------------------------

FINA_NUMERIC = [
    # per-share
    "eps", "dt_eps", "total_revenue_ps", "revenue_ps",
    "capital_rese_ps", "surplus_rese_ps", "undist_profit_ps",
    "extra_item", "profit_dedt", "gross_margin",
    "diluted2_eps", "bps", "ocfps", "retainedps", "cfps",
    "ebit_ps", "fcff_ps", "fcfe_ps",
    # solvency
    "current_ratio", "quick_ratio", "cash_ratio",
    # turnover
    "invturn_days", "arturn_days", "inv_turn", "ar_turn",
    "ca_turn", "fa_turn", "assets_turn",
    # income items
    "op_income", "valuechange_income", "interst_income", "daa",
    "ebit", "ebitda", "fcff", "fcfe",
    "current_exint", "noncurrent_exint", "interestdebt", "netdebt",
    "tangible_asset", "working_capital", "networking_capital",
    "invest_capital", "retained_earnings",
    # profitability
    "netprofit_margin", "grossprofit_margin",
    "cogs_of_sales", "expense_of_sales",
    "profit_to_gr", "saleexp_to_gr", "adminexp_of_gr", "finaexp_of_gr",
    "impai_ttm", "gc_of_gr", "op_of_gr", "ebit_of_gr",
    "roe", "roe_waa", "roe_dt", "roa", "npta", "roic",
    "roe_yearly", "roa2_yearly", "roe_avg",
    # earnings composition
    "opincome_of_ebt", "investincome_of_ebt", "n_op_profit_of_ebt",
    "tax_to_ebt", "dtprofit_to_profit",
    # cash flow ratios
    "salescash_to_or", "ocf_to_or", "ocf_to_opincome", "capitalized_to_da",
    # capital structure
    "debt_to_assets", "assets_to_eqt", "dp_assets_to_eqt",
    "ca_to_assets", "nca_to_assets", "tbassets_to_totalassets",
    "int_to_talcap", "eqt_to_talcapital",
    "currentdebt_to_debt", "longdeb_to_debt", "ocf_to_shortdebt",
    "debt_to_eqt", "eqt_to_debt", "eqt_to_interestdebt",
    "tangibleasset_to_debt", "tangasset_to_intdebt", "tangibleasset_to_netdebt",
    "ocf_to_debt", "ocf_to_interestdebt", "ocf_to_netdebt",
    "ebit_to_interest", "longdebt_to_workingcapital", "ebitda_to_debt",
    # misc
    "turn_days", "roa_yearly", "roa_dp",
    "fixed_assets", "profit_prefin_exp", "non_op_profit",
    "op_to_ebt", "nop_to_ebt", "ocf_to_profit",
    "cash_to_liqdebt", "cash_to_liqdebt_withinterest",
    "op_to_liqdebt", "op_to_debt", "roic_yearly", "total_fa_trun",
    "profit_to_op",
    # single-quarter
    "q_opincome", "q_investincome", "q_dtprofit", "q_eps",
    "q_netprofit_margin", "q_gsprofit_margin", "q_exp_to_sales",
    "q_profit_to_gr", "q_saleexp_to_gr", "q_adminexp_to_gr",
    "q_finaexp_to_gr", "q_impair_to_gr_ttm", "q_gc_to_gr",
    "q_op_to_gr", "q_roe", "q_dt_roe", "q_npta",
    "q_opincome_to_ebt", "q_investincome_to_ebt", "q_dtprofit_to_profit",
    "q_salescash_to_or", "q_ocf_to_sales", "q_ocf_to_or",
    # YoY growth
    "basic_eps_yoy", "dt_eps_yoy", "cfps_yoy", "op_yoy",
    "ebt_yoy", "netprofit_yoy", "dt_netprofit_yoy",
    "ocf_yoy", "roe_yoy", "bps_yoy",
    "assets_yoy", "eqt_yoy", "tr_yoy", "or_yoy",
    # single-quarter YoY/QoQ
    "q_gr_yoy", "q_gr_qoq", "q_sales_yoy", "q_sales_qoq",
    "q_op_yoy", "q_op_qoq", "q_profit_yoy", "q_profit_qoq",
    "q_netprofit_yoy", "q_netprofit_qoq",
    # equity & R&D
    "equity_yoy", "rd_exp",
]

FINA_COLUMNS = ["symbol", "end_date", "ann_date", *FINA_NUMERIC, "update_flag"]

_FINA_NUMERIC_DEFS = ",\n    ".join(f"{c:30s} DOUBLE" for c in FINA_NUMERIC)
FINA_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS fina_indicator_quarterly (
    symbol      VARCHAR,
    end_date    VARCHAR,
    ann_date    VARCHAR,
    {_FINA_NUMERIC_DEFS},
    update_flag VARCHAR,
    PRIMARY KEY (symbol, end_date)
)
"""

# ---------------------------------------------------------------------------
# Schema: dividends event table
# ---------------------------------------------------------------------------

DIVIDEND_COLUMNS = [
    "symbol", "end_date", "ann_date", "ex_date", "record_date", "pay_date",
    "cash_div", "cash_div_tax", "stk_div", "stk_bo_rate", "div_proc",
]

DIVIDEND_SCHEMA = """
CREATE TABLE IF NOT EXISTS dividends (
    symbol       VARCHAR,
    end_date     VARCHAR,
    ann_date     VARCHAR,
    ex_date      VARCHAR,
    record_date  VARCHAR,
    pay_date     VARCHAR,
    cash_div     DOUBLE,
    cash_div_tax DOUBLE,
    stk_div      DOUBLE,
    stk_bo_rate  DOUBLE,
    div_proc     VARCHAR,
    PRIMARY KEY (symbol, end_date)
)
"""


class MarketStorage:
    """DuckDB storage for market_daily, fina_indicator_quarterly, dividends."""

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = DB_PATH
        self.conn = duckdb.connect(str(DB_PATH))
        self._init_tables()

    def _init_tables(self):
        self.conn.execute(DAILY_SCHEMA)
        self.conn.execute(FINA_SCHEMA)
        self.conn.execute(DIVIDEND_SCHEMA)
        self._add_double_columns("market_daily", DAILY_COLUMNS[2:])
        self._add_double_columns("fina_indicator_quarterly", FINA_NUMERIC)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        self.conn.close()

    # -- generic helpers --------------------------------------------------

    @contextmanager
    def _registered(self, name: str, df: pd.DataFrame):
        """Register df as a DuckDB view; unregister on exit (success or failure)."""
        self.conn.register(name, df)
        try:
            yield name
        finally:
            self.conn.unregister(name)

    def _add_double_columns(self, table: str, cols):
        """Add any missing DOUBLE columns (idempotent — diff against schema first)."""
        existing = {
            r[0] for r in self.conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ?",
                [table],
            ).fetchall()
        }
        for col in cols:
            if col not in existing:
                self.conn.execute(f'ALTER TABLE {table} ADD COLUMN "{col}" DOUBLE')

    def _max_column(self, table: str, col: str):
        return self.conn.execute(f"SELECT MAX({col}) FROM {table}").fetchone()[0]

    def _distinct_symbols(self, table: str) -> set[str]:
        rows = self.conn.execute(f"SELECT DISTINCT symbol FROM {table}").fetchall()
        return {r[0] for r in rows}

    def _table_stats(self, table: str, *, date_col: str) -> dict:
        row = self.conn.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT symbol), "
            f"MIN({date_col}), MAX({date_col}) FROM {table}"
        ).fetchone()
        return {
            "total_rows": row[0],
            "total_symbols": row[1],
            f"min_{date_col}": row[2],
            f"max_{date_col}": row[3],
        }

    def _upsert(
        self,
        df: pd.DataFrame,
        *,
        table: str,
        pk_cols: tuple[str, ...],
        schema_cols,
    ):
        """INSERT ... ON CONFLICT DO UPDATE, only writing columns present in df."""
        if df.empty:
            return
        cols = [c for c in df.columns if c in schema_cols]
        update_cols = [c for c in cols if c not in pk_cols]
        if not update_cols:
            return  # df has only PK columns; nothing to write

        cols_sql = ", ".join(f'"{c}"' for c in cols)
        update_sql = ", ".join(f'"{c}" = excluded."{c}"' for c in update_cols)
        pk_sql = ", ".join(pk_cols)

        with self._registered(f"_upsert_{table}", df) as view:
            self.conn.execute(f"""
                INSERT INTO {table} ({cols_sql})
                SELECT {cols_sql} FROM {view}
                ON CONFLICT ({pk_sql}) DO UPDATE SET
                    {update_sql}
            """)

    # -- market_daily -----------------------------------------------------

    def get_max_date(self) -> str | None:
        """Return max date in market_daily as YYYYMMDD, or None if empty."""
        result = self.conn.execute("SELECT MAX(date) FROM market_daily").fetchone()
        if result[0]:
            return result[0].strftime("%Y%m%d")
        return None

    def get_existing_dates(self) -> set[str]:
        """Return the set of trade dates already in market_daily as YYYYMMDD strings."""
        rows = self.conn.execute(
            "SELECT DISTINCT date FROM market_daily"
        ).fetchall()
        return {r[0].strftime("%Y%m%d") for r in rows}

    def get_stats(self) -> dict:
        return self._table_stats("market_daily", date_col="date")

    def insert_daily(self, df: pd.DataFrame):
        """UPSERT daily rows. Only columns present in df are written; the rest
        keep their previous values (enables column-by-column backfill)."""
        self._upsert(
            df,
            table="market_daily",
            pk_cols=("date", "symbol"),
            schema_cols=DAILY_COLUMNS,
        )

    # -- fina_indicator ---------------------------------------------------

    def get_max_fina_ann_date(self) -> str | None:
        return self._max_column("fina_indicator_quarterly", "ann_date")

    def get_symbols_in_fina(self) -> set[str]:
        return self._distinct_symbols("fina_indicator_quarterly")

    def get_fina_stats(self) -> dict:
        return self._table_stats("fina_indicator_quarterly", date_col="ann_date")

    def insert_fina(self, df: pd.DataFrame):
        self._upsert(
            df,
            table="fina_indicator_quarterly",
            pk_cols=("symbol", "end_date"),
            schema_cols=FINA_COLUMNS,
        )

    # -- dividends --------------------------------------------------------

    def get_max_dividend_ann_date(self) -> str | None:
        return self._max_column("dividends", "ann_date")

    def get_symbols_in_dividends(self) -> set[str]:
        return self._distinct_symbols("dividends")

    def get_dividend_stats(self) -> dict:
        return self._table_stats("dividends", date_col="ann_date")

    def insert_dividends(self, df: pd.DataFrame):
        self._upsert(
            df,
            table="dividends",
            pk_cols=("symbol", "end_date"),
            schema_cols=DIVIDEND_COLUMNS,
        )
