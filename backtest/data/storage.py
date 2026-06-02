"""DuckDB storage for market daily data, fundamental statements, and dividends."""

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
    # -- margin trading detail (pro.margin_detail) -----------------------------
    "margin_rzye",
    "margin_rqye",
    "margin_rzmre",
    "margin_rqyl",
    "margin_rzche",
    "margin_rqchl",
    "margin_rqmcl",
    "margin_rzrqye",
    # -- capital flow (pro.moneyflow) ------------------------------------------
    "mf_buy_sm_vol",
    "mf_buy_sm_amount",
    "mf_sell_sm_vol",
    "mf_sell_sm_amount",
    "mf_buy_md_vol",
    "mf_buy_md_amount",
    "mf_sell_md_vol",
    "mf_sell_md_amount",
    "mf_buy_lg_vol",
    "mf_buy_lg_amount",
    "mf_sell_lg_vol",
    "mf_sell_lg_amount",
    "mf_buy_elg_vol",
    "mf_buy_elg_amount",
    "mf_sell_elg_vol",
    "mf_sell_elg_amount",
    "mf_net_mf_vol",
    "mf_net_mf_amount",
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
# Schema: income_q / balancesheet_q / cashflow_q (three independent tables)
# ---------------------------------------------------------------------------
# Physical tables keep Tushare raw column names (no prefix at storage).
#   Prefixes (inc_/bs_/cf_) are added only by get_fina_snapshot() on read.
# PK includes update_flag because Tushare sometimes returns both
#   update_flag=0 and update_flag=1 for the same (symbol, end_date, f_ann_date).
# ---------------------------------------------------------------------------

INCOME_NUMERIC = [
    "basic_eps", "diluted_eps", "total_revenue", "revenue", "int_income",
    "prem_earned", "comm_income", "n_commis_income", "n_oth_income", "n_oth_b_income",
    "prem_income", "out_prem", "une_prem_reser", "reins_income", "n_sec_tb_income",
    "n_sec_uw_income", "n_asset_mg_income", "oth_b_income", "fv_value_chg_gain", "invest_income",
    "ass_invest_income", "forex_gain", "total_cogs", "oper_cost", "int_exp",
    "comm_exp", "biz_tax_surchg", "sell_exp", "admin_exp", "fin_exp",
    "assets_impair_loss", "prem_refund", "compens_payout", "reser_insur_liab", "div_payt",
    "reins_exp", "oper_exp", "compens_payout_refu", "insur_reser_refu", "reins_cost_refund",
    "other_bus_cost", "operate_profit", "non_oper_income", "non_oper_exp", "nca_disploss",
    "total_profit", "income_tax", "n_income", "n_income_attr_p", "minority_gain",
    "oth_compr_income", "t_compr_income", "compr_inc_attr_p", "compr_inc_attr_m_s", "ebit",
    "ebitda", "insurance_exp", "undist_profit", "distable_profit", "rd_exp",
    "fin_exp_int_exp", "fin_exp_int_inc", "transfer_surplus_rese", "transfer_housing_imprest", "transfer_oth",
    "adj_lossgain", "withdra_legal_surplus", "withdra_legal_pubfund", "withdra_biz_devfund", "withdra_rese_fund",
    "withdra_oth_ersu", "workers_welfare", "distr_profit_shrhder", "prfshare_payable_dvd", "comshare_payable_dvd",
    "capit_comstock_div", "continued_net_profit",
]

BALANCESHEET_NUMERIC = [
    "total_share", "cap_rese", "undistr_porfit", "surplus_rese", "special_rese",
    "money_cap", "trad_asset", "notes_receiv", "accounts_receiv", "oth_receiv",
    "prepayment", "div_receiv", "int_receiv", "inventories", "amor_exp",
    "nca_within_1y", "sett_rsrv", "loanto_oth_bank_fi", "premium_receiv", "reinsur_receiv",
    "reinsur_res_receiv", "pur_resale_fa", "oth_cur_assets", "total_cur_assets", "fa_avail_for_sale",
    "htm_invest", "lt_eqt_invest", "invest_real_estate", "time_deposits", "oth_assets",
    "lt_rec", "fix_assets", "cip", "const_materials", "fixed_assets_disp",
    "produc_bio_assets", "oil_and_gas_assets", "intan_assets", "r_and_d", "goodwill",
    "lt_amor_exp", "defer_tax_assets", "decr_in_disbur", "oth_nca", "total_nca",
    "cash_reser_cb", "depos_in_oth_bfi", "prec_metals", "deriv_assets", "rr_reins_une_prem",
    "rr_reins_outstd_cla", "rr_reins_lins_liab", "rr_reins_lthins_liab", "refund_depos", "ph_pledge_loans",
    "refund_cap_depos", "indep_acct_assets", "client_depos", "client_prov", "transac_seat_fee",
    "invest_as_receiv", "total_assets", "lt_borr", "st_borr", "cb_borr",
    "depos_ib_deposits", "loan_oth_bank", "trading_fl", "notes_payable", "acct_payable",
    "adv_receipts", "sold_for_repur_fa", "comm_payable", "payroll_payable", "taxes_payable",
    "int_payable", "div_payable", "oth_payable", "acc_exp", "deferred_inc",
    "st_bonds_payable", "payable_to_reinsurer", "rsrv_insur_cont", "acting_trading_sec", "acting_uw_sec",
    "non_cur_liab_due_1y", "oth_cur_liab", "total_cur_liab", "bond_payable", "lt_payable",
    "specific_payables", "estimated_liab", "defer_tax_liab", "defer_inc_non_cur_liab", "oth_ncl",
    "total_ncl", "depos_oth_bfi", "deriv_liab", "depos", "agency_bus_liab",
    "oth_liab", "prem_receiv_adva", "depos_received", "ph_invest", "reser_une_prem",
    "reser_outstd_claims", "reser_lins_liab", "reser_lthins_liab", "indept_acc_liab", "pledge_borr",
    "indem_payable", "policy_div_payable", "total_liab", "treasury_share", "ordin_risk_reser",
    "forex_differ", "invest_loss_unconf", "minority_int", "total_hldr_eqy_exc_min_int", "total_hldr_eqy_inc_min_int",
    "total_liab_hldr_eqy", "lt_payroll_payable", "oth_comp_income", "oth_eqt_tools", "oth_eqt_tools_p_shr",
    "lending_funds", "acc_receivable", "st_fin_payable", "payables", "hfs_assets",
    "hfs_sales", "cost_fin_assets", "fair_value_fin_assets", "contract_assets", "contract_liab",
    "accounts_receiv_bill", "accounts_pay", "oth_rcv_total", "fix_assets_total", "cip_total",
    "oth_pay_total", "long_pay_total", "debt_invest", "oth_debt_invest",
]

CASHFLOW_NUMERIC = [
    "net_profit", "finan_exp", "c_fr_sale_sg", "recp_tax_rends", "n_depos_incr_fi",
    "n_incr_loans_cb", "n_inc_borr_oth_fi", "prem_fr_orig_contr", "n_incr_insured_dep", "n_reinsur_prem",
    "n_incr_disp_tfa", "ifc_cash_incr", "n_incr_disp_faas", "n_incr_loans_oth_bank", "n_cap_incr_repur",
    "c_fr_oth_operate_a", "c_inf_fr_operate_a", "c_paid_goods_s", "c_paid_to_for_empl", "c_paid_for_taxes",
    "n_incr_clt_loan_adv", "n_incr_dep_cbob", "c_pay_claims_orig_inco", "pay_handling_chrg", "pay_comm_insur_plcy",
    "oth_cash_pay_oper_act", "st_cash_out_act", "n_cashflow_act", "oth_recp_ral_inv_act", "c_disp_withdrwl_invest",
    "c_recp_return_invest", "n_recp_disp_fiolta", "n_recp_disp_sobu", "stot_inflows_inv_act", "c_pay_acq_const_fiolta",
    "c_paid_invest", "n_disp_subs_oth_biz", "oth_pay_ral_inv_act", "n_incr_pledge_loan", "stot_out_inv_act",
    "n_cashflow_inv_act", "c_recp_borrow", "proc_issue_bonds", "oth_cash_recp_ral_fnc_act", "stot_cash_in_fnc_act",
    "free_cashflow", "c_prepay_amt_borr", "c_pay_dist_dpcp_int_exp", "incl_dvd_profit_paid_sc_ms", "oth_cashpay_ral_fnc_act",
    "stot_cashout_fnc_act", "n_cash_flows_fnc_act", "eff_fx_flu_cash", "n_incr_cash_cash_equ", "c_cash_equ_beg_period",
    "c_cash_equ_end_period", "c_recp_cap_contrib", "incl_cash_rec_saims", "uncon_invest_loss", "prov_depr_assets",
    "depr_fa_coga_dpba", "amort_intang_assets", "lt_amort_deferred_exp", "decr_deferred_exp", "incr_acc_exp",
    "loss_disp_fiolta", "loss_scr_fa", "loss_fv_chg", "invest_loss", "decr_def_inc_tax_assets",
    "incr_def_inc_tax_liab", "decr_inventories", "decr_oper_payable", "incr_oper_payable", "others",
    "im_net_cashflow_oper_act", "conv_debt_into_cap", "conv_copbonds_due_within_1y", "fa_fnc_leases", "im_n_incr_cash_equ",
    "net_dism_capital_add", "net_cash_rece_sec", "credit_impa_loss", "use_right_asset_dep", "oth_loss_asset",
    "end_bal_cash", "beg_bal_cash", "end_bal_cash_equ", "beg_bal_cash_equ",
]


FUNDAMENTAL_META = [
    "symbol", "end_date", "ann_date", "f_ann_date",
    "report_type", "comp_type", "end_type", "update_flag",
]

FUNDAMENTAL_PK = ("symbol", "end_date", "f_ann_date", "update_flag", "report_type")
FUNDAMENTAL_TABLES = ("income_q", "balancesheet_q", "cashflow_q")
_FINA_JOIN_KEYS = ("symbol", "end_date")

_FUNDAMENTAL_COLS_MAP = {
    "income_q": FUNDAMENTAL_META + INCOME_NUMERIC,
    "balancesheet_q": FUNDAMENTAL_META + BALANCESHEET_NUMERIC,
    "cashflow_q": FUNDAMENTAL_META + CASHFLOW_NUMERIC,
}


def _build_fundamental_schema(name: str, numeric_cols: list[str]) -> str:
    numeric_defs = ",\n    ".join(f"{c:30s} DOUBLE" for c in numeric_cols)
    pk_sql = ", ".join(FUNDAMENTAL_PK)
    return f"""
CREATE TABLE IF NOT EXISTS {name} (
    symbol      VARCHAR,
    end_date    VARCHAR,
    ann_date    VARCHAR,
    f_ann_date  VARCHAR,
    report_type VARCHAR,
    comp_type   VARCHAR,
    end_type    VARCHAR,
    update_flag VARCHAR,
    {numeric_defs},
    PRIMARY KEY ({pk_sql})
)
"""


INCOME_SCHEMA = _build_fundamental_schema("income_q", INCOME_NUMERIC)
BALANCESHEET_SCHEMA = _build_fundamental_schema("balancesheet_q", BALANCESHEET_NUMERIC)
CASHFLOW_SCHEMA = _build_fundamental_schema("cashflow_q", CASHFLOW_NUMERIC)

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
    PRIMARY KEY (symbol, end_date, ann_date, ex_date)
)
"""

# ---------------------------------------------------------------------------
# Schema: index_daily (benchmark indices: 000300.SH, 000905.SH, ...)
# ---------------------------------------------------------------------------

INDEX_DAILY_COLUMNS = [
    "date", "symbol",
    "open", "high", "low", "close",
    "pre_close", "change", "pct_chg",
    "volume", "amount",
]

INDEX_DAILY_SCHEMA = """
CREATE TABLE IF NOT EXISTS index_daily (
    date       DATE,
    symbol     VARCHAR,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    pre_close  DOUBLE,
    change     DOUBLE,
    pct_chg    DOUBLE,
    volume     DOUBLE,
    amount     DOUBLE,
    PRIMARY KEY (date, symbol)
)
"""

# ---------------------------------------------------------------------------
# Schema: sw_industry (申万行业归属历史, SW2021 体系)
# ---------------------------------------------------------------------------
# 同一 (symbol, level) 在不同时段可能属于不同行业,也可能多次进出同一行业。
# PK (symbol, level, industry_code, in_date) 唯一区分每段所属;
# out_date IS NULL 表示截至最新数据日仍在该行业。
# 数据源: pro.index_classify (拿 industry_code → industry_name 映射)
#         pro.index_member   (拿成分股历史,含 in_date / out_date)
# ---------------------------------------------------------------------------

SW_INDUSTRY_COLUMNS = [
    "symbol", "level", "industry_code", "industry_name", "in_date", "out_date",
]

SW_INDUSTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS sw_industry (
    symbol        VARCHAR,
    level         VARCHAR,
    industry_code VARCHAR,
    industry_name VARCHAR,
    in_date       DATE,
    out_date      DATE,
    PRIMARY KEY (symbol, level, industry_code, in_date)
)
"""

# ---------------------------------------------------------------------------
# Schema: index_members (宽基指数成分股, 已密集化到每个交易日)
# ---------------------------------------------------------------------------
# 数据源 pro.index_weight 是月度快照(每月一次再平衡日发布权重)。回填时
# 把月度快照展开到下次发布日前的每个交易日,日期等值查询直接可用,
# 不需要 as-of 逻辑。
# PK (index_code, symbol, trade_date) 唯一识别"某只股票在 D 日属于哪只指数"。
# ---------------------------------------------------------------------------

INDEX_MEMBERS_COLUMNS = ["index_code", "symbol", "trade_date", "weight"]

INDEX_MEMBERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS index_members (
    index_code  VARCHAR,
    symbol      VARCHAR,
    trade_date  DATE,
    weight      DOUBLE,
    PRIMARY KEY (index_code, symbol, trade_date)
)
"""

# ---------------------------------------------------------------------------
# Schema: trade_calendar (交易日历,含周/月首末预计算标志)
# ---------------------------------------------------------------------------

TRADE_CALENDAR_SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_calendar (
    cal_date       DATE PRIMARY KEY,
    is_open        BOOLEAN NOT NULL,
    is_week_first  BOOLEAN NOT NULL,
    is_week_last   BOOLEAN NOT NULL,
    is_month_first BOOLEAN NOT NULL,
    is_month_last  BOOLEAN NOT NULL
)
"""

TRADE_CALENDAR_COLUMNS = [
    "cal_date", "is_open", "is_week_first",
    "is_week_last", "is_month_first", "is_month_last",
]


class MarketStorage:
    """DuckDB storage for market_daily, fundamental tables, and dividends."""

    def __init__(self, *, read_only: bool = False):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = DB_PATH
        self.conn = duckdb.connect(str(DB_PATH), read_only=read_only)
        if not read_only:
            # Cap DuckDB working set + spill to disk on overshoot. Default
            # is ≈ 80% of RAM which lets range-joins OOM before spilling.
            self.conn.execute("PRAGMA memory_limit='6GB'")
            # temp_directory is database-level and not re-settable in the
            # same process; a second MarketStorage in the same run inherits
            # whatever the first set.
            try:
                self.conn.execute("PRAGMA temp_directory='/tmp/duckdb_spill'")
            except duckdb.NotImplementedException:
                pass
            self._init_tables()

    def _init_tables(self):
        self.conn.execute(DAILY_SCHEMA)
        self._drop_fundamental_tables_if_legacy_pk()
        self._drop_dividends_if_legacy_pk()
        self.conn.execute(INCOME_SCHEMA)
        self.conn.execute(BALANCESHEET_SCHEMA)
        self.conn.execute(CASHFLOW_SCHEMA)
        self.conn.execute(DIVIDEND_SCHEMA)
        self.conn.execute(INDEX_DAILY_SCHEMA)
        self.conn.execute(SW_INDUSTRY_SCHEMA)
        self.conn.execute(INDEX_MEMBERS_SCHEMA)
        self.conn.execute(TRADE_CALENDAR_SCHEMA)
        self._add_double_columns("market_daily", DAILY_COLUMNS[2:])
        for table in FUNDAMENTAL_TABLES:
            self._add_double_columns(table, _FUNDAMENTAL_COLS_MAP[table][len(FUNDAMENTAL_META):])

    def _drop_dividends_if_legacy_pk(self):
        """Drop legacy dividends table whose PK is only (symbol, end_date).

        Required because DuckDB has no ``ALTER PRIMARY KEY`` — switching from
        the 2-column PK to 4 columns needs a rebuild.
        Callers must re-run backfill afterwards.
        """
        rows = self.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = 'dividends'"
        ).fetchall()
        if not rows:
            return  # CREATE TABLE below handles fresh DBs
        info = self.conn.execute("PRAGMA table_info('dividends')").fetchall()
        pk_cols = {r[1] for r in info if r[5]}  # col name where pk flag > 0
        if len(pk_cols) == 2 and pk_cols == {"symbol", "end_date"}:
            print(
                "[storage] legacy PK on dividends (symbol, end_date); "
                "dropping — re-run backfill to repopulate"
            )
            self.conn.execute("DROP TABLE dividends")

    def _drop_fundamental_tables_if_legacy_pk(self):
        """Drop legacy fundamental tables whose PK lacks ``report_type``.

        Required because DuckDB has no ``ALTER PRIMARY KEY`` — switching from
        the 4-column PK to ``FUNDAMENTAL_PK`` (5 columns) needs a rebuild.
        Callers must re-run backfill afterwards.
        """
        existing = {
            r[0] for r in self.conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_name IN ('income_q','balancesheet_q','cashflow_q')"
            ).fetchall()
        }
        for table in FUNDAMENTAL_TABLES:
            if table not in existing:
                continue  # CREATE TABLE below handles fresh DBs
            rows = self.conn.execute(f"PRAGMA table_info('{table}')").fetchall()
            pk_cols = {r[1] for r in rows if r[5]}  # col name where pk flag > 0
            if "report_type" not in pk_cols:
                print(
                    f"[storage] legacy PK on {table} (missing report_type); "
                    f"dropping — re-run backfill to repopulate"
                )
                self.conn.execute(f"DROP TABLE {table}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        self.conn.close()

    # -- generic helpers ------------------------------------------------------

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

    # -- market_daily ---------------------------------------------------------

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

    def get_panel(self, date: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Return a cross-section of market_daily for a single trade date.

        *date* is a YYYYMMDD string.  If *columns* is None all columns are
        returned except the PK pair (date, symbol).
        """
        if columns:
            cols_sql = ", ".join(f'"{c}"' for c in columns if c in DAILY_COLUMNS)
        else:
            cols_sql = ", ".join(f'"{c}"' for c in DAILY_COLUMNS if c not in ("date", "symbol"))
        sql = f"""
            SELECT date, symbol, {cols_sql}
            FROM market_daily
            WHERE date = strptime(?, '%Y%m%d')::DATE
            ORDER BY symbol
        """
        return self.conn.execute(sql, [date]).fetchdf()

    def get_bars(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return time-series bars from market_daily.

        Parameters
        ----------
        symbols : list[str] | None
            Filter to these symbols.  If None, all symbols.
        start, end : str | None
            YYYYMMDD inclusive bounds.  If None, no bound on that side.
        columns : list[str] | None
            Subset of columns to return.  If None, all non-PK columns.
        """
        if columns:
            cols_sql = ", ".join(f'"{c}"' for c in columns if c in DAILY_COLUMNS)
        else:
            cols_sql = ", ".join(f'"{c}"' for c in DAILY_COLUMNS if c not in ("date", "symbol"))

        conditions = []
        params: list = []
        if start:
            conditions.append("date >= strptime(?, '%Y%m%d')::DATE")
            params.append(start)
        if end:
            conditions.append("date <= strptime(?, '%Y%m%d')::DATE")
            params.append(end)
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT date, symbol, {cols_sql}
            FROM market_daily
            {where_clause}
            ORDER BY date, symbol
        """
        return self.conn.execute(sql, params).fetchdf()

    # -- fundamentals (three independent tables) ------------------------------

    # Thin wrappers around generic helpers; table name is the only variable.

    def _fundamental_pk(self) -> tuple[str, ...]:
        return FUNDAMENTAL_PK

    def _fundamental_cols(self, table: str) -> list[str]:
        return _FUNDAMENTAL_COLS_MAP[table]

    def insert_income(self, df: pd.DataFrame):
        self._upsert(df, table="income_q", pk_cols=self._fundamental_pk(), schema_cols=self._fundamental_cols("income_q"))

    def insert_balancesheet(self, df: pd.DataFrame):
        self._upsert(df, table="balancesheet_q", pk_cols=self._fundamental_pk(), schema_cols=self._fundamental_cols("balancesheet_q"))

    def insert_cashflow(self, df: pd.DataFrame):
        self._upsert(df, table="cashflow_q", pk_cols=self._fundamental_pk(), schema_cols=self._fundamental_cols("cashflow_q"))

    def get_max_f_ann_date(self, table: str) -> str | None:
        return self._max_column(table, "f_ann_date")

    def get_symbols_in_fundamentals(self, table: str) -> set[str]:
        return self._distinct_symbols(table)

    def get_fundamentals_stats(self, table: str) -> dict:
        return self._table_stats(table, date_col="f_ann_date")

    def get_fina_snapshot(
        self,
        as_of_date: str,
        symbols: list[str] | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """D 日财报 wide 快照(PIT 安全)。

        对 income_q / balancesheet_q / cashflow_q 三张表分别做
        ``WHERE f_ann_date <= D`` + ``QUALIFY`` 取每个 (symbol, end_date) 的最新可见版本,
        再按 ``(symbol, end_date)`` outer-join 成 wide DataFrame。

        非 join-key 列(包括 ``ann_date / f_ann_date / report_type / comp_type /
        end_type / update_flag`` 等 meta 与数值列)自动加 ``inc_/bs_/cf_`` 前缀。
        三表 outer-join 仅依据 ``(symbol, end_date)``,因为多 ``report_type`` 共存下
        三表各自的 meta 不必相同(例如 inc 取到 type=4 而 bs 仍是 type=1)。

        ``columns`` 按 ``inc_/bs_/cf_`` 前缀指定要返回的数值列,推到 SQL 层只读
        必要列(避免每个快照都加载 330 列)。``end_date`` 永远保留。
        """
        tables = {
            "inc": "income_q",
            "bs": "balancesheet_q",
            "cf": "cashflow_q",
        }

        # Decide per-table column whitelist from the requested prefixed names.
        # If columns is None we keep the SELECT * behaviour for back-compat.
        per_table_cols: dict[str, list[str] | None] = {}
        if columns:
            for prefix in tables:
                wanted = [
                    c.removeprefix(f"{prefix}_") for c in columns
                    if c.startswith(f"{prefix}_")
                ]
                # Always need the join keys and end_date is part of them already.
                # f_ann_date / update_flag are needed for the QUALIFY; we pull
                # them then drop afterwards if the caller didn't ask.
                per_table_cols[prefix] = wanted
        else:
            for prefix in tables:
                per_table_cols[prefix] = None

        symbol_filter = ""
        params = [as_of_date]
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            symbol_filter = f"AND symbol IN ({placeholders})"
            params.extend(symbols)

        join_keys = set(_FINA_JOIN_KEYS)
        dfs: dict[str, pd.DataFrame] = {}
        for prefix, table in tables.items():
            wanted = per_table_cols[prefix]
            # Skip tables the caller didn't request any columns from.
            if wanted == []:
                dfs[prefix] = pd.DataFrame()
                continue
            if wanted is None:
                select_clause = "*"
            else:
                # Always include join keys + QUALIFY-required columns.
                # f_ann_date and update_flag drive the row-number window.
                needed = list(_FINA_JOIN_KEYS) + ["f_ann_date", "update_flag"] + wanted
                # Dedupe while preserving order.
                seen: set[str] = set()
                kept: list[str] = []
                for c in needed:
                    if c not in seen:
                        kept.append(c)
                        seen.add(c)
                select_clause = ", ".join(f'"{c}"' for c in kept)
            sql = f"""
                SELECT {select_clause}
                FROM {table}
                WHERE f_ann_date <= ? {symbol_filter}
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY symbol, end_date
                    ORDER BY f_ann_date DESC, update_flag DESC
                ) = 1
            """
            df = self.conn.execute(sql, params).fetchdf()
            if df.empty:
                dfs[prefix] = df
                continue
            rename_map = {c: f"{prefix}_{c}" for c in df.columns if c not in join_keys}
            df = df.rename(columns=rename_map)
            dfs[prefix] = df

        merged = dfs["inc"]
        for prefix in ("bs", "cf"):
            if merged.empty:
                merged = dfs[prefix]
            elif dfs[prefix].empty:
                continue
            else:
                merged = merged.merge(
                    dfs[prefix],
                    on=list(_FINA_JOIN_KEYS),
                    how="outer",
                )

        if columns:
            keep = list(_FINA_JOIN_KEYS)
            keep += [c for c in columns if c in merged.columns]
            merged = merged[[c for c in keep if c in merged.columns]].copy()
            # Backfill any requested columns that weren't present (e.g. early-
            # period queries with no balancesheet rows yet) as all-NaN. Callers
            # rely on stable column shape.
            for c in columns:
                if c not in merged.columns:
                    merged[c] = pd.NA

        return merged

    def get_fina_snapshot_range(
        self,
        start: str,
        end: str,
        symbols: list[str] | None = None,
        columns: list[str] | None = None,
        last_n_quarters: int | None = None,
        delay: int = 0,
    ) -> pd.DataFrame:
        """Batched PIT snapshots for every trade date in ``[start, end]``.

        Semantically equivalent to::

            for D in trade_dates(start, end):
                yield get_fina_snapshot(D, ...)

        but executed as **one range-join SQL per table per chunk** instead of
        ``N_dates × 3`` independent QUALIFY scans. The trick:

        1. For each ``(symbol, end_date)`` group compute ``superseded_at =
           LEAD(f_ann_date)`` — the next version's publication date — once.
        2. Range-join against ``trade_calendar``: a version row ``v`` is the
           D-day-visible-latest iff ``v.f_ann_date + delay <= D < v.superseded_at + delay``
           (NULL ``superseded_at`` means "still current").

        ``delay`` (default 0) shifts the visibility window forward by *delay*
        calendar days.  For A-share factors this should be set to 1 so that a
        report announced on ``f_ann_date`` only becomes visible on the **next**
        trade date, matching the real-world timing: announcement after market
        close → usable the following day.

        ``last_n_quarters`` (if given) caps to the most recent N ``end_date``
        rows per ``(date, symbol)``. Slope / TTM factors only need the last
        ~20 quarters; trimming serves two purposes:

        - **Memory**: the version CTE pre-filters ``end_date >= start −
          ceil(N/4) years`` so the range join doesn't fan out across 35
          years of history when only the recent 5 are needed.
        - **Output size**: a QUALIFY trims final rows to top-N per
          ``(date, symbol)``.

        Internally the date range is split into half-year sub-windows so a
        single range-join's intermediate (``N_dates × N_versions`` rows
        before QUALIFY trims) stays under DuckDB's memory limit.

        Output columns: ``date, symbol, end_date, <requested cols...>``
        (long format, one row per ``(date, symbol, end_date)``). The three
        fundamental tables are still outer-joined on ``(symbol, end_date)``
        so a single row carries ``inc_/bs_/cf_`` values aligned. ``date`` is
        a ``DATE`` column ready for downstream merge with ``market_daily``.
        """
        from datetime import datetime as _dt, timedelta as _td

        _CHUNK_MONTHS = 6

        start_dt = _dt.strptime(start, "%Y%m%d")
        end_dt = _dt.strptime(end, "%Y%m%d")
        pieces: list[pd.DataFrame] = []
        cur = start_dt
        while cur <= end_dt:
            # Walk forward _CHUNK_MONTHS months keeping the day at 1 then
            # subtract 1 day for the inclusive end. Avoids dateutil dep.
            next_year = cur.year + (cur.month - 1 + _CHUNK_MONTHS) // 12
            next_month = (cur.month - 1 + _CHUNK_MONTHS) % 12 + 1
            sub_end = _dt(next_year, next_month, 1) - _td(days=1)
            if sub_end > end_dt:
                sub_end = end_dt
            sub = self._get_fina_snapshot_window(
                cur.strftime("%Y%m%d"),
                sub_end.strftime("%Y%m%d"),
                symbols=symbols,
                columns=columns,
                last_n_quarters=last_n_quarters,
                delay=delay,
            )
            if not sub.empty:
                pieces.append(sub)
            cur = sub_end + _td(days=1)

        if not pieces:
            # Maintain the stable-column contract even on empty results.
            cols = ["date", "symbol", "end_date"] + (list(columns) if columns else [])
            return pd.DataFrame(columns=cols)
        return pd.concat(pieces, ignore_index=True)

    def _get_fina_snapshot_window(
        self,
        start: str,
        end: str,
        symbols: list[str] | None = None,
        columns: list[str] | None = None,
        last_n_quarters: int | None = None,
        delay: int = 0,
    ) -> pd.DataFrame:
        """One range-join SQL per fina table for the ``[start, end]`` window.

        Caller (``get_fina_snapshot_range``) sub-divides long ranges to keep
        the per-window intermediate manageable. This function itself runs a
        single range-join per table — three SQL round trips total — and
        outer-joins their results on ``(date, symbol, end_date)``.
        """
        tables = {
            "inc": "income_q",
            "bs": "balancesheet_q",
            "cf": "cashflow_q",
        }

        # Pre-filter version table by end_date when last_n_quarters is set.
        # 20 quarters back = 5 years; add a 1-year cushion for late-published
        # versions whose end_date is years before f_ann_date.
        end_date_floor_sql = ""
        if last_n_quarters is not None:
            lookback_years = int(last_n_quarters / 4) + 2
            start_year = int(start[:4])
            floor_year = max(start_year - lookback_years, 1990)
            end_date_floor_sql = f"end_date >= '{floor_year}0101'"

        per_table_cols: dict[str, list[str] | None] = {}
        if columns:
            for prefix in tables:
                per_table_cols[prefix] = [
                    c.removeprefix(f"{prefix}_") for c in columns
                    if c.startswith(f"{prefix}_")
                ]
        else:
            for prefix in tables:
                per_table_cols[prefix] = None

        symbol_clause = ""
        symbol_params: list[str] = []
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            symbol_clause = f"symbol IN ({placeholders})"
            symbol_params = list(symbols)

        where_parts = [p for p in (symbol_clause, end_date_floor_sql) if p]
        where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        join_keys = set(_FINA_JOIN_KEYS)
        dfs: dict[str, pd.DataFrame] = {}
        for prefix, table in tables.items():
            wanted = per_table_cols[prefix]
            # Skip tables the caller didn't request any columns from —
            # avoids the SQL + outer-join cost when only one of inc/bs/cf
            # is needed.
            if wanted == []:
                dfs[prefix] = pd.DataFrame()
                continue
            if wanted is None:
                version_select = "*"
            else:
                needed = (
                    list(_FINA_JOIN_KEYS) + ["f_ann_date", "update_flag"] + wanted
                )
                seen: set[str] = set()
                kept: list[str] = []
                for c in needed:
                    if c not in seen:
                        kept.append(c)
                        seen.add(c)
                version_select = ", ".join(f'"{c}"' for c in kept)

            sql = f"""
                WITH versions AS (
                    SELECT {version_select},
                        LEAD(f_ann_date) OVER (
                            PARTITION BY symbol, end_date
                            ORDER BY f_ann_date ASC, update_flag ASC
                        ) AS superseded_at
                    FROM {table}
                    {where_clause}
                )
                SELECT td.cal_date AS date, v.*
                FROM versions v
                JOIN trade_calendar td
                  ON td.cal_date >= strptime(v.f_ann_date, '%Y%m%d')::DATE + INTERVAL '{delay}' DAY
                 AND (v.superseded_at IS NULL
                      OR td.cal_date < strptime(v.superseded_at, '%Y%m%d')::DATE + INTERVAL '{delay}' DAY)
                 AND td.cal_date BETWEEN strptime(?, '%Y%m%d')::DATE
                                     AND strptime(?, '%Y%m%d')::DATE
                 AND td.is_open
            """
            if last_n_quarters is not None:
                sql += f"""
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY td.cal_date, v.symbol
                    ORDER BY v.end_date DESC
                ) <= {int(last_n_quarters)}
                """
            params = list(symbol_params) + [start, end]
            df = self.conn.execute(sql, params).fetchdf()
            if df.empty:
                dfs[prefix] = df
                continue
            if "superseded_at" in df.columns:
                df = df.drop(columns=["superseded_at"])
            rename_map = {
                c: f"{prefix}_{c}" for c in df.columns
                if c not in join_keys and c != "date"
            }
            df = df.rename(columns=rename_map)
            dfs[prefix] = df

        merge_keys = ["date", "symbol", "end_date"]
        merged = dfs["inc"]
        for prefix in ("bs", "cf"):
            other = dfs[prefix]
            if merged.empty:
                merged = other
            elif other.empty:
                continue
            else:
                merged = merged.merge(other, on=merge_keys, how="outer")

        if columns:
            keep = list(merge_keys) + [c for c in columns if c in merged.columns]
            merged = merged[[c for c in keep if c in merged.columns]].copy()
            for c in columns:
                if c not in merged.columns:
                    merged[c] = pd.NA

        return merged

    def get_fina_event_panel(
        self,
        start: str,
        end: str,
        columns: list[str],
        last_n_quarters: int = 20,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Per-(symbol, end_date) event panel for event-driven fina factors.

        Replaces the per-trade-date materialisation pattern used by
        :meth:`get_fina_snapshot_range` for factors whose value only
        changes when a new announcement supersedes the previous one
        (Growth EGRO, Quality AGRO/ROA/GP). The naive per-(date, symbol)
        panel duplicates the same N-quarter history once per trade date
        — 60× output amplification on a daily backtest universe — so TTM
        / slope are re-computed redundantly.

        Returns a long-format frame with one row per ``(symbol,
        announce_end_date, history_end_date)``:

        ============     ===========================================
        column           meaning
        ============     ===========================================
        symbol           A-share ticker
        announce_end_date  end_date of the **announcement** (event)
        f_ann_date       publication date of this announcement (PIT)
        next_f_ann_date  publication date of the next announcement
                         for this symbol (or NULL); together with
                         ``f_ann_date`` defines the trade-date
                         interval this view is current for
        end_date         end_date of one historical quarter visible
                         at this announcement (≤ announce_end_date)
        <fina cols>      numeric values at end_date
        ============     ===========================================

        Downstream usage::

            df.groupby(['symbol', 'announce_end_date']) gives one group
            per event, each carrying the last_n_quarters history rows.
            Compute per-event scalar (TTM, slope, ratio) once on the
            ~22k events/year, then expand_events_to_dates to materialise
            (date, symbol, value) at full 1.3M-row resolution.

        Currently single-table: ``columns`` must all share one prefix
        (``inc_`` / ``bs_`` / ``cf_``). Multi-table composites (Quality)
        call this once per source table and merge on
        ``(symbol, announce_end_date, end_date)``.
        """
        if not columns:
            raise ValueError("get_fina_event_panel requires explicit columns")

        prefix_table = {
            "inc_": "income_q", "bs_": "balancesheet_q", "cf_": "cashflow_q",
        }
        prefixes = {c.split("_", 1)[0] + "_" for c in columns}
        if len(prefixes) != 1:
            raise ValueError(
                "get_fina_event_panel requires all columns from one table; "
                f"got prefixes {sorted(prefixes)}"
            )
        prefix = next(iter(prefixes))
        if prefix not in prefix_table:
            raise ValueError(f"Unknown column prefix {prefix!r}")
        table = prefix_table[prefix]
        bare_cols = [c.removeprefix(prefix) for c in columns]

        # Quarters-of-history floor so the version CTE stays small. 20q
        # ≈ 5y back; +2y cushion for late restatements with old end_dates.
        lookback_years = int(last_n_quarters / 4) + 2
        floor_year = max(int(start[:4]) - lookback_years, 1990)

        symbol_clause = ""
        symbol_params: list[str] = []
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            symbol_clause = f"AND symbol IN ({placeholders})"
            symbol_params = list(symbols)

        col_select = ", ".join(f'"{c}"' for c in bare_cols)
        v_col_select = ", ".join(f'v."{c}"' for c in bare_cols)

        # Two-step plan:
        # 1) `latest` = newest visible version per (symbol, end_date)
        #    (the QUALIFY ROW_NUMBER pattern from get_fina_snapshot).
        # 2) `events` = the announcements whose [f_ann_date, next) interval
        #    overlaps [start, end], with `next_f_ann_date` computed as
        #    LEAD(f_ann_date) over the per-symbol announcement sequence.
        # 3) Cross-join each event with the up-to-N preceding latest
        #    versions (history) on (symbol, end_date <= announce_end_date),
        #    QUALIFY ROW_NUMBER <= N to cap.
        sql = f"""
            WITH latest AS (
                SELECT symbol, end_date, f_ann_date, update_flag, {col_select}
                FROM {table}
                WHERE end_date >= '{floor_year}0101' {symbol_clause}
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY symbol, end_date
                    ORDER BY f_ann_date DESC, update_flag DESC
                ) = 1
            ),
            ordered AS (
                SELECT symbol, end_date AS announce_end_date,
                       f_ann_date,
                       LEAD(f_ann_date) OVER (
                           PARTITION BY symbol
                           ORDER BY f_ann_date ASC, end_date ASC
                       ) AS next_f_ann_date
                FROM latest
            ),
            events AS (
                SELECT * FROM ordered
                WHERE f_ann_date <= ?
                  AND (next_f_ann_date IS NULL OR next_f_ann_date > ?)
            )
            SELECT e.symbol, e.announce_end_date, e.f_ann_date,
                   e.next_f_ann_date,
                   v.end_date,
                   {v_col_select}
            FROM events e
            JOIN latest v
              ON v.symbol = e.symbol
             AND v.end_date <= e.announce_end_date
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY e.symbol, e.announce_end_date
                ORDER BY v.end_date DESC
            ) <= {int(last_n_quarters)}
            ORDER BY e.symbol, e.announce_end_date, v.end_date ASC
        """
        params = list(symbol_params) + [end, start]
        df = self.conn.execute(sql, params).fetchdf()
        if df.empty:
            return df
        df = df.rename(columns={c: prefix + c for c in bare_cols})
        return df

    # -- dividends ------------------------------------------------------------

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
            pk_cols=("symbol", "end_date", "ann_date", "ex_date"),
            schema_cols=DIVIDEND_COLUMNS,
        )

    def get_dividends(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return dividend events with ex_date inside ``[start, end]``.

        Filters on ``ex_date`` because that's the trading-relevant date —
        ``DividendHandler`` keys off it for share/cash adjustments.
        """
        cols_sql = ", ".join(f'"{c}"' for c in DIVIDEND_COLUMNS)

        conditions: list[str] = []
        params: list = []
        if start:
            conditions.append("ex_date >= ?")
            params.append(start)
        if end:
            conditions.append("ex_date <= ?")
            params.append(end)
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT {cols_sql}
            FROM dividends
            {where_clause}
            ORDER BY ex_date, symbol
        """
        return self.conn.execute(sql, params).fetchdf()

    # -- index_daily ----------------------------------------------------------

    def insert_index_daily(self, df: pd.DataFrame):
        self._upsert(
            df,
            table="index_daily",
            pk_cols=("date", "symbol"),
            schema_cols=INDEX_DAILY_COLUMNS,
        )

    def get_max_index_date(self, symbol: str) -> str | None:
        """Max trade date for a single index ts_code, as YYYYMMDD, or None if empty."""
        row = self.conn.execute(
            "SELECT MAX(date) FROM index_daily WHERE symbol = ?", [symbol]
        ).fetchone()
        if row and row[0]:
            return row[0].strftime("%Y%m%d")
        return None

    def get_index_bars(
        self,
        symbols: list[str] | str | None = None,
        start: str | None = None,
        end: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Time-series bars from index_daily.

        Mirrors get_bars() (market_daily) but reads the index table.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        if columns:
            cols_sql = ", ".join(f'"{c}"' for c in columns if c in INDEX_DAILY_COLUMNS)
        else:
            cols_sql = ", ".join(
                f'"{c}"' for c in INDEX_DAILY_COLUMNS if c not in ("date", "symbol")
            )

        conditions = []
        params: list = []
        if start:
            conditions.append("date >= strptime(?, '%Y%m%d')::DATE")
            params.append(start)
        if end:
            conditions.append("date <= strptime(?, '%Y%m%d')::DATE")
            params.append(end)
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"""
            SELECT date, symbol, {cols_sql}
            FROM index_daily
            {where_clause}
            ORDER BY date, symbol
        """
        return self.conn.execute(sql, params).fetchdf()

    # -- sw_industry ----------------------------------------------------------

    def insert_sw_industry(self, df: pd.DataFrame):
        """UPSERT 申万行业归属记录。

        df 列至少包含 SW_INDUSTRY_COLUMNS 的子集;in_date/out_date 必须为
        pandas.Timestamp 或 datetime.date,空值用 None / NaT 表示。
        """
        self._upsert(
            df,
            table="sw_industry",
            pk_cols=("symbol", "level", "industry_code", "in_date"),
            schema_cols=SW_INDUSTRY_COLUMNS,
        )

    def get_sw_industry_stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol), "
            "COUNT(DISTINCT industry_code) FILTER (WHERE level='L1'), "
            "COUNT(DISTINCT industry_code) FILTER (WHERE level='L2') "
            "FROM sw_industry"
        ).fetchone()
        return {
            "total_rows": row[0],
            "total_symbols": row[1],
            "n_l1_industries": row[2],
            "n_l2_industries": row[3],
        }

    def get_industry_panel(
        self,
        date: str,
        level: str = "L1",
    ) -> pd.DataFrame:
        """返回 ``date`` 日各股票的申万行业归属横截面。

        Parameters
        ----------
        date : str
            YYYYMMDD 日期(归属判定日)。
        level : str
            ``'L1'`` 或 ``'L2'``。

        Returns
        -------
        pd.DataFrame
            列 ``[symbol, industry_code, industry_name]``。无对应记录的股票不出现。
            当一只股票在该日同时命中多条分段(``sw_industry`` 存在重叠分段)时,
            按 "最新 ``in_date`` 胜出" 规则取一条。
        """
        sql = """
            SELECT symbol, industry_code, industry_name
            FROM (
                SELECT symbol, industry_code, industry_name,
                       ROW_NUMBER() OVER (
                           PARTITION BY symbol
                           ORDER BY in_date DESC
                       ) AS rn
                FROM sw_industry
                WHERE level = ?
                  AND in_date <= strptime(?, '%Y%m%d')::DATE
                  AND (out_date IS NULL OR out_date > strptime(?, '%Y%m%d')::DATE)
            )
            WHERE rn = 1
            ORDER BY symbol
        """
        return self.conn.execute(sql, [level, date, date]).fetchdf()

    def get_industry_history(
        self,
        symbol: str,
        level: str | None = None,
    ) -> pd.DataFrame:
        """返回某股票的申万行业归属全历史(各分段)。

        Parameters
        ----------
        symbol : str
            股票 ts_code。
        level : str | None
            ``'L1'`` / ``'L2'`` 过滤,None 表示两级都返回。
        """
        conditions = ["symbol = ?"]
        params: list = [symbol]
        if level:
            conditions.append("level = ?")
            params.append(level)
        where = " AND ".join(conditions)
        sql = f"""
            SELECT symbol, level, industry_code, industry_name, in_date, out_date
            FROM sw_industry
            WHERE {where}
            ORDER BY level, in_date
        """
        return self.conn.execute(sql, params).fetchdf()

    def get_industry_panel_range(
        self,
        start: str,
        end: str,
        level: str = "L1",
        *,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """``[start, end]`` 区间内每个 ``(date, symbol)`` 的行业归属。

        以 ``market_daily`` 的实际交易日为基准,把 ``sw_industry`` 的分段归属
        展开为每日记录。用于因子层的批量行业中性化。

        Returns DataFrame ``[date, symbol, industry_code, industry_name]``,
        缺失行业的 ``(date, symbol)`` 不出现。

        当某 ``(symbol, level)`` 在 ``sw_industry`` 中存在分段重叠(``in_date``
        范围互相覆盖)时,按 "最新 ``in_date`` 胜出" 规则取一条,保证返回的
        ``(date, symbol)`` 唯一。
        """
        symbol_filter = ""
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            symbol_filter = f"AND m.symbol IN ({placeholders})"
        sql = f"""
            SELECT date, symbol, industry_code, industry_name
            FROM (
                SELECT m.date AS date, m.symbol AS symbol,
                       s.industry_code AS industry_code,
                       s.industry_name AS industry_name,
                       ROW_NUMBER() OVER (
                           PARTITION BY m.date, m.symbol
                           ORDER BY s.in_date DESC
                       ) AS rn
                FROM market_daily m
                INNER JOIN sw_industry s
                    ON s.symbol = m.symbol
                   AND s.level = ?
                   AND s.in_date <= m.date
                   AND (s.out_date IS NULL OR s.out_date > m.date)
                WHERE m.date >= strptime(?, '%Y%m%d')::DATE
                  AND m.date <= strptime(?, '%Y%m%d')::DATE
                  {symbol_filter}
            )
            WHERE rn = 1
            ORDER BY date, symbol
        """
        params = [level, start, end] + (symbols or [])
        return self.conn.execute(sql, params).fetchdf()

    # -- index_members --------------------------------------------------------

    def insert_index_members(self, df: pd.DataFrame):
        """UPSERT 宽基指数成分股记录(每个交易日一行,已密集化)。

        df 至少包含 ``index_code, symbol, trade_date``,``weight`` 可选。
        ``trade_date`` 必须为 ``pandas.Timestamp`` / ``datetime.date``。
        """
        self._upsert(
            df,
            table="index_members",
            pk_cols=("index_code", "symbol", "trade_date"),
            schema_cols=INDEX_MEMBERS_COLUMNS,
        )

    def get_max_index_member_date(self, index_code: str):
        """单个指数已写入的最大 trade_date,用于增量回填。无数据返回 None。"""
        row = self.conn.execute(
            "SELECT MAX(trade_date) FROM index_members WHERE index_code = ?",
            [index_code],
        ).fetchone()
        return row[0] if row and row[0] else None

    def get_index_members(self, index_code: str, date: str) -> set[str]:
        """返回 ``index_code`` 在 ``date`` (YYYYMMDD) 的成分股 symbol 集合。

        前提:``index_members`` 已通过 ``backfill_index_members`` 密集化到每个
        交易日;如果该日无记录则返回空集。
        """
        result = self.conn.execute(
            "SELECT symbol FROM index_members "
            "WHERE index_code = ? AND trade_date = strptime(?, '%Y%m%d')::DATE",
            [index_code, date],
        ).fetchdf()
        return set(result["symbol"]) if not result.empty else set()

    def get_index_members_stats(self) -> pd.DataFrame:
        """按 index_code 汇总:行数 + 覆盖区间(用于 CLI banner)。"""
        return self.conn.execute(
            "SELECT index_code, COUNT(*) AS rows, "
            "COUNT(DISTINCT symbol) AS n_symbols, "
            "MIN(trade_date) AS min_date, MAX(trade_date) AS max_date "
            "FROM index_members GROUP BY index_code ORDER BY index_code"
        ).fetchdf()

    # -- trade_calendar -------------------------------------------------------

    def get_max_cal_date(self) -> str | None:
        """Return max cal_date in trade_calendar as YYYYMMDD, or None if empty."""
        result = self.conn.execute(
            "SELECT MAX(cal_date) FROM trade_calendar"
        ).fetchone()
        if result[0]:
            return result[0].strftime("%Y%m%d")
        return None

    def get_trade_dates_from_db(self, start: str, end: str) -> list[str]:
        """Return open trade dates in [start, end] from trade_calendar."""
        rows = self.conn.execute(
            "SELECT cal_date FROM trade_calendar "
            "WHERE is_open = true "
            "  AND cal_date >= strptime(?, '%Y%m%d')::DATE "
            "  AND cal_date <= strptime(?, '%Y%m%d')::DATE "
            "ORDER BY cal_date",
            [start, end],
        ).fetchall()
        return [r[0].strftime("%Y%m%d") for r in rows]

    def get_rebalance_dates_from_db(self, start: str, end: str, freq: str) -> list[str]:
        """Return rebalancing dates in [start, end] for given frequency.

        Parameters
        ----------
        start, end : str
            YYYYMMDD bounds.
        freq : str
            ``"1D"``, ``"5D"``, ``"1W"``, ``"2W"``, ``"1M"``, ``"EOM"``.
        """
        if freq == "1D":
            return self.get_trade_dates_from_db(start, end)

        if freq == "5D":
            dates = self.get_trade_dates_from_db(start, end)
            return dates[::5]

        if freq == "1W":
            sql = (
                "SELECT cal_date FROM trade_calendar "
                "WHERE is_open = true AND is_week_first = true "
                "  AND cal_date >= strptime(?, '%Y%m%d')::DATE "
                "  AND cal_date <= strptime(?, '%Y%m%d')::DATE "
                "ORDER BY cal_date"
            )
        elif freq == "2W":
            sql = (
                "SELECT cal_date FROM trade_calendar "
                "WHERE is_open = true AND is_week_first = true "
                "  AND cal_date >= strptime(?, '%Y%m%d')::DATE "
                "  AND cal_date <= strptime(?, '%Y%m%d')::DATE "
                "ORDER BY cal_date"
            )
        elif freq == "1M":
            sql = (
                "SELECT cal_date FROM trade_calendar "
                "WHERE is_open = true AND is_month_first = true "
                "  AND cal_date >= strptime(?, '%Y%m%d')::DATE "
                "  AND cal_date <= strptime(?, '%Y%m%d')::DATE "
                "ORDER BY cal_date"
            )
        elif freq == "EOM":
            sql = (
                "SELECT cal_date FROM trade_calendar "
                "WHERE is_open = true AND is_month_last = true "
                "  AND cal_date >= strptime(?, '%Y%m%d')::DATE "
                "  AND cal_date <= strptime(?, '%Y%m%d')::DATE "
                "ORDER BY cal_date"
            )
        else:
            raise ValueError(f"Unknown rebalance frequency: {freq}")

        rows = self.conn.execute(sql, [start, end]).fetchall()
        dates = [r[0] for r in rows]

        if freq == "2W":
            # 取 ISO 周号为偶数的周
            return [d.strftime("%Y%m%d") for d in dates if d.isocalendar()[1] % 2 == 0]

        return [d.strftime("%Y%m%d") for d in dates]

    def insert_trade_calendar(self, df: pd.DataFrame):
        """UPSERT trade calendar rows."""
        self._upsert(
            df,
            table="trade_calendar",
            pk_cols=("cal_date",),
            schema_cols=TRADE_CALENDAR_COLUMNS,
        )

    def get_trade_dates_in_db(self, start: str, end: str) -> list:
        """市场实际交易日(``market_daily.date``)在 [start, end] 内的升序列表。

        与 ``backtest.data.trade_calendar.get_trade_dates`` 不同 —— 本方法以
        本地 ``market_daily`` 为真理源,不调 Tushare;返回 ``datetime.date``
        而不是 YYYYMMDD 字符串(下游 ``pd.merge_asof`` 需要 datetime 类型)。
        """
        rows = self.conn.execute(
            "SELECT DISTINCT date FROM market_daily "
            "WHERE date >= strptime(?, '%Y%m%d')::DATE "
            "  AND date <= strptime(?, '%Y%m%d')::DATE "
            "ORDER BY date",
            [start, end],
        ).fetchall()
        return [r[0] for r in rows]
