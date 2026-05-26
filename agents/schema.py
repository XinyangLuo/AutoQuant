"""Data schema query helpers for Claude Code-driven factor iteration.

Provides the column lists Claude needs to generate factor code without
hallucinating column names.  No LLM prompt rendering — that's Claude's job now.
"""

from __future__ import annotations


def get_panel_columns_for_data_sources(
    data_sources: list[str],
) -> dict[str, list[str]]:
    """Return column names present in ``panel`` for the given data sources.

    Maps logical source names (e.g. ``"income_q"``) to the physical column
    names visible in the ``panel`` DataFrame passed to factor functions.
    Financial columns carry prefixes (``inc_`` / ``bs_`` / ``cf_``) so they
    remain distinguishable after the three tables are merged.
    """
    result: dict[str, list[str]] = {}

    _market_cols = [
        "open", "high", "low", "close", "volume", "amount",
        "pre_close", "change", "pct_chg", "adj_factor",
        "is_st", "list_date", "limit_up", "limit_down",
        "turnover_rate", "turnover_rate_f", "volume_ratio",
        "pe", "pe_ttm", "pb", "ps", "ps_ttm",
        "dv_ratio", "dv_ttm",
        "total_share", "float_share", "free_share",
        "total_mv", "circ_mv",
        "mf_buy_elg_vol", "mf_buy_elg_amount",
        "mf_sell_elg_vol", "mf_sell_elg_amount",
        "mf_buy_lg_vol", "mf_buy_lg_amount",
        "mf_sell_lg_vol", "mf_sell_lg_amount",
        "mf_buy_md_vol", "mf_buy_md_amount",
        "mf_sell_md_vol", "mf_sell_md_amount",
        "mf_buy_sm_vol", "mf_buy_sm_amount",
        "mf_sell_sm_vol", "mf_sell_sm_amount",
        "mf_net_mf_vol", "mf_net_mf_amount",
    ]

    _income_cols = [
        "inc_basic_eps", "inc_diluted_eps", "inc_total_revenue",
        "inc_revenue", "inc_n_income", "inc_n_income_attr_p",
        "inc_operate_profit", "inc_total_profit", "inc_income_tax",
        "inc_ebit", "inc_ebitda", "inc_total_cogs", "inc_oper_cost",
        "inc_sell_exp", "inc_admin_exp", "inc_fin_exp", "inc_rd_exp",
        "inc_non_oper_income", "inc_non_oper_exp", "inc_minority_gain",
        "inc_oth_compr_income", "inc_t_compr_income",
        "inc_compr_inc_attr_p", "inc_continued_net_profit",
    ]

    _bs_cols = [
        "bs_total_assets", "bs_total_liab",
        "bs_total_hldr_eqy_inc_min_int", "bs_total_hldr_eqy_exc_min_int",
        "bs_total_cur_assets", "bs_total_nca",
        "bs_money_cap", "bs_trad_asset", "bs_inventories",
        "bs_accounts_receiv", "bs_notes_receiv", "bs_oth_receiv",
        "bs_fix_assets", "bs_cip", "bs_intan_assets", "bs_goodwill",
        "bs_lt_eqt_invest", "bs_fa_avail_for_sale",
        "bs_total_cur_liab", "bs_total_ncl",
        "bs_st_borr", "bs_lt_borr", "bs_bond_payable",
        "bs_notes_payable", "bs_acct_payable",
        "bs_st_bonds_payable", "bs_lt_payable",
        "bs_deferred_inc", "bs_defer_tax_liab",
        "bs_surplus_rese", "bs_undistr_porfit", "bs_cap_rese",
        "bs_treasury_share", "bs_minority_int",
    ]

    _cf_cols = [
        "cf_n_cashflow_act", "cf_n_cashflow_inv_act",
        "cf_n_cash_flows_fnc_act", "cf_free_cashflow",
        "cf_net_profit", "cf_c_fr_sale_sg",
        "cf_c_inf_fr_operate_a", "cf_st_cash_out_act",
        "cf_stot_inflows_inv_act", "cf_stot_out_inv_act",
        "cf_stot_cash_in_fnc_act", "cf_stot_cashout_fnc_act",
        "cf_c_recp_borrow", "cf_proc_issue_bonds",
        "cf_c_recp_cap_contrib", "cf_n_incr_cash_cash_equ",
        "cf_c_cash_equ_beg_period", "cf_c_cash_equ_end_period",
    ]

    for src in data_sources:
        if src == "market_daily":
            result["market_daily"] = _market_cols
        elif src == "income_q":
            result["income_q (prefix: inc_)"] = _income_cols
        elif src == "balancesheet_q":
            result["balancesheet_q (prefix: bs_)"] = _bs_cols
        elif src == "cashflow_q":
            result["cashflow_q (prefix: cf_)"] = _cf_cols
        elif src == "financial_statements_q":
            result["financial_statements_q (prefixes: inc_/bs_/cf_)"] = (
                _income_cols[:10]
                + _bs_cols[:10]
                + _cf_cols[:8]
                + [f"... ({len(_income_cols) + len(_bs_cols) + len(_cf_cols)} total columns, "
                   f"all prefixed inc_/bs_/cf_; see FACTOR_CODE_GUIDE.md for full list)"]
            )
        elif src == "factors_daily":
            result["factors_daily"] = [
                "Other admitted factor columns (accessed via factor_storage, "
                "not in panel — declare event_driven=True or use composite pattern)"
            ]

    has_fina = any(
        s in ("income_q", "balancesheet_q", "cashflow_q", "financial_statements_q")
        for s in data_sources
    )
    if has_fina:
        result["_meta"] = ["end_date (quarter end date, present when fina data is included)"]

    return result


# Common column aliases that LLMs tend to hallucinate — map to real names.
COLUMN_ALIASES: dict[str, str] = {
    "buy_sm": "mf_buy_sm_amount",
    "sell_sm": "mf_sell_sm_amount",
    "buy_md": "mf_buy_md_amount",
    "sell_md": "mf_sell_md_amount",
    "buy_lg": "mf_buy_lg_amount",
    "sell_lg": "mf_sell_lg_amount",
    "buy_elg": "mf_buy_elg_amount",
    "sell_elg": "mf_sell_elg_amount",
    "net_mf": "mf_net_mf_amount",
    "ts_zscore": "z_score",
    "cs_rank": "rank",
}
