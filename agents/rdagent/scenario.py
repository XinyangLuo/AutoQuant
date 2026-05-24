"""A-share quantitative scenario implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backtest.config_loader import get_admission_thresholds, get_pipeline_thresholds
from backtest.factor.variants import DEFAULT_VARIANT, VALID_VARIANTS

from .core.scenario import Scenario
from .core.utils import render_prompt


def _format_dict_as_markdown(data: dict[str, Any], level: int = 0) -> str:
    """Recursively format a nested dict into readable markdown.

    Lists of strings → comma-separated line or bullet list.
    Nested dicts → indented bullet sections.
    Scalar values → inline after the key.
    """
    indent = "  " * level
    lines: list[str] = []

    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{indent}- **{key}**:")
            lines.append(_format_dict_as_markdown(value, level + 1))
        elif isinstance(value, list):
            # If list of strings and short, join them; otherwise bullet list
            if value and all(isinstance(v, str) for v in value):
                if len(value) <= 6:
                    lines.append(f"{indent}- **{key}**: {', '.join(value)}")
                else:
                    lines.append(f"{indent}- **{key}**:")
                    for v in value:
                        lines.append(f"{indent}  - `{v}`")
            else:
                lines.append(f"{indent}- **{key}**:")
                for v in value:
                    if isinstance(v, dict):
                        lines.append(_format_dict_as_markdown(v, level + 1))
                    else:
                        lines.append(f"{indent}  - {v}")
        elif isinstance(value, bool):
            lines.append(f"{indent}- **{key}**: {value}")
        elif isinstance(value, (int, float)):
            lines.append(f"{indent}- **{key}**: {value}")
        else:
            lines.append(f"{indent}- **{key}**: {value}")

    return "\n".join(lines)


@dataclass
class AShareQuantScenario(Scenario):
    """A-share quantitative research scenario.

    Provides the complete context an LLM needs to generate meaningful
    factor hypotheses: data schema, trading rules, evaluation thresholds,
    available operators, and neutralization options.
    """

    _prompt_dir: Path = field(default_factory=lambda: Path(__file__).parent / "prompts")

    # ------------------------------------------------------------------
    # Data schema
    # ------------------------------------------------------------------
    def get_data_schema(self) -> dict[str, Any]:
        return {
            "market_daily": {
                "description": "Daily market data, primary table for factor research",
                "columns": [
                    "open", "high", "low", "close", "volume", "amount",
                    "pre_close", "change", "pct_chg", "adj_factor",
                    "is_st", "list_date", "limit_up", "limit_down",
                    "turnover_rate", "turnover_rate_f", "volume_ratio",
                    "pe", "pe_ttm", "pb", "ps", "ps_ttm",
                    "dv_ratio", "dv_ttm",
                    "total_share", "float_share", "free_share",
                    "total_mv", "circ_mv",
                    # Moneyflow (capital flow) — extra-large/large/medium/small orders
                    "mf_buy_elg_vol", "mf_buy_elg_amount",
                    "mf_sell_elg_vol", "mf_sell_elg_amount",
                    "mf_buy_lg_vol", "mf_buy_lg_amount",
                    "mf_sell_lg_vol", "mf_sell_lg_amount",
                    "mf_buy_md_vol", "mf_buy_md_amount",
                    "mf_sell_md_vol", "mf_sell_md_amount",
                    "mf_buy_sm_vol", "mf_buy_sm_amount",
                    "mf_sell_sm_vol", "mf_sell_sm_amount",
                    "mf_net_mf_vol", "mf_net_mf_amount",
                ],
            },
            "income_q": {
                "description": "Quarterly income statement (PIT-safe)",
                "key_columns": [
                    "basic_eps", "total_revenue", "n_income",
                    "n_income_attr_p", "operate_profit",
                ],
            },
            "balancesheet_q": {
                "description": "Quarterly balance sheet (PIT-safe)",
                "key_columns": [
                    "total_assets", "total_liab",
                    "total_hldr_eqy_inc_min_int", "total_cur_assets",
                ],
            },
            "cashflow_q": {
                "description": "Quarterly cash flow statement (PIT-safe)",
                "key_columns": [
                    "n_cashflow_act", "n_cashflow_inv_act",
                    "n_cash_flows_fnc_act", "free_cashflow",
                ],
            },
        }

    def get_panel_columns_for_data_sources(
        self, data_sources: list[str]
    ) -> dict[str, list[str]]:
        """Return actual available column names in `panel` for given data sources.

        This maps logical source names (e.g. ``"income_q"``) to the physical
        column names that appear in the ``panel`` DataFrame passed to factor
        compute functions.  Financial columns carry prefixes so they are
        distinguishable after the three tables are merged.
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
            # Moneyflow (capital flow)
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

        # When any fina source is present, end_date is also injected
        has_fina = any(
            s in ("income_q", "balancesheet_q", "cashflow_q", "financial_statements_q")
            for s in data_sources
        )
        if has_fina:
            result["_meta"] = ["end_date (quarter end date, present when fina data is included)"]

        return result

    # ------------------------------------------------------------------
    # Trading rules
    # ------------------------------------------------------------------
    def get_trading_rules(self) -> dict[str, Any]:
        return {
            "settlement": "T+1 (buy today, sell tomorrow)",
            "price_limits": {
                "normal_stocks": "±10% from previous close",
                "star_market": "±20% from previous close (688/300/301 prefix)",
            },
            "exclusions": [
                "ST / *ST stocks (is_st flag)",
                "IPO within 60 trading days (list_date filter)",
                "Suspended stocks (volume = 0)",
            ],
            "delisting_risk": "Skip stocks with delisting warnings",
        }

    # ------------------------------------------------------------------
    # Evaluation criteria
    # ------------------------------------------------------------------
    def get_evaluation_criteria(self) -> dict[str, Any]:
        """Evaluation criteria sourced from admission + pipeline configs."""
        adm = get_admission_thresholds()
        pipe = get_pipeline_thresholds()
        return {
            "candidate_thresholds": {
                "min_rankicir": adm["min_rankicir"],
                "min_ic_positive_ratio": adm["min_ic_positive_ratio"],
                "max_turnover": adm["max_turnover"],
                "max_corr_with_existing": adm["max_corr"],
            },
            "pipeline_gates": {
                "min_sharpe_simple": pipe["simple_backtest"]["min_sharpe"],
                "max_max_drawdown": pipe["simple_backtest"]["max_max_drawdown"],
                "min_calmar_simple": pipe["simple_backtest"]["min_calmar"],
                "min_annual_return_simple": pipe["simple_backtest"]["min_annual_return"],
                "min_sharpe_detailed": pipe["detailed_backtest"]["min_sharpe"],
                "min_annual_return_detailed": pipe["detailed_backtest"]["min_annual_return"],
                "min_monotonicity": pipe["monotonicity"]["min_monotonicity"],
            },
            "high_bar": {
                "min_sharpe_simple": 1.0,
            },
            "primary_horizon": adm["primary_horizon"],
            "ret_type": adm["ret_type"],
            "evaluation_period": "At least 3 years of history recommended",
        }

    # ------------------------------------------------------------------
    # Factor taxonomy
    # ------------------------------------------------------------------
    def get_factor_categories(self) -> list[str]:
        return [
            "reversal",      # mean-reversion, short-term price reversal
            "momentum",      # trend-following, medium-term momentum
            "value",         # valuation ratios
            "quality",       # profitability, earnings stability
            "growth",        # earnings/revenue growth
            "liquidity",     # turnover, volume patterns
            "volatility",    # price volatility measures
        ]

    # ------------------------------------------------------------------
    # Available operators
    # ------------------------------------------------------------------
    def get_available_operators(self) -> list[str]:
        """Return the list of operators available in backtest.factor.transforms."""
        return [
            # Cross-sectional
            "rank", "z_score", "cs_zscore", "cs_demean",
            "cs_winsorize", "cs_mad_winsorize",
            "industry_neutralize", "cap_neutralize", "industry_median_fill",
            # Time-series
            "ts_rank", "ts_mean", "ts_std", "ts_sum",
            "ts_min", "ts_max", "ts_argmax", "ts_argmin",
            "ts_delta", "ts_delay", "ts_pct_change", "ts_product",
            "ts_skewness", "ts_kurtosis", "ts_ir",
            "ts_decay_linear", "ts_decay_exp",
            "ts_corr", "ts_covariance",
            # Element-wise
            "abs_", "sign", "log", "sqrt", "signed_power", "inverse", "if_else",
            # Fundamental helpers
            "single_quarter", "ttm", "yoy",
        ]

    # ------------------------------------------------------------------
    # Neutralization options
    # ------------------------------------------------------------------
    def get_neutralization_options(self) -> list[str]:
        return list(VALID_VARIANTS)

    def get_default_variant(self) -> str:
        return DEFAULT_VARIANT

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------
    def render_scenario_prompt(self) -> str:
        """Render the full scenario description for LLM prompts."""
        template_path = self._prompt_dir / "scenario_desc.md"
        if not template_path.exists():
            return self._build_fallback_prompt()
        return render_prompt(
            template_path,
            data_schema=_format_dict_as_markdown(self.get_data_schema()),
            trading_rules=_format_dict_as_markdown(self.get_trading_rules()),
            evaluation_criteria=_format_dict_as_markdown(self.get_evaluation_criteria()),
            factor_categories=", ".join(self.get_factor_categories()),
            available_operators=", ".join(self.get_available_operators()),
            neutralization_options=", ".join(self.get_neutralization_options()),
            default_variant=self.get_default_variant(),
        )

    def _build_fallback_prompt(self) -> str:
        """Build a plain-text prompt when the template file is missing."""
        lines = [
            "# A-Share Quantitative Research Scenario",
            "",
            "## Data Schema",
            _format_dict_as_markdown(self.get_data_schema()),
            "",
            "## Trading Rules",
            _format_dict_as_markdown(self.get_trading_rules()),
            "",
            "## Evaluation Criteria",
            _format_dict_as_markdown(self.get_evaluation_criteria()),
            "",
            f"## Factor Categories: {', '.join(self.get_factor_categories())}",
            "",
            f"## Available Operators: {', '.join(self.get_available_operators())}",
            "",
            f"## Neutralization Options: {', '.join(self.get_neutralization_options())} (default: {self.get_default_variant()})",
        ]
        return "\n".join(lines)
