"""A-share quantitative scenario implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backtest.config_loader import get_admission_thresholds, get_pipeline_thresholds
from backtest.factor.variants import DEFAULT_VARIANT, VALID_VARIANTS

from .core.scenario import Scenario
from .core.utils import render_prompt


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
            data_schema=str(self.get_data_schema()),
            trading_rules=str(self.get_trading_rules()),
            evaluation_criteria=str(self.get_evaluation_criteria()),
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
            str(self.get_data_schema()),
            "",
            "## Trading Rules",
            str(self.get_trading_rules()),
            "",
            "## Evaluation Criteria",
            str(self.get_evaluation_criteria()),
            "",
            f"## Factor Categories: {', '.join(self.get_factor_categories())}",
            "",
            f"## Available Operators: {', '.join(self.get_available_operators())}",
            "",
            f"## Neutralization Options: {', '.join(self.get_neutralization_options())} (default: {self.get_default_variant()})",
        ]
        return "\n".join(lines)
