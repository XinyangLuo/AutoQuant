"""Single-factor strategy: topK, long-short, and decile selection."""

from __future__ import annotations

import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.strategy.base import StrategyBase
from backtest.strategy.config import StrategyConfig
from backtest.strategy.selection import build_signals


class SingleFactorStrategy(StrategyBase):
    """Single-factor strategy: rank stocks by one factor and select.

    因子值已经在因子层完成中性化(由 registry 的 ``variant`` 字段记录),
    策略层不再做中性化。

    Supports three selection modes:
      - **topk**: Long the top-K stocks (equal or cap-weighted).
      - **long_short**: Long top-K + short bottom-K.
      - **decile**: Divide into 10 quantile groups (for analysis, not trading).
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        if len(config.factors) != 1:
            raise ValueError(
                "SingleFactorStrategy requires exactly one factor, "
                f"got {len(config.factors)}"
            )
        self.factor_config = config.factors[0]

    def generate_signals(
        self,
        factor_panel: pd.DataFrame,
        market_panel: pd.DataFrame,
        rebalance_dates: list[str],
        market_storage: MarketStorage | None = None,
    ) -> pd.DataFrame:
        """Generate signals for each rebalancing date."""
        factor_id = self.factor_config.id
        if factor_id not in factor_panel.columns:
            raise ValueError(f"Factor {factor_id} not found in factor panel")

        # Pre-compute trading-day index map for the New-IPO filter so
        # UniverseFilter.filter() doesn't call get_trade_dates() per date.
        trade_date_to_idx = self.universe_filter.warmup(
            rebalance_dates[0], rebalance_dates[-1],
        )

        signal_rows: list[dict] = []

        # Batch: subset both panels to rebalance dates once, then group.
        reb_dates_set = frozenset(pd.to_datetime(d) for d in rebalance_dates)
        reb_factor = factor_panel[
            factor_panel["date"].isin(reb_dates_set)
        ].dropna(subset=[factor_id])
        reb_market = market_panel[
            market_panel["date"].isin(reb_dates_set)
        ]

        if reb_factor.empty:
            return pd.DataFrame(columns=["date", "symbol", "target_weight"])

        # Single merge for all rebalance dates.
        merged_all = reb_factor.merge(
            reb_market, on=["date", "symbol"], how="left",
        )

        ascending = self.factor_config.direction == "asc"
        factor_col = factor_id if self.config.weighting.method == "factor_value" else None

        for date, grp in merged_all.groupby("date", sort=False):
            date_str = date.strftime("%Y%m%d")
            if date_str not in rebalance_dates:
                continue

            filtered = self.universe_filter.filter(
                date_str,
                grp,
                market_storage=market_storage,
                trade_date_to_idx=trade_date_to_idx,
            )
            if filtered.empty:
                continue

            factor_values = filtered.set_index("symbol")[factor_id]
            sorted_scores = factor_values.sort_values(ascending=ascending)

            rows = build_signals(
                date,
                sorted_scores,
                filtered,
                self.config.selection,
                self.config.weighting,
                factor_col=factor_col,
            )
            signal_rows.extend(rows)

        signals = pd.DataFrame(signal_rows)
        if signals.empty:
            return pd.DataFrame(columns=["date", "symbol", "target_weight"])
        return signals
