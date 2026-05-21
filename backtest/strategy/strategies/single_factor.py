"""Single-factor strategy: topK, long-short, and decile selection."""

from __future__ import annotations

import pandas as pd

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
    ) -> pd.DataFrame:
        """Generate signals for each rebalancing date."""
        factor_id = self.factor_config.id
        if factor_id not in factor_panel.columns:
            raise ValueError(f"Factor {factor_id} not found in factor panel")

        signal_rows: list[dict] = []

        for date_str in rebalance_dates:
            date = pd.Timestamp(date_str)

            day_factors = factor_panel[factor_panel["date"] == date].copy()
            day_factors = day_factors.dropna(subset=[factor_id])
            if day_factors.empty:
                continue

            day_market = market_panel[market_panel["date"] == date].copy()
            merged = day_factors.merge(day_market, on=["date", "symbol"], how="left")

            filtered = self.universe_filter.filter(date_str, merged)
            if filtered.empty:
                continue

            factor_values = filtered.set_index("symbol")[factor_id]

            ascending = self.factor_config.direction == "asc"
            sorted_scores = factor_values.sort_values(ascending=ascending)

            factor_col = factor_id if self.config.weighting.method == "factor_value" else None
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
