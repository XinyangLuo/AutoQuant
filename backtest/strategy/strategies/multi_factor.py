"""Multi-factor strategy: combine multiple factors into a composite score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.strategy.base import StrategyBase
from backtest.strategy.config import StrategyConfig
from backtest.strategy.selection import build_signals


class MultiFactorStrategy(StrategyBase):
    """Multi-factor combination strategy.

    Combines multiple factors into a single composite score, then applies
    the same selection logic as SingleFactorStrategy. 因子值已经在因子层
    完成中性化(由 registry 的 ``variant`` 字段记录),所以策略层不再做中性化。

    Supported combination methods:
      - **zscore_equal**: Z-score each factor cross-sectionally, then equal-weight sum.
      - **ic_weighted**: Weight by rolling IC (runtime computed).
      - **icir_weighted**: Weight by rolling ICIR (runtime computed).
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        if len(config.factors) < 2:
            raise ValueError(
                "MultiFactorStrategy requires at least 2 factors, "
                f"got {len(config.factors)}"
            )

    def generate_signals(
        self,
        factor_panel: pd.DataFrame,
        market_panel: pd.DataFrame,
        rebalance_dates: list[str],
    ) -> pd.DataFrame:
        """Generate signals using multi-factor composite score."""
        signal_rows: list[dict] = []

        for date_str in rebalance_dates:
            date = pd.Timestamp(date_str)

            day_factors = factor_panel[factor_panel["date"] == date].copy()
            if day_factors.empty:
                continue

            day_market = market_panel[market_panel["date"] == date].copy()
            merged = day_factors.merge(day_market, on=["date", "symbol"], how="left")

            filtered = self.universe_filter.filter(date_str, merged)
            if filtered.empty:
                continue

            composite = self._build_composite(filtered, date_str)
            if composite.empty:
                continue

            sorted_scores = composite.sort_values(ascending=False)

            rows = build_signals(
                date,
                sorted_scores,
                filtered,
                self.config.selection,
                self.config.weighting,
            )
            signal_rows.extend(rows)

        signals = pd.DataFrame(signal_rows)
        if signals.empty:
            return pd.DataFrame(columns=["date", "symbol", "target_weight"])
        return signals

    def _build_composite(
        self,
        filtered: pd.DataFrame,
        date_str: str,
    ) -> pd.Series:
        """Combine multiple factors into a single composite score."""
        combine_method = self.config.combine_method
        factor_ids = [f.id for f in self.config.factors]

        scores = pd.DataFrame(index=filtered["symbol"])

        for fc in self.config.factors:
            fid = fc.id
            if fid not in filtered.columns:
                continue
            vals = filtered.set_index("symbol")[fid]
            vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
            if vals.empty:
                continue

            mean = vals.mean()
            std = vals.std()
            if std > 0 and not pd.isna(std):
                zscore = (vals - mean) / std
            else:
                zscore = pd.Series(0.0, index=vals.index)

            if fc.direction == "asc":
                zscore = -zscore

            scores[fid] = zscore

        if scores.empty:
            return pd.Series(dtype=float)

        scores = scores.fillna(0)

        if combine_method == "zscore_equal":
            weights = {fc.id: fc.weight for fc in self.config.factors if fc.id in scores.columns}
            total_weight = sum(weights.values())
            if total_weight == 0:
                return pd.Series(dtype=float)
            composite = pd.Series(0.0, index=scores.index)
            for fid, w in weights.items():
                if fid in scores.columns:
                    composite += scores[fid] * (w / total_weight)
            return composite

        if combine_method in ("ic_weighted", "icir_weighted"):
            ic_weights = self._compute_ic_weights(factor_ids, date_str, combine_method)
            total_w = sum(ic_weights.values())
            if total_w == 0:
                return pd.Series(dtype=float)
            composite = pd.Series(0.0, index=scores.index)
            for fid, w in ic_weights.items():
                if fid in scores.columns:
                    composite += scores[fid] * (w / total_w)
            return composite

        raise ValueError(f"Unknown combine method: {combine_method}")

    def _compute_ic_weights(
        self,
        factor_ids: list[str],
        date_str: str,
        method: str,
        window: int = 252,
    ) -> dict[str, float]:
        """Compute IC-based weights for each factor using a rolling window."""
        from backtest.factor.evaluation import evaluate
        from datetime import datetime, timedelta

        end_dt = datetime.strptime(date_str, "%Y%m%d")
        start_dt = end_dt - timedelta(days=int(window * 1.5))
        start_str = start_dt.strftime("%Y%m%d")

        weights: dict[str, float] = {}
        for fid in factor_ids:
            try:
                result = evaluate(fid, start_str, date_str, horizons=[1])
                metrics = result.ic_metrics.get(1, {})
                if method == "ic_weighted":
                    weights[fid] = max(0, metrics.get("ic_mean", 0))
                else:
                    weights[fid] = max(0, metrics.get("icir", 0))
            except Exception:
                weights[fid] = 0.0

        return weights
