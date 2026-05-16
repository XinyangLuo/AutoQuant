"""Weight allocation methods: equal, market-cap, factor-value."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.strategy.config import WeightingConfig


class WeightAllocator:
    """Allocate portfolio weights to selected stocks."""

    def __init__(self, config: WeightingConfig):
        self.config = config

    def allocate(
        self,
        df: pd.DataFrame,
        factor_col: str | None = None,
    ) -> pd.Series:
        """Return target weights for selected stocks.

        Parameters
        ----------
        df : pd.DataFrame
            Selected stocks panel. Must contain ``symbol`` and optionally
            ``circ_mv`` (for market-cap weighting) or ``factor_col``.
        factor_col : str | None
            Column name of the factor value (for factor-value weighting).

        Returns
        -------
        pd.Series
            Index = symbol, values = target weights (sum to 1.0 for long-only).
        """
        method = self.config.method

        if method == "equal":
            return self._equal_weight(df)
        if method == "market_cap":
            return self._market_cap_weight(df)
        if method == "factor_value":
            return self._factor_value_weight(df, factor_col)

        raise ValueError(f"Unknown weighting method: {method}")

    @staticmethod
    def _equal_weight(df: pd.DataFrame) -> pd.Series:
        n = len(df)
        if n == 0:
            return pd.Series(dtype=float)
        w = 1.0 / n
        return pd.Series(w, index=df["symbol"])

    @staticmethod
    def _market_cap_weight(df: pd.DataFrame) -> pd.Series:
        if "circ_mv" not in df.columns:
            raise ValueError("market_cap weighting requires 'circ_mv' column")
        caps = df.set_index("symbol")["circ_mv"]
        caps = caps.replace(0, np.nan).dropna()
        if caps.sum() == 0 or caps.empty:
            return pd.Series(dtype=float)
        return caps / caps.sum()

    @staticmethod
    def _factor_value_weight(
        df: pd.DataFrame, factor_col: str | None
    ) -> pd.Series:
        if factor_col is None or factor_col not in df.columns:
            raise ValueError(
                "factor_value weighting requires a valid factor_col"
            )
        vals = df.set_index("symbol")[factor_col]
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            return pd.Series(dtype=float)
        # Use absolute values and normalize
        abs_vals = vals.abs()
        if abs_vals.sum() == 0:
            return pd.Series(dtype=float)
        return abs_vals / abs_vals.sum()
