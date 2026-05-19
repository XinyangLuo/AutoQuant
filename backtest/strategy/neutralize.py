"""Industry and market-cap neutralization utilities.

.. deprecated::
    中性化已下沉到因子层。生产代码不应再调用 :class:`Neutralizer`。请通过
    :class:`backtest.strategy.config.FactorConfig` 的 ``variant`` 选择已经
    中性化的因子值;算子实现见 :mod:`backtest.factor.transforms`
    (``industry_neutralize`` / ``cap_neutralize``)与
    :mod:`backtest.factor.variants`(命名规范)。

    此模块仅保留作为兼容垫片,内部 API 可能在后续移除。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Neutralizer:
    """Neutralize factor values by industry and/or market cap."""

    @staticmethod
    def industry_neutralize(
        factor_values: pd.Series,
        industries: pd.Series,
        method: str = "group_rank",
    ) -> pd.Series:
        """Neutralize factor values within each industry group.

        Parameters
        ----------
        factor_values : pd.Series
            Index = symbol, values = raw factor values.
        industries : pd.Series
            Index = symbol, values = industry code/name.
        method : str
            - ``group_rank``: rank within each group, then scale to [0, 1].
            - ``group_zscore``: z-score within each group.

        Returns
        -------
        pd.Series
            Index = symbol, values = neutralized scores.
        """
        df = pd.DataFrame({
            "factor": factor_values,
            "industry": industries,
        }).dropna()

        if df.empty:
            return pd.Series(dtype=float)

        if method == "group_rank":
            df["rank"] = df.groupby("industry")["factor"].rank(pct=True)
            # Fill NaN (groups with 1 member) with 0.5
            df["rank"] = df["rank"].fillna(0.5)
            return df["rank"]

        if method == "group_demean":
            # Within each industry group, subtract the group mean so that
            # the mean of each group is zero.  This is the standard
            # WorldQuant-style subindustry neutralization.
            df["demeaned"] = df.groupby("industry")["factor"].transform(
                lambda x: x - x.mean()
            )
            return df["demeaned"]

        if method == "group_zscore":
            def _zscore(x: pd.Series) -> pd.Series:
                mean = x.mean()
                std = x.std()
                if std == 0 or pd.isna(std):
                    return pd.Series(0.0, index=x.index)
                return (x - mean) / std

            df["zscore"] = df.groupby("industry")["factor"].transform(_zscore)
            return df["zscore"]

        raise ValueError(f"Unknown industry neutralization method: {method}")

    @staticmethod
    def market_cap_neutralize(
        factor_values: pd.Series,
        market_cap: pd.Series,
    ) -> pd.Series:
        """Remove market-cap exposure from factor values via regression residual.

        Runs a cross-sectional regression:
            factor = alpha + beta * log(circ_mv) + residual
        Returns the residual (factor with market-cap effect removed).

        Parameters
        ----------
        factor_values : pd.Series
            Index = symbol, values = raw factor values.
        market_cap : pd.Series
            Index = symbol, values = market cap (circ_mv).

        Returns
        -------
        pd.Series
            Index = symbol, values = residual (neutralized factor).
        """
        df = pd.DataFrame({
            "factor": factor_values,
            "cap": market_cap,
        }).dropna()

        if len(df) < 3:
            return pd.Series(dtype=float)

        log_cap = np.log(df["cap"].replace(0, np.nan).dropna())
        valid_idx = log_cap.index
        y = df.loc[valid_idx, "factor"]
        x = log_cap

        # OLS: y = alpha + beta * x
        x_mean = x.mean()
        y_mean = y.mean()
        beta = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        if np.isnan(beta):
            beta = 0.0
        alpha = y_mean - beta * x_mean
        residual = y - (alpha + beta * x)
        return residual
