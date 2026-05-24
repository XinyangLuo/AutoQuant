"""Abstract strategy base class and runner logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_rebalance_dates, get_trade_dates
from backtest.factor.storage import FactorStorage
from backtest.strategy.config import StrategyConfig
from backtest.strategy.universe import UniverseFilter
from backtest.strategy.weight import WeightAllocator

if TYPE_CHECKING:
    pass


class StrategyBase(ABC):
    """Abstract base class for all strategies.

    Subclasses implement ``generate_signals()`` to produce a target-weight
    DataFrame.  The ``run()`` method orchestrates data loading and delegates
    signal generation to the subclass.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.universe_filter = UniverseFilter(config.universe)
        self.weight_allocator = WeightAllocator(config.weighting)

    @abstractmethod
    def generate_signals(
        self,
        factor_panel: pd.DataFrame,
        market_panel: pd.DataFrame,
        rebalance_dates: list[str],
    ) -> pd.DataFrame:
        """Generate target position signals.

        Parameters
        ----------
        factor_panel : pd.DataFrame
            Wide DataFrame with columns ``[date, symbol, f_001, f_002, ...]``.
        market_panel : pd.DataFrame
            Wide DataFrame with columns ``[date, symbol, close, circ_mv, ...]``.
        rebalance_dates : list[str]
            List of YYYYMMDD rebalancing dates.

        Returns
        -------
        pd.DataFrame
            Columns ``[date, symbol, target_weight]``.
            ``date`` is the **effective date** (already accounts for ``delay``).
        """
        ...

    def run(
        self,
        start_date: str,
        end_date: str,
        factor_storage: FactorStorage | None = None,
        market_storage: MarketStorage | None = None,
    ) -> pd.DataFrame:
        """Full pipeline: fetch factors + market data → generate signals.

        Returns
        -------
        pd.DataFrame
            Columns ``[date, symbol, target_weight]`` where ``date`` is the
            effective holding date (signal date + delay).
        """
        own_factor = factor_storage is None
        own_market = market_storage is None

        try:
            if factor_storage is None:
                factor_storage = FactorStorage()
            if market_storage is None:
                market_storage = MarketStorage(read_only=True)

            factor_ids = [f.id for f in self.config.factors]
            if not factor_ids:
                raise ValueError("No factors configured")

            # Load all factor data for the date range.
            factor_panel = self._load_factor_panel(
                factor_ids, start_date, end_date, factor_storage
            )
            if factor_panel.empty:
                raise ValueError("No factor data found for the given date range")

            # Load market data (for universe filtering, neutralization, etc.)
            market_panel = market_storage.get_bars(
                symbols=None,
                start=start_date,
                end=end_date,
                columns=["close", "open", "high", "low", "circ_mv", "amount",
                         "is_st", "list_date", "limit_up", "limit_down"],
            )

            rebalance_dates = get_rebalance_dates(
                start_date, end_date, self.config.rebalance_freq
            )

            # Apply decay smoothing to factor values (reduces turnover)
            if self.config.decay is not None:
                factor_panel = self._apply_decay(factor_panel, self.config.decay)

            signals = self.generate_signals(factor_panel, market_panel, rebalance_dates)

            # Apply delay: signal computed on rebalance_date → effective on rebalance_date + delay
            if self.config.delay > 0 and not signals.empty:
                signals = self._apply_delay(signals, rebalance_dates, self.config.delay)

            # Expand to daily signals: forward-fill weights between rebalance effective dates
            if not signals.empty:
                signals = self._to_daily_signals(signals, start_date, end_date)

            return signals

        finally:
            if own_factor and factor_storage is not None:
                factor_storage.close()
            if own_market and market_storage is not None:
                market_storage.close()

    @staticmethod
    def _to_daily_signals(
        signals: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """将调仓日信号扩展为日频信号：调仓日之间前向填充权重。"""
        if signals.empty:
            return signals

        trade_dates = get_trade_dates(start_date, end_date)
        if not trade_dates:
            return signals

        # pivot → 扩展到所有交易日 → ffill → melt
        wide = signals.pivot(index="date", columns="symbol", values="target_weight")
        all_dates = pd.to_datetime(trade_dates)
        wide = wide.reindex(all_dates)
        wide.index.name = "date"

        # 调仓生效日：被调出的股票显式置 0
        rebalance_dates = set(signals["date"])
        mask = wide.index.isin(rebalance_dates)
        if mask.any():
            wide.loc[mask] = wide.loc[mask].fillna(0)

        wide = wide.ffill().fillna(0)
        long_df = wide.reset_index().melt(
            id_vars=["date"], var_name="symbol", value_name="target_weight"
        )
        return long_df[long_df["target_weight"] != 0].reset_index(drop=True)

    @staticmethod
    def _apply_decay(
        factor_panel: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        """Apply linear decay smoothing to factor values.

        decay(x, n) = (x[date] * n + x[date-1] * (n-1) + ... + x[date-n+1] * 1)
                      / (n + (n-1) + ... + 1)

        Weights recent values more heavily, producing smoother factor series
        and lower portfolio turnover.

        Parameters
        ----------
        factor_panel : pd.DataFrame
            Wide DataFrame with columns ``[date, symbol, f_001, f_002, ...]``.
        n : int
            Decay window length (must be >= 1).

        Returns
        -------
        pd.DataFrame
            Decay-smoothed factor panel with same shape.
        """
        if n < 1:
            return factor_panel

        df = factor_panel.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["symbol", "date"])

        factor_cols = [c for c in df.columns if c not in ("date", "symbol")]
        if not factor_cols:
            return factor_panel

        # Build linear decay weights: [n, n-1, ..., 1]
        weights = np.arange(n, 0, -1, dtype=float)

        def _decay_series(s: pd.Series) -> pd.Series:
            def _apply(x: np.ndarray) -> float:
                w = weights[-len(x):]
                # Rolling window order is [oldest, ..., newest];
                # weights are [newest_weight=n, ..., oldest_weight=1].
                return float(np.dot(x[::-1], w) / w.sum())

            return s.rolling(window=n, min_periods=1).apply(_apply, raw=True)

        for col in factor_cols:
            df[col] = df.groupby("symbol", group_keys=False)[col].apply(_decay_series)

        return df

    @staticmethod
    def _load_factor_panel(
        factor_ids: list[str],
        start_date: str,
        end_date: str,
        factor_storage: FactorStorage,
    ) -> pd.DataFrame:
        """Load and merge multiple factors into a wide panel.

        Each factor is read at whichever neutralization variant it was
        registered with — variant is a property of the factor, not the
        strategy.
        """
        all_factors: list[pd.DataFrame] = []
        for fid in factor_ids:
            df = factor_storage.get_factor(fid, start_date, end_date)
            if df.empty:
                continue
            df = df.rename(columns={"value": fid})
            all_factors.append(df[["date", "symbol", fid]])

        if not all_factors:
            return pd.DataFrame()

        merged = all_factors[0]
        for df in all_factors[1:]:
            merged = merged.merge(df, on=["date", "symbol"], how="outer")

        return merged

    @staticmethod
    def _apply_delay(
        signals: pd.DataFrame,
        rebalance_dates: list[str],
        delay: int,
    ) -> pd.DataFrame:
        """Shift signal dates forward by ``delay`` trading days."""
        max_signal_date = signals["date"].max().strftime("%Y%m%d")
        # Buffer of ~delay+5 calendar days is enough for trading-day shift
        from datetime import datetime, timedelta

        end_dt = datetime.strptime(max_signal_date, "%Y%m%d") + timedelta(days=delay + 7)
        end_bound = end_dt.strftime("%Y%m%d")

        trade_dates = get_trade_dates(rebalance_dates[0], end_bound)
        date_to_idx = {d: i for i, d in enumerate(trade_dates)}

        def _shift_date(d: pd.Timestamp) -> pd.Timestamp | None:
            d_str = d.strftime("%Y%m%d")
            idx = date_to_idx.get(d_str)
            if idx is None:
                return d
            new_idx = idx + delay
            if new_idx >= len(trade_dates):
                return None
            return pd.Timestamp(trade_dates[new_idx])

        signals["date"] = signals["date"].apply(_shift_date)
        return signals.dropna(subset=["date"])
