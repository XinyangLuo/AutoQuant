"""Abstract strategy base class and runner logic."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
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


def _try_read_cache(env_var: str) -> pd.DataFrame | None:
    """Return a DataFrame from the parquet path in *env_var*, or None."""
    path = os.environ.get(env_var)
    if path and Path(path).exists():
        return pd.read_parquet(path)
    return None


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
        market_storage: MarketStorage | None = None,
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
        market_storage : MarketStorage | None
            Optional market store for filters that need DB-backed metadata,
            such as index membership.

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
        factor_panel: pd.DataFrame | None = None,
        market_panel: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Full pipeline: fetch factors + market data → generate signals.

        When the environment variables ``AQ_FACTOR_CACHE`` /
        ``AQ_MARKET_CACHE`` point to existing parquet files, those are
        read instead of querying DuckDB. Callers may also pass preloaded
        ``factor_panel`` / ``market_panel`` directly to reuse data already
        loaded in a surrounding pipeline step.

        Returns
        -------
        pd.DataFrame
            Columns ``[date, symbol, target_weight]`` where ``date`` is the
            effective holding date (signal date + delay).
        """
        own_factor = factor_storage is None and factor_panel is None
        needs_market_storage = market_panel is None or self.config.universe.index_members is not None
        own_market = market_storage is None and needs_market_storage

        try:
            if factor_storage is None and factor_panel is None:
                factor_storage = FactorStorage(read_only=True)
            if market_storage is None and needs_market_storage:
                market_storage = MarketStorage(read_only=True)

            factor_ids = [f.id for f in self.config.factors]
            if not factor_ids:
                raise ValueError("No factors configured")

            # --- factor panel: try parquet cache first --------------------
            if factor_panel is None:
                factor_panel = _try_read_cache("AQ_FACTOR_CACHE")
            if factor_panel is None:
                factor_panel = self._load_factor_panel(
                    factor_ids, start_date, end_date, factor_storage
                )
            if factor_panel.empty:
                raise ValueError("No factor data found for the given date range")

            # --- market panel: try parquet cache first --------------------
            if market_panel is None:
                market_panel = _try_read_cache("AQ_MARKET_CACHE")
            if market_panel is None:
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

            signals = self.generate_signals(
                factor_panel,
                market_panel,
                rebalance_dates,
                market_storage=market_storage,
            )

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

        Uses ``sliding_window_view`` + dot-product per group instead of
        pandas ``rolling().apply()`` — avoids ~9M Python lambda calls and is
        ~50-100× faster.

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

        # Linear decay weights: [newest=n, ..., oldest=1].
        weights = np.arange(n, 0, -1, dtype=float)
        w_full = weights.sum()

        # Normalisers for partial windows: position i uses weights[-(i+1):] = [i+1, ..., 1]
        # → sum = (i+1)*(i+2)/2
        partial_norms: np.ndarray | None = None
        if n > 1:
            partial_norms = np.array(
                [weights[-(i + 1) :].sum() for i in range(n - 1)]
            )

        for col in factor_cols:
            result_parts: list[pd.Series] = []
            for _sym, grp in df.groupby("symbol", sort=False)[col]:
                vals = grp.values.astype(float, copy=False)
                m = len(vals)
                if m == 0:
                    result_parts.append(grp)
                    continue

                out = np.empty(m, dtype=float)

                # --- full windows (positions n-1 … m-1) -------------------
                if m >= n:
                    # sliding_window_view(vals, n)[i] = [vals[i], ..., vals[i+n-1]]
                    # (oldest … newest in natural order).
                    windows = np.lib.stride_tricks.sliding_window_view(vals, n)
                    # The original formula: x[newest] * n + ... + x[oldest] * 1
                    # = dot(row[::-1], weights) = dot(row, weights[::-1])
                    out[n - 1 :] = windows.dot(weights[::-1]) / w_full

                # --- partial ramp-up (positions 0 … n-2) -------------------
                limit = min(n - 1, m)
                if limit > 0 and partial_norms is not None:
                    for i in range(limit):
                        # Window: vals[0 … i], weights used: last i+1 entries.
                        w = weights[-(i + 1) :]
                        out[i] = np.dot(vals[: i + 1][::-1], w) / partial_norms[i]

                result_parts.append(
                    pd.Series(out, index=grp.index, name=col)
                )

            df[col] = pd.concat(result_parts)

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
