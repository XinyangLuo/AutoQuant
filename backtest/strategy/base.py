"""Abstract strategy base class and runner logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates
from backtest.factor.storage import FactorStorage
from backtest.strategy.config import StrategyConfig
from backtest.strategy.universe import UniverseFilter
from backtest.strategy.weight import WeightAllocator

if TYPE_CHECKING:
    pass


def _get_rebalance_dates(
    start_date: str,
    end_date: str,
    freq: str,
) -> list[str]:
    """Generate rebalancing trade dates from a frequency code.

    Parameters
    ----------
    start_date, end_date : str
        YYYYMMDD bounds.
    freq : str
        ``"1D"``, ``"1W"``, ``"2W"``, ``"1M"``, ``"EOM"``.

    Returns
    -------
    list[str]
        Sorted list of YYYYMMDD rebalancing dates.
    """
    trade_dates = get_trade_dates(start_date, end_date)
    if not trade_dates:
        return []

    if freq == "1D":
        return trade_dates

    rebalance_dates: list[str] = []

    if freq == "1W":
        # First trade day of each week
        for i, d in enumerate(trade_dates):
            if i == 0:
                rebalance_dates.append(d)
            else:
                prev_dt = pd.Timestamp(trade_dates[i - 1])
                curr_dt = pd.Timestamp(d)
                if curr_dt.isocalendar()[1] != prev_dt.isocalendar()[1]:
                    rebalance_dates.append(d)

    elif freq == "2W":
        # Every other week's first trade day
        week_groups: list[list[str]] = []
        current_week: list[str] = [trade_dates[0]]
        for d in trade_dates[1:]:
            curr_dt = pd.Timestamp(d)
            prev_dt = pd.Timestamp(current_week[-1])
            if curr_dt.isocalendar()[1] != prev_dt.isocalendar()[1]:
                week_groups.append(current_week)
                current_week = [d]
            else:
                current_week.append(d)
        if current_week:
            week_groups.append(current_week)
        rebalance_dates = [g[0] for g in week_groups[::2]]

    elif freq == "1M":
        # First trade day of each month
        for i, d in enumerate(trade_dates):
            if i == 0:
                rebalance_dates.append(d)
            else:
                prev_dt = pd.Timestamp(trade_dates[i - 1])
                curr_dt = pd.Timestamp(d)
                if curr_dt.month != prev_dt.month:
                    rebalance_dates.append(d)

    elif freq == "EOM":
        # Last trade day of each month
        for i, d in enumerate(trade_dates):
            if i == len(trade_dates) - 1:
                rebalance_dates.append(d)
            else:
                next_dt = pd.Timestamp(trade_dates[i + 1])
                curr_dt = pd.Timestamp(d)
                if next_dt.month != curr_dt.month:
                    rebalance_dates.append(d)

    else:
        raise ValueError(f"Unknown rebalance frequency: {freq}")

    return rebalance_dates


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
                market_storage = MarketStorage()

            factor_ids = [f.id for f in self.config.factors]
            if not factor_ids:
                raise ValueError("No factors configured")

            # Load all factor data for the date range
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

            rebalance_dates = _get_rebalance_dates(
                start_date, end_date, self.config.rebalance_freq
            )

            signals = self.generate_signals(factor_panel, market_panel, rebalance_dates)

            # Apply delay: signal computed on rebalance_date → effective on rebalance_date + delay
            if self.config.delay > 0 and not signals.empty:
                signals = self._apply_delay(signals, rebalance_dates, self.config.delay)

            return signals

        finally:
            if own_factor and factor_storage is not None:
                factor_storage.close()
            if own_market and market_storage is not None:
                market_storage.close()

    @staticmethod
    def _load_factor_panel(
        factor_ids: list[str],
        start_date: str,
        end_date: str,
        factor_storage: FactorStorage,
    ) -> pd.DataFrame:
        """Load and merge multiple factors into a wide panel."""
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
