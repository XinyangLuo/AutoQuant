"""Universe filter: ST, new IPO, board (CYB/KCB), index members, liquidity."""

from __future__ import annotations

import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates
from backtest.strategy.config import UniverseConfig


class UniverseFilter:
    """Daily universe filter applied before stock selection.

    Filters are applied in order:
      1. ST/*ST removal
      2. New IPO removal (by trading days since listing)
      3. Board filter (CYB 30xxxx / KCB 68xxxx)
      4. Index membership filter
      5. Liquidity filter (min market cap / avg amount)
    """

    def __init__(self, config: UniverseConfig):
        self.config = config

    def filter(
        self,
        date: str,
        panel: pd.DataFrame,
        market_storage: MarketStorage | None = None,
    ) -> pd.DataFrame:
        """Filter a cross-section panel to the tradable universe.

        Parameters
        ----------
        date : str
            Trade date in YYYYMMDD format.
        panel : pd.DataFrame
            Cross-section from ``MarketStorage.get_panel(date)``.
            Expected columns: ``symbol, is_st, list_date, circ_mv, amount``.
        market_storage : MarketStorage | None
            Used for index-membership queries when ``config.index_members`` is set.

        Returns
        -------
        pd.DataFrame
            Filtered panel containing only stocks passing all filters.
        """
        df = panel.copy()
        if df.empty:
            return df

        # 1. ST/*ST filter
        if self.config.exclude_st and "is_st" in df.columns:
            df = df[df["is_st"].fillna(0).astype(int) == 0]

        # 2. New IPO filter
        if self.config.exclude_new_ipo_days and "list_date" in df.columns:
            df = df[df["list_date"].notna()]
            # Count trading days from list_date to current date
            # Use calendar days as a cheap proxy; precise implementation
            # would query the trade calendar.
            list_dt = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
            current_dt = pd.Timestamp(date)
            days_since_list = (current_dt - list_dt).dt.days
            # Trading days ≈ calendar days * 0.7; add slack
            min_calendar_days = int(self.config.exclude_new_ipo_days / 0.65)
            df = df[days_since_list >= min_calendar_days]

        # 3. Board filter
        if not self.config.include_cyb:
            df = df[~df["symbol"].str.startswith("30")]
        if not self.config.include_kcb:
            df = df[~df["symbol"].str.startswith("68")]

        # 4. Index membership filter
        if self.config.index_members and market_storage is not None:
            members = self._get_index_members(date, self.config.index_members, market_storage)
            df = df[df["symbol"].isin(members)]

        # 5. Liquidity filter
        if self.config.min_market_cap and "circ_mv" in df.columns:
            # circ_mv is stored in 万元 (Tushare convention) — convert to 元
            # so the threshold is unit-aligned with min_avg_amount.
            df = df[df["circ_mv"] * 10_000 >= self.config.min_market_cap]

        if self.config.min_avg_amount:
            df = self._filter_by_avg_amount(df, date, market_storage)

        return df.reset_index(drop=True)

    def _get_index_members(
        self,
        date: str,
        index_code: str,
        market_storage: MarketStorage,
    ) -> set[str]:
        """Return the set of symbols belonging to an index on a given date.

        Reads from ``MarketStorage.get_index_members`` which expects the
        ``index_members`` table to already be densified to every trade date
        (see ``backtest.data.backfill_index_members``).
        """
        tables = {
            r[0] for r in market_storage.conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        if "index_members" not in tables:
            raise NotImplementedError(
                "index_members table not found. Run "
                "`python -m backtest.data.backfill_index_members` first."
            )
        return market_storage.get_index_members(index_code, date)

    def _filter_by_avg_amount(
        self,
        df: pd.DataFrame,
        date: str,
        market_storage: MarketStorage | None,
    ) -> pd.DataFrame:
        """Filter by 20-day average amount."""
        if market_storage is None or self.config.min_avg_amount is None:
            return df

        # Query past 20 trading days of amount using a bounded calendar window.
        from datetime import datetime, timedelta

        from backtest.data.trade_calendar import get_trade_dates

        end_dt = datetime.strptime(date, "%Y%m%d")
        start_dt = (end_dt - timedelta(days=60)).strftime("%Y%m%d")
        trade_dates = get_trade_dates(start_dt, date)

        if date not in trade_dates:
            return df
        idx = trade_dates.index(date)
        start_idx = max(0, idx - 19)
        lookback_dates = trade_dates[start_idx : idx + 1]

        if len(lookback_dates) < 5:
            return df

        symbols = df["symbol"].tolist()
        bars = market_storage.get_bars(
            symbols=symbols,
            start=lookback_dates[0],
            end=lookback_dates[-1],
            columns=["amount"],
        )
        if bars.empty:
            return df

        avg_amount = bars.groupby("symbol")["amount"].mean()
        valid_symbols = avg_amount[avg_amount >= self.config.min_avg_amount].index
        return df[df["symbol"].isin(valid_symbols)]
