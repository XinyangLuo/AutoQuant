"""Tests for trade_calendar module."""

from __future__ import annotations

import pandas as pd
import pytest


def _make_test_data() -> pd.DataFrame:
    """Build Jan 2024 trade calendar test data."""
    dates = pd.to_datetime([
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
        "2024-01-12", "2024-01-15", "2024-01-16", "2024-01-17",
        "2024-01-18", "2024-01-19", "2024-01-22", "2024-01-23",
        "2024-01-24", "2024-01-25", "2024-01-26", "2024-01-29",
        "2024-01-30", "2024-01-31",
    ]).date

    is_week_first = [
        True, False, False, False,
        True, False, False, False, False,
        True, False, False, False, False,
        True, False, False, False, False,
        True, False, False,
    ]
    is_week_last = [
        False, False, False, True,
        False, False, False, False, True,
        False, False, False, False, True,
        False, False, False, False, True,
        False, False, True,
    ]
    is_month_first = [True] + [False] * 21
    is_month_last = [False] * 21 + [True]

    return pd.DataFrame({
        "cal_date": dates,
        "is_open": [True] * 22,
        "is_week_first": is_week_first,
        "is_week_last": is_week_last,
        "is_month_first": is_month_first,
        "is_month_last": is_month_last,
    })


@pytest.fixture
def temp_storage(monkeypatch, tmp_path):
    """Provide a MarketStorage backed by a temporary DuckDB."""
    import backtest.data.storage as storage_module

    monkeypatch.setattr(storage_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(storage_module, "DB_PATH", tmp_path / "test.duckdb")

    from backtest.data.storage import MarketStorage

    with MarketStorage() as storage:
        storage.insert_trade_calendar(_make_test_data())
        yield storage


class TestGetRebalanceDates:
    """Test get_rebalance_dates with DB-backed data."""

    def test_1d(self, temp_storage):
        """1D returns all trade dates."""
        from backtest.data.trade_calendar import get_rebalance_dates

        dates = get_rebalance_dates("20240102", "20240110", "1D")
        assert len(dates) == 7
        assert dates[0] == "20240102"
        assert dates[-1] == "20240110"

    def test_5d(self, temp_storage):
        """5D returns every 5th trade date."""
        from backtest.data.trade_calendar import get_rebalance_dates

        dates = get_rebalance_dates("20240102", "20240131", "5D")
        assert dates[0] == "20240102"
        assert dates[1] == "20240109"
        assert dates[2] == "20240116"

    def test_1w(self, temp_storage):
        """1W returns first trade day of each ISO week."""
        from backtest.data.trade_calendar import get_rebalance_dates

        dates = get_rebalance_dates("20240102", "20240131", "1W")
        assert len(dates) == 5
        assert dates == ["20240102", "20240108", "20240115", "20240122", "20240129"]

    def test_2w(self, temp_storage):
        """2W returns first trade day of even ISO weeks."""
        from backtest.data.trade_calendar import get_rebalance_dates

        dates = get_rebalance_dates("20240102", "20240131", "2W")
        # 2024-01-02 is ISO week 1 (odd), 2024-01-08 is week 2 (even), etc.
        # Only even weeks: week 2 (20240108), week 4 (20240122)
        assert dates == ["20240108", "20240122"]

    def test_1m(self, temp_storage):
        """1M returns first trade day of each month."""
        from backtest.data.trade_calendar import get_rebalance_dates

        dates = get_rebalance_dates("20240102", "20240131", "1M")
        assert dates == ["20240102"]

    def test_eom(self, temp_storage):
        """EOM returns last trade day of each month."""
        from backtest.data.trade_calendar import get_rebalance_dates

        dates = get_rebalance_dates("20240102", "20240131", "EOM")
        assert dates == ["20240131"]

    def test_unknown_freq(self):
        """Unknown frequency raises ValueError."""
        from backtest.data.trade_calendar import get_rebalance_dates

        with pytest.raises(ValueError, match="Unknown rebalance frequency"):
            get_rebalance_dates("20240101", "20240131", "3W")


class TestGetTradeDates:
    """Test get_trade_dates DuckDB-first behavior."""

    def test_reads_from_db(self, temp_storage):
        """Should return dates from DB when data exists."""
        from backtest.data.trade_calendar import get_trade_dates

        dates = get_trade_dates("20240102", "20240105")
        assert dates == ["20240102", "20240103", "20240104", "20240105"]

    def test_empty_range(self, temp_storage, monkeypatch):
        """Should return empty list when no dates in range."""
        from backtest.data import trade_calendar as tc_module
        from backtest.data.trade_calendar import get_trade_dates

        monkeypatch.setattr(tc_module, "_fetch_and_write", lambda s, e: [])
        dates = get_trade_dates("20231201", "20231231")
        assert dates == []
