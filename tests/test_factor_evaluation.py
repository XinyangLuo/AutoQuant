"""Tests for factor evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.factor.evaluation import (
    _compute_forward_returns,
    _compute_ic_stats,
    _group_returns,
    _ic_series,
    _rank_ic_series,
    _turnover,
)


class TestICSeries:
    def test_perfect_positive_correlation(self):
        f = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _ic_series(f, r) == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative_correlation(self):
        f = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        assert _ic_series(f, r) == pytest.approx(-1.0, abs=1e-6)

    def test_no_correlation(self):
        f = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
        assert np.isnan(_ic_series(f, r))

    def test_insufficient_data(self):
        f = pd.Series([1.0, 2.0])
        r = pd.Series([1.0, 2.0])
        assert np.isnan(_ic_series(f, r))

    def test_with_nans(self):
        f = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        r = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _ic_series(f, r) == pytest.approx(1.0, abs=1e-6)


class TestRankICSeries:
    def test_perfect_positive_rank_correlation(self):
        f = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        r = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _rank_ic_series(f, r) == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative_rank_correlation(self):
        f = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        r = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        assert _rank_ic_series(f, r) == pytest.approx(-1.0, abs=1e-6)

    def test_nonlinear_monotonic(self):
        """Spearman detects monotonic relationship even if non-linear."""
        f = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = pd.Series([1.0, 4.0, 9.0, 16.0, 25.0])
        assert _rank_ic_series(f, r) == pytest.approx(1.0, abs=1e-6)


class TestComputeICStats:
    def test_basic(self):
        ic = pd.Series([0.1, 0.2, 0.1, -0.1, 0.0])
        stats = _compute_ic_stats(ic)
        assert stats["ic_mean"] == pytest.approx(0.06, abs=1e-6)
        assert stats["ic_count"] == 5
        assert stats["ic_positive_ratio"] == pytest.approx(0.6, abs=1e-6)

    def test_all_positive(self):
        ic = pd.Series([0.1, 0.2, 0.3])
        stats = _compute_ic_stats(ic)
        assert stats["ic_positive_ratio"] == 1.0

    def test_empty(self):
        ic = pd.Series([], dtype=float)
        stats = _compute_ic_stats(ic)
        assert np.isnan(stats["ic_mean"])
        assert stats["ic_count"] == 0


class TestForwardReturns:
    def test_close_to_close(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "symbol": "A",
            "close": [100.0, 102.0, 101.0, 103.0, 105.0],
            "open": [100.0, 101.0, 102.0, 100.0, 104.0],
        })
        result = _compute_forward_returns(df, [1, 2], "close")

        assert result.loc[0, "ret_1"] == pytest.approx(102.0 / 100.0 - 1, abs=1e-6)
        assert result.loc[1, "ret_1"] == pytest.approx(101.0 / 102.0 - 1, abs=1e-6)
        assert result.loc[0, "ret_2"] == pytest.approx(101.0 / 100.0 - 1, abs=1e-6)

    def test_open_to_open(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "symbol": "A",
            "close": [100.0] * 5,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        })
        result = _compute_forward_returns(df, [1], "open")

        assert result.loc[0, "ret_1"] == pytest.approx(102.0 / 101.0 - 1, abs=1e-6)
        assert result.loc[1, "ret_1"] == pytest.approx(103.0 / 102.0 - 1, abs=1e-6)

    def test_multiple_symbols(self):
        df = pd.DataFrame({
            "date": list(pd.date_range("2024-01-01", periods=3)) * 2,
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "close": [100.0, 110.0, 120.0, 200.0, 190.0, 210.0],
            "open": [100.0, 110.0, 120.0, 200.0, 190.0, 210.0],
        })
        result = _compute_forward_returns(df, [1], "close")

        a_ret = result[(result["symbol"] == "A") & (result["date"] == "2024-01-01")]["ret_1"].iloc[0]
        assert a_ret == pytest.approx(0.1, abs=1e-6)

        b_ret = result[(result["symbol"] == "B") & (result["date"] == "2024-01-01")]["ret_1"].iloc[0]
        assert b_ret == pytest.approx(-0.05, abs=1e-6)

    def test_last_rows_dropped(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "symbol": "A",
            "close": [100.0, 102.0, 101.0],
            "open": [100.0, 102.0, 101.0],
        })
        result = _compute_forward_returns(df, [2], "close")
        assert len(result) == 1


class TestTurnover:
    def test_perfect_stability(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "symbol": ["A", "B", "A", "B"],
            "value": [1.0, 2.0, 1.0, 2.0],
        })
        assert _turnover(df) == pytest.approx(0.0, abs=1e-6)

    def test_complete_reversal(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "symbol": ["A", "B", "A", "B"],
            "value": [1.0, 2.0, 2.0, 1.0],
        })
        turnover = _turnover(df)
        assert turnover > 0.5


class TestGroupReturns:
    @pytest.mark.parametrize("ret_direction", ["same", "opposite"])
    def test_monotonic_direction(self, ret_direction):
        df = pd.DataFrame({
            "date": ["2024-01-01"] * 100,
            "symbol": [f"S{i}" for i in range(100)],
            "value": list(range(100)),
            "ret_1": list(range(100)) if ret_direction == "same" else list(reversed(range(100))),
        })
        result = _group_returns(df, "ret_1", n_groups=10)

        if ret_direction == "same":
            assert result["mean_ret"].iloc[-1] > result["mean_ret"].iloc[0]
        else:
            assert result["mean_ret"].iloc[0] > result["mean_ret"].iloc[-1]

    def test_multiple_dates(self):
        df = pd.DataFrame({
            "date": ["2024-01-01"] * 10 + ["2024-01-02"] * 10,
            "symbol": [f"S{i}" for i in range(10)] * 2,
            "value": list(range(10)) * 2,
            "ret_1": list(range(10)) * 2,
        })
        result = _group_returns(df, "ret_1", n_groups=5)

        assert len(result) == 5
        for i in range(1, 5):
            assert result["mean_ret"].iloc[i] >= result["mean_ret"].iloc[i - 1]
