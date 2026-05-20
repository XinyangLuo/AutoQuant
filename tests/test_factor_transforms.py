"""Tests for factor transforms operators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.factor.transforms import (
    abs_,
    cs_demean,
    cs_winsorize,
    cs_zscore,
    if_else,
    inverse,
    log,
    rank,
    sign,
    signed_power,
    sqrt,
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_covariance,
    ts_decay_exp,
    ts_decay_linear,
    ts_delta,
    ts_delay,
    ts_ir,
    ts_kurtosis,
    ts_max,
    ts_mean,
    ts_min,
    ts_pct_change,
    ts_product,
    ts_rank,
    ts_skewness,
    ts_std,
    ts_sum,
    z_score,
)


def _make_index(pairs: list[tuple[str, str]]) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(pairs, names=["date", "symbol"])


# ---------------------------------------------------------------------------
# rank (existing, keep)
# ---------------------------------------------------------------------------

class TestRank:
    def test_basic_ascending(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([10.0, 20.0, 30.0], index=idx)
        result = rank(s)
        assert result.tolist() == pytest.approx([0.0, 0.5, 1.0])

    def test_descending(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([10.0, 20.0, 30.0], index=idx)
        result = rank(s, ascending=False)
        assert result.tolist() == pytest.approx([1.0, 0.5, 0.0])

    def test_ties_use_average_rank(self):
        """Two tied minima share rank 1.5 -> (1.5-1)/(3-1) = 0.25."""
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([10.0, 10.0, 30.0], index=idx)
        result = rank(s)
        assert result.tolist() == pytest.approx([0.25, 0.25, 1.0])

    def test_nan_preserved(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([10.0, np.nan, 30.0], index=idx)
        result = rank(s)
        assert result.iloc[0] == pytest.approx(0.0)
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(1.0)

    def test_single_non_nan_yields_half(self):
        """A date with one non-NaN value can't be ranked; map to 0.5."""
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([10.0, np.nan, np.nan], index=idx)
        result = rank(s)
        assert result.iloc[0] == pytest.approx(0.5)
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])

    def test_range_is_bounded(self):
        rng = np.random.default_rng(42)
        n_days, n_syms = 30, 20
        pairs = [
            (f"2024-{m:02d}-01", f"S{i:02d}")
            for m in range(1, n_days + 1)
            for i in range(n_syms)
        ]
        idx = _make_index(pairs)
        s = pd.Series(rng.standard_normal(n_days * n_syms), index=idx)
        result = rank(s)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_independent_per_date(self):
        """Day 2's scale must not leak into day 1's ranks."""
        idx = _make_index([
            ("2024-01-01", "A"), ("2024-01-01", "B"),
            ("2024-01-02", "A"), ("2024-01-02", "B"),
        ])
        s = pd.Series([1.0, 2.0, 100.0, 200.0], index=idx)
        result = rank(s)
        assert result.loc[("2024-01-01", "A")] == pytest.approx(0.0)
        assert result.loc[("2024-01-01", "B")] == pytest.approx(1.0)
        assert result.loc[("2024-01-02", "A")] == pytest.approx(0.0)
        assert result.loc[("2024-01-02", "B")] == pytest.approx(1.0)

    def test_preserves_index(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B")])
        s = pd.Series([1.0, 2.0], index=idx)
        result = rank(s)
        assert result.index.equals(idx)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            rank(s)


# ---------------------------------------------------------------------------
# z_score (existing, keep)
# ---------------------------------------------------------------------------

class TestZScore:
    def test_basic_per_symbol(self):
        """Window=3, arithmetic sequence per symbol -> trailing z = 1.0."""
        dates = [f"2024-01-0{i}" for i in range(1, 8)]
        pairs = [(d, s) for d in dates for s in ("A", "B")]
        idx = _make_index(pairs)
        vals = []
        for i in range(1, 8):
            vals.extend([float(i), float(i * 100)])
        s = pd.Series(vals, index=idx)

        result = z_score(s, window=3)

        assert np.isnan(result.loc[("2024-01-01", "A")])
        assert np.isnan(result.loc[("2024-01-02", "A")])
        assert result.loc[("2024-01-03", "A")] == pytest.approx(1.0)
        assert result.loc[("2024-01-03", "B")] == pytest.approx(1.0)
        assert result.loc[("2024-01-07", "A")] == pytest.approx(1.0)

    def test_constant_series_yields_nan(self):
        """std=0 must yield NaN, not inf, from the safe-divide path."""
        idx = _make_index([(f"2024-01-0{i}", "A") for i in range(1, 6)])
        s = pd.Series([5.0] * 5, index=idx)
        result = z_score(s, window=3)
        assert result.iloc[2:].isna().all()

    def test_min_periods_override(self):
        """min_periods=2 produces a z-score at index 1 (window not yet full)."""
        idx = _make_index([(f"2024-01-0{i}", "A") for i in range(1, 6)])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        result = z_score(s, window=5, min_periods=2)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx((2.0 - 1.5) / np.sqrt(0.5))

    def test_preserves_input_index_order(self):
        """Sort is internal; the returned Series matches the caller's order."""
        idx = _make_index([
            ("2024-01-02", "A"),
            ("2024-01-01", "A"),
            ("2024-01-03", "A"),
        ])
        s = pd.Series([2.0, 1.0, 3.0], index=idx)
        result = z_score(s, window=2, min_periods=2)
        assert list(result.index) == list(idx)

    def test_window_below_two_raises(self):
        idx = _make_index([("2024-01-01", "A")])
        s = pd.Series([1.0], index=idx)
        with pytest.raises(ValueError, match="window"):
            z_score(s, window=1)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            z_score(s, window=2)

    def test_symbols_are_independent(self):
        """A's history must not leak into B's z-score under groupby+rolling."""
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        pairs = [(d, s) for d in dates for s in ("A", "B")]
        idx = _make_index(pairs)
        vals = []
        for i in range(1, 6):
            vals.extend([10.0, float(i)])
        s = pd.Series(vals, index=idx)

        result = z_score(s, window=3)

        assert np.isnan(result.loc[("2024-01-03", "A")])
        assert result.loc[("2024-01-03", "B")] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ts_mean / ts_std / ts_rank (existing, keep)
# ---------------------------------------------------------------------------

class TestTsMean:
    def test_basic(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

        result = ts_mean(s, window=3)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(3.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_symbols_are_independent(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        pairs = [(d, s) for d in dates for s in ("A", "B")]
        idx = _make_index(pairs)
        vals = []
        for i in range(1, 6):
            vals.extend([float(i), float(i * 10)])
        s = pd.Series(vals, index=idx)

        result = ts_mean(s, window=3)

        assert result.loc[("2024-01-05", "A")] == pytest.approx(4.0)
        assert result.loc[("2024-01-05", "B")] == pytest.approx(40.0)

    def test_preserves_input_index_order(self):
        idx = _make_index([
            ("2024-01-02", "A"),
            ("2024-01-01", "A"),
            ("2024-01-03", "A"),
        ])
        s = pd.Series([2.0, 1.0, 3.0], index=idx)
        result = ts_mean(s, window=2, min_periods=2)
        assert list(result.index) == list(idx)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_mean(s, window=2)


class TestTsStd:
    def test_basic(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

        result = ts_std(s, window=3)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # std([1,2,3]) = 1.0
        assert result.iloc[2] == pytest.approx(1.0)
        assert result.iloc[3] == pytest.approx(1.0)
        assert result.iloc[4] == pytest.approx(1.0)

    def test_constant_series(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([5.0] * 5, index=idx)

        result = ts_std(s, window=3)

        # pandas rolling std returns 0.0 for constant windows, not NaN
        assert (result.iloc[2:] == 0.0).all()

    def test_symbols_are_independent(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        pairs = [(d, s) for d in dates for s in ("A", "B")]
        idx = _make_index(pairs)
        vals = []
        for i in range(1, 6):
            vals.extend([float(i), 10.0])
        s = pd.Series(vals, index=idx)

        result = ts_std(s, window=3)

        # A varies, B is constant
        assert result.loc[("2024-01-05", "A")] == pytest.approx(1.0)
        assert result.loc[("2024-01-05", "B")] == pytest.approx(0.0)

    def test_preserves_input_index_order(self):
        idx = _make_index([
            ("2024-01-02", "A"),
            ("2024-01-01", "A"),
            ("2024-01-03", "A"),
        ])
        s = pd.Series([2.0, 1.0, 3.0], index=idx)
        result = ts_std(s, window=2, min_periods=2)
        assert list(result.index) == list(idx)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_std(s, window=2)

    def test_window_below_two_raises(self):
        idx = _make_index([("2024-01-01", "A")])
        s = pd.Series([1.0], index=idx)
        with pytest.raises(ValueError, match="window"):
            ts_std(s, window=1)


class TestTsRank:
    def test_basic(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

        result = ts_rank(s, window=3)

        # window=3: first 2 are NaN (min_periods defaults to 3)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # 2024-01-03: [1, 2, 3] -> rank(3)=3 -> (3-1)/(3-1)*2-1 = 1.0
        assert result.iloc[2] == pytest.approx(1.0)
        # 2024-01-04: [2, 3, 4] -> rank(4)=3 -> 1.0
        assert result.iloc[3] == pytest.approx(1.0)
        # 2024-01-05: [3, 4, 5] -> rank(5)=3 -> 1.0
        assert result.iloc[4] == pytest.approx(1.0)

    def test_min_maps_to_minus_one(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=idx)

        result = ts_rank(s, window=3)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # 2024-01-03: [5, 4, 3] -> rank(3)=1 -> (1-1)/(3-1)*2-1 = -1.0
        assert result.iloc[2] == pytest.approx(-1.0)
        # 2024-01-04: [4, 3, 2] -> rank(2)=1 -> -1.0
        assert result.iloc[3] == pytest.approx(-1.0)
        # 2024-01-05: [3, 2, 1] -> rank(1)=1 -> -1.0
        assert result.iloc[4] == pytest.approx(-1.0)

    def test_middle_value(self):
        dates = [f"2024-01-0{i}" for i in range(1, 5)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 3.0, 2.0], index=idx)

        result = ts_rank(s, window=3)

        # 2024-01-03: [1, 2, 3] -> rank(3)=3 -> 1.0
        assert result.iloc[2] == pytest.approx(1.0)
        # 2024-01-04: [2, 3, 2] -> ties: ranks are 1.5, 3, 1.5 -> rank(2)=1.5
        # -> (1.5-1)/(3-1)*2-1 = 0.5/2*2-1 = -0.5
        assert result.iloc[3] == pytest.approx(-0.5)

    def test_single_element_window(self):
        """Window with only 1 valid element -> 0.0."""
        dates = [f"2024-01-0{i}" for i in range(1, 3)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([5.0, 10.0], index=idx)

        result = ts_rank(s, window=3, min_periods=1)

        # First valid: only 5.0 -> 0.0
        assert result.iloc[0] == pytest.approx(0.0)
        # Second: [5, 10] -> rank(10)=2 -> (2-1)/(2-1)*2-1 = 1.0
        assert result.iloc[1] == pytest.approx(1.0)

    def test_range_bounded(self):
        rng = np.random.default_rng(42)
        n_days, n_syms = 30, 10
        pairs = [
            (f"2024-{m:02d}-01", f"S{i:02d}")
            for m in range(1, n_days + 1)
            for i in range(n_syms)
        ]
        idx = _make_index(pairs)
        s = pd.Series(rng.standard_normal(n_days * n_syms), index=idx)
        result = ts_rank(s, window=5)
        valid = result.dropna()
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_symbols_are_independent(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        pairs = [(d, s) for d in dates for s in ("A", "B")]
        idx = _make_index(pairs)
        vals = []
        for i in range(1, 6):
            vals.extend([float(i), float(10 - i)])
        s = pd.Series(vals, index=idx)

        result = ts_rank(s, window=3)

        # A is ascending -> latest is max -> 1.0
        assert result.loc[("2024-01-05", "A")] == pytest.approx(1.0)
        # B is descending -> latest is min -> -1.0
        assert result.loc[("2024-01-05", "B")] == pytest.approx(-1.0)

    def test_preserves_input_index_order(self):
        idx = _make_index([
            ("2024-01-02", "A"),
            ("2024-01-01", "A"),
            ("2024-01-03", "A"),
        ])
        s = pd.Series([2.0, 1.0, 3.0], index=idx)
        result = ts_rank(s, window=2, min_periods=2)
        assert list(result.index) == list(idx)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_rank(s, window=2)

    def test_window_below_two_raises(self):
        idx = _make_index([("2024-01-01", "A")])
        s = pd.Series([1.0], index=idx)
        with pytest.raises(ValueError, match="window"):
            ts_rank(s, window=1)

    def test_with_nan_in_window(self):
        """NaN inside rolling window: pandas min_periods counts non-NaN values."""
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0], index=idx)

        result = ts_rank(s, window=3)

        # window=3, min_periods=3: [nan, 3, 4] has only 2 non-NaN -> NaN
        assert np.isnan(result.iloc[3])
        # [3, 4, 5] has 3 non-NaN -> rank(5)=3 -> 1.0
        assert result.iloc[4] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# New operators - batch 1: core time-series
# ---------------------------------------------------------------------------

class TestTsSum:
    def test_basic(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

        result = ts_sum(s, window=3)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(6.0)
        assert result.iloc[3] == pytest.approx(9.0)
        assert result.iloc[4] == pytest.approx(12.0)

    def test_symbols_are_independent(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        pairs = [(d, s) for d in dates for s in ("A", "B")]
        idx = _make_index(pairs)
        vals = []
        for i in range(1, 6):
            vals.extend([float(i), float(i * 10)])
        s = pd.Series(vals, index=idx)

        result = ts_sum(s, window=3)

        assert result.loc[("2024-01-05", "A")] == pytest.approx(12.0)
        assert result.loc[("2024-01-05", "B")] == pytest.approx(120.0)

    def test_preserves_input_index_order(self):
        idx = _make_index([
            ("2024-01-02", "A"),
            ("2024-01-01", "A"),
            ("2024-01-03", "A"),
        ])
        s = pd.Series([2.0, 1.0, 3.0], index=idx)
        result = ts_sum(s, window=2, min_periods=2)
        assert list(result.index) == list(idx)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_sum(s, window=2)


class TestTsMin:
    def test_basic(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([5.0, 4.0, 3.0, 4.0, 5.0], index=idx)

        result = ts_min(s, window=3)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(3.0)
        assert result.iloc[3] == pytest.approx(3.0)
        assert result.iloc[4] == pytest.approx(3.0)

    def test_symbols_are_independent(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        pairs = [(d, s) for d in dates for s in ("A", "B")]
        idx = _make_index(pairs)
        vals = []
        for i in range(1, 6):
            vals.extend([float(i), float(10 - i)])
        s = pd.Series(vals, index=idx)

        result = ts_min(s, window=3)

        assert result.loc[("2024-01-05", "A")] == pytest.approx(3.0)
        assert result.loc[("2024-01-05", "B")] == pytest.approx(5.0)


class TestTsMax:
    def test_basic(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0], index=idx)

        result = ts_max(s, window=3)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(3.0)
        assert result.iloc[3] == pytest.approx(3.0)
        assert result.iloc[4] == pytest.approx(3.0)


class TestTsArgmax:
    def test_current_is_max(self):
        """When current is the max, distance from end is 0."""
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

        result = ts_argmax(s, window=3)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # [1,2,3] -> max=3 at end -> distance 0
        assert result.iloc[2] == pytest.approx(0.0)
        # [2,3,4] -> max=4 at end -> distance 0
        assert result.iloc[3] == pytest.approx(0.0)
        # [3,4,5] -> max=5 at end -> distance 0
        assert result.iloc[4] == pytest.approx(0.0)

    def test_earliest_is_max(self):
        """When earliest is the max, distance from end is window-1."""
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=idx)

        result = ts_argmax(s, window=3)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # [5,4,3] -> max=5 at position 0 -> distance from end = 2
        assert result.iloc[2] == pytest.approx(2.0)
        # [4,3,2] -> max=4 at position 0 -> distance = 2
        assert result.iloc[3] == pytest.approx(2.0)
        # [3,2,1] -> max=3 at position 0 -> distance = 2
        assert result.iloc[4] == pytest.approx(2.0)

    def test_max_in_middle(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 5.0, 3.0, 4.0, 2.0], index=idx)

        result = ts_argmax(s, window=3)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # [1,5,3] -> max=5 at position 1 (middle) -> distance from end = 1
        assert result.iloc[2] == pytest.approx(1.0)
        # [5,3,4] -> max=5 at position 0 -> distance = 2
        assert result.iloc[3] == pytest.approx(2.0)
        # [3,4,2] -> max=4 at position 1 -> distance = 1
        assert result.iloc[4] == pytest.approx(1.0)


class TestTsArgmin:
    def test_current_is_min(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=idx)

        result = ts_argmin(s, window=3)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # [5,4,3] -> min=3 at end -> distance 0
        assert result.iloc[2] == pytest.approx(0.0)

    def test_earliest_is_min(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

        result = ts_argmin(s, window=3)

        # [1,2,3] -> min=1 at position 0 -> distance = 2
        assert result.iloc[2] == pytest.approx(2.0)


class TestTsDelta:
    def test_basic(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 5.0, 4.0, 6.0], index=idx)

        result = ts_delta(s, d=2)

        # d=2: first 2 are NaN
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # 5.0 - 1.0 = 4.0
        assert result.iloc[2] == pytest.approx(4.0)
        # 4.0 - 2.0 = 2.0
        assert result.iloc[3] == pytest.approx(2.0)
        # 6.0 - 5.0 = 1.0
        assert result.iloc[4] == pytest.approx(1.0)

    def test_d_one(self):
        dates = [f"2024-01-0{i}" for i in range(1, 5)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 3.0, 2.0, 4.0], index=idx)

        result = ts_delta(s, d=1)

        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(2.0)
        assert result.iloc[2] == pytest.approx(-1.0)
        assert result.iloc[3] == pytest.approx(2.0)

    def test_symbols_are_independent(self):
        dates = [f"2024-01-0{i}" for i in range(1, 5)]
        pairs = [(d, s) for d in dates for s in ("A", "B")]
        idx = _make_index(pairs)
        vals = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]
        s = pd.Series(vals, index=idx)

        result = ts_delta(s, d=1)

        assert np.isnan(result.loc[("2024-01-01", "A")])
        assert result.loc[("2024-01-02", "A")] == pytest.approx(1.0)
        assert result.loc[("2024-01-02", "B")] == pytest.approx(10.0)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_delta(s, d=1)

    def test_d_must_be_positive(self):
        idx = _make_index([("2024-01-01", "A")])
        s = pd.Series([1.0], index=idx)
        with pytest.raises(ValueError, match="d must be >= 1"):
            ts_delta(s, d=0)


class TestTsDelay:
    def test_basic(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

        result = ts_delay(s, d=2)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(1.0)
        assert result.iloc[3] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(3.0)

    def test_d_one(self):
        dates = [f"2024-01-0{i}" for i in range(1, 5)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 3.0, 2.0, 4.0], index=idx)

        result = ts_delay(s, d=1)

        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(1.0)
        assert result.iloc[2] == pytest.approx(3.0)
        assert result.iloc[3] == pytest.approx(2.0)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_delay(s, d=1)

    def test_d_must_be_positive(self):
        idx = _make_index([("2024-01-01", "A")])
        s = pd.Series([1.0], index=idx)
        with pytest.raises(ValueError, match="d must be >= 1"):
            ts_delay(s, d=0)


class TestTsPctChange:
    def test_basic(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([100.0, 110.0, 121.0, 110.0, 99.0], index=idx)

        result = ts_pct_change(s, d=1)

        assert np.isnan(result.iloc[0])
        # (110 - 100) / 100 = 0.10
        assert result.iloc[1] == pytest.approx(0.10)
        # (121 - 110) / 110 = 0.10
        assert result.iloc[2] == pytest.approx(0.10)
        # (110 - 121) / 121 = -0.0909...
        assert result.iloc[3] == pytest.approx(-11 / 121)

    def test_d_two(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([100.0, 110.0, 121.0, 110.0, 99.0], index=idx)

        result = ts_pct_change(s, d=2)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # (121 - 100) / 100 = 0.21
        assert result.iloc[2] == pytest.approx(0.21)

    def test_divide_by_zero(self):
        dates = [f"2024-01-0{i}" for i in range(1, 4)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([0.0, 0.0, 5.0], index=idx)

        result = ts_pct_change(s, d=1)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(np.inf)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_pct_change(s, d=1)


class TestTsProduct:
    def test_basic(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

        result = ts_product(s, window=3)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(6.0)   # 1*2*3
        assert result.iloc[3] == pytest.approx(24.0)  # 2*3*4
        assert result.iloc[4] == pytest.approx(60.0)  # 3*4*5

    def test_with_nan(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0], index=idx)

        result = ts_product(s, window=3)

        # window=3, min_periods=3: [nan, 3, 4] has only 2 non-NaN -> NaN
        assert np.isnan(result.iloc[3])
        # [3, 4, 5] has 3 non-NaN -> 60
        assert result.iloc[4] == pytest.approx(60.0)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_product(s, window=2)


# ---------------------------------------------------------------------------
# New operators - batch 2: statistical time-series
# ---------------------------------------------------------------------------

class TestTsSkewness:
    def test_symmetric_yields_near_zero(self):
        dates = [f"2024-01-{i:02d}" for i in range(1, 8)]
        idx = _make_index([(d, "A") for d in dates])
        # Symmetric around 0: [-3, -2, -1, 0, 1, 2, 3]
        s = pd.Series([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], index=idx)

        result = ts_skewness(s, window=5)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
        # [-3, -2, -1, 0, 1] is skewed right? No, roughly symmetric
        # Just check it's a reasonable number
        assert abs(result.iloc[4]) < 1.0

    def test_right_skew(self):
        dates = [f"2024-01-{i:02d}" for i in range(1, 8)]
        idx = _make_index([(d, "A") for d in dates])
        # Right-skewed: most values low, a few high
        s = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 10.0], index=idx)

        result = ts_skewness(s, window=5)

        # Last window [1,1,2,10]... wait that's only 4. Full window: [1,1,1,2,10]
        # This should be positively skewed
        assert result.iloc[6] > 0.5

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_skewness(s, window=3)

    def test_window_below_three_raises(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-02", "A")])
        s = pd.Series([1.0, 2.0], index=idx)
        with pytest.raises(ValueError, match="window"):
            ts_skewness(s, window=2)


class TestTsKurtosis:
    def test_basic(self):
        dates = [f"2024-01-{i:02d}" for i in range(1, 9)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], index=idx)

        result = ts_kurtosis(s, window=5)

        # First 4 should be NaN (need 5 obs, but kurt needs 4 so first 3 are NaN)
        # Actually rolling kurt with window=5, min_periods=5: first 4 NaN
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
        assert np.isnan(result.iloc[3])

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_kurtosis(s, window=4)

    def test_window_below_four_raises(self):
        idx = _make_index([("2024-01-01", "A") for _ in range(3)])
        s = pd.Series([1.0, 2.0, 3.0], index=idx)
        with pytest.raises(ValueError, match="window"):
            ts_kurtosis(s, window=3)


class TestTsIr:
    def test_basic(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

        result = ts_ir(s, window=3)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # mean([1,2,3])=2, std([1,2,3])=1 -> IR=2
        assert result.iloc[2] == pytest.approx(2.0)
        # mean([2,3,4])=3, std=1 -> IR=3
        assert result.iloc[3] == pytest.approx(3.0)

    def test_constant_yields_nan(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([5.0] * 5, index=idx)

        result = ts_ir(s, window=3)

        # std=0 -> IR=NaN (safe divide)
        assert result.iloc[2:].isna().all()

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_ir(s, window=2)

    def test_window_below_two_raises(self):
        idx = _make_index([("2024-01-01", "A")])
        s = pd.Series([1.0], index=idx)
        with pytest.raises(ValueError, match="window"):
            ts_ir(s, window=1)


class TestTsCorr:
    def test_perfect_positive_corr(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)

        result = ts_corr(x, y, window=3)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(1.0)
        assert result.iloc[3] == pytest.approx(1.0)
        assert result.iloc[4] == pytest.approx(1.0)

    def test_perfect_negative_corr(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        y = pd.Series([50.0, 40.0, 30.0, 20.0, 10.0], index=idx)

        result = ts_corr(x, y, window=3)

        assert result.iloc[2] == pytest.approx(-1.0)
        assert result.iloc[3] == pytest.approx(-1.0)
        assert result.iloc[4] == pytest.approx(-1.0)

    def test_symbols_are_independent(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        # idx order: (d1,A), (d1,B), (d2,A), (d2,B), ...
        pairs = [(d, s) for d in dates for s in ("A", "B")]
        idx = _make_index(pairs)
        # A: x=[1,2,3,4,5], y=[10,20,30,40,50] -> perfectly correlated
        # B: x=[1,2,3,4,5], y=[50,40,30,20,10] -> perfectly negatively correlated
        x_vals = []
        y_vals = []
        for i in range(1, 6):
            x_vals.extend([float(i), float(i)])
            y_vals.extend([float(i * 10), float((6 - i) * 10)])
        x = pd.Series(x_vals, index=idx)
        y = pd.Series(y_vals, index=idx)

        result = ts_corr(x, y, window=3)

        assert result.loc[("2024-01-05", "A")] == pytest.approx(1.0)
        assert result.loc[("2024-01-05", "B")] == pytest.approx(-1.0)

    def test_mismatched_index_raises(self):
        idx1 = _make_index([("2024-01-01", "A"), ("2024-01-02", "A")])
        idx2 = _make_index([("2024-01-01", "A"), ("2024-01-02", "B")])
        x = pd.Series([1.0, 2.0], index=idx1)
        y = pd.Series([1.0, 2.0], index=idx2)

        with pytest.raises(ValueError, match="identical MultiIndex"):
            ts_corr(x, y, window=2)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_corr(s, s, window=2)


class TestTsCovariance:
    def test_basic(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)

        result = ts_covariance(x, y, window=3)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # cov([1,2,3], [10,20,30]) = 1.0 * 100 = 100? Let's compute:
        # mean_x = 2, mean_y = 20
        # (1-2)*(10-20) + (2-2)*(20-20) + (3-2)*(30-20) = 10 + 0 + 10 = 20
        # / (3-1) = 10
        assert result.iloc[2] == pytest.approx(10.0)

    def test_mismatched_index_raises(self):
        idx1 = _make_index([("2024-01-01", "A"), ("2024-01-02", "A")])
        idx2 = _make_index([("2024-01-01", "A"), ("2024-01-02", "B")])
        x = pd.Series([1.0, 2.0], index=idx1)
        y = pd.Series([1.0, 2.0], index=idx2)

        with pytest.raises(ValueError, match="identical MultiIndex"):
            ts_covariance(x, y, window=2)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_covariance(s, s, window=2)


# ---------------------------------------------------------------------------
# New operators - batch 3: decay / weighted
# ---------------------------------------------------------------------------

class TestTsDecayLinear:
    def test_weights_increase_toward_end(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=idx)

        result = ts_decay_linear(s, window=3)

        # All 1s -> weighted avg = 1 regardless of weights
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(1.0)

    def test_weights_favor_recent(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

        result = ts_decay_linear(s, window=3)

        # window [3,4,5]: weights [1,2,3]/6
        # (3*1 + 4*2 + 5*3) / 6 = (3 + 8 + 15) / 6 = 26/6 = 4.333...
        assert result.iloc[4] == pytest.approx(26.0 / 6.0)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_decay_linear(s, window=2)


class TestTsDecayExp:
    def test_all_ones(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=idx)

        result = ts_decay_exp(s, window=3, halflife=10)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(1.0)

    def test_halflife_effect(self):
        dates = [f"2024-01-0{i}" for i in range(1, 6)]
        idx = _make_index([(d, "A") for d in dates])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

        result = ts_decay_exp(s, window=3, halflife=1.0)

        # With small halflife, recent obs get much more weight
        # weights for window [oldest, middle, newest] with halflife=1:
        # ages = [2, 1, 0], weights = [0.25, 0.5, 1.0]
        # normalized: [0.25, 0.5, 1.0] / 1.75
        # For [3,4,5]: (3*0.25 + 4*0.5 + 5*1.0) / 1.75 = (0.75 + 2 + 5) / 1.75 = 7.75/1.75
        expected = 7.75 / 1.75
        assert result.iloc[4] == pytest.approx(expected)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ts_decay_exp(s, window=2)

    def test_invalid_halflife(self):
        idx = _make_index([("2024-01-01", "A")])
        s = pd.Series([1.0], index=idx)
        with pytest.raises(ValueError, match="halflife"):
            ts_decay_exp(s, window=1, halflife=0)


# ---------------------------------------------------------------------------
# New operators - batch 4: cross-sectional
# ---------------------------------------------------------------------------

class TestCsZscore:
    def test_basic(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([1.0, 2.0, 3.0], index=idx)

        result = cs_zscore(s)

        # mean=2, std=1 -> [-1, 0, 1]
        assert result.iloc[0] == pytest.approx(-1.0)
        assert result.iloc[1] == pytest.approx(0.0)
        assert result.iloc[2] == pytest.approx(1.0)

    def test_nan_preserved(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([1.0, np.nan, 3.0], index=idx)

        result = cs_zscore(s)

        assert np.isnan(result.iloc[1])
        # mean of [1,3]=2, std=sqrt(2) -> (1-2)/sqrt(2), (3-2)/sqrt(2)
        assert result.iloc[0] == pytest.approx(-1.0 / np.sqrt(2))
        assert result.iloc[2] == pytest.approx(1.0 / np.sqrt(2))

    def test_single_value_maps_to_zero(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([5.0, np.nan, np.nan], index=idx)

        result = cs_zscore(s)

        # Only one non-NaN -> mapped to 0
        assert result.iloc[0] == pytest.approx(0.0)

    def test_constant_maps_to_zero(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([5.0, 5.0, 5.0], index=idx)

        result = cs_zscore(s)

        # std=0 -> all mapped to 0
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(0.0)
        assert result.iloc[2] == pytest.approx(0.0)

    def test_independent_per_date(self):
        idx = _make_index([
            ("2024-01-01", "A"), ("2024-01-01", "B"),
            ("2024-01-02", "A"), ("2024-01-02", "B"),
        ])
        s = pd.Series([1.0, 3.0, 100.0, 200.0], index=idx)

        result = cs_zscore(s)

        # Day 1: mean=2, sample std=sqrt(2) -> [-0.707, 0.707]
        assert result.loc[("2024-01-01", "A")] == pytest.approx(-1.0 / np.sqrt(2))
        assert result.loc[("2024-01-01", "B")] == pytest.approx(1.0 / np.sqrt(2))
        # Day 2: mean=150, sample std=50*sqrt(2) -> [-0.707, 0.707]
        assert result.loc[("2024-01-02", "A")] == pytest.approx(-1.0 / np.sqrt(2))
        assert result.loc[("2024-01-02", "B")] == pytest.approx(1.0 / np.sqrt(2))

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            cs_zscore(s)


class TestCsDemean:
    def test_basic(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([1.0, 2.0, 3.0], index=idx)

        result = cs_demean(s)

        # mean=2 -> [-1, 0, 1]
        assert result.iloc[0] == pytest.approx(-1.0)
        assert result.iloc[1] == pytest.approx(0.0)
        assert result.iloc[2] == pytest.approx(1.0)

    def test_nan_preserved(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([1.0, np.nan, 3.0], index=idx)

        result = cs_demean(s)

        assert np.isnan(result.iloc[1])
        # mean of [1,3]=2 -> [-1, nan, 1]
        assert result.iloc[0] == pytest.approx(-1.0)
        assert result.iloc[2] == pytest.approx(1.0)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            cs_demean(s)


class TestCsWinsorize:
    def test_basic_clip(self):
        idx = _make_index([
            ("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C"),
            ("2024-01-01", "D"), ("2024-01-01", "E"),
        ])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0], index=idx)

        result = cs_winsorize(s, lower=0.2, upper=0.8)

        # 5 values: [1, 2, 3, 4, 100]
        # pandas default linear interpolation for quantile
        # lower=0.2 -> 1.8, upper=0.8 -> 23.2
        assert result.iloc[0] == pytest.approx(1.8)   # clipped up
        assert result.iloc[4] == pytest.approx(23.2)  # clipped down
        # Middle values untouched
        assert result.iloc[1] == pytest.approx(2.0)
        assert result.iloc[2] == pytest.approx(3.0)
        assert result.iloc[3] == pytest.approx(4.0)

    def test_nan_preserved(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([1.0, np.nan, 3.0], index=idx)

        result = cs_winsorize(s)

        assert np.isnan(result.iloc[1])

    def test_invalid_bounds(self):
        idx = _make_index([("2024-01-01", "A")])
        s = pd.Series([1.0], index=idx)
        with pytest.raises(ValueError, match="lower"):
            cs_winsorize(s, lower=0.5, upper=0.3)
        with pytest.raises(ValueError, match="lower"):
            cs_winsorize(s, lower=-0.1, upper=0.5)
        with pytest.raises(ValueError, match="lower"):
            cs_winsorize(s, lower=0.5, upper=1.1)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            cs_winsorize(s)


# ---------------------------------------------------------------------------
# New operators - batch 5: math / conditional
# ---------------------------------------------------------------------------

class TestAbs:
    def test_basic(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B")])
        s = pd.Series([-3.0, 4.0], index=idx)

        result = abs_(s)

        assert result.iloc[0] == pytest.approx(3.0)
        assert result.iloc[1] == pytest.approx(4.0)

    def test_nan_preserved(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B")])
        s = pd.Series([-3.0, np.nan], index=idx)

        result = abs_(s)

        assert result.iloc[0] == pytest.approx(3.0)
        assert np.isnan(result.iloc[1])

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            abs_(s)


class TestSign:
    def test_basic(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([-5.0, 0.0, 3.0], index=idx)

        result = sign(s)

        assert result.iloc[0] == pytest.approx(-1.0)
        assert result.iloc[1] == pytest.approx(0.0)
        assert result.iloc[2] == pytest.approx(1.0)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            sign(s)


class TestLog:
    def test_basic(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B")])
        s = pd.Series([1.0, np.e], index=idx)

        result = log(s)

        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(1.0)

    def test_non_positive_yields_nan(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([-1.0, 0.0, 2.0], index=idx)

        result = log(s)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(np.log(2.0))

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            log(s)


class TestSqrt:
    def test_basic(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B")])
        s = pd.Series([4.0, 9.0], index=idx)

        result = sqrt(s)

        assert result.iloc[0] == pytest.approx(2.0)
        assert result.iloc[1] == pytest.approx(3.0)

    def test_negative_yields_nan(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B")])
        s = pd.Series([-4.0, 4.0], index=idx)

        result = sqrt(s)

        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(2.0)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            sqrt(s)


class TestSignedPower:
    def test_basic(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        s = pd.Series([-2.0, 0.0, 3.0], index=idx)

        result = signed_power(s, power=2)

        # sign(-2) * |-2|^2 = -1 * 4 = -4
        assert result.iloc[0] == pytest.approx(-4.0)
        # sign(0) * |0|^2 = 0
        assert result.iloc[1] == pytest.approx(0.0)
        # sign(3) * |3|^2 = 1 * 9 = 9
        assert result.iloc[2] == pytest.approx(9.0)

    def test_fractional_power(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B")])
        s = pd.Series([-4.0, 9.0], index=idx)

        result = signed_power(s, power=0.5)

        # sign(-4) * |-4|^0.5 = -1 * 2 = -2
        assert result.iloc[0] == pytest.approx(-2.0)
        # sign(9) * |9|^0.5 = 1 * 3 = 3
        assert result.iloc[1] == pytest.approx(3.0)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            signed_power(s, power=2)


class TestInverse:
    def test_basic(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B")])
        s = pd.Series([2.0, 4.0], index=idx)

        result = inverse(s)

        assert result.iloc[0] == pytest.approx(0.5)
        assert result.iloc[1] == pytest.approx(0.25)

    def test_zero_yields_nan(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B")])
        s = pd.Series([0.0, 2.0], index=idx)

        result = inverse(s)

        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(0.5)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            inverse(s)


class TestIfElse:
    def test_basic(self):
        idx = _make_index([("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")])
        cond = pd.Series([True, False, True], index=idx)
        true_vals = pd.Series([100.0, 200.0, 300.0], index=idx)
        false_vals = pd.Series([1.0, 2.0, 3.0], index=idx)

        result = if_else(cond, true_vals, false_vals)

        assert result.iloc[0] == pytest.approx(100.0)
        assert result.iloc[1] == pytest.approx(2.0)
        assert result.iloc[2] == pytest.approx(300.0)

    def test_mismatched_true_raises(self):
        idx1 = _make_index([("2024-01-01", "A"), ("2024-01-01", "B")])
        idx2 = _make_index([("2024-01-01", "A"), ("2024-01-01", "C")])
        cond = pd.Series([True, False], index=idx1)
        true_vals = pd.Series([1.0, 2.0], index=idx2)
        false_vals = pd.Series([1.0, 2.0], index=idx1)

        with pytest.raises(ValueError, match="true_values"):
            if_else(cond, true_vals, false_vals)

    def test_mismatched_false_raises(self):
        idx1 = _make_index([("2024-01-01", "A"), ("2024-01-01", "B")])
        idx2 = _make_index([("2024-01-01", "A"), ("2024-01-01", "C")])
        cond = pd.Series([True, False], index=idx1)
        true_vals = pd.Series([1.0, 2.0], index=idx1)
        false_vals = pd.Series([1.0, 2.0], index=idx2)

        with pytest.raises(ValueError, match="false_values"):
            if_else(cond, true_vals, false_vals)

    def test_rejects_non_multiindex(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            if_else(s, s, s)
