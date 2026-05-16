"""Tests for factor transforms: rank and z_score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.factor.transforms import rank, z_score


def _make_index(pairs: list[tuple[str, str]]) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(pairs, names=["date", "symbol"])


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
        """Two tied minima share rank 1.5 → (1.5-1)/(3-1) = 0.25."""
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


class TestZScore:
    def test_basic_per_symbol(self):
        """Window=3, arithmetic sequence per symbol → trailing z = 1.0."""
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
