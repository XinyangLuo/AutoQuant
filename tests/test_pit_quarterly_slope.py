"""Equivalence test: vectorised pit_quarterly_slope vs the legacy loop.

The naive per-group ``groupby.apply(regress_slope_over_mean)`` was the
biggest bottleneck on fina-heavy factor backfills (5 min/year). The new
matrix form should produce bit-identical results on a deterministic
small panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.factor.builtin.barra._common import (
    pit_quarterly_slope,
    regress_slope_over_mean,
)


def _legacy_pit_quarterly_slope(
    panel: pd.DataFrame, value_col: str, *, n: int = 20, sign: float = 1.0,
) -> pd.DataFrame:
    df = panel.dropna(subset=[value_col, "end_date"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "symbol", "value"])
    df["end_date"] = df["end_date"].astype(str)
    df = df.sort_values(["symbol", "date", "end_date"])

    def _score(arr: np.ndarray) -> float:
        return regress_slope_over_mean(arr[-n:]) * sign

    grouped = df.groupby(["symbol", "date"], sort=False)[value_col].apply(
        lambda s: _score(s.to_numpy())
    )
    return grouped.rename("value").reset_index()[["date", "symbol", "value"]]


def _make_panel(seed: int = 0, n_symbols: int = 20, n_dates: int = 6,
                n_quarters: int = 24) -> pd.DataFrame:
    """Build a PIT-shaped panel: each (date, symbol) has up to n_quarters end_dates."""
    rng = np.random.default_rng(seed)
    rows = []
    base_quarters = pd.date_range("2018-03-31", periods=n_quarters, freq="QE")
    base_quarters_str = base_quarters.strftime("%Y%m%d").tolist()
    trade_dates = pd.date_range("2023-01-03", periods=n_dates, freq="B")
    for sym in (f"SYM{i:03d}" for i in range(n_symbols)):
        # Each symbol has a (slightly different) random walk EPS series.
        eps = rng.normal(loc=0.5, scale=0.2, size=n_quarters).cumsum()
        # Insert occasional NaNs so the mask paths are exercised.
        eps[rng.uniform(size=n_quarters) < 0.1] = np.nan
        for d in trade_dates:
            # Visible end_dates on D are those with end_date < D — for the
            # test panel we just expose the trailing 20 quarters.
            for end_date, val in zip(base_quarters_str[-20:], eps[-20:]):
                rows.append((d, sym, end_date, val))
    return pd.DataFrame(rows, columns=["date", "symbol", "end_date", "eps"])


def test_vectorised_matches_legacy_random_panel():
    """Same (date, symbol, value) tuples on a 20-symbol × 6-date panel."""
    panel = _make_panel(seed=42)
    legacy = _legacy_pit_quarterly_slope(panel, "eps", n=20, sign=1.0)
    new = pit_quarterly_slope(panel, "eps", n=20, sign=1.0)

    legacy = legacy.sort_values(["date", "symbol"]).reset_index(drop=True)
    new = new.sort_values(["date", "symbol"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(legacy, new, check_dtype=False, rtol=1e-10, atol=1e-12)


def test_sign_flip_negates_result():
    """sign=-1.0 should produce the negation of sign=1.0 (modulo NaN)."""
    panel = _make_panel(seed=7)
    pos = pit_quarterly_slope(panel, "eps", n=20, sign=1.0)
    neg = pit_quarterly_slope(panel, "eps", n=20, sign=-1.0)
    keys = ["date", "symbol"]
    merged = pos.merge(neg, on=keys, suffixes=("_pos", "_neg"))
    both_valid = merged["value_pos"].notna() & merged["value_neg"].notna()
    np.testing.assert_allclose(
        merged.loc[both_valid, "value_pos"],
        -merged.loc[both_valid, "value_neg"],
        rtol=1e-12, atol=1e-14,
    )


def test_empty_panel_returns_empty_frame():
    empty = pd.DataFrame(columns=["date", "symbol", "end_date", "eps"])
    out = pit_quarterly_slope(empty, "eps")
    assert out.empty
    assert list(out.columns) == ["date", "symbol", "value"]


def test_short_history_yields_nan():
    """Groups with <4 valid points should produce NaN (matches legacy)."""
    panel = pd.DataFrame([
        ("2024-01-02", "A", "20231231", 1.0),
        ("2024-01-02", "A", "20240331", 1.5),
        ("2024-01-02", "A", "20240630", 2.0),  # only 3 points
    ], columns=["date", "symbol", "end_date", "eps"])
    out = pit_quarterly_slope(panel, "eps", n=20)
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["value"])
