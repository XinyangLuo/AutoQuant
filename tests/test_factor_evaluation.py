"""Tests for factor evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.factor.evaluation import (
    EvaluationResult,
    _compute_forward_returns,
    _compute_ic_stats,
    _corr_with_existing,
    _group_returns,
    _ic_series,
    _rank_ic_series,
    _turnover,
)
from backtest.factor.storage import FactorStorage


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

    def test_ties_use_average_rank(self):
        f = pd.Series([1.0, 1.0, 2.0, 3.0])
        r = pd.Series([1.0, 2.0, 2.0, 3.0])
        expected = f.rank(method="average").corr(r.rank(method="average"))
        assert _rank_ic_series(f, r) == pytest.approx(expected, abs=1e-6)


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


# ---------------------------------------------------------------------------
# Correlation with existing factors (duplicate detection)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_factor_storage(tmp_path):
    """A FactorStorage backed by a temporary DuckDB file."""
    fs = FactorStorage(db_path=tmp_path / "test_factors.duckdb")
    try:
        yield fs
    finally:
        fs.close()


def _make_long_factor(factor_id: str, values_per_day: dict[str, list[float]],
                      symbols: list[str]) -> pd.DataFrame:
    """Build a long-form factor DataFrame: (date, symbol, factor_id, value)."""
    rows = []
    for date, vals in values_per_day.items():
        for sym, val in zip(symbols, vals):
            rows.append({
                "date": pd.Timestamp(date),
                "symbol": sym,
                "factor_id": factor_id,
                "value": val,
            })
    return pd.DataFrame(rows)


class TestCorrWithExisting:
    def test_empty_when_no_other_factors(self, tmp_factor_storage):
        factor_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "value": [1.0, 2.0, 1.1, 2.1],
        })
        result = _corr_with_existing(factor_df, "f_new", tmp_factor_storage)
        assert result.empty
        assert list(result.columns) == ["factor_id", "corr", "n_dates"]

    def test_top_k_zero_disables(self, tmp_factor_storage):
        symbols = [f"S{i}" for i in range(5)]
        tmp_factor_storage.insert_factors(_make_long_factor(
            "f_existing",
            {"2024-01-01": list(range(5))},
            symbols,
        ))
        new_df = _make_long_factor(
            "f_new", {"2024-01-01": list(range(5))}, symbols,
        )[["date", "symbol", "value"]]
        result = _corr_with_existing(new_df, "f_new", tmp_factor_storage, top_k=0)
        assert result.empty

    def test_excludes_self(self, tmp_factor_storage):
        symbols = [f"S{i}" for i in range(10)]
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        vals = {d: list(range(10)) for d in dates}

        df_self = _make_long_factor("f_001", vals, symbols)
        tmp_factor_storage.insert_factors(df_self)

        factor_only = df_self[["date", "symbol", "value"]].copy()
        result = _corr_with_existing(factor_only, "f_001", tmp_factor_storage)
        assert result.empty

    def test_identical_factor_has_corr_one(self, tmp_factor_storage):
        symbols = [f"S{i}" for i in range(10)]
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        vals = {d: [float(i) + idx * 0.1 for i, idx in enumerate(range(10))] for d in dates}

        tmp_factor_storage.insert_factors(_make_long_factor("f_existing", vals, symbols))

        new_df = _make_long_factor("f_new", vals, symbols)[["date", "symbol", "value"]]
        result = _corr_with_existing(new_df, "f_new", tmp_factor_storage)

        assert len(result) == 1
        assert result.iloc[0]["factor_id"] == "f_existing"
        assert result.iloc[0]["corr"] == pytest.approx(1.0, abs=1e-6)
        assert result.iloc[0]["n_dates"] == 3

    def test_perfectly_anti_correlated(self, tmp_factor_storage):
        symbols = [f"S{i}" for i in range(10)]
        dates = ["2024-01-01", "2024-01-02"]
        existing_vals = {d: list(range(10)) for d in dates}
        new_vals = {d: list(reversed(range(10))) for d in dates}

        tmp_factor_storage.insert_factors(_make_long_factor("f_existing", existing_vals, symbols))
        new_df = _make_long_factor("f_new", new_vals, symbols)[["date", "symbol", "value"]]

        result = _corr_with_existing(new_df, "f_new", tmp_factor_storage)
        assert result.iloc[0]["corr"] == pytest.approx(-1.0, abs=1e-6)

    def test_sorted_by_abs_corr(self, tmp_factor_storage):
        symbols = [f"S{i}" for i in range(10)]
        dates = ["2024-01-01", "2024-01-02"]

        f_close = {d: list(range(10)) for d in dates}
        f_far = {d: [3, 1, 4, 1, 5, 9, 2, 6, 5, 3] for d in dates}
        f_new = {d: list(range(10)) for d in dates}

        tmp_factor_storage.insert_factors(_make_long_factor("f_close", f_close, symbols))
        tmp_factor_storage.insert_factors(_make_long_factor("f_far", f_far, symbols))

        new_df = _make_long_factor("f_new", f_new, symbols)[["date", "symbol", "value"]]
        result = _corr_with_existing(new_df, "f_new", tmp_factor_storage)

        assert len(result) == 2
        assert result.iloc[0]["factor_id"] == "f_close"
        assert abs(result.iloc[0]["corr"]) >= abs(result.iloc[1]["corr"])

    def test_top_k_truncation(self, tmp_factor_storage):
        symbols = [f"S{i}" for i in range(10)]
        dates = ["2024-01-01"]

        for fid in ("f_a", "f_b", "f_c"):
            v = {d: list(np.random.default_rng(hash(fid) % (2**32)).standard_normal(10)) for d in dates}
            tmp_factor_storage.insert_factors(_make_long_factor(fid, v, symbols))

        new_df = _make_long_factor(
            "f_new", {d: list(range(10)) for d in dates}, symbols
        )[["date", "symbol", "value"]]

        result = _corr_with_existing(new_df, "f_new", tmp_factor_storage, top_k=2)
        assert len(result) == 2

    def test_skips_factor_with_no_date_overlap(self, tmp_factor_storage):
        symbols = [f"S{i}" for i in range(5)]
        tmp_factor_storage.insert_factors(_make_long_factor(
            "f_old",
            {"2023-01-01": list(range(5))},
            symbols,
        ))
        new_df = _make_long_factor(
            "f_new", {"2024-01-01": list(range(5))}, symbols,
        )[["date", "symbol", "value"]]
        result = _corr_with_existing(new_df, "f_new", tmp_factor_storage)
        assert result.empty


class TestMaxCorr:
    def test_returns_none_when_empty(self):
        result = EvaluationResult(
            factor_id="f_x",
        
            ret_type="close",
            horizons=[1],
            start="20240101",
            end="20240131",
            ic_metrics={},
            rank_ic_metrics={},
            decay={},
            turnover=0.0,
            group_returns={},
            corr_with_existing=pd.DataFrame(columns=["factor_id", "corr", "n_dates"]),
        )
        assert result.max_corr() is None

    def test_returns_top_row(self):
        corr_df = pd.DataFrame([
            {"factor_id": "f_a", "corr": 0.95, "n_dates": 100},
            {"factor_id": "f_b", "corr": -0.30, "n_dates": 100},
        ])
        result = EvaluationResult(
            factor_id="f_x",
        
            ret_type="close",
            horizons=[1],
            start="20240101",
            end="20240131",
            ic_metrics={},
            rank_ic_metrics={},
            decay={},
            turnover=0.0,
            group_returns={},
            corr_with_existing=corr_df,
        )
        top = result.max_corr()
        assert top == ("f_a", pytest.approx(0.95))
