"""Tests for the backtest evaluation module.

Covers:
- Pure-function metrics on synthetic NAV (linear up, V-shape, flat).
- Edge cases (n_days < 2, missing trades/metrics).
- Benchmark beta/alpha recovery on synthetic data.
- ``evaluate()`` end-to-end against parquet artefacts on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.evaluation.benchmark import compute_benchmark_metrics
from backtest.evaluation.loader import BacktestArtifacts, load_result
from backtest.evaluation.metrics import (
    compute_all_metrics,
    compute_drawdown_series,
    compute_monthly_return_matrix,
    compute_return_metrics,
    compute_risk_adjusted,
    compute_risk_metrics,
    compute_rolling_sharpe,
    compute_trading_stats,
    compute_winrate_metrics,
    compute_yearly_returns,
)
from backtest.evaluation.report import evaluate, render_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nav_df(dates: pd.DatetimeIndex, nav_values: np.ndarray) -> pd.DataFrame:
    """Build a minimal nav.parquet-shaped DataFrame from a NAV array."""
    daily_ret = pd.Series(nav_values).pct_change().fillna(0.0).to_numpy()
    return pd.DataFrame({
        "date": dates,
        "nav": nav_values.astype(float),
        "daily_return": daily_ret,
    })


def _trading_calendar(n: int, start: str = "2022-01-03") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _make_artifacts(
    nav_df: pd.DataFrame,
    *,
    initial_cash: float = 1e8,
    trades: pd.DataFrame | None = None,
    metrics_df: pd.DataFrame | None = None,
    tmp_path: Path | None = None,
) -> BacktestArtifacts:
    return BacktestArtifacts(
        result_dir=tmp_path or Path("."),
        nav=nav_df,
        positions=None,
        trades=trades,
        metrics=metrics_df,
        metadata={},
        initial_cash=initial_cash,
        start=pd.Timestamp(nav_df["date"].iloc[0]),
        end=pd.Timestamp(nav_df["date"].iloc[-1]),
    )


# ---------------------------------------------------------------------------
# Test 1: Linear monotonic NAV — drawdown is zero, sharpe is positive.
# ---------------------------------------------------------------------------


class TestLinearAscendingNav:
    """Monotone-increasing NAV: drawdown 0, positive sharpe, monthly matrix shape OK."""

    @pytest.fixture()
    def nav_df(self) -> pd.DataFrame:
        dates = _trading_calendar(252)                        # ~ one trading year
        nav = np.linspace(1.0, 1.10, len(dates))               # +10% straight line
        return _make_nav_df(dates, nav)

    def test_total_return(self, nav_df):
        m = compute_return_metrics(nav_df)
        assert m["total_return"] == pytest.approx(0.10, abs=1e-9)

    def test_max_drawdown_is_zero(self, nav_df):
        m = compute_risk_metrics(nav_df)
        # Monotone increasing => no drawdown.
        assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-12)

    def test_sharpe_positive(self, nav_df):
        m = compute_risk_adjusted(nav_df)
        assert m["sharpe"] > 0.0
        # No downside returns => sortino is NaN by definition.
        assert np.isnan(m["sortino"])

    def test_monthly_matrix_shape(self, nav_df):
        matrix = compute_monthly_return_matrix(nav_df)
        # Always 12-column wide, every cell non-negative for a strict uptrend.
        assert matrix.shape[1] == 12
        non_nan = matrix.values[~np.isnan(matrix.values)]
        assert (non_nan >= -1e-12).all()

    def test_yearly_returns_non_empty(self, nav_df):
        yr = compute_yearly_returns(nav_df)
        assert not yr.empty
        assert (yr >= -1e-12).all()

    def test_daily_win_rate_full(self, nav_df):
        m = compute_winrate_metrics(nav_df)
        # Every day is a positive day on a strict linear ramp.
        assert m["daily_win_rate"] == pytest.approx(1.0)

    def test_drawdown_series_non_positive(self, nav_df):
        dd = compute_drawdown_series(nav_df)
        assert (dd["drawdown"] <= 1e-12).all()


# ---------------------------------------------------------------------------
# Test 2: V-shape NAV — drawdown start/end and recovery days.
# ---------------------------------------------------------------------------


class TestVShapeNav:
    """Crash-then-recover: max_drawdown_end is the trough, recovery_days correct."""

    @pytest.fixture()
    def nav_df(self) -> pd.DataFrame:
        dates = _trading_calendar(100)
        # Days 0..39: hold at 1.0
        # Days 40..49: linear drop 1.0 -> 0.8 (20% drawdown)
        # Days 50..99: linear recovery 0.8 -> 1.05
        nav = np.empty(100)
        nav[:40] = 1.0
        nav[40:50] = np.linspace(1.0, 0.8, 10)
        nav[50:] = np.linspace(0.8, 1.05, 50)
        return _make_nav_df(dates, nav)

    def test_max_drawdown_value(self, nav_df):
        m = compute_risk_metrics(nav_df)
        # NAV trough = 0.8, peak = 1.0 => -20% drawdown.
        assert m["max_drawdown"] == pytest.approx(-0.20, abs=1e-9)

    def test_mdd_start_and_end_dates(self, nav_df):
        m = compute_risk_metrics(nav_df)
        # Trough sits at index 49 (last entry of the linspace 1.0 -> 0.8).
        # Plateau is at 1.0 for the first 40 entries — idxmax picks the first.
        expected_start = nav_df["date"].iloc[0].strftime("%Y-%m-%d")
        expected_end = nav_df["date"].iloc[49].strftime("%Y-%m-%d")
        assert m["max_drawdown_start"] == expected_start
        assert m["max_drawdown_end"] == expected_end

    def test_recovery_days_finite(self, nav_df):
        m = compute_risk_metrics(nav_df)
        # NAV crosses 1.0 again somewhere during the recovery half.
        assert m["recovery_days"] is not None
        assert m["recovery_days"] > 0

    def test_drawdown_series_starts_at_zero(self, nav_df):
        dd = compute_drawdown_series(nav_df)
        assert dd["drawdown"].iloc[0] == pytest.approx(0.0)
        # The trough date in the long series matches the metric.
        trough_date = dd.loc[dd["drawdown"].idxmin(), "date"]
        assert trough_date == nav_df["date"].iloc[49]


class TestUnrecoveredDrawdown:
    """If the NAV never makes a new high, recovery_days is None."""

    def test_recovery_none(self):
        dates = _trading_calendar(60)
        nav = np.empty(60)
        nav[:20] = 1.0
        nav[20:] = np.linspace(1.0, 0.7, 40)  # straight monotone down
        df = _make_nav_df(dates, nav)
        m = compute_risk_metrics(df)
        assert m["recovery_days"] is None
        assert m["max_drawdown"] == pytest.approx(-0.30, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 3: Edge cases — n_days < 2 / missing trades / missing metrics.
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """``compute_*`` and ``evaluate()`` must degrade gracefully on tiny inputs."""

    def test_single_day_returns_nan_metrics(self):
        dates = _trading_calendar(1)
        df = _make_nav_df(dates, np.array([1.0]))
        ret_m = compute_return_metrics(df)
        risk_m = compute_risk_metrics(df)
        # Every flat metric should be NaN — no error raised.
        for k, v in ret_m.items():
            assert isinstance(v, float)
            assert np.isnan(v), f"{k} should be NaN on 1-day NAV"
        for k, v in risk_m.items():
            # Some risk keys are strings/None for date/recovery so just smoke-check
            if isinstance(v, float):
                assert np.isnan(v) or v == 0.0

    def test_empty_nav_df_safe(self):
        df = pd.DataFrame({"date": [], "nav": [], "daily_return": []})
        assert compute_return_metrics(df)["total_return"] != compute_return_metrics(df)["total_return"]  # NaN
        assert compute_monthly_return_matrix(df).empty
        assert compute_yearly_returns(df).empty

    def test_rolling_sharpe_short_series(self):
        # Series shorter than window must return an empty series, not crash.
        dates = _trading_calendar(30)
        df = _make_nav_df(dates, np.linspace(1.0, 1.05, 30))
        rolling = compute_rolling_sharpe(df, window=90)
        assert rolling.empty

    def test_trading_stats_with_none(self):
        out = compute_trading_stats(None, None, initial_cash=1e8)
        assert all(np.isnan(v) for v in out.values())

    def test_trading_stats_with_trades_only(self):
        trades = pd.DataFrame({
            "trade_date": _trading_calendar(2),
            "symbol": ["000001.SZ", "000002.SZ"],
            "direction": ["buy", "sell"],
            "shares": [100, 100],
            "price": [10.0, 11.0],
            "amount": [1000.0, 1100.0],
            "commission": [0.3, 0.33],
            "reason": ["normal", "normal"],
        })
        out = compute_trading_stats(trades, None, initial_cash=1e8)
        assert out["total_trades"] == 2
        assert out["total_commission"] == pytest.approx(0.63, abs=1e-9)
        # Fallback stamp-duty path picks up the single sell.
        assert out["total_stamp_duty"] == pytest.approx(1100.0 * 0.001, abs=1e-9)

    def test_evaluate_handles_single_day_nav(self, tmp_path):
        """``evaluate()`` must not crash on a 1-row nav.parquet."""
        dates = _trading_calendar(1)
        df = _make_nav_df(dates, np.array([1.0]))
        df.to_parquet(tmp_path / "nav.parquet")
        report = evaluate(tmp_path, plot=False)
        # All numeric metrics should be NaN/None — no exception.
        assert isinstance(report.metrics, dict)
        # n_days passthrough sanity
        assert report.metrics["n_days"] == 1


# ---------------------------------------------------------------------------
# Test 4: Synthetic benchmark — beta / alpha / correlation recovery.
# ---------------------------------------------------------------------------


class TestBenchmarkRegression:
    """``compute_benchmark_metrics`` should recover known beta/alpha via OLS."""

    @pytest.fixture()
    def beta_alpha_setup(self):
        rng = np.random.default_rng(42)
        n = 252
        dates = _trading_calendar(n)
        bench_r = rng.normal(0.0005, 0.01, n)
        true_beta = 1.3
        true_alpha_daily = 0.0002
        noise = rng.normal(0.0, 0.005, n)
        strat_r = true_alpha_daily + true_beta * bench_r + noise

        # Anchor day 0 so pct_change() starts at zero (matches NAV convention).
        bench_r[0] = 0.0
        strat_r[0] = 0.0

        bench_nav_vals = np.cumprod(1.0 + bench_r)
        strat_nav_vals = np.cumprod(1.0 + strat_r)
        bench_series = pd.Series(bench_nav_vals, index=dates, name="000300.SH")
        strat_df = _make_nav_df(dates, strat_nav_vals)
        return {
            "strat_df": strat_df,
            "bench_series": bench_series,
            "true_beta": true_beta,
            "true_alpha_daily": true_alpha_daily,
        }

    def test_beta_recovered(self, beta_alpha_setup):
        m = compute_benchmark_metrics(
            beta_alpha_setup["strat_df"],
            beta_alpha_setup["bench_series"],
        )
        assert m["beta"] == pytest.approx(beta_alpha_setup["true_beta"], abs=0.05)

    def test_alpha_recovered(self, beta_alpha_setup):
        m = compute_benchmark_metrics(
            beta_alpha_setup["strat_df"],
            beta_alpha_setup["bench_series"],
        )
        # Annualised alpha = daily * 252; tolerance is loose because noise std ≈ 0.5%.
        expected_alpha_annual = beta_alpha_setup["true_alpha_daily"] * 252
        assert m["alpha_annual"] == pytest.approx(expected_alpha_annual, abs=0.06)

    def test_correlation_in_unit_interval(self, beta_alpha_setup):
        m = compute_benchmark_metrics(
            beta_alpha_setup["strat_df"],
            beta_alpha_setup["bench_series"],
        )
        assert -1.0 <= m["corr"] <= 1.0
        # With true beta=1.3 dominating, correlation should be high.
        assert m["corr"] > 0.6

    def test_information_ratio_finite(self, beta_alpha_setup):
        m = compute_benchmark_metrics(
            beta_alpha_setup["strat_df"],
            beta_alpha_setup["bench_series"],
        )
        assert not np.isnan(m["information_ratio"])
        assert m["tracking_error"] > 0


# ---------------------------------------------------------------------------
# Test 5: evaluate() end-to-end with parquet artefacts.
# ---------------------------------------------------------------------------


class TestEvaluateEndToEnd:
    """Smoke-test the full evaluate() pipeline against on-disk parquet files."""

    @pytest.fixture()
    def result_dir(self, tmp_path) -> Path:
        dates = _trading_calendar(252)
        rng = np.random.default_rng(7)
        r = rng.normal(0.0005, 0.01, 252)
        r[0] = 0.0
        nav = np.cumprod(1.0 + r)
        nav_df = _make_nav_df(dates, nav)
        nav_df["total_value"] = nav * 1e8
        nav_df["cash"] = 0.0
        nav_df["position_value"] = nav * 1e8

        trades = pd.DataFrame({
            "trade_date": dates[:5],
            "symbol": ["000001.SZ"] * 5,
            "direction": ["buy", "buy", "sell", "buy", "sell"],
            "shares": [100, 200, 100, 100, 200],
            "price": [10.0, 10.5, 11.0, 11.2, 11.5],
            "amount": [1000.0, 2100.0, 1100.0, 1120.0, 2300.0],
            "commission": [0.3, 0.6, 0.33, 0.34, 0.69],
            "reason": ["normal", "normal", "normal", "normal", "limit_up_traded"],
        })

        metrics_df = pd.DataFrame({
            "date": dates,
            "turnover": np.full(252, 0.2),
            "stamp_duty": np.full(252, 100.0),
            "transfer_fee": np.full(252, 1.0),
            "position_count": np.full(252, 50.0),
            "long_count": np.full(252, 50.0),
            "short_count": np.zeros(252),
            "cash_ratio": np.full(252, 0.05),
            "gross_exposure": np.full(252, 0.95),
            "net_exposure": np.full(252, 0.95),
            "herfindahl": np.full(252, 0.02),
            "top5_weight": np.full(252, 0.10),
            "top10_weight": np.full(252, 0.20),
        })

        nav_df.to_parquet(tmp_path / "nav.parquet")
        trades.to_parquet(tmp_path / "trades.parquet")
        metrics_df.to_parquet(tmp_path / "metrics.parquet")

        meta = {
            "strategy": {"name": "synthetic_unit_test"},
            "simulation": {"engine": "Synthetic", "initial_cash": 1e8},
            "period": {"start_date": str(dates[0].date()), "end_date": str(dates[-1].date())},
        }
        (tmp_path / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return tmp_path

    def test_load_result_roundtrip(self, result_dir):
        arts = load_result(result_dir)
        assert arts.nav is not None and len(arts.nav) == 252
        assert arts.trades is not None and len(arts.trades) == 5
        assert arts.metrics is not None and len(arts.metrics) == 252
        assert arts.initial_cash == 1e8
        assert arts.strategy_id == "synthetic_unit_test"

    def test_evaluate_writes_outputs(self, result_dir):
        report = evaluate(result_dir, plot=False)
        # Side-effect files
        assert (result_dir / "summary.json").exists()
        assert (result_dir / "summary.csv").exists()
        # report.png intentionally skipped because plot=False
        assert not (result_dir / "report.png").exists()

        # JSON shape
        data = json.loads((result_dir / "summary.json").read_text())
        assert "metrics" in data
        assert data["metrics"]["n_days"] == 252
        assert data["start_date"] == report.artifacts.start.strftime("%Y-%m-%d")

    def test_render_table_contains_sections(self, result_dir):
        report = evaluate(result_dir, plot=False)
        text = render_table(report)
        for section in ("Return", "Risk-Adjusted", "Risk", "Win Rate", "Trading", "Holdings"):
            assert f"## {section}" in text

    def test_compute_all_metrics_consistency(self, result_dir):
        """``compute_all_metrics`` results must match what ``evaluate()`` returns."""
        arts = load_result(result_dir)
        flat = compute_all_metrics(arts)
        report = evaluate(result_dir, plot=False)
        for key in ("total_return", "annual_return", "max_drawdown", "sharpe"):
            a, b = flat[key], report.metrics[key]
            if np.isnan(a) and np.isnan(b):
                continue
            assert a == pytest.approx(b, abs=1e-12), f"{key} differs"
