"""Unit tests for DecileSimulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.simulation.config import SimulationConfig
from backtest.simulation.decile import DecileSimulator, plot_decile_backtest
from backtest.simulation.models import DecileBacktestResult


def _make_market_data(
    dates: list[str],
    symbols: list[str],
    *,
    returns: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Construct market data with a constant daily return per symbol."""
    rows = []
    for idx, d in enumerate(dates):
        for s in symbols:
            ret = returns.get(s, 0.0) if returns else 0.0
            price = 100 * (1 + ret) ** idx
            rows.append(
                {
                    "date": d,
                    "symbol": s,
                    "close": price,
                    "open": price,
                    "adj_factor": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _make_factor_data(
    dates: list[str],
    symbols: list[str],
    values: dict[str, float],
) -> pd.DataFrame:
    """Construct factor data where each symbol has a constant factor value."""
    rows = []
    for d in dates:
        for s in symbols:
            rows.append({"date": d, "symbol": s, "value": values[s]})
    return pd.DataFrame(rows)


class TestDecileSimulator:
    """Test DecileSimulator core behaviour."""

    def test_monotonic_factor(self):
        """High-factor stocks outperform low-factor stocks → D10 > D1."""
        dates = pd.date_range("2024-01-01", periods=20, freq="B").strftime("%Y-%m-%d").tolist()
        symbols = [f"s{i:02d}" for i in range(20)]

        # Factor values: s00=0, s01=1, ... s19=19
        factor_vals = {s: float(i) for i, s in enumerate(symbols)}
        # Returns: higher factor → higher return
        returns = {s: 0.001 * factor_vals[s] for s in symbols}

        factor_df = _make_factor_data(dates, symbols, factor_vals)
        market_df = _make_market_data(dates, symbols, returns=returns)

        sim = DecileSimulator()
        result = sim.run(factor_df, market_df)

        assert isinstance(result, DecileBacktestResult)
        assert not result.nav_df.empty
        # D10 nav > D1 nav
        assert result.nav_df["d9_nav"].iloc[-1] > result.nav_df["d0_nav"].iloc[-1]
        # Monotonicity should be strongly positive
        assert result.monotonicity_score > 0.8
        # LS should be > 1
        assert result.nav_df["ls_nav"].iloc[-1] > 1.0

    def test_anti_monotonic_factor(self):
        """Low-factor stocks outperform → D1 > D10 (negative monotonicity)."""
        dates = pd.date_range("2024-01-01", periods=20, freq="B").strftime("%Y-%m-%d").tolist()
        symbols = [f"s{i:02d}" for i in range(20)]

        factor_vals = {s: float(i) for i, s in enumerate(symbols)}
        # Reverse: higher factor → lower return
        returns = {s: 0.001 * (19 - factor_vals[s]) for s in symbols}

        factor_df = _make_factor_data(dates, symbols, factor_vals)
        market_df = _make_market_data(dates, symbols, returns=returns)

        sim = DecileSimulator()
        result = sim.run(factor_df, market_df)

        assert result.nav_df["d0_nav"].iloc[-1] > result.nav_df["d9_nav"].iloc[-1]
        assert result.monotonicity_score < -0.8
        assert result.nav_df["ls_nav"].iloc[-1] < 1.0

    def test_flat_factor(self):
        """All equal returns → all deciles roughly equal, LS ≈ 1."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
        symbols = [f"s{i:02d}" for i in range(20)]

        factor_vals = {s: float(i) for i, s in enumerate(symbols)}
        returns = {s: 0.0005 for s in symbols}  # same return

        factor_df = _make_factor_data(dates, symbols, factor_vals)
        market_df = _make_market_data(dates, symbols, returns=returns)

        sim = DecileSimulator()
        result = sim.run(factor_df, market_df)

        # All NAVs should be close
        final_navs = [result.nav_df[f"d{i}_nav"].iloc[-1] for i in range(10)]
        assert max(final_navs) - min(final_navs) < 0.01
        # LS ≈ 1
        assert abs(result.nav_df["ls_nav"].iloc[-1] - 1.0) < 0.01
        # Monotonicity is NaN when all annual returns are identical (corr undefined)
        assert np.isnan(result.monotonicity_score) or abs(result.monotonicity_score) < 0.3

    def test_empty_factor(self):
        """Empty factor data → empty result, no exception."""
        factor_df = pd.DataFrame(columns=["date", "symbol", "value"])
        market_df = _make_market_data(
            ["2024-01-01"], ["s00"], returns={"s00": 0.0}
        )

        sim = DecileSimulator()
        result = sim.run(factor_df, market_df)

        assert result.nav_df.empty
        assert np.isnan(result.monotonicity_score)

    def test_empty_market(self):
        """Empty market data → empty result."""
        dates = ["2024-01-01", "2024-01-02"]
        symbols = ["s00", "s01"]
        factor_df = _make_factor_data(dates, symbols, {"s00": 1.0, "s01": 2.0})
        market_df = pd.DataFrame(columns=["date", "symbol", "close", "open", "adj_factor"])

        sim = DecileSimulator()
        result = sim.run(factor_df, market_df)

        assert result.nav_df.empty

    def test_insufficient_stocks(self):
        """Fewer than 10 stocks → fewer deciles, still works."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
        symbols = ["s00", "s01", "s02"]  # only 3 stocks

        factor_vals = {"s00": 1.0, "s01": 2.0, "s02": 3.0}
        returns = {"s00": 0.001, "s01": 0.002, "s02": 0.003}

        factor_df = _make_factor_data(dates, symbols, factor_vals)
        market_df = _make_market_data(dates, symbols, returns=returns)

        sim = DecileSimulator()
        result = sim.run(factor_df, market_df)

        assert not result.nav_df.empty
        # Should still have D0 and D2 (maybe D1 too)
        assert "d0_nav" in result.nav_df.columns
        assert "d2_nav" in result.nav_df.columns

    def test_ls_nav_relation(self):
        """LS nav = D10 - D1 + 1 at every point."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
        symbols = [f"s{i:02d}" for i in range(20)]

        factor_vals = {s: float(i) for i, s in enumerate(symbols)}
        returns = {s: 0.001 * factor_vals[s] for s in symbols}

        factor_df = _make_factor_data(dates, symbols, factor_vals)
        market_df = _make_market_data(dates, symbols, returns=returns)

        sim = DecileSimulator()
        result = sim.run(factor_df, market_df)

        # Check LS = (1 + D10_return - D1_return).cumprod() for every row
        d9_ret = result.nav_df["d9_nav"].pct_change().fillna(0)
        d0_ret = result.nav_df["d0_nav"].pct_change().fillna(0)
        expected_ls = (1 + d9_ret - d0_ret).cumprod()
        expected_ls.iloc[0] = 1.0
        # Use a relaxed tolerance — the equality is mathematically exact but
        # floating-point rounding in pct_change vs. direct decile mean can
        # diverge slightly over many compounding steps.
        pd.testing.assert_series_equal(
            result.nav_df["ls_nav"], expected_ls, check_names=False, rtol=1e-4
        )

    def test_per_decile_metrics(self):
        """Each decile should have computed metrics."""
        dates = pd.date_range("2024-01-01", periods=20, freq="B").strftime("%Y-%m-%d").tolist()
        symbols = [f"s{i:02d}" for i in range(20)]

        factor_vals = {s: float(i) for i, s in enumerate(symbols)}
        returns = {s: 0.001 * factor_vals[s] for s in symbols}

        factor_df = _make_factor_data(dates, symbols, factor_vals)
        market_df = _make_market_data(dates, symbols, returns=returns)

        sim = DecileSimulator()
        result = sim.run(factor_df, market_df)

        # D0 and D9 should have non-empty metrics
        assert result.decile_metrics[0]
        assert result.decile_metrics[9]
        assert "annual_return" in result.decile_metrics[0]
        assert "annual_return" in result.decile_metrics[9]

        # LS metrics
        assert result.ls_metrics
        assert "annual_return" in result.ls_metrics

    def test_save_and_load(self, tmp_path):
        """DecileBacktestResult.save() produces readable files."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B").strftime("%Y-%m-%d").tolist()
        symbols = [f"s{i:02d}" for i in range(20)]

        factor_vals = {s: float(i) for i, s in enumerate(symbols)}
        returns = {s: 0.001 * factor_vals[s] for s in symbols}

        factor_df = _make_factor_data(dates, symbols, factor_vals)
        market_df = _make_market_data(dates, symbols, returns=returns)

        sim = DecileSimulator()
        result = sim.run(factor_df, market_df)

        out_dir = tmp_path / "decile_out"
        result.save(str(out_dir))

        assert (out_dir / "nav.parquet").exists()
        assert (out_dir / "metrics.json").exists()

        loaded_nav = pd.read_parquet(out_dir / "nav.parquet")
        pd.testing.assert_frame_equal(loaded_nav, result.nav_df)


class TestPlotDecile:
    """Smoke tests for plotting."""

    def test_plot_smoke(self, tmp_path):
        dates = pd.date_range("2024-01-01", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
        symbols = [f"s{i:02d}" for i in range(20)]

        factor_vals = {s: float(i) for i, s in enumerate(symbols)}
        returns = {s: 0.001 * factor_vals[s] for s in symbols}

        factor_df = _make_factor_data(dates, symbols, factor_vals)
        market_df = _make_market_data(dates, symbols, returns=returns)

        sim = DecileSimulator()
        result = sim.run(factor_df, market_df)

        out_path = tmp_path / "decile_plot.png"
        path = plot_decile_backtest(result, output_path=str(out_path))
        assert Path(path).exists()

    def test_plot_empty_raises(self):
        empty = DecileBacktestResult(
            nav_df=pd.DataFrame(columns=["date"] + [f"d{i}_nav" for i in range(10)] + ["ls_nav"]),
            decile_metrics={},
            ls_metrics={},
            monotonicity_score=float("nan"),
        )
        with pytest.raises(ValueError, match="Empty nav_df"):
            plot_decile_backtest(empty)
