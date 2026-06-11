"""Decile-layered backtest: split universe into 10 groups by factor value."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import pandas as pd

from backtest.evaluation.metrics import compute_single_nav_metrics
from backtest.simulation.config import SimulationConfig
from backtest.simulation.models import DecileBacktestResult
from backtest.simulation.utils import cumulate_nav

_DECILE_NAV_COLUMNS = ["date"] + [f"d{i}_nav" for i in range(10)] + ["ls_nav"]


def _decile_cut(x: pd.Series) -> pd.Series:
    """Assign decile labels (0-9) to a cross-section of factor values.

    Uses numpy ``argsort`` (stable) for ranking — ≈2× faster than
    ``pd.qcut``.  NaN values receive NaN labels.  Each decile gets
    an approximately equal number of stocks.
    """
    mask = x.notna()
    n = mask.sum()
    if n < 2:
        return pd.Series(np.nan, index=x.index)
    # Stable argsort → equal-count decile labels via (rank-1)/n * 10.
    vals = x[mask].values
    order = np.empty(n, dtype=np.float64)
    order[np.argsort(vals, kind="stable")] = np.arange(n, dtype=np.float64)
    pcts = order / n  # 0 … (n-1)/n  — never exactly 1.0
    result = np.full(len(x), np.nan)
    result[mask.values] = (pcts * 10).astype(int)
    return pd.Series(result, index=x.index)


class DecileSimulator:
    """Lightweight decile backtest.

    Reuses the vectorisation logic from :class:`SimpleSimulator` but computes
    10 equal-weight portfolios simultaneously — one per decile of the factor
    distribution on each date.
    """

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()

    def run(
        self,
        factor_df: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> "DecileBacktestResult":
        """Run decile backtest.

        Parameters
        ----------
        factor_df : pd.DataFrame
            ``[date, symbol, value]`` — factor values per trading day.
        market_data : pd.DataFrame
            ``[date, symbol, close, open, adj_factor]`` — market data.

        Returns
        -------
        DecileBacktestResult
        """
        # 1. Merge factor + market data
        cols = ["date", "symbol", "close", "open", "adj_factor"]
        df = market_data[[c for c in cols if c in market_data.columns]].copy()

        merged = factor_df.merge(df, on=["date", "symbol"], how="inner")
        if merged.empty:
            return _empty_result()

        # 2. Forward daily return (delay=1 safe), adjusted for dividends.
        #    Factor is computed at T close → trade at T+1 open/close →
        #    return to T+2 open/close.  With decile.shift(1) the row-T
        #    decile comes from factor at T-1, so the return at row T
        #    must be price_{T+1} / price_T - 1 — the period that starts
        #    at T (after factor_{T-1} is already known).
        merged = merged.sort_values(["symbol", "date"])
        price_col = "open" if self.config.price_type == "o2o" else "close"
        merged["adj_price"] = merged[price_col] * merged["adj_factor"]
        merged["daily_return"] = (
            merged.groupby("symbol")["adj_price"].shift(-1)
            / merged["adj_price"] - 1.0
        )

        # 3. Assign decile labels per date, delay=1: use T-1 factor for T's row.
        merged["decile"] = merged.groupby("date")["value"].transform(_decile_cut)
        merged["decile"] = merged.groupby("symbol")["decile"].shift(1)

        valid = merged[merged["decile"].notna()].copy()
        if valid.empty:
            return _empty_result()

        # 4. Equal-weight daily return per decile
        decile_returns = (
            valid.groupby(["date", "decile"])["daily_return"]
            .mean()
            .unstack()
        )
        # Reindex to ensure columns 0-9 exist; missing ones stay flat at nav=1
        decile_returns = decile_returns.reindex(columns=range(10), fill_value=0.0)
        decile_returns = decile_returns.sort_index()

        # 5. Cumulate NAV.  decile_returns[t] is the holding-period return
        #    starting at row date t, so it is realised on the next NAV row.
        realised_returns = decile_returns.shift(1).fillna(0.0)
        nav = cumulate_nav(realised_returns)

        # Long-short: only use deciles that actually have data
        present_deciles = [c for c in decile_returns.columns if c in valid["decile"].values]
        max_label = max(present_deciles)
        min_label = min(present_deciles)
        ls_daily = realised_returns[max_label] - realised_returns[min_label]
        ls_nav = cumulate_nav(ls_daily)
        nav["ls"] = ls_nav.values

        nav_df = nav.reset_index()
        nav_df.columns = _DECILE_NAV_COLUMNS

        # 6. Compute per-decile and long-short metrics
        decile_metrics: dict[int, dict] = {}
        for d in range(10):
            col = f"d{d}_nav"
            single = nav_df[["date", col]].rename(columns={col: "nav"})
            if single["nav"].isna().all():
                decile_metrics[d] = {}
                continue
            decile_metrics[d] = compute_single_nav_metrics(single)

        ls = nav_df[["date", "ls_nav"]].rename(columns={"ls_nav": "nav"})
        ls_metrics = compute_single_nav_metrics(ls)

        # Monotonicity: Spearman between decile rank and annual return
        annual_rets = [decile_metrics[d].get("annual_return", np.nan) for d in range(10)]
        valid_pairs = [(i, r) for i, r in enumerate(annual_rets) if pd.notna(r)]
        if len(valid_pairs) >= 3:
            ranks = np.arange(len(valid_pairs))
            rets = np.array([r for _, r in valid_pairs])
            monotonicity = float(np.corrcoef(ranks, rets)[0, 1])
        else:
            monotonicity = float("nan")

        return DecileBacktestResult(
            nav_df=nav_df,
            decile_metrics=decile_metrics,
            ls_metrics=ls_metrics,
            monotonicity_score=monotonicity,
        )


def _empty_result() -> DecileBacktestResult:
    return DecileBacktestResult(
        nav_df=pd.DataFrame(columns=_DECILE_NAV_COLUMNS),
        decile_metrics={},
        ls_metrics={},
        monotonicity_score=float("nan"),
    )


def plot_decile_backtest(
    result: "DecileBacktestResult",
    output_path: str | None = None,
) -> str:
    """Plot decile NAV curves and long-short spread.

    Parameters
    ----------
    result : DecileBacktestResult
    output_path : str, optional
        If None, writes to ``results/<factor_id>/decile_backtest/...``.

    Returns
    -------
    str
        Path to saved figure.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    # Guard: only force Agg backend when not already set (prevents
    # clobbering an interactive backend in notebooks / tests).
    if matplotlib.get_backend() != "Agg":
        matplotlib.use("Agg")

    nav_df = result.nav_df.copy()
    if nav_df.empty:
        raise ValueError("Empty nav_df, nothing to plot")
    nav_df["date"] = pd.to_datetime(nav_df["date"])

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]}
    )

    # --- Top panel: 10 decile NAV curves ---
    ax = axes[0]
    cmap = matplotlib.colormaps["RdYlGn"]
    for d in range(10):
        col = f"d{d}_nav"
        if col not in nav_df.columns:
            continue
        color = cmap(d / 9)
        lw = 1.5 if d in (0, 9) else 1.0
        ax.plot(
            nav_df["date"],
            nav_df[col],
            color=color,
            linewidth=lw,
            label=f"D{d + 1}",
        )
    ax.set_ylabel("NAV (log)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_title("Decile NAV Curves")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    # --- Bottom panel: Long-Short ---
    ax = axes[1]
    if "ls_nav" in nav_df.columns:
        ax.plot(
            nav_df["date"],
            nav_df["ls_nav"],
            color="steelblue",
            linewidth=1.2,
        )
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
        ax.fill_between(
            nav_df["date"],
            nav_df["ls_nav"],
            1.0,
            alpha=0.15,
            color="steelblue",
        )
        ax.set_ylabel("Long-Short NAV")
        ax.set_xlabel("Date")
        ax.set_title(
            f"Long-Short (D10 - D1)  |  Monotonicity={result.monotonicity_score:+.3f}"
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is None:
        out_dir = Path("results") / "decile_backtest"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "decile_backtest.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
