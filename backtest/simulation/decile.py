"""Decile-layered backtest: split universe into 10 groups by factor value."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from backtest.simulation.config import SimulationConfig
from backtest.simulation.utils import compute_adj_price

if TYPE_CHECKING:
    from backtest.simulation.models import DecileBacktestResult


def _decile_cut(x: pd.Series) -> pd.Series:
    """Assign decile labels (0-9) to a cross-section of factor values.

    Falls back to fewer groups when there are not enough unique values.
    """
    n = len(x.dropna())
    if n < 2:
        return pd.Series(np.nan, index=x.index)
    n_groups = min(10, n)
    return pd.qcut(x, n_groups, labels=False, duplicates="drop")


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
        required = ["date", "symbol", "close", "adj_factor"]
        df = market_data[[c for c in required if c in market_data.columns]].copy()
        if "open" in market_data.columns:
            df["open"] = market_data["open"]

        merged = factor_df.merge(df, on=["date", "symbol"], how="inner")
        if merged.empty:
            return _empty_result()

        # 2. Adjusted price & daily return
        merged["adj_price"] = compute_adj_price(merged, self.config.price_type)

        merged = merged.sort_values(["symbol", "date"])
        merged["daily_return"] = merged.groupby("symbol")["adj_price"].pct_change()

        # 3. Assign decile labels per date
        merged["decile"] = merged.groupby("date")["value"].transform(_decile_cut)

        # delay=1: T-day factor → T+1-day decile assignment
        merged["decile"] = merged.groupby("symbol")["decile"].shift(1)

        valid = merged[merged["decile"].notna()].copy()
        if valid.empty:
            return _empty_result()

        # 4. Equal-weight daily return per decile
        decile_returns = (
            valid.groupby(["date", "decile"])["daily_return"]
            .mean()
            .unstack(fill_value=0.0)
        )

        # Ensure columns 0-9 exist (missing ones stay flat at nav=1)
        decile_returns = decile_returns.reindex(columns=range(10), fill_value=0.0)
        decile_returns = decile_returns.sort_index()

        # 5. Cumulate NAV
        nav = (1 + decile_returns).cumprod()
        nav.iloc[0] = 1.0

        max_label = int(valid["decile"].max())
        min_label = int(valid["decile"].min())
        # Long-short: daily return = D_max_return - D_min_return, then cumprod
        ls_daily = decile_returns[max_label] - decile_returns[min_label]
        ls_nav = (1 + ls_daily).cumprod().copy()
        ls_nav.iloc[0] = 1.0
        nav["ls"] = ls_nav.values

        nav_df = nav.reset_index()
        nav_df.columns = ["date"] + [f"d{i}_nav" for i in range(10)] + ["ls_nav"]

        # 6. Compute per-decile and long-short metrics
        from backtest.evaluation.metrics import compute_single_nav_metrics

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

        from backtest.simulation.models import DecileBacktestResult

        return DecileBacktestResult(
            nav_df=nav_df,
            decile_metrics=decile_metrics,
            ls_metrics=ls_metrics,
            monotonicity_score=monotonicity,
        )


def _empty_result() -> "DecileBacktestResult":
    from backtest.simulation.models import DecileBacktestResult

    return DecileBacktestResult(
        nav_df=pd.DataFrame(
            columns=["date"] + [f"d{i}_nav" for i in range(10)] + ["ls_nav"]
        ),
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

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    # Two-column legend to keep it compact
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_title("Decile NAV Curves")
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
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is None:
        out_dir = Path("results") / "decile_backtest"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "decile_backtest.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
