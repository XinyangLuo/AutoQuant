#!/usr/bin/env python3
"""Backtest f_rev_05 (reversal_zscore_combo) with a simple top-N long strategy.

Usage:
    conda activate AutoQuant
    python scripts/backtest_f_rev_05.py

Strategy:
    - Daily rebalance, delay=1 (T-day signal -> T+1 day execution)
    - Long top-N stocks by f_rev_05 factor value, equal weight
    - No short, no leverage
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.factor.storage import FactorStorage
from backtest.simulation import SimpleSimulator, SimulationConfig

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
FACTOR_ID = "f_rev_05"
START_DATE = "20210101"
END_DATE = "20241231"
TOP_N = 50  # number of stocks to hold
INITIAL_CASH = 100_000_000.0


def generate_signals(factor_id: str, start: str, end: str, top_n: int) -> pd.DataFrame:
    """Generate daily target_weight signals from factor values.

    Returns DataFrame with columns [date, symbol, target_weight].
    """
    with FactorStorage() as fs:
        factor_df = fs.get_factor(factor_id, start, end)

    if factor_df.empty:
        raise ValueError(f"No factor data for {factor_id}")

    signals = []
    for date, group in factor_df.groupby("date"):
        # Drop NaN, sort ascending (f_rev_05 is reversal: low = past losers = future winners)
        # Wait: f_rev_05 = z_score(-ret) * z_score(turnover). Higher = more overbought = lower future returns
        # So we want to SHORT high values, LONG low values.
        # For a simple long-only strategy: take bottom N (most oversold / likely to rebound)
        valid = group.dropna(subset=["value"])
        if len(valid) < top_n:
            continue
        bottom_n = valid.nsmallest(top_n, "value")
        weight = 1.0 / top_n
        for _, row in bottom_n.iterrows():
            signals.append({
                "date": row["date"],
                "symbol": row["symbol"],
                "target_weight": weight,
            })

    return pd.DataFrame(signals)


def get_market_data(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch market data for backtest."""
    with MarketStorage() as ms:
        df = ms.get_bars(
            symbols=symbols,
            start=start,
            end=end,
            columns=["open", "high", "low", "close", "adj_factor"],
        )
    return df


def compute_metrics(nav_df: pd.DataFrame) -> dict:
    """Compute backtest performance metrics."""
    nav = nav_df["nav"].values
    returns = nav_df["daily_return"].dropna().values

    if len(returns) == 0:
        return {}

    total_return = nav[-1] / nav[0] - 1
    n_days = len(returns)
    annual_return = (1 + total_return) ** (252 / n_days) - 1
    annual_vol = float(np.std(returns) * np.sqrt(252))
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # Max drawdown
    cum_max = np.maximum.accumulate(nav)
    drawdowns = (nav - cum_max) / cum_max
    max_dd = float(np.min(drawdowns))
    max_dd_idx = np.argmin(drawdowns)
    peak_idx = np.argmax(nav[: max_dd_idx + 1]) if max_dd_idx > 0 else 0

    # Calmar
    calmar = -annual_return / max_dd if max_dd < 0 else np.inf

    # Win rate
    win_rate = (returns > 0).sum() / len(returns)

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_dd_start": str(nav_df["date"].iloc[peak_idx])[:10],
        "max_dd_end": str(nav_df["date"].iloc[max_dd_idx])[:10],
        "calmar": calmar,
        "win_rate": win_rate,
        "n_days": n_days,
    }


def main():
    print(f"=" * 60)
    print(f"Backtest: {FACTOR_ID}")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    print(f"Strategy: Long bottom-{TOP_N} equal weight, delay=1")
    print(f"=" * 60)

    # 1. Generate signals
    print("\n[1/4] Loading factor data...")
    signals = generate_signals(FACTOR_ID, START_DATE, END_DATE, TOP_N)
    print(f"       Signals: {len(signals):,} rows, {signals['date'].nunique()} trading days")

    # 2. Load market data
    print("\n[2/4] Loading market data...")
    all_symbols = signals["symbol"].unique().tolist()
    # Extend date range for delay=1 execution
    signal_end = pd.to_datetime(signals["date"].max())
    market_end = (signal_end + pd.Timedelta(days=5)).strftime("%Y%m%d")
    market_data = get_market_data(all_symbols, START_DATE, market_end)
    print(f"       Market data: {len(market_data):,} rows")

    # 3. Run backtest
    print("\n[3/4] Running backtest...")
    config = SimulationConfig(initial_cash=INITIAL_CASH)
    sim = SimpleSimulator(config)
    result = sim.run(signals, market_data)
    print(f"       Done. Nav periods: {len(result.nav_df)}")

    # 4. Print results
    print("\n[4/4] Performance Metrics")
    print("-" * 40)
    metrics = compute_metrics(result.nav_df)

    print(f"  Total Return      : {metrics['total_return']:+.2%}")
    print(f"  Annualized Return : {metrics['annual_return']:+.2%}")
    print(f"  Annualized Vol    : {metrics['annual_vol']:.2%}")
    print(f"  Sharpe Ratio      : {metrics['sharpe']:.3f}")
    print(f"  Max Drawdown      : {metrics['max_drawdown']:.2%}")
    print(f"    From {metrics['max_dd_start']} to {metrics['max_dd_end']}")
    print(f"  Calmar Ratio      : {metrics['calmar']:.3f}")
    print(f"  Daily Win Rate    : {metrics['win_rate']:.1%}")
    print(f"  Trading Days      : {metrics['n_days']}")

    # Yearly breakdown
    print("\n  --- Yearly Returns ---")
    nav_df = result.nav_df.copy()
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df["year"] = nav_df["date"].dt.year
    yearly = nav_df.groupby("year").apply(
        lambda g: g["nav"].iloc[-1] / g["nav"].iloc[0] - 1,
        include_groups=False,
    )
    for year, ret in yearly.items():
        print(f"  {year}: {ret:+.2%}")

    print(f"\n{'=' * 60}")

    # Optional: save result
    output_dir = PROJECT_ROOT / "results" / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.nav_df.to_csv(output_dir / f"{FACTOR_ID}_nav.csv", index=False)
    print(f"Nav curve saved to: {output_dir / f'{FACTOR_ID}_nav.csv'}")


if __name__ == "__main__":
    main()
