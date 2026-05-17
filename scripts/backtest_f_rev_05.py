#!/usr/bin/env python3
"""Backtest f_rev_05 (reversal_zscore_combo) with a simple top-N long strategy.

Usage:
    conda activate AutoQuant
    python scripts/backtest_f_rev_05.py

Strategy:
    - Daily rebalance, delay=1 (T-day signal -> T+1 day execution)
    - Long bottom-N stocks by f_rev_05 factor value, equal weight
    - No short, no leverage
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.evaluation import evaluate, render_table
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
        # f_rev_05 = z_score(-ret) * z_score(turnover); higher → more overbought.
        # For long-only, take the most oversold bottom_n.
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


def main():
    print(f"=" * 60)
    print(f"Backtest: {FACTOR_ID}")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    print(f"Strategy: Long bottom-{TOP_N} equal weight, delay=1")
    print(f"=" * 60)

    print("\n[1/4] Loading factor data...")
    signals = generate_signals(FACTOR_ID, START_DATE, END_DATE, TOP_N)
    print(f"       Signals: {len(signals):,} rows, {signals['date'].nunique()} trading days")

    print("\n[2/4] Loading market data...")
    all_symbols = signals["symbol"].unique().tolist()
    signal_end = pd.to_datetime(signals["date"].max())
    market_end = (signal_end + pd.Timedelta(days=5)).strftime("%Y%m%d")
    market_data = get_market_data(all_symbols, START_DATE, market_end)
    print(f"       Market data: {len(market_data):,} rows")

    print("\n[3/4] Running backtest...")
    config = SimulationConfig(initial_cash=INITIAL_CASH)
    sim = SimpleSimulator(config)
    result = sim.run(signals, market_data)
    print(f"       Done. Nav periods: {len(result.nav_df)}")

    output_dir = PROJECT_ROOT / "results" / "backtest" / FACTOR_ID
    result.save(str(output_dir), metadata={
        "strategy": {
            "name": f"{FACTOR_ID}_bottom{TOP_N}_equal",
            "factor": FACTOR_ID,
            "top_n": TOP_N,
            "weighting": "equal",
        },
        "simulation": {
            "engine": "SimpleSimulator",
            "initial_cash": INITIAL_CASH,
        },
        "period": {"start_date": START_DATE, "end_date": END_DATE},
    })

    print("\n[4/4] Evaluating...")
    report = evaluate(output_dir, plot=True)
    print(render_table(report))
    print(f"\nArtifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()
