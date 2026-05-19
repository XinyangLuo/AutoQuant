"""Live smoke test for the xtquant realtime adapter.

Run under Wine Python 3.9 with xtquant installed (see
``backtest/data/realtime/README.md``). On macOS / Linux native Python this
will exit immediately with an actionable ImportError.

This script intentionally uses only **anonymous benchmark tickers** —
no account credentials, no portfolio holdings.
"""

from __future__ import annotations

import sys
from datetime import datetime

from backtest.data.realtime import xtquant_quote


WATCH = [
    "000001.SZ",  # 平安银行
    "600000.SH",  # 浦发银行
    "600519.SH",  # 贵州茅台
    "300750.SZ",  # 宁德时代
    "000300.SH",  # 沪深300 指数
]


def _banner(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> int:
    print(f"now: {datetime.now():%Y-%m-%d %H:%M:%S}")

    _banner("1. realtime snapshot (fetch_full_tick)")
    try:
        df = xtquant_quote.fetch_full_tick(WATCH)
        if df.empty:
            print("  (no quotes — market closed or QMT not connected?)")
        else:
            print(df.to_string(index=False))
    except ImportError as exc:
        print(f"  SKIP: {exc}")
        return 0

    _banner("2. instrument details")
    print(xtquant_quote.fetch_instrument_details(WATCH[:3]).to_string(index=False))

    _banner("3. daily bars — 平安银行 last 10")
    bars = xtquant_quote.fetch_bars(["000001.SZ"], period="1d", count=10)
    if "000001.SZ" in bars:
        print(bars["000001.SZ"].tail(10))

    _banner("4. 1m bars — 茅台 last 5")
    bars = xtquant_quote.fetch_bars(["600519.SH"], period="1m", count=5)
    if "600519.SH" in bars:
        print(bars["600519.SH"].tail(5))

    return 0


if __name__ == "__main__":
    sys.exit(main())
