#!/usr/bin/env python3
"""Unified daily-update entry point.

Runs two sub-pipelines sequentially:

  1. ``backtest.data.update_daily``  — market_daily / fundamentals /
     dividends / index_daily / index_members (and optionally sw_industry).
  2. ``backtest.factor.update``      — admitted factors only (writes to the
     library DB). Pending / unadmitted factors in the work DB are *not*
     touched; refresh those manually with ``backtest.factor.backfill``.

Use this for the once-a-day cron. For ad-hoc partial refresh keep calling
the underlying modules directly.

Usage:
    python -m backtest.update
    python -m backtest.update --include-sw-industry
    python -m backtest.update --skip-data       # factors only
    python -m backtest.update --skip-factors    # data only
"""

from __future__ import annotations

import argparse

from backtest.data import update_daily as data_update
from backtest.factor import update as factor_update


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-data", action="store_true",
        help="Skip the data update phase.",
    )
    parser.add_argument(
        "--skip-factors", action="store_true",
        help="Skip the admitted-factor update phase.",
    )
    parser.add_argument(
        "--include-sw-industry", action="store_true",
        help="Forwarded to data update: also rebuild sw_industry.",
    )
    args = parser.parse_args()

    if not args.skip_data:
        print("=" * 50)
        print("[1/2] Data update")
        print("=" * 50)
        data_argv = ["--include-sw-industry"] if args.include_sw_industry else []
        data_update.main(data_argv)

    if not args.skip_factors:
        print("\n" + "=" * 50)
        print("[2/2] Admitted-factor update (library DB)")
        print("=" * 50)
        factor_update.main()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
