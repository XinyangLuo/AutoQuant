#!/usr/bin/env python3
"""One-shot migration: drop the legacy fina_indicator_quarterly table.

Run this after the new income_q / balancesheet_q / cashflow_q tables
have been backfilled and verified.

Usage:
    python scripts/drop_legacy_fina.py
"""

from backtest.data.storage import MarketStorage


def main():
    with MarketStorage() as storage:
        storage.conn.execute("DROP TABLE IF EXISTS fina_indicator_quarterly")
    print("Dropped fina_indicator_quarterly (if it existed).")


if __name__ == "__main__":
    main()
