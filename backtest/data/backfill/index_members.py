#!/usr/bin/env python3
"""Backfill 宽基指数成分股到 market.duckdb / index_members 表。

``pro.index_weight`` 返回月度快照(每月一次,在再平衡日发布)。本脚本:

1. 按 ``--indices`` 遍历指数代码;
2. 拉月度快照(用 :func:`fetch_index_weights`);
3. 用 ``market_daily`` 的实际交易日历做前向填充,把每月一次的快照展开成每个
   交易日一行 → 写入 ``index_members``;
4. 增量模式:从 ``get_max_index_member_date(idx)`` + 1 开始拉。

Usage:
    python -m backtest.data.backfill_index_members
    python -m backtest.data.backfill_index_members --indices 000300.SH,000905.SH
    python -m backtest.data.backfill_index_members --start 20240101 --end 20241231

注意:
- 创业板指 ``399006.SZ`` 的 ``pro.index_weight`` 数据 Tushare 限制较多,如需
  请确认账户权限;默认列表暂未含。
- 每月快照展开后行数约 ×30,日线 BT 中 universe 过滤可直接用 daily 等值查询。
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

from tqdm import tqdm

from backtest.data.fetcher.index_members_fetcher import (
    densify_to_trade_dates,
    fetch_index_weights,
)
from backtest.data.storage import MarketStorage


DEFAULT_INDICES = [
    "000300.SH",   # 沪深 300
    "000905.SH",   # 中证 500
    "000852.SH",   # 中证 1000
    "932000.CSI",  # 中证 2000
]


def _next_day(d: date) -> str:
    return (d + timedelta(days=1)).strftime("%Y%m%d")


def backfill_index_members(
    indices: list[str],
    start_override: str | None = None,
    end_override: str | None = None,
) -> None:
    today = datetime.today().strftime("%Y%m%d")
    end = end_override or today

    with MarketStorage() as storage:
        for idx in tqdm(indices, desc="Backfill index_members"):
            if start_override:
                start = start_override
            else:
                max_d = storage.get_max_index_member_date(idx)
                start = _next_day(max_d) if max_d else "20100101"

            if start > end:
                tqdm.write(f"  {idx}: already up to date (max={start})")
                continue

            try:
                # 多拉前 45 天,确保 [start, end] 内每个交易日都能找到 ≤ 它的 snapshot
                fetch_start = (
                    datetime.strptime(start, "%Y%m%d") - timedelta(days=45)
                ).strftime("%Y%m%d")
                monthly = fetch_index_weights(idx, fetch_start, end)
            except Exception as exc:
                tqdm.write(f"  {idx}: fetch failed ({exc})")
                continue

            if monthly.empty:
                tqdm.write(f"  {idx}: no monthly snapshots in [{fetch_start}, {end}]")
                continue

            trade_dates = storage.get_trade_dates_in_db(start, end)
            if not trade_dates:
                tqdm.write(
                    f"  {idx}: no trade_dates in market_daily for [{start}, {end}] "
                    f"— run market_daily backfill first"
                )
                continue

            dense = densify_to_trade_dates(monthly, trade_dates)
            if dense.empty:
                tqdm.write(f"  {idx}: no dense rows after asof merge")
                continue

            storage.insert_index_members(dense)
            tqdm.write(
                f"  {idx}: inserted {len(dense):,} rows "
                f"({dense['trade_date'].min()} ~ {dense['trade_date'].max()}, "
                f"{dense['symbol'].nunique()} symbols)"
            )

        stats = storage.get_index_members_stats()
        if not stats.empty:
            tqdm.write("\nindex_members 现状:")
            tqdm.write(stats.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backtest.data.backfill_index_members",
        description="Backfill 宽基指数成分股(月度快照展开到日频) → index_members.",
    )
    parser.add_argument(
        "--indices", "-i",
        default=",".join(DEFAULT_INDICES),
        help=f"Comma-separated index ts_codes. Default: {','.join(DEFAULT_INDICES)}",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Force start date YYYYMMDD (覆盖各指数的 max trade_date,触发全量重拉)。",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYYMMDD,默认今天。",
    )
    args = parser.parse_args(argv)

    indices = [s.strip() for s in args.indices.split(",") if s.strip()]
    if not indices:
        parser.error("at least one index required")
    backfill_index_members(indices, start_override=args.start, end_override=args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
