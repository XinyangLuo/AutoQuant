#!/usr/bin/env python3
"""Backfill 申万 (SW2021) 行业归属到 market.duckdb / sw_industry 表。

行业变更不频繁 (一年改不了几次),默认全量重拉:
1. 取 L1 / L2 行业代码清单 (pro.index_classify)
2. 遍历每个行业 index_code,拉成分股历史 (pro.index_member)
3. 合并 → UPSERT 到 sw_industry

Usage:
    python -m backtest.data.backfill_sw_industry
    python -m backtest.data.backfill_sw_industry --levels L1
"""

from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from backtest.data.storage import MarketStorage
from backtest.data.fetcher.sw_industry_fetcher import (
    build_sw_industry_rows,
    fetch_industry_classify,
    fetch_industry_members,
)


def backfill_sw_industry(levels: list[str]) -> None:
    with MarketStorage() as storage:
        for level in levels:
            classify = fetch_industry_classify(level)
            if classify.empty:
                tqdm.write(f"[{level}] classify 空,跳过")
                continue
            tqdm.write(f"[{level}] {len(classify)} 个行业")

            total_rows = 0
            for _, row in tqdm(
                classify.iterrows(),
                total=len(classify),
                desc=f"sw_industry {level}",
            ):
                index_code = row["index_code"]
                try:
                    members = fetch_industry_members(index_code)
                except Exception as exc:
                    tqdm.write(f"  {index_code} ({row['industry_name']}): fetch failed ({exc})")
                    continue
                if members.empty:
                    continue
                df = build_sw_industry_rows(classify, members, level)
                if df.empty:
                    continue
                storage.insert_sw_industry(df)
                total_rows += len(df)

            tqdm.write(f"[{level}] 总写入 {total_rows:,} 行")

        stats = storage.get_sw_industry_stats()
        tqdm.write(
            f"sw_industry 现状: total_rows={stats['total_rows']:,} "
            f"symbols={stats['total_symbols']:,} "
            f"L1={stats['n_l1_industries']} L2={stats['n_l2_industries']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backtest.data.backfill_sw_industry",
        description="Backfill SW2021 industry membership (L1 / L2) into sw_industry table.",
    )
    parser.add_argument(
        "--levels", "-l",
        default="L1,L2",
        help="Comma-separated levels to backfill. Default: L1,L2",
    )
    args = parser.parse_args(argv)
    levels = [s.strip() for s in args.levels.split(",") if s.strip()]
    if not levels:
        parser.error("at least one level required")
    backfill_sw_industry(levels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
