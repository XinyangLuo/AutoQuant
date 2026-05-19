"""宽基指数成分股(月度快照)抓取 + 密集化辅助。

数据源 ``pro.index_weight`` — 在每个再平衡日(通常每月一次)发布该指数的
成分股 + 权重。一次调用最多返回约 5000 行,跨年大区间需按月切片调用。

下游 :func:`densify_to_trade_dates` 把月度快照沿交易日历前向填充,得到
``(index_code, symbol, trade_date, weight)`` 的日频长表,直接落 DuckDB
``index_members``。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from backtest.data.tushare_client import api_call, pro


_FETCH_COLUMNS = ["index_code", "symbol", "trade_date", "weight"]


def _month_starts(start: str, end: str) -> list[tuple[str, str]]:
    """把 [start, end] 切成 (chunk_start, chunk_end) 列表。

    单 chunk 约一年。``pro.index_weight`` 月度发布,300 成分股 × 12 月 ≈ 3600
    行,远低于 5000 行单调用上限,一年一调用最稳。日期均为 YYYYMMDD。
    """
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    if s > e:
        return []
    chunks: list[tuple[str, str]] = []
    cur = s
    while cur <= e:
        nxt = min(cur + timedelta(days=365), e)
        chunks.append((cur.strftime("%Y%m%d"), nxt.strftime("%Y%m%d")))
        cur = nxt + timedelta(days=1)
    return chunks


def fetch_index_weights(index_code: str, start: str, end: str) -> pd.DataFrame:
    """拉取 ``index_code`` 在 [start, end] 区间的月度成分股快照。

    Returns DataFrame ``[index_code, symbol, trade_date, weight]``,其中
    ``trade_date`` 已转换为 ``datetime.date``,``weight`` 单位为 %(Tushare 原始)。
    若区间无数据,返回空 DataFrame(列已声明)。
    """
    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _month_starts(start, end):
        df = api_call(
            pro.index_weight,
            index_code=index_code,
            start_date=chunk_start,
            end_date=chunk_end,
        )
        if df is None or df.empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=_FETCH_COLUMNS)

    raw = pd.concat(frames, ignore_index=True)
    out = raw.rename(columns={"con_code": "symbol"}).copy()
    out["trade_date"] = pd.to_datetime(
        out["trade_date"], format="%Y%m%d", errors="coerce"
    ).dt.date
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.dropna(subset=["index_code", "symbol", "trade_date"])
    out = out[_FETCH_COLUMNS].drop_duplicates(
        subset=["index_code", "symbol", "trade_date"]
    )
    return out.reset_index(drop=True)


def densify_to_trade_dates(
    monthly_df: pd.DataFrame,
    trade_dates: list,
) -> pd.DataFrame:
    """把月度快照前向填充到每个交易日。

    Parameters
    ----------
    monthly_df : pd.DataFrame
        :func:`fetch_index_weights` 的返回值。同一 ``index_code`` 即可,
        函数对单个 index_code 也工作。
    trade_dates : list[datetime.date]
        升序排列的交易日(应覆盖 monthly_df 的快照日到最新需要的日期)。

    Returns
    -------
    pd.DataFrame
        列 ``[index_code, symbol, trade_date, weight]``,密集化后的日频长表。
    """
    if monthly_df.empty or not trade_dates:
        return pd.DataFrame(columns=_FETCH_COLUMNS)

    # merge_asof requires datetime64 dtype on both sides, so work in
    # pd.Timestamp internally and cast back to date at the end.
    td = pd.to_datetime(pd.Series(list(trade_dates))).sort_values().reset_index(drop=True)

    out_frames: list[pd.DataFrame] = []
    for idx_code, grp in monthly_df.groupby("index_code", sort=False):
        grp = grp.copy()
        grp["trade_date"] = pd.to_datetime(grp["trade_date"])
        snapshot_dates = grp["trade_date"].drop_duplicates().sort_values()
        if snapshot_dates.empty:
            continue
        first_snapshot = snapshot_dates.iloc[0]

        eligible = td[td >= first_snapshot]
        if eligible.empty:
            continue

        keys = pd.DataFrame({"trade_date": eligible.to_list()})
        snap_unique = pd.DataFrame({"snapshot_date": snapshot_dates.to_list()})
        matched = pd.merge_asof(
            keys.sort_values("trade_date"),
            snap_unique.sort_values("snapshot_date"),
            left_on="trade_date",
            right_on="snapshot_date",
            direction="backward",
        ).dropna(subset=["snapshot_date"])

        snap_rows = grp.rename(columns={"trade_date": "snapshot_date"})[
            ["snapshot_date", "symbol", "weight"]
        ]
        merged = matched.merge(snap_rows, on="snapshot_date", how="left")
        merged["index_code"] = idx_code
        merged["trade_date"] = merged["trade_date"].dt.date
        out_frames.append(merged[_FETCH_COLUMNS])

    if not out_frames:
        return pd.DataFrame(columns=_FETCH_COLUMNS)
    out = pd.concat(out_frames, ignore_index=True)
    return out.dropna(subset=["symbol"]).reset_index(drop=True)
