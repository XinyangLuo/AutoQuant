"""申万 (SW2021) 行业归属抓取。

两个 Tushare 接口配合:
- ``pro.index_classify`` — 行业代码清单 + 名称 (按 level)
- ``pro.index_member``  — 成分股历史 (含 in_date / out_date,is_new='N' 的为历史变更行)

注意:**不要用 ``pro.index_member_all``**,它只返回 is_new='Y' 的当前归属,丢失历史。
"""

from __future__ import annotations

import pandas as pd

from backtest.data.tushare_client import api_call, pro


_SRC = "SW2021"


def fetch_industry_classify(level: str) -> pd.DataFrame:
    """拿 SW2021 体系下 L1 或 L2 的行业代码清单。

    Returns DataFrame 列: ``index_code, industry_name, industry_code``。
    """
    df = api_call(pro.index_classify, level=level, src=_SRC)
    if df is None or df.empty:
        return pd.DataFrame()
    return df[["index_code", "industry_name", "industry_code"]].copy()


def fetch_industry_members(index_code: str) -> pd.DataFrame:
    """拿一个行业的成分股历史(含 in_date / out_date)。

    Tushare ``pro.index_member`` 返回列 ``index_code, con_code, in_date, out_date, is_new``。
    is_new='N' 行带 out_date(历史归属),is_new='Y' 行 out_date 为 NaN(当前在该行业)。
    """
    df = api_call(pro.index_member, index_code=index_code)
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def build_sw_industry_rows(
    classify_df: pd.DataFrame,
    members_df: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    """合并 classify 名称 + members 成分史 → sw_industry 表行格式。

    Returns columns: ``symbol, level, industry_code, industry_name, in_date, out_date``
    """
    if members_df.empty:
        return pd.DataFrame(columns=[
            "symbol", "level", "industry_code", "industry_name", "in_date", "out_date",
        ])
    name_map = dict(zip(classify_df["index_code"], classify_df["industry_name"]))

    df = members_df.rename(columns={
        "con_code": "symbol",
        "index_code": "industry_code",
    }).copy()
    df["level"] = level
    df["industry_name"] = df["industry_code"].map(name_map)
    df["in_date"] = pd.to_datetime(df["in_date"], format="%Y%m%d", errors="coerce").dt.date
    df["out_date"] = pd.to_datetime(df["out_date"], format="%Y%m%d", errors="coerce").dt.date

    out = df[["symbol", "level", "industry_code", "industry_name", "in_date", "out_date"]]
    out = out.dropna(subset=["symbol", "industry_code", "in_date"])
    return out.reset_index(drop=True)
