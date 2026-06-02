from __future__ import annotations

from typing import Literal

import pandas as pd


def compute_adj_price(df: pd.DataFrame, price_type: Literal["o2o", "c2c"]) -> pd.Series:
    """Compute adjusted price based on price type.

    Uses ``open * adj_factor`` for ``o2o`` when ``open`` is available,
    otherwise falls back to ``close * adj_factor``.
    """
    if price_type == "o2o" and "open" in df.columns:
        return df["open"] * df["adj_factor"]
    return df["close"] * df["adj_factor"]


def detect_board(symbol: str) -> Literal["default", "kcb", "bj"]:
    """根据股票代码识别板块。

    - 688xxx.SH → kcb (科创板：200股起，1股递增)
    - 8xxxxx.BJ / 4xxxxx.BJ → bj (北交所：100股起，1股递增)
    - 其他 → default (主板/创业板：100股整数倍)
    """
    if symbol.startswith("688"):
        return "kcb"
    if symbol.startswith(("8", "4")) and symbol.endswith(".BJ"):
        return "bj"
    return "default"


def round_lot(shares: float, board: Literal["default", "kcb", "bj"] = "default") -> int:
    """按板块规则取整到合法交易单位。

    - default: 100股整数倍，向下取整。shares < 100 → 0（不够一手不买）
    - kcb: 200股起，超过200股按1股取整。shares < 200 → 0
    - bj: 100股起，超过100股按1股取整。shares < 100 → 0
    """
    if board == "kcb":
        if shares < 200:
            return 0
        return int(shares)
    if board == "bj":
        if shares < 100:
            return 0
        return int(shares)
    # default
    if shares < 100:
        return 0
    return int(shares // 100) * 100


def round_lot_for_symbol(shares: float, symbol: str) -> int:
    """根据股票代码自动判断板块并取整。"""
    return round_lot(shares, detect_board(symbol))


def round_lot_for_symbol_vec(shares: pd.Series, symbols: pd.Series) -> pd.Series:
    """Vectorised ``round_lot_for_symbol``.

    Parameters
    ----------
    shares : pd.Series
        Raw share counts (float, before rounding).
    symbols : pd.Series
        Stock codes aligned with *shares*.

    Returns
    -------
    pd.Series
        Rounded share counts (int), same index as *shares*.
    """
    is_kcb = symbols.str.startswith("688")
    is_bj = symbols.str.startswith(("8", "4")) & symbols.str.endswith(".BJ")

    result = pd.Series(0, index=shares.index, dtype=int)

    kcb_mask = is_kcb & (shares >= 200)
    result[kcb_mask] = shares[kcb_mask].astype(int)

    bj_mask = is_bj & (shares >= 100) & ~kcb_mask
    result[bj_mask] = shares[bj_mask].astype(int)

    default_mask = ~is_kcb & ~is_bj & (shares >= 100)
    result[default_mask] = (shares[default_mask] // 100 * 100).astype(int)

    return result


def cumulate_nav(daily_returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Cumulate daily returns into NAV, starting at 1.0.

    Handles NaN at both ends: the first row (from pct_change) and the
    last row (from forward-return shift(-1)). Leading NaN is replaced
    with 1.0; trailing NaN is forward-filled.
    """
    nav = (1 + daily_returns).cumprod()
    if hasattr(nav, "iloc") and nav.shape[0] > 0:
        # First row: no prior day → return is NaN → cumprod gives NaN.
        first = nav.iloc[0]
        if hasattr(first, "iloc"):
            first = first.iloc[0] if len(first) > 0 else first
        if pd.isna(first):
            nav.iloc[0] = 1.0
    # Forward-fill trailing NaN from shift(-1).
    nav = nav.ffill()
    return nav
