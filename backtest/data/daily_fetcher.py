"""Per-date market data fetcher and pipeline."""

import pandas as pd

from backtest.data.tushare_client import api_call, pro


# ---------------------------------------------------------------------------
# Low-level fetch helpers
# ---------------------------------------------------------------------------

def _fetch_by_date(api_func, trade_date: str) -> pd.DataFrame:
    """Call a Tushare API with trade_date param and normalize empty/None to empty DF."""
    df = api_call(api_func, trade_date=trade_date)
    return df if df is not None and not df.empty else pd.DataFrame()


def fetch_daily(trade_date: str) -> pd.DataFrame:
    """Fetch all stocks' daily OHLCV for a single trade date."""
    return _fetch_by_date(pro.daily, trade_date)


def fetch_adj_factor(trade_date: str) -> pd.DataFrame:
    """Fetch all stocks' adj_factor for a single trade date."""
    return _fetch_by_date(pro.adj_factor, trade_date)


def fetch_st_status(trade_date: str) -> pd.DataFrame:
    """Fetch ST stock list for a single trade date."""
    return _fetch_by_date(pro.stock_st, trade_date)


def fetch_limit_prices(trade_date: str) -> pd.DataFrame:
    """Fetch all stocks' limit_up / limit_down prices for a single trade date."""
    return _fetch_by_date(pro.stk_limit, trade_date)


def fetch_daily_basic(trade_date: str) -> pd.DataFrame:
    """Fetch all stocks' daily basic indicators (turnover, pe, pb, mv, etc.)."""
    return _fetch_by_date(pro.daily_basic, trade_date)


# ---------------------------------------------------------------------------
# Transform & merge helpers
# ---------------------------------------------------------------------------

def transform_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Tushare daily output to market_daily schema."""
    if df.empty:
        return df

    df = df.rename(columns={
        "trade_date": "date",
        "ts_code": "symbol",
        "vol": "volume",
    })
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.date
    df["volume"] = (df["volume"] * 100).round().astype("int64")
    df["amount"] = (df["amount"] * 1000).round(3)

    cols = [
        "date", "symbol", "open", "high", "low", "close",
        "pre_close", "change", "pct_chg", "volume", "amount", "adj_factor"
    ]
    return df[[c for c in cols if c in df.columns]]


def merge_adj_factor(daily_df: pd.DataFrame, adj_df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge adj_factor into daily DataFrame on (date, symbol)."""
    if daily_df.empty:
        return daily_df

    if adj_df.empty:
        daily_df["adj_factor"] = None
        return daily_df

    adj = adj_df.rename(columns={"trade_date": "date", "ts_code": "symbol"})
    adj["date"] = pd.to_datetime(adj["date"], format="%Y%m%d").dt.date
    adj = adj[["date", "symbol", "adj_factor"]]

    return daily_df.merge(adj, on=["date", "symbol"], how="left")


def merge_st_status(daily_df: pd.DataFrame, st_df: pd.DataFrame) -> pd.DataFrame:
    """Mark is_st=True for symbols present in the daily ST list."""
    if daily_df.empty:
        return daily_df

    daily_df["is_st"] = False

    if st_df.empty or "ts_code" not in st_df.columns:
        return daily_df

    st_symbols = set(st_df["ts_code"])
    daily_df["is_st"] = daily_df["symbol"].isin(st_symbols)
    return daily_df


def merge_limit_prices(daily_df: pd.DataFrame, limit_df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge limit_up / limit_down into daily DataFrame on (date, symbol)."""
    if daily_df.empty:
        return daily_df

    if limit_df.empty:
        daily_df["limit_up"] = None
        daily_df["limit_down"] = None
        return daily_df

    limit = limit_df.rename(columns={"trade_date": "date", "ts_code": "symbol"})
    limit["date"] = pd.to_datetime(limit["date"], format="%Y%m%d").dt.date
    limit = limit[["date", "symbol", "up_limit", "down_limit"]]

    return daily_df.merge(limit, on=["date", "symbol"], how="left")


_DAILY_BASIC_COLS = [
    "turnover_rate", "turnover_rate_f", "volume_ratio",
    "pe", "pe_ttm", "pb", "ps", "ps_ttm",
    "dv_ratio", "dv_ttm",
    "total_share", "float_share", "free_share",
    "total_mv", "circ_mv",
]


def merge_daily_basic(daily_df: pd.DataFrame, basic_df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge daily_basic indicators into daily DataFrame on (date, symbol)."""
    if daily_df.empty:
        return daily_df

    if basic_df.empty:
        for col in _DAILY_BASIC_COLS:
            daily_df[col] = None
        return daily_df

    basic = basic_df.rename(columns={"trade_date": "date", "ts_code": "symbol"})
    basic["date"] = pd.to_datetime(basic["date"], format="%Y%m%d").dt.date
    # Exclude 'close' because it duplicates pro.daily close
    cols = ["date", "symbol"] + [c for c in _DAILY_BASIC_COLS if c in basic.columns]
    basic = basic[[c for c in cols if c in basic.columns]]

    return daily_df.merge(basic, on=["date", "symbol"], how="left")


# ---------------------------------------------------------------------------
# Stock-list metadata helpers
# ---------------------------------------------------------------------------

def build_list_date_map(stock_list: pd.DataFrame) -> dict:
    """Pre-build {symbol -> list_date} dict for fast lookup in hot path."""
    dates = pd.to_datetime(stock_list["list_date"], format="%Y%m%d").dt.date
    return dict(zip(stock_list["ts_code"], dates))


def merge_stock_info(df: pd.DataFrame, list_date_map: dict) -> pd.DataFrame:
    """Map list_date via dict. Drops rows where symbol is not in the map."""
    if df.empty:
        return df

    df["list_date"] = df["symbol"].map(list_date_map)
    return df.dropna(subset=["list_date"])


# ---------------------------------------------------------------------------
# Single-date pipeline (used by cold_start and update_daily)
# ---------------------------------------------------------------------------

def process_trade_date(trade_date: str, list_date_map: dict) -> pd.DataFrame:
    """
    Fetch and transform all data for a single trade date.
    Returns an empty DataFrame if no trading data for that date.
    """
    daily_df = fetch_daily(trade_date)
    if daily_df.empty:
        return pd.DataFrame()

    adj_df = fetch_adj_factor(trade_date)
    st_df = fetch_st_status(trade_date)
    limit_df = fetch_limit_prices(trade_date)
    basic_df = fetch_daily_basic(trade_date)

    daily_df = transform_daily(daily_df)
    daily_df = merge_adj_factor(daily_df, adj_df)
    daily_df = merge_st_status(daily_df, st_df)
    daily_df = merge_limit_prices(daily_df, limit_df)
    daily_df = merge_daily_basic(daily_df, basic_df)
    daily_df = merge_stock_info(daily_df, list_date_map)
    return daily_df
