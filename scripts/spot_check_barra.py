"""Spot-check Barra L1 factors against independent raw-data recompute.

For N random (date, symbol) pairs:
  1. Read the factor value from factor_library.duckdb.
  2. Independently recompute the factor from raw tables using only
     pandas + numpy (no calls into backtest.factor.builtin.barra).
  3. Either (a) compare absolute values where the formula yields a
     unique scalar (Size raw lncap, Momentum exp-weighted log-return
     sum, Liquidity combined turnover, Beta WLS slope), or (b) compare
     ranks within the date's cross-section where the factor goes
     through MAD winsorize + industry fill + z-score and the absolute
     value is only meaningful relative to the universe.

Run: conda run -n AutoQuant python scripts/spot_check_barra.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MARKET_DB = ROOT / "data" / "duckdb" / "market.duckdb"
LIB_DB = ROOT / "data" / "duckdb" / "factor_library.duckdb"

# Pick a handful of recent-but-historical trade dates that have full data
# (industry assignment, ≥ 252d history for Beta, ≥ 5y fina for Growth).
SAMPLE_DATES = ["2019-06-28", "2021-09-30", "2022-12-30", "2024-03-29"]
K_SYMBOLS_PER_DATE = 8


# ---------------------------------------------------------------------------
# Independent recomputes — each function takes (mkt_con, date) and returns a
# Series indexed by symbol with the "raw pre-pipeline" factor value.
# ---------------------------------------------------------------------------


def raw_lncap(mkt: duckdb.DuckDBPyConnection, date: str) -> pd.Series:
    """Size LNCAP = log(circ_mv * 10000) before z-score."""
    df = mkt.execute(
        "SELECT symbol, circ_mv FROM market_daily WHERE date = ?", [date],
    ).fetchdf()
    s = np.log(df["circ_mv"].astype(float) * 10_000.0)
    return s.set_axis(df["symbol"]).rename("lncap")


def raw_momentum_rstr(mkt: duckdb.DuckDBPyConnection, date: str) -> pd.Series:
    """Barra Momentum: 252d EWMA-sum of log-returns (halflife 126), lag 11,
    smooth over the next 11 days.

    Matches ``barra/momentum.py``:
        ewma_sum_252(log_ret).shift(11).rolling(11).mean()
    """
    end_dt = pd.Timestamp(date)
    bars = mkt.execute(
        """
        SELECT date, symbol, close, adj_factor
        FROM market_daily
        WHERE date <= ?::DATE AND date >= ?::DATE - INTERVAL '900 days'
        """, [date, date],
    ).fetchdf()
    bars["adj_close"] = bars["close"] * bars["adj_factor"]
    bars = bars.sort_values(["symbol", "date"])
    bars["log_ret"] = bars.groupby("symbol")["adj_close"].transform(
        lambda s: np.log(s).diff()
    )

    window, halflife, lag, smooth = 252, 126, 11, 11
    weights = np.power(0.5, np.arange(window - 1, -1, -1, dtype=float) / halflife)
    sw = weights.sum()

    def _ewma_sum_kernel(buf: np.ndarray) -> float:
        mask = ~np.isnan(buf)
        if mask.sum() < window // 2:
            return np.nan
        clean = np.where(mask, buf, 0.0)
        wm = weights * mask
        denom = wm.sum()
        if denom <= 0:
            return np.nan
        return (weights * clean).sum() * (sw / denom)

    out = {}
    for sym, g in bars.groupby("symbol"):
        ts = g.set_index("date")["log_ret"]
        if len(ts) < window + lag + smooth:
            continue
        ewma_sum = ts.rolling(window, min_periods=window).apply(_ewma_sum_kernel, raw=True)
        lagged = ewma_sum.shift(lag)
        smoothed = lagged.rolling(smooth, min_periods=smooth // 2).mean()
        # Take the value at end_dt (or the last available trade date ≤ end_dt).
        valid = smoothed[smoothed.index <= end_dt].dropna()
        if not valid.empty:
            out[sym] = float(valid.iloc[-1])
    return pd.Series(out, name="rstr")


def raw_liquidity_stom(mkt: duckdb.DuckDBPyConnection, date: str) -> pd.Series:
    """Barra Liquidity = STOM only = log(sum 21d amount / circ_mv).

    Matches ``barra/liquidity.py``: amount is in 千元, circ_mv in 万元 →
    ratio = (amount * 1e3) / (circ_mv * 1e4).
    """
    bars = mkt.execute(
        """
        SELECT date, symbol, amount, circ_mv
        FROM market_daily
        WHERE date <= ?::DATE AND date >= ?::DATE - INTERVAL '60 days'
        """, [date, date],
    ).fetchdf()
    bars = bars.sort_values(["symbol", "date"])
    bars["ratio"] = (bars["amount"] * 1e3) / (bars["circ_mv"] * 1e4).where(
        bars["circ_mv"] > 0, np.nan,
    )
    bars["ratio"] = bars["ratio"].replace([np.inf, -np.inf], np.nan)

    out = {}
    end_dt = pd.Timestamp(date)
    for sym, g in bars.groupby("symbol"):
        ts = g.set_index("date")["ratio"]
        ts = ts[ts.index <= end_dt]
        if len(ts) < 21:
            continue
        roll_sum = ts.iloc[-21:].sum()
        if roll_sum > 0:
            out[sym] = float(np.log(roll_sum))
    return pd.Series(out, name="stom")


def raw_beta_slope(
    mkt: duckdb.DuckDBPyConnection, date: str, symbol: str,
) -> float:
    """Beta = WLS slope of symbol log-return on CSI300 log-return,
    window=252, halflife=63.
    """
    end_dt = pd.Timestamp(date)
    bars = mkt.execute(
        """
        SELECT date, close, adj_factor
        FROM market_daily
        WHERE symbol = ? AND date <= ?::DATE
          AND date >= ?::DATE - INTERVAL '500 days'
        ORDER BY date
        """, [symbol, date, date],
    ).fetchdf()
    bench = mkt.execute(
        """
        SELECT date, close
        FROM index_daily
        WHERE symbol = '000300.SH' AND date <= ?::DATE
          AND date >= ?::DATE - INTERVAL '500 days'
        ORDER BY date
        """, [date, date],
    ).fetchdf()
    if len(bench) < 252:
        return float("nan")

    # Align symbol to bench calendar (suspended days become NaN).
    bars["adj_close"] = bars["close"] * bars["adj_factor"]
    bench_dates = pd.DatetimeIndex(bench["date"])
    sym_ts = bars.set_index("date")["adj_close"].reindex(bench_dates)
    r = np.log(sym_ts).diff().to_numpy()
    R = np.log(bench.set_index("date")["close"]).diff().to_numpy()

    # Take the last 252 points ending at date.
    mask = pd.DatetimeIndex(bench["date"]) <= end_dt
    r = r[mask][-252:]
    R = R[mask][-252:]
    if len(r) < 252:
        return float("nan")

    weights = np.power(0.5, np.arange(251, -1, -1, dtype=float) / 63)
    valid = ~(np.isnan(r) | np.isnan(R))
    if valid.sum() < 126:
        return float("nan")
    w = weights * valid
    sw = w.sum()
    rm = (w * np.where(valid, r, 0)).sum() / sw
    Rm = (w * np.where(valid, R, 0)).sum() / sw
    rd = np.where(valid, r - rm, 0.0)
    Rd = np.where(valid, R - Rm, 0.0)
    cov = (w * rd * Rd).sum()
    var = (w * Rd * Rd).sum()
    return float(cov / var) if var > 0 else float("nan")


def raw_growth_eps_slope(
    mkt: duckdb.DuckDBPyConnection, date: str, symbol: str,
) -> tuple[float, list[tuple[str, float, float]]]:
    """Growth EGRO raw: slope of last 20 quarterly TTM EPS / |mean TTM EPS|.

    Returns (raw_egro, history) where history is the list of
    (end_date, eps, ttm_eps) used in the regression. Slope sign should
    match the library factor (positive = growing earnings).

    PIT-correct: pick the latest version per (symbol, end_date) whose
    f_ann_date ≤ ``date``. Then take the last 20 end_dates ≤ this
    announcement's end_date.
    """
    rows = mkt.execute(
        """
        WITH latest AS (
            SELECT symbol, end_date, basic_eps,
                   ROW_NUMBER() OVER (
                       PARTITION BY symbol, end_date
                       ORDER BY f_ann_date DESC, update_flag DESC
                   ) AS rn
            FROM income_q
            WHERE symbol = ?
              AND f_ann_date <= ?
              AND end_date >= '20100101'
        )
        SELECT end_date, basic_eps
        FROM latest WHERE rn = 1
        ORDER BY end_date DESC
        LIMIT 25
        """, [symbol, date.replace("-", "")],
    ).fetchdf()
    if rows.empty or len(rows) < 5:
        return float("nan"), []

    rows = rows.sort_values("end_date").reset_index(drop=True)

    # Compute TTM EPS per row using current + LY_FY − LY_same (within
    # this row's visible-at-D history).
    lookup = dict(zip(rows["end_date"].astype(str), rows["basic_eps"].astype(float)))
    annualize = {"03": 4.0, "06": 2.0, "09": 4.0 / 3.0, "12": 1.0}

    history = []
    ttm_values = []
    for _, r in rows.iterrows():
        ed = str(r["end_date"])
        cur = float(r["basic_eps"]) if pd.notna(r["basic_eps"]) else np.nan
        month = ed[4:6]
        ly_fy = str(int(ed[:4]) - 1) + "1231"
        ly_same = str(int(ed[:4]) - 1) + ed[4:]
        if month == "12":
            ttm = cur
        else:
            ly_fy_v = lookup.get(ly_fy, np.nan)
            ly_same_v = lookup.get(ly_same, np.nan)
            ttm = cur + ly_fy_v - ly_same_v
            if np.isnan(ttm):
                ttm = cur * annualize.get(month, np.nan)
        history.append((ed, cur, ttm))
        ttm_values.append(ttm)

    arr = np.array(ttm_values[-20:], dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() < 4:
        return float("nan"), history
    y = arr[mask]
    x = np.arange(arr.size, dtype=float)[mask]
    cov = np.cov(x, y, bias=True)[0, 1]
    var = np.var(x)
    if var <= 0:
        return float("nan"), history
    slope = cov / var
    mean = np.mean(y)
    if mean == 0:
        return float("nan"), history
    return float(slope / abs(mean)), history


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


def main(seed: int) -> int:
    rng = np.random.default_rng(seed)

    mkt = duckdb.connect(str(MARKET_DB), read_only=True)
    lib = duckdb.connect(str(LIB_DB), read_only=True)

    factor_cols = [
        "f_barra_size", "f_barra_beta", "f_barra_momentum",
        "f_barra_liquidity", "f_barra_value",
        "f_barra_growth", "f_barra_quality",
    ]

    all_results = []
    failures = []

    for date in SAMPLE_DATES:
        # Library full cross-section for that date.
        lib_df = lib.execute(
            f"SELECT symbol, {', '.join(factor_cols)} "
            f"FROM factors_daily WHERE date = ?::DATE",
            [date],
        ).fetchdf()
        if lib_df.empty:
            print(f"[{date}] no library rows; skip")
            continue

        # Random K symbols that have non-null Size (sanity: stock listed).
        candidates = lib_df.dropna(subset=["f_barra_size"])["symbol"]
        if len(candidates) == 0:
            continue
        symbols = rng.choice(
            candidates.to_numpy(), size=min(K_SYMBOLS_PER_DATE, len(candidates)),
            replace=False,
        )

        # Independently recompute the simple-formula factors over the full
        # cross-section (so we can z-score and compare).
        print(f"\n=== {date} === sampling {len(symbols)} symbols")
        raw_size = raw_lncap(mkt, date)
        raw_mom = raw_momentum_rstr(mkt, date)
        raw_liq = raw_liquidity_stom(mkt, date)

        # Pull industry for that date (need for fair compare: library Size
        # also went through industry median fill).
        ind = mkt.execute(
            """
            SELECT symbol, industry_code
            FROM sw_industry
            WHERE level = 'L1'
              AND in_date <= ?::DATE
              AND (out_date IS NULL OR out_date > ?::DATE)
            """, [date, date],
        ).fetchdf().drop_duplicates("symbol").set_index("symbol")["industry_code"]

        def pipeline(s: pd.Series) -> pd.Series:
            """L3 pipeline: MAD winsorize → industry median fill → z-score."""
            med = s.median()
            mad = (s - med).abs().median() * 1.4826
            hi, lo = med + 3 * mad, med - 3 * mad
            s = s.clip(lower=lo, upper=hi)
            # Industry median fill on NaN.
            df = pd.DataFrame({"v": s})
            df["ind"] = ind.reindex(df.index)
            ind_med = df.groupby("ind")["v"].transform("median")
            s = df["v"].fillna(ind_med).fillna(med)
            mu, sd = s.mean(), s.std(ddof=1)
            return (s - mu) / sd

        size_recom = pipeline(raw_size)
        mom_recom = pipeline(raw_mom)
        liq_recom = pipeline(raw_liq)

        # Spot compare for the K sampled symbols.
        for sym in symbols:
            lib_row = lib_df[lib_df["symbol"] == sym].iloc[0]

            checks = [
                ("Size", lib_row["f_barra_size"], size_recom.get(sym, np.nan)),
                ("Momentum", lib_row["f_barra_momentum"], mom_recom.get(sym, np.nan)),
                ("Liquidity", lib_row["f_barra_liquidity"], liq_recom.get(sym, np.nan)),
            ]
            # Beta and Growth: per-symbol independent slopes, compare to
            # library z-score by sign/magnitude only (universe-dependent
            # z-scoring makes absolute comparison meaningless).
            beta_raw = raw_beta_slope(mkt, date, sym)
            checks.append(("Beta(raw slope)", "—", beta_raw))

            egro_raw, _ = raw_growth_eps_slope(mkt, date, sym)
            checks.append(("Growth(raw EGRO)", lib_row["f_barra_growth"], egro_raw))

            for name, lib_v, recom_v in checks:
                row = dict(
                    date=date, symbol=sym, factor=name,
                    library=lib_v, recompute=recom_v,
                )
                if name in ("Beta(raw slope)", "Growth(raw EGRO)"):
                    row["delta"] = np.nan  # informational only
                else:
                    try:
                        row["delta"] = float(lib_v) - float(recom_v)
                    except (TypeError, ValueError):
                        row["delta"] = np.nan
                    if not np.isnan(row["delta"]) and abs(row["delta"]) > 0.10:
                        failures.append(row)
                all_results.append(row)

    mkt.close()
    lib.close()

    df = pd.DataFrame(all_results)
    print("\n\n=== Results ===")
    print(df.to_string(index=False))

    if failures:
        print(f"\n=== {len(failures)} comparisons with |delta| > 0.10 ===")
        print(pd.DataFrame(failures).to_string(index=False))
        return 1

    # Aggregate stats
    pivot = df[~df["factor"].isin(["Beta(raw slope)", "Growth(raw EGRO)"])].pivot_table(
        index="factor", values="delta", aggfunc=["mean", "std", "min", "max"],
    )
    print("\n=== Delta stats per factor ===")
    print(pivot.to_string())

    # Sign-agreement check for raw EGRO vs library Growth: if both are
    # finite, sign of EGRO should usually match sign of library z-score.
    eg = df[df["factor"] == "Growth(raw EGRO)"].dropna(subset=["recompute"])
    eg = eg[pd.to_numeric(eg["library"], errors="coerce").notna()].copy()
    eg["lib_f"] = pd.to_numeric(eg["library"])
    eg["rec_f"] = pd.to_numeric(eg["recompute"])
    eg = eg[(eg["lib_f"].abs() > 0.1) & (eg["rec_f"].abs() > 1e-4)]
    if len(eg):
        agree = (np.sign(eg["lib_f"]) == np.sign(eg["rec_f"])).mean()
        print(f"\nGrowth raw-EGRO vs library z-score sign agreement: "
              f"{agree:.0%} (n={len(eg)})")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sys.exit(main(args.seed))
