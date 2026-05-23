"""Spot-check Barra L1 factors against an independent raw-data recompute.

For a fixed set of (date, symbol) pairs:

  1. Read the factor value from ``factor_library.duckdb``.
  2. Independently recompute the factor from raw tables (``market_daily``,
     ``index_daily``, ``income_q``, ``sw_industry``) using only pandas +
     numpy — no calls into ``backtest.factor.builtin.barra``.
  3. Compare either:
     * absolute values for factors with a unique scalar formula (Size LNCAP,
       Momentum RSTR, Liquidity STOM), put through the same L3 pipeline
       (MAD winsorize → SW-L1 industry median fill → cs_zscore) the library
       applies, OR
     * sign agreement on the raw slope vs the library z-score for factors
       whose absolute value depends on the universe (Growth EGRO).

  Beta is sanity-checked by computing the raw WLS slope per symbol; the
  library value is universe-z-scored so we don't assert on its magnitude.

The local ``_l3_pipeline`` deliberately re-implements
``backtest.factor.builtin.barra._common.apply_l3_pipeline`` instead of
importing it — calling the production helper would turn this into a
self-check. Keep the MAD scale 1.4826 and the winsorize → industry-fill →
zscore order in sync with that module by hand.

Marked ``integration`` because both DuckDBs must be present and admitted.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MARKET_DB = ROOT / "data" / "duckdb" / "market.duckdb"
LIB_DB = ROOT / "data" / "duckdb" / "factor_library.duckdb"

# Quarter-ends with full sw_industry coverage and ≥ 5y of fina history.
# Rotate manually when extending the library's backfill window.
SAMPLE_DATES = ["2019-06-28", "2021-09-30", "2022-12-30", "2024-03-29"]
K_SYMBOLS_PER_DATE = 8
SEED = 42

# Empirical: cross-section z-score std is ~1, so |Δ| > 0.10 means the
# recompute diverged by ≥ 10% of one stdev — well outside float / industry-
# median-fill noise but small enough to catch formula errors.
ABS_DELTA_THRESHOLD = 0.10

# 85% leaves headroom for the ~5% of names whose library Growth lands near
# zero (where sign is noise-dominated). Pre-fix this was ~93%; the TTM
# lookahead bug that motivated this whole test dropped it to ~70%.
GROWTH_SIGN_AGREE_MIN = 0.85

# Factor labels — single source so a typo (e.g. "Momentun") fails import
# instead of silently producing an empty sub-frame at assert time.
F_SIZE = "Size"
F_MOMENTUM = "Momentum"
F_LIQUIDITY = "Liquidity"
F_BETA_RAW = "Beta(raw_slope)"
F_GROWTH_RAW = "Growth(raw_egro)"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (MARKET_DB.exists() and LIB_DB.exists()),
        reason="requires populated market.duckdb + admitted factor_library.duckdb",
    ),
]


# ---------------------------------------------------------------------------
# Independent recomputes — pandas/numpy only, no imports from backtest.factor.
# ---------------------------------------------------------------------------


def raw_lncap(mkt: duckdb.DuckDBPyConnection, date: str) -> pd.Series:
    df = mkt.execute(
        "SELECT symbol, circ_mv FROM market_daily WHERE date = ?", [date],
    ).fetchdf()
    s = np.log(df["circ_mv"].astype(float) * 10_000.0)
    return s.set_axis(df["symbol"]).rename("lncap")


def raw_momentum_rstr(mkt: duckdb.DuckDBPyConnection, date: str) -> pd.Series:
    """252d EWMA-sum of log-returns (halflife 126), lag 11, smooth 11."""
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

    def _kernel(buf: np.ndarray) -> float:
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
        ewma_sum = ts.rolling(window, min_periods=window).apply(_kernel, raw=True)
        lagged = ewma_sum.shift(lag)
        smoothed = lagged.rolling(smooth, min_periods=smooth // 2).mean()
        valid = smoothed[smoothed.index <= end_dt].dropna()
        if not valid.empty:
            out[sym] = float(valid.iloc[-1])
    return pd.Series(out, name="rstr")


def raw_liquidity_stom(mkt: duckdb.DuckDBPyConnection, date: str) -> pd.Series:
    """STOM = log(rolling-21d sum of amount/circ_mv)."""
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


def _fetch_csi300_bench(mkt: duckdb.DuckDBPyConnection, date: str) -> pd.DataFrame:
    """One-time CSI300 fetch per sample date — shared across all per-symbol
    Beta computations on that date."""
    return mkt.execute(
        """
        SELECT date, close
        FROM index_daily
        WHERE symbol = '000300.SH' AND date <= ?::DATE
          AND date >= ?::DATE - INTERVAL '500 days'
        ORDER BY date
        """, [date, date],
    ).fetchdf()


def _fetch_symbols_bars(
    mkt: duckdb.DuckDBPyConnection, date: str, symbols: list[str],
) -> pd.DataFrame:
    """Batched ``market_daily`` fetch for the sampled symbols on one date."""
    placeholders = ",".join("?" for _ in symbols)
    return mkt.execute(
        f"""
        SELECT symbol, date, close, adj_factor
        FROM market_daily
        WHERE symbol IN ({placeholders})
          AND date <= ?::DATE
          AND date >= ?::DATE - INTERVAL '500 days'
        ORDER BY symbol, date
        """, [*symbols, date, date],
    ).fetchdf()


def raw_beta_slope(
    sym_bars: pd.DataFrame, bench: pd.DataFrame, date: str,
) -> float:
    """WLS slope of symbol log-return on CSI300, window=252, halflife=63.

    ``sym_bars`` is the slice of ``_fetch_symbols_bars`` for one symbol;
    ``bench`` is the pre-fetched CSI300 frame for the date."""
    if len(bench) < 252:
        return float("nan")
    end_dt = pd.Timestamp(date)
    bench_dates = pd.DatetimeIndex(bench["date"])

    adj_close = (sym_bars["close"] * sym_bars["adj_factor"])
    sym_ts = adj_close.set_axis(pd.DatetimeIndex(sym_bars["date"])).reindex(bench_dates)
    r = np.log(sym_ts).diff().to_numpy()
    R = np.log(bench.set_index("date")["close"]).diff().to_numpy()

    mask = bench_dates <= end_dt
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


def _fetch_eps_history(
    mkt: duckdb.DuckDBPyConnection, date: str, symbols: list[str],
) -> dict[str, pd.DataFrame]:
    """Batched fetch of the latest-per-(symbol, end_date) ≤ 25-row EPS history
    for all sampled symbols on one date. Returns ``{symbol: rows_frame}``."""
    placeholders = ",".join("?" for _ in symbols)
    df = mkt.execute(
        f"""
        WITH latest AS (
            SELECT symbol, end_date, basic_eps,
                   ROW_NUMBER() OVER (
                       PARTITION BY symbol, end_date
                       ORDER BY f_ann_date DESC, update_flag DESC
                   ) AS rn
            FROM income_q
            WHERE symbol IN ({placeholders})
              AND f_ann_date <= ?
              AND end_date >= '20100101'
        ),
        ranked AS (
            SELECT symbol, end_date, basic_eps,
                   ROW_NUMBER() OVER (
                       PARTITION BY symbol ORDER BY end_date DESC
                   ) AS recency
            FROM latest WHERE rn = 1
        )
        SELECT symbol, end_date, basic_eps
        FROM ranked
        WHERE recency <= 25
        ORDER BY symbol, end_date
        """, [*symbols, date.replace("-", "")],
    ).fetchdf()
    return {sym: g.reset_index(drop=True) for sym, g in df.groupby("symbol")}


def raw_growth_eps_slope(rows: pd.DataFrame) -> float:
    """EGRO = slope(last 20 quarterly TTM EPS) / |mean TTM EPS|.

    Caller provides the per-symbol slice of ``_fetch_eps_history``. We need
    25 quarters so TTM lookups for the earliest of the last 20 still have
    their LY_FY / LY_same neighbours inside the window — that cushion is
    what the production Growth factor's bug fix added."""
    if rows.empty or len(rows) < 5:
        return float("nan")

    lookup = dict(zip(rows["end_date"].astype(str), rows["basic_eps"].astype(float)))
    annualize = {"03": 4.0, "06": 2.0, "09": 4.0 / 3.0, "12": 1.0}

    ttm_values: list[float] = []
    for _, r in rows.iterrows():
        ed = str(r["end_date"])
        cur = float(r["basic_eps"]) if pd.notna(r["basic_eps"]) else np.nan
        month = ed[4:6]
        if month == "12":
            ttm = cur
        else:
            ly_fy = lookup.get(str(int(ed[:4]) - 1) + "1231", np.nan)
            ly_same = lookup.get(str(int(ed[:4]) - 1) + ed[4:], np.nan)
            ttm = cur + ly_fy - ly_same
            if np.isnan(ttm):
                ttm = cur * annualize.get(month, np.nan)
        ttm_values.append(ttm)

    arr = np.array(ttm_values[-20:], dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() < 4:
        return float("nan")
    y = arr[mask]
    x = np.arange(arr.size, dtype=float)[mask]
    cov = np.cov(x, y, bias=True)[0, 1]
    var = np.var(x)
    if var <= 0:
        return float("nan")
    slope = cov / var
    mean = np.mean(y)
    if mean == 0:
        return float("nan")
    return float(slope / abs(mean))


# ---------------------------------------------------------------------------
# Pipeline used for absolute-value compare. See module docstring for why
# this duplicates apply_l3_pipeline instead of importing it.
# ---------------------------------------------------------------------------


def _l3_pipeline(s: pd.Series, ind: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    hi, lo = med + 3 * mad, med - 3 * mad
    s = s.clip(lower=lo, upper=hi)
    df = pd.DataFrame({"v": s})
    df["ind"] = ind.reindex(df.index)
    ind_med = df.groupby("ind")["v"].transform("median")
    s = df["v"].fillna(ind_med).fillna(med)
    mu, sd = s.mean(), s.std(ddof=1)
    return (s - mu) / sd


def _fetch_industry(mkt: duckdb.DuckDBPyConnection, date: str) -> pd.Series:
    """SW-L1 industry assignment as of ``date``. On overlapping segments
    pick the latest ``in_date`` to match production's ROW_NUMBER tie-break."""
    df = mkt.execute(
        """
        SELECT symbol, industry_code
        FROM (
            SELECT symbol, industry_code,
                   ROW_NUMBER() OVER (
                       PARTITION BY symbol ORDER BY in_date DESC
                   ) AS rn
            FROM sw_industry
            WHERE level = 'L1'
              AND in_date <= ?::DATE
              AND (out_date IS NULL OR out_date > ?::DATE)
        ) WHERE rn = 1
        """, [date, date],
    ).fetchdf()
    return df.set_index("symbol")["industry_code"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def conns():
    mkt = duckdb.connect(str(MARKET_DB), read_only=True)
    lib = duckdb.connect(str(LIB_DB), read_only=True)
    try:
        yield mkt, lib
    finally:
        mkt.close()
        lib.close()


@pytest.fixture(scope="module")
def sample_results(conns):
    """Produce the per-(date, symbol, factor) comparison rows once for all
    tests. Each test then filters and asserts on its slice."""
    mkt, lib = conns
    rng = np.random.default_rng(SEED)

    factor_cols = [
        "f_barra_size", "f_barra_beta", "f_barra_momentum",
        "f_barra_liquidity", "f_barra_value",
        "f_barra_growth", "f_barra_quality",
    ]

    rows: list[dict] = []
    for date in SAMPLE_DATES:
        lib_df = lib.execute(
            f"SELECT symbol, {', '.join(factor_cols)} "
            f"FROM factors_daily WHERE date = ?::DATE",
            [date],
        ).fetchdf()
        if lib_df.empty:
            continue

        candidates = lib_df.dropna(subset=["f_barra_size"])["symbol"]
        if len(candidates) == 0:
            continue
        symbols = list(rng.choice(
            candidates.to_numpy(),
            size=min(K_SYMBOLS_PER_DATE, len(candidates)),
            replace=False,
        ))

        raw_size = raw_lncap(mkt, date)
        raw_mom = raw_momentum_rstr(mkt, date)
        raw_liq = raw_liquidity_stom(mkt, date)
        ind = _fetch_industry(mkt, date)

        size_recom = _l3_pipeline(raw_size, ind)
        mom_recom = _l3_pipeline(raw_mom, ind)
        liq_recom = _l3_pipeline(raw_liq, ind)

        # Batch the per-symbol Beta / Growth fetches: one SQL each per date
        # instead of K queries × 2 factors = 16 round-trips per date.
        bench = _fetch_csi300_bench(mkt, date)
        sym_bars_all = _fetch_symbols_bars(mkt, date, symbols)
        bars_by_sym = {s: g for s, g in sym_bars_all.groupby("symbol")}
        eps_by_sym = _fetch_eps_history(mkt, date, symbols)

        for sym in symbols:
            lib_row = lib_df[lib_df["symbol"] == sym].iloc[0]
            sym_bars = bars_by_sym.get(sym, pd.DataFrame(columns=sym_bars_all.columns))
            beta = raw_beta_slope(sym_bars, bench, date)
            egro = raw_growth_eps_slope(eps_by_sym.get(sym, pd.DataFrame()))

            entries: list[tuple[str, float, float]] = [
                (F_SIZE, lib_row["f_barra_size"], size_recom.get(sym, np.nan)),
                (F_MOMENTUM, lib_row["f_barra_momentum"], mom_recom.get(sym, np.nan)),
                (F_LIQUIDITY, lib_row["f_barra_liquidity"], liq_recom.get(sym, np.nan)),
                (F_BETA_RAW, float("nan"), beta),
                (F_GROWTH_RAW, lib_row["f_barra_growth"], egro),
            ]
            for name, lib_v, recom_v in entries:
                rows.append({
                    "date": date, "symbol": sym, "factor": name,
                    "library": lib_v, "recompute": recom_v,
                })

    if not rows:
        pytest.skip("library has no rows on any sample date — backfill incomplete")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [F_SIZE, F_MOMENTUM, F_LIQUIDITY])
def test_z_scored_factor_matches_independent_recompute(sample_results, factor):
    """For each sampled (date, symbol), the library z-score should be
    within ``ABS_DELTA_THRESHOLD`` of the independently-computed z-score."""
    sub = sample_results[sample_results["factor"] == factor].copy()
    sub["library"] = pd.to_numeric(sub["library"], errors="coerce")
    sub["recompute"] = pd.to_numeric(sub["recompute"], errors="coerce")
    sub = sub.dropna(subset=["library", "recompute"])
    assert not sub.empty, f"no comparable rows for {factor}"

    sub["delta"] = (sub["library"] - sub["recompute"]).abs()
    failures = sub[sub["delta"] > ABS_DELTA_THRESHOLD]
    assert failures.empty, (
        f"{factor}: {len(failures)} rows exceed |Δ|={ABS_DELTA_THRESHOLD}:\n"
        f"{failures.to_string(index=False)}"
    )


def test_beta_raw_slope_is_finite(sample_results):
    """The raw WLS slope should be finite (numeric) for most sampled symbols
    that have ≥ 252d of bench-aligned history. This catches CSI300 path or
    column-name regressions in the recompute, not the library factor itself."""
    sub = sample_results[sample_results["factor"] == F_BETA_RAW]
    sub = sub[pd.to_numeric(sub["recompute"], errors="coerce").notna()]
    assert len(sub) >= 0.5 * len(SAMPLE_DATES) * K_SYMBOLS_PER_DATE, (
        f"too few sampled symbols had a finite raw Beta slope: {len(sub)}"
    )


def test_growth_sign_agrees_with_library(sample_results):
    """The raw EGRO slope sign should agree with the library z-score sign on
    the dominant majority of sampled symbols (drops where either is near zero
    are noise-dominated and excluded)."""
    sub = sample_results[sample_results["factor"] == F_GROWTH_RAW].copy()
    sub["lib_f"] = pd.to_numeric(sub["library"], errors="coerce")
    sub["rec_f"] = pd.to_numeric(sub["recompute"], errors="coerce")
    sub = sub.dropna(subset=["lib_f", "rec_f"])
    sub = sub[(sub["lib_f"].abs() > 0.1) & (sub["rec_f"].abs() > 1e-4)]
    if sub.empty:
        pytest.skip("no Growth pairs with both sides far enough from zero")

    agree = (np.sign(sub["lib_f"]) == np.sign(sub["rec_f"])).mean()
    assert agree >= GROWTH_SIGN_AGREE_MIN, (
        f"Growth sign agreement {agree:.0%} < {GROWTH_SIGN_AGREE_MIN:.0%} "
        f"(n={len(sub)}). This previously caught a TTM lookahead bug where "
        f"the first ~4 of the 20 slope inputs fell back to annualize because "
        f"their LY_FY / LY_same neighbours were truncated. If this fails, "
        f"check that ``barra_growth.py`` still fetches a cushion of "
        f"end_dates before slicing to ``last_n_quarters``."
    )
