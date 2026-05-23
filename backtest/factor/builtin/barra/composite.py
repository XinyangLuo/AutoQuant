"""Barra L1 composite factors — equal-weight averages of pre-z-scored L3 inputs.

Each L1 factor:

1. Calls the relevant L3 helper(s) on the input ``panel`` — those are now
   plain Python functions in this package, not registered factors.
2. Runs each raw L3 series through the standard L3 pipeline
   (``apply_l3_pipeline``: MAD winsorize → SW-L1 industry median fill →
   cs_zscore) so each component becomes a comparable z-score.
3. Averages the z-scored components equal-weight; the result is the L1
   composite that goes into ``factor_library.duckdb``.

Composition (PLAN.md §2.1):
    Size      = LNCAP
    Beta      = BETA
    Momentum  = RSTR
    Value     = (BTOP + ETOP + DTOP) / 3
    Quality   = (ROA + GP + AGRO) / 3
    Liquidity = STOM
    Growth    = EGRO

``variant="none"`` for all L1s — post-processing already happened on the
L3 components inside this module, so the variant pipeline shouldn't touch
the composite again.
"""

from __future__ import annotations

import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.factor.builtin.barra._common import apply_l3_pipeline
from backtest.factor.builtin.barra.beta import barra_beta_beta
from backtest.factor.builtin.barra.growth import barra_growth_egro
from backtest.factor.builtin.barra.liquidity import barra_liquidity_stom
from backtest.factor.builtin.barra.momentum import barra_momentum_rstr
from backtest.factor.builtin.barra.quality import (
    barra_quality_agro,
    barra_quality_gp,
    barra_quality_roa,
)
from backtest.factor.builtin.barra.size import barra_size_lncap
from backtest.factor.builtin.barra.value import (
    barra_value_btop,
    barra_value_dtop,
    barra_value_etop,
)
from backtest.factor.registry import register
from backtest.factor.variants import CATEGORY_BARRA_L1, NONE_VARIANT


def _z(
    raw: pd.Series,
    market_storage: MarketStorage,
    *,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Shortcut for the per-component L3 pipeline used by every composite."""
    return apply_l3_pipeline(raw, market_storage, start=start_date, end=end_date)


def _average(components: list[pd.Series]) -> pd.Series:
    """Equal-weight average of z-scored components on their union index.

    Each component is a ``(date, symbol)`` Series. ``pd.concat`` aligns by
    index; ``.mean(axis=1)`` ignores NaNs so a symbol missing one component
    still gets a value from the remaining ones.
    """
    if len(components) == 1:
        return components[0]
    wide = pd.concat(components, axis=1)
    return wide.mean(axis=1)


# ---------------------------------------------------------------------------
# Single-component composites
# ---------------------------------------------------------------------------


@register(
    "f_barra_size",
    name="Barra L1 — Size",
    category=CATEGORY_BARRA_L1,
    data_sources=["market_daily"],
    description="z-scored ln(circ_mv). Single-input composite (= LNCAP).",
    variant=NONE_VARIANT,
    frequency="D",
)
def barra_size(
    panel: pd.DataFrame,
    *,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
) -> pd.Series:
    raw = barra_size_lncap(panel)
    return _z(raw, market_storage, start_date=start_date, end_date=end_date)


@register(
    "f_barra_beta",
    name="Barra L1 — Beta",
    category=CATEGORY_BARRA_L1,
    data_sources=["market_daily"],
    description=(
        "z-scored WLS slope of daily log-returns on CSI 300 "
        "(window=252d, half-life=63d). Single-input composite (= BETA)."
    ),
    variant=NONE_VARIANT,
    frequency="D",
    parameters={"window": 274},  # BETA_WINDOW + 22d cushion
)
def barra_beta(
    panel: pd.DataFrame,
    *,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
) -> pd.Series:
    raw = barra_beta_beta(panel, market_storage=market_storage)
    return _z(raw, market_storage, start_date=start_date, end_date=end_date)


@register(
    "f_barra_momentum",
    name="Barra L1 — Momentum",
    category=CATEGORY_BARRA_L1,
    data_sources=["market_daily"],
    description=(
        "z-scored CNE6 RSTR: EWMA(252d, half-life=126d) of ln(1+r_t), "
        "lagged 11d then 11d smoothed. Single-input composite (= RSTR)."
    ),
    variant=NONE_VARIANT,
    frequency="D",
    parameters={"window": 274},  # 252 + 11 + 11
)
def barra_momentum(
    panel: pd.DataFrame,
    *,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
) -> pd.Series:
    raw = barra_momentum_rstr(panel)
    return _z(raw, market_storage, start_date=start_date, end_date=end_date)


@register(
    "f_barra_liquidity",
    name="Barra L1 — Liquidity",
    category=CATEGORY_BARRA_L1,
    data_sources=["market_daily"],
    description="z-scored ln(rolling-21d sum of amount/circ_mv). Single-input composite (= STOM).",
    variant=NONE_VARIANT,
    frequency="D",
    parameters={"window": 21},
)
def barra_liquidity(
    panel: pd.DataFrame,
    *,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
) -> pd.Series:
    raw = barra_liquidity_stom(panel)
    return _z(raw, market_storage, start_date=start_date, end_date=end_date)


@register(
    "f_barra_growth",
    name="Barra L1 — Growth",
    category=CATEGORY_BARRA_L1,
    data_sources=["market_daily", "income_q"],
    description=(
        "z-scored slope of last 20 quarterly TTM EPS on time / |mean|. "
        "Single-input composite (= EGRO)."
    ),
    variant=NONE_VARIANT,
    frequency="D",
)
def barra_growth(
    panel: pd.DataFrame,
    *,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
) -> pd.Series:
    raw = barra_growth_egro(panel)
    return _z(raw, market_storage, start_date=start_date, end_date=end_date)


# ---------------------------------------------------------------------------
# Multi-component composites
# ---------------------------------------------------------------------------


@register(
    "f_barra_value",
    name="Barra L1 — Value",
    category=CATEGORY_BARRA_L1,
    data_sources=["market_daily", "income_q", "balancesheet_q"],
    description=(
        "Equal-weight average of z-scored {BTOP, ETOP, DTOP}. "
        "BTOP = book_equity_incl_minority / circ_mv. "
        "ETOP = TTM net_income_attr_p / circ_mv. "
        "DTOP = trailing-12m cash_div / pre_close."
    ),
    variant=NONE_VARIANT,
    frequency="D",
)
def barra_value(
    panel: pd.DataFrame,
    *,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
) -> pd.Series:
    btop_z = _z(barra_value_btop(panel), market_storage, start_date=start_date, end_date=end_date)
    etop_z = _z(barra_value_etop(panel), market_storage, start_date=start_date, end_date=end_date)
    dtop_z = _z(
        barra_value_dtop(
            panel, market_storage=market_storage,
            start_date=start_date, end_date=end_date,
        ),
        market_storage, start_date=start_date, end_date=end_date,
    )
    return _average([btop_z, etop_z, dtop_z])


@register(
    "f_barra_quality",
    name="Barra L1 — Quality",
    category=CATEGORY_BARRA_L1,
    data_sources=["market_daily", "income_q", "balancesheet_q"],
    description=(
        "Equal-weight average of z-scored {ROA, GP, AGRO}. "
        "ROA = TTM net_income / total_assets. "
        "GP = (TTM revenue - TTM oper_cost) / total_assets. "
        "AGRO = -slope(20 quarterly total_assets) / |mean| (sign-flipped so "
        "rapid asset growth ⇒ lower quality)."
    ),
    variant=NONE_VARIANT,
    frequency="D",
)
def barra_quality(
    panel: pd.DataFrame,
    *,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
) -> pd.Series:
    roa_z = _z(barra_quality_roa(panel), market_storage, start_date=start_date, end_date=end_date)
    gp_z = _z(barra_quality_gp(panel), market_storage, start_date=start_date, end_date=end_date)
    agro_z = _z(barra_quality_agro(panel), market_storage, start_date=start_date, end_date=end_date)
    return _average([roa_z, gp_z, agro_z])
