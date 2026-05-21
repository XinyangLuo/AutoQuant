"""Barra L1 composite factors — equal-weight averages of pre-z-scored L3 inputs.

Read L3 values directly from ``factors_daily`` (already MAD-winsorized,
industry-median-filled, cross-section z-scored by the ``barra_l3`` variant
pipeline). No further post-processing — ``variant="none"``.

Composition (PLAN.md §2.1):
    Size      = LNCAP
    Beta      = BETA
    Momentum  = RSTR
    Value     = (BTOP + ETOP + DTOP) / 3
    Quality   = (ROA + GP + AGRO) / 3
    Liquidity = STOM
    Growth    = EGRO
"""

from __future__ import annotations

import pandas as pd

from backtest.factor.registry import register
from backtest.factor.storage import FactorStorage
from backtest.factor.variants import NONE_VARIANT


def _composite_from_l3(
    factor_storage: FactorStorage,
    l3_ids: list[str],
    start_date: str | None,
    end_date: str | None,
) -> pd.Series:
    """Read L3 columns from the work DB and return their equal-weight average."""
    panel = factor_storage.get_factors_long(
        factor_ids=l3_ids, start=start_date, end=end_date,
    )
    if panel.empty:
        return pd.Series(dtype=float)
    if len(l3_ids) == 1:
        return panel.set_index(["date", "symbol"])["value"]
    return panel.groupby(["date", "symbol"])["value"].mean()


def _register_composite(
    factor_id: str,
    name: str,
    description: str,
    l3_ids: list[str],
):
    @register(
        factor_id,
        name=name,
        category="barra_l1",
        data_sources=["factors_daily"],
        description=description,
        variant=NONE_VARIANT,
        frequency="D",
    )
    def _fn(
        panel: pd.DataFrame,
        *,
        factor_storage: FactorStorage,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        del panel  # unused — composites read from factor_storage
        series = _composite_from_l3(factor_storage, l3_ids, start_date, end_date)
        series.name = factor_id.removeprefix("f_barra_")
        return series
    _fn.__name__ = f"barra_{factor_id.removeprefix('f_barra_')}"
    return _fn


barra_size = _register_composite(
    "f_barra_size",
    "Barra L1 — Size",
    "Equal-weight of {LNCAP}. Single-input composite.",
    ["f_barra_size_lncap"],
)

barra_beta = _register_composite(
    "f_barra_beta",
    "Barra L1 — Beta",
    "Equal-weight of {BETA}. Single-input composite.",
    ["f_barra_beta_beta"],
)

barra_momentum = _register_composite(
    "f_barra_momentum",
    "Barra L1 — Momentum",
    "Equal-weight of {RSTR}. Single-input composite.",
    ["f_barra_momentum_rstr"],
)

barra_value = _register_composite(
    "f_barra_value",
    "Barra L1 — Value",
    "Equal-weight of {BTOP, ETOP, DTOP}.",
    ["f_barra_value_btop", "f_barra_value_etop", "f_barra_value_dtop"],
)

barra_quality = _register_composite(
    "f_barra_quality",
    "Barra L1 — Quality",
    "Equal-weight of {ROA, GP, AGRO}.",
    ["f_barra_quality_roa", "f_barra_quality_gp", "f_barra_quality_agro"],
)

barra_liquidity = _register_composite(
    "f_barra_liquidity",
    "Barra L1 — Liquidity",
    "Equal-weight of {STOM}. Single-input composite.",
    ["f_barra_liquidity_stom"],
)

barra_growth = _register_composite(
    "f_barra_growth",
    "Barra L1 — Growth",
    "Equal-weight of {EGRO}. Single-input composite.",
    ["f_barra_growth_egro"],
)
