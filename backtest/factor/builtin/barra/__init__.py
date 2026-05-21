"""Barra-style risk factors (CNE6 trimmed subset).

11 L3 (style-exposure) factors → 7 L1 composites. See PLAN.md §2.1 for the
canonical formula table and ``backtest/factor/DESIGN.md`` for storage layout.

L3 factors (``variant="barra_l3"``)
    Pipeline: MAD winsorize → SW-L1 industry median fill → cs_zscore.
    The stored value is the z-scored style exposure, ready to act as a
    regressor in alpha-neutralization (``variant="barra_ind_size"``).

L1 composites (``variant="none"``)
    Equal-weight averages of pre-z-scored L3 inputs read directly from
    ``factors_daily``. No further post-processing.

Naming:
    L3 ``f_barra_<l1>_<l3>``  e.g. ``f_barra_size_lncap``
    L1 ``f_barra_<l1>``       e.g. ``f_barra_size``
"""

from backtest.factor.builtin.barra import (
    beta,
    composite,
    growth,
    liquidity,
    momentum,
    quality,
    size,
    value,
)

__all__ = [
    "beta",
    "composite",
    "growth",
    "liquidity",
    "momentum",
    "quality",
    "size",
    "value",
]
