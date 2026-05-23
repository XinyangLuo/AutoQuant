"""Barra-style risk factors (CNE6 trimmed subset) — 7 L1 composites.

Only the 7 L1 composites are registered factors that land in
``factor_library.duckdb``: Size, Beta, Momentum, Value, Quality, Liquidity,
Growth. The 11 underlying L3 building blocks (LNCAP / BTOP / ETOP / DTOP /
BETA / RSTR / STOM / EGRO / ROA / GP / AGRO) are internal Python helpers
living in this package — they are NOT registered, NOT in the registry,
and NOT in either work or library DB.

Each L1 composite:
    1. Calls its L3 helper(s) on the input panel.
    2. z-scores each L3 series via the L3 pipeline
       (MAD winsorize → SW-L1 industry median fill → cs_zscore).
    3. Equal-weight averages the z-scored components.

Composite-side ``variant="none"`` — the L3 pipeline already happened
on each component.
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
