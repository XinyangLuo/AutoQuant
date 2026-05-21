"""Factor neutralization variant naming.

Each factor gets exactly **one** version of values in ``factors_daily``. The
value reflects whichever neutralization pipeline was applied during compute;
``variant`` is just a label that records *which* pipeline produced it, stored
in ``registry.json`` alongside the factor metadata.

So ``variant`` lives in metadata, not in the data table schema. The table
PK is ``(date, symbol)``; the column is the ``factor_id``; the variant label
travels with the registry entry. To re-compute a factor under a different
variant, you re-run backfill — the new values overwrite the old.

Valid variant names
-------------------

``"none"``
    No post-processing. The stored value is whatever the compute function
    returned. Used by Barra L1 composites (already z-scored inputs combined
    equal-weight) and by any factor that opts out of the pipeline.

``"barra_l3"``
    The CNE6 L3 style-exposure pipeline: MAD winsorize → SW-L1 industry
    median fill → cs_zscore. Used by all 11 Barra L3 factors. The output
    is a z-scored *style exposure* — NOT a residual against other styles.
    Style factors are themselves regressors elsewhere; they don't get
    industry/size neutralized.

``"barra_ind_size"``
    The PLAN.md §2.2 alpha-neutralization pipeline: MAD winsorize → SW-L1
    industry median fill → cs_zscore → cross-section OLS regression on
    industry dummies + ``log(circ_mv)`` → residual → re-cs_zscore.
    **Default** for user-registered alphas — strips industry and size
    style exposure so what remains is pure alpha. Future variants follow
    the same ``barra_<inputs>`` naming if more regressors are added.
"""

from __future__ import annotations


NONE_VARIANT: str = "none"
BARRA_L3_VARIANT: str = "barra_l3"
BARRA_IND_SIZE_VARIANT: str = "barra_ind_size"

VALID_VARIANTS: tuple[str, ...] = (
    NONE_VARIANT,
    BARRA_L3_VARIANT,
    BARRA_IND_SIZE_VARIANT,
)

#: ``@register`` default variant — user alphas get the unified Barra-style
#: neutralization unless they explicitly pass ``variant="none"``.
DEFAULT_VARIANT: str = BARRA_IND_SIZE_VARIANT


def validate_variant(variant: str) -> str:
    """Return ``variant`` unchanged if valid, else raise ``ValueError``."""
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}. Valid: {VALID_VARIANTS}"
        )
    return variant


__all__ = [
    "NONE_VARIANT",
    "BARRA_L3_VARIANT",
    "BARRA_IND_SIZE_VARIANT",
    "DEFAULT_VARIANT",
    "VALID_VARIANTS",
    "validate_variant",
]
