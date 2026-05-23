"""Ridge R² admission check — classify a candidate factor against Barra L1.

After backfill + offline evaluation pass, before ``admit()`` writes a
factor into the library, we regress the candidate's panel values on the
6 admitted Barra L1 style factors (Size and Industry excluded — both are
already stripped during the ``barra_ind_size`` neutralization pipeline).
The pooled R² tells us how much of the candidate is just a linear
combination of existing style risks.

Tiers are defined in ``config.yaml`` (``thresholds.admission.ridge_r2``):

    R² < pure_alpha_max     -> pure_alpha  (orthogonal — keep)
    pure_alpha_max ≤ R²
      < smart_beta_max      -> smart_beta  (partial style — keep)
    R² ≥ smart_beta_max     -> reject      (style clone — drop)

Ridge (small α) instead of OLS so the 6 Barra L1 — which are themselves
moderately correlated by construction — don't blow up via collinearity.

This module is read-only against the factor stores. It returns a verdict;
``admission.admit()`` owns the side effects (registry write, library
promotion, reject path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from backtest.config_loader import get_section
from backtest.factor.storage import FactorLibrary, FactorStorage


BARRA_L1_REGRESSORS: tuple[str, ...] = (
    "f_barra_beta",
    "f_barra_momentum",
    "f_barra_value",
    "f_barra_quality",
    "f_barra_liquidity",
    "f_barra_growth",
)

TIER_PURE_ALPHA: str = "pure_alpha"
TIER_SMART_BETA: str = "smart_beta"
TIER_REJECT: str = "reject"

Tier = Literal["pure_alpha", "smart_beta", "reject"]


def _get_ridge_thresholds():
    """Read ridge R² thresholds from config.yaml (single source of truth)."""
    return get_section("thresholds", "admission", "ridge_r2")


class RidgeCheckError(ValueError):
    """Base class for ridge-check failures.

    Subclassed so admit()'s CLI / callers can distinguish *infrastructure*
    problems (library not bootstrapped, candidate not backfilled, not
    enough overlap) from the *verdict-driven* style-clone rejection.
    """


class LibraryNotBootstrappedError(RidgeCheckError):
    """A Barra L1 regressor is missing from the library DB."""


class CandidateNotBackfilledError(RidgeCheckError):
    """The candidate factor has no rows in the work DB."""


class InsufficientOverlapError(RidgeCheckError):
    """Candidate and regressors don't share enough rows for the fit."""


class StyleCloneRejectedError(RidgeCheckError):
    """The candidate's R² landed in the reject tier."""


@dataclass(frozen=True)
class RidgeCheckResult:
    factor_id: str
    r2: float
    tier: Tier
    n_obs: int
    n_regressors: int

    def as_meta(self) -> dict:
        """Subset suitable for stamping onto ``registry.json`` meta."""
        return {
            "r2": float(self.r2),
            "tier": self.tier,
            "n_obs": self.n_obs,
        }


def _classify(r2: float) -> Tier:
    th = _get_ridge_thresholds()
    if r2 < th["pure_alpha_max"]:
        return TIER_PURE_ALPHA
    if r2 < th["smart_beta_max"]:
        return TIER_SMART_BETA
    return TIER_REJECT


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Closed-form ridge with intercept; returns (beta_no_intercept, intercept).

    Centers X and y to absorb the intercept, then solves
    ``(XᵀX + α I) β = Xᵀy`` on the centered data.
    """
    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - x_mean
    yc = y - y_mean
    XtX = Xc.T @ Xc
    XtX[np.diag_indices_from(XtX)] += alpha
    beta = np.linalg.solve(XtX, Xc.T @ yc)
    intercept = float(y_mean - x_mean @ beta)
    return beta, intercept


def _pooled_r2(
    candidate: pd.DataFrame,
    regressors: pd.DataFrame,
    *,
    alpha: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Inner-join candidate to regressors, fit ridge, return (r2, residual, keys).

    ``candidate`` columns: ``[date, symbol, value]``.
    ``regressors`` columns: ``[date, symbol, <regressor_id>, ...]``.
    Rows with any NaN are dropped before fitting. ``keys`` is the
    ``(n_obs, 2)`` (date, symbol) array parallel to the residual.
    """
    merged = candidate.merge(regressors, on=["date", "symbol"], how="inner")
    reg_cols = [c for c in regressors.columns if c not in ("date", "symbol")]
    merged = merged.dropna(subset=["value", *reg_cols])
    if len(merged) < len(reg_cols) + 2:
        raise InsufficientOverlapError(
            f"Too few overlapping rows for ridge fit: got {len(merged)}, "
            f"need >= {len(reg_cols) + 2} for {len(reg_cols)} regressors."
        )

    X = merged[reg_cols].to_numpy(dtype=float)
    y = merged["value"].to_numpy(dtype=float)
    beta, intercept = _ridge_fit(X, y, alpha=alpha)
    y_hat = X @ beta + intercept
    residual = y - y_hat
    ss_res = float((residual ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    keys = merged[["date", "symbol"]].to_numpy()
    return r2, residual, keys


def ridge_r2_check(
    factor_id: str,
    *,
    alpha: float = 1.0,
    start: str | None = None,
    end: str | None = None,
    factor_storage: FactorStorage | None = None,
    library: FactorLibrary | None = None,
    regressors: tuple[str, ...] = BARRA_L1_REGRESSORS,
) -> RidgeCheckResult:
    """Classify a work-DB candidate against the 6 Barra L1 styles in the library.

    Parameters
    ----------
    factor_id
        Candidate factor in the **work DB**.
    alpha
        Ridge regularization strength. ``1.0`` is a mild default.
    start, end
        Optional date window (``YYYYMMDD``). Defaults to the full overlap of
        candidate and regressors.
    factor_storage, library
        Optional pre-opened handles. The function opens (and closes) its own
        connections when these are ``None``.
    regressors
        Override the regressor list (mostly useful for tests).

    Returns
    -------
    RidgeCheckResult
    """
    own_fs = factor_storage is None
    own_lib = library is None
    try:
        if factor_storage is None:
            factor_storage = FactorStorage()
        if library is None:
            library = FactorLibrary()

        candidate = factor_storage.get_factor(factor_id, start=start, end=end)
        if candidate.empty:
            raise CandidateNotBackfilledError(
                f"Candidate {factor_id} has no rows in the work DB for "
                f"range {start}~{end}. Backfill first."
            )

        wide_parts: list[pd.DataFrame] = []
        for reg_id in regressors:
            sub = library.get_factor(reg_id, start=start, end=end)
            if sub.empty:
                raise LibraryNotBootstrappedError(
                    f"Regressor {reg_id} missing from library. Admit the "
                    f"Barra L1 composites first."
                )
            wide_parts.append(sub.rename(columns={"value": reg_id}))

        reg_df = wide_parts[0]
        for sub in wide_parts[1:]:
            reg_df = reg_df.merge(sub, on=["date", "symbol"], how="outer")

        r2, _residual, keys = _pooled_r2(candidate, reg_df, alpha=alpha)
        tier = _classify(r2)

        return RidgeCheckResult(
            factor_id=factor_id,
            r2=float(r2),
            tier=tier,
            n_obs=int(keys.shape[0]),
            n_regressors=len(regressors),
        )
    finally:
        if own_fs and factor_storage is not None:
            factor_storage.close()
        if own_lib and library is not None:
            library.close()


__all__ = [
    "BARRA_L1_REGRESSORS",
    "CandidateNotBackfilledError",
    "InsufficientOverlapError",
    "LibraryNotBootstrappedError",
    "RidgeCheckError",
    "RidgeCheckResult",
    "StyleCloneRejectedError",
    "TIER_PURE_ALPHA",
    "TIER_REJECT",
    "TIER_SMART_BETA",
    "Tier",
    "ridge_r2_check",
]
