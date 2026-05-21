"""Ridge R² admission check — classify a candidate factor against Barra L1.

PLAN.md §2.3 / §4 step 8. After backfill + offline evaluation pass, before
``admit()`` writes a factor into the library, we regress the candidate's
panel values on the 6 admitted Barra L1 style factors (Size and Industry
excluded — both are already stripped during the ``barra_ind_size``
neutralization pipeline). The pooled R² tells us how much of the candidate
is just a linear combination of existing style risks:

    R² < 0.10              -> pure_alpha       (orthogonal — keep)
    0.10 <= R² < 0.50      -> smart_beta       (partial style — keep)
    0.50 <= R² < 0.80      -> edge_smart_beta  (heavy style; keep only
                                                if residual ICIR clears
                                                day/month threshold)
    R² >= 0.80             -> reject           (style clone — drop)

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

from backtest.factor.evaluation import _compute_ic_stats, _ic_series, _load_market_data
from backtest.factor.registry import get_factor_meta
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
TIER_EDGE_SMART_BETA: str = "edge_smart_beta"
TIER_REJECT: str = "reject"

Tier = Literal["pure_alpha", "smart_beta", "edge_smart_beta", "reject"]

R2_PURE_ALPHA_MAX: float = 0.10
R2_SMART_BETA_MAX: float = 0.50
R2_EDGE_SMART_BETA_MAX: float = 0.80

# Residual ICIR floors for the edge_smart_beta tier (PLAN.md §4 step 8).
RESIDUAL_ICIR_FLOOR_DAILY: float = 1.0
RESIDUAL_ICIR_FLOOR_MONTHLY: float = 0.8

# Extra calendar days to pad on the right when loading market data to ensure
# we have enough rows for the H-day forward-return shift. Matches the
# convention used by evaluation._load_factor_and_returns.
_RETURN_LOAD_BUFFER_DAYS: int = 5


class RidgeCheckError(ValueError):
    """Base class for ridge-check failures.

    Subclassed so admit()'s CLI / callers can distinguish *infrastructure*
    problems (library not bootstrapped, candidate not backfilled, not
    enough overlap) from the *verdict-driven* style-clone rejection.
    Stays a ``ValueError`` for backwards compatibility with existing
    ``except ValueError`` paths.
    """


class LibraryNotBootstrappedError(RidgeCheckError):
    """A Barra L1 regressor is missing from the library DB."""


class CandidateNotBackfilledError(RidgeCheckError):
    """The candidate factor has no rows in the work DB."""


class InsufficientOverlapError(RidgeCheckError):
    """Candidate and regressors don't share enough rows for the fit."""


class StyleCloneRejectedError(RidgeCheckError):
    """The candidate's R² landed in the reject tier (≥ 0.80)."""


@dataclass(frozen=True)
class RidgeCheckResult:
    factor_id: str
    r2: float
    tier: Tier
    residual_icir: float | None
    n_obs: int
    n_regressors: int

    def as_meta(self) -> dict:
        """Subset suitable for stamping onto ``registry.json`` meta."""
        return {
            "r2": float(self.r2),
            "tier": self.tier,
            "residual_icir": (None if self.residual_icir is None
                              else float(self.residual_icir)),
            "n_obs": self.n_obs,
        }


def _classify(r2: float) -> Tier:
    if r2 < R2_PURE_ALPHA_MAX:
        return TIER_PURE_ALPHA
    if r2 < R2_SMART_BETA_MAX:
        return TIER_SMART_BETA
    if r2 < R2_EDGE_SMART_BETA_MAX:
        return TIER_EDGE_SMART_BETA
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
    # Need at least p+2 rows so the centered design has more rows than
    # parameters AND the SS_tot denominator has > 1 degree of freedom.
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


def _residual_icir(
    residual: np.ndarray,
    keys: np.ndarray,
    frequency: str,
) -> float | None:
    """Per-date IC between residual and 1-day forward return, then ICIR.

    ``keys`` is an ``(n, 2)`` array of (date, symbol) parallel to ``residual``.
    Forward returns are loaded once via :func:`_load_market_data`. Frequency
    selects the IC horizon (1 trading day for ``frequency=='D'``, 21 for
    monthly).
    """
    horizon = 21 if frequency == "M" else 1

    df = pd.DataFrame({
        "date": pd.to_datetime(keys[:, 0]),
        "symbol": keys[:, 1].astype(str),
        "resid": residual,
    })
    if df.empty:
        return None

    start = df["date"].min().strftime("%Y%m%d")
    end = (df["date"].max()
           + pd.Timedelta(days=horizon + _RETURN_LOAD_BUFFER_DAYS)).strftime("%Y%m%d")
    _, returns_df, _ = _load_market_data(
        symbols=df["symbol"].unique().tolist(),
        start=start, end=end, horizons=[horizon], ret_type="close",
    )

    ret_col = f"ret_{horizon}"
    merged = df.merge(returns_df[["date", "symbol", ret_col]], on=["date", "symbol"])
    if merged.empty:
        return None

    ic_per_day = merged.groupby("date", group_keys=False).apply(
        lambda g: _ic_series(g["resid"], g[ret_col]),
        include_groups=False,
    )
    icir = _compute_ic_stats(ic_per_day)["icir"]
    return None if (icir is None or np.isnan(icir)) else float(icir)


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
        Ridge regularization strength. ``1.0`` is a mild default that
        stabilises the moderately-correlated Barra L1 design without
        meaningfully shrinking the fit on a ~25M-row pool.
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
        ``{factor_id, r2, tier, residual_icir, n_obs, n_regressors}``.
        ``residual_icir`` is only computed for the ``edge_smart_beta`` band;
        otherwise ``None``.
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

        r2, residual, keys = _pooled_r2(candidate, reg_df, alpha=alpha)
        tier = _classify(r2)

        residual_icir: float | None = None
        if tier == TIER_EDGE_SMART_BETA:
            meta = get_factor_meta(factor_id)
            frequency = str(meta.get("frequency", "D"))
            residual_icir = _residual_icir(residual, keys, frequency=frequency)
            floor = (RESIDUAL_ICIR_FLOOR_MONTHLY if frequency == "M"
                     else RESIDUAL_ICIR_FLOOR_DAILY)
            if residual_icir is None or abs(residual_icir) < floor:
                tier = TIER_REJECT

        return RidgeCheckResult(
            factor_id=factor_id,
            r2=float(r2),
            tier=tier,
            residual_icir=residual_icir,
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
    "R2_EDGE_SMART_BETA_MAX",
    "R2_PURE_ALPHA_MAX",
    "R2_SMART_BETA_MAX",
    "RESIDUAL_ICIR_FLOOR_DAILY",
    "RESIDUAL_ICIR_FLOOR_MONTHLY",
    "RidgeCheckError",
    "RidgeCheckResult",
    "StyleCloneRejectedError",
    "TIER_EDGE_SMART_BETA",
    "TIER_PURE_ALPHA",
    "TIER_REJECT",
    "TIER_SMART_BETA",
    "Tier",
    "ridge_r2_check",
]
