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


# ---------------------------------------------------------------------------
# Residual ICIR incremental-information check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualICIRResult:
    """Outcome of the residual ICIR incremental-info admission check.

    Regresses the candidate against ALL admitted factors per-date (Ridge),
    then computes the RankICIR of the residuals against forward returns.
    """

    factor_id: str
    residual_rank_icirs: dict[int, float]
    residual_rank_ic_means: dict[int, float]
    residual_rank_ic_stds: dict[int, float]
    residual_rank_ic_pos_ratios: dict[int, float]
    annual_icirs: dict[int, float]
    n_regressors: int
    n_dates: int
    n_obs_total: int
    threshold: float
    ic_mean_threshold: float
    passed: bool

    def as_meta(self) -> dict:
        return {
            "residual_rank_icirs": self.residual_rank_icirs,
            "annual_icirs": self.annual_icirs,
            "residual_rank_ic_means": self.residual_rank_ic_means,
            "n_regressors": self.n_regressors,
            "n_dates": self.n_dates,
            "n_obs_total": self.n_obs_total,
            "threshold": self.threshold,
            "ic_mean_threshold": self.ic_mean_threshold,
            "passed": self.passed,
        }


def _per_date_ridge_residuals(
    candidate: pd.DataFrame,
    regressors: pd.DataFrame,
    *,
    alpha: float = 1.0,
    min_obs_per_date: int | None = None,
) -> pd.DataFrame:
    """For each date, fit ridge on the cross-section and return per-row residual.

    Parameters
    ----------
    candidate : pd.DataFrame
        Columns ``[date, symbol, value]``.
    regressors : pd.DataFrame
        Columns ``[date, symbol, <regressor_id>, ...]``.
    alpha : float
        Ridge regularisation strength.
    min_obs_per_date : int | None
        Minimum observations per date. Defaults to ``n_regressors + 2``.

    Returns
    -------
    pd.DataFrame
        Columns ``[date, symbol, residual]``.

    Raises
    ------
    InsufficientOverlapError
        If no date has enough overlapping observations.
    """
    merged = candidate.merge(regressors, on=["date", "symbol"], how="inner")
    if merged.empty:
        raise InsufficientOverlapError(
            "Candidate and regressors share zero overlapping (date, symbol) rows."
        )

    reg_cols = [c for c in regressors.columns if c not in ("date", "symbol")]
    n_reg = len(reg_cols)
    if min_obs_per_date is None:
        min_obs_per_date = n_reg + 2

    residuals_parts: list[pd.DataFrame] = []
    for date, group in merged.groupby("date"):
        sub = group.dropna(subset=["value", *reg_cols])
        if len(sub) < min_obs_per_date:
            continue
        X = sub[reg_cols].to_numpy(dtype=float)
        y = sub["value"].to_numpy(dtype=float)
        try:
            # OLS first — strips linear signal completely.
            # Fall back to Ridge only on rank-deficient days.
            if alpha == 0.0:
                beta, resid, _rank, _s = np.linalg.lstsq(
                    np.column_stack([np.ones(len(X)), X]), y, rcond=None
                )
                intercept = float(beta[0])
                beta = beta[1:]
            else:
                beta, intercept = _ridge_fit(X, y, alpha=alpha)
        except np.linalg.LinAlgError:
            # Fall back to Ridge with configured alpha; use mild
            # regularisation (1.0) if configured alpha is also 0.0.
            fallback_alpha = alpha if alpha != 0.0 else 1.0
            beta, intercept = _ridge_fit(X, y, alpha=fallback_alpha)
        y_hat = X @ beta + intercept
        residual = y - y_hat
        residuals_parts.append(
            pd.DataFrame(
                {"date": sub["date"], "symbol": sub["symbol"], "residual": residual}
            )
        )

    if not residuals_parts:
        raise InsufficientOverlapError(
            f"No date had >= {min_obs_per_date} valid observations "
            f"for {n_reg} regressors."
        )

    return pd.concat(residuals_parts, ignore_index=True)


def _load_all_admitted_regressors(
    factor_id: str,
    start: str,
    end: str,
    library: FactorLibrary,
) -> pd.DataFrame:
    """Load ALL admitted factors from library as a wide regressor DataFrame.

    Excludes ``factor_id`` (in case it already exists in the library).
    Returns an empty DataFrame (no columns beyond date/symbol) if the
    library has no admitted factors.
    """
    others = library.get_factors_long(start=start, end=end, exclude=factor_id)
    if others.empty:
        return pd.DataFrame(columns=["date", "symbol"])

    wide = others.pivot(
        index=["date", "symbol"], columns="factor_id", values="value"
    ).reset_index()
    wide.columns.name = None
    return wide


def _load_forward_returns_for_check(
    symbols: list[str],
    start: str,
    end: str,
    horizons: list[int],
    ret_type: str = "open",
) -> pd.DataFrame:
    """Load market data and compute forward returns for residual ICIR check."""
    from backtest.data.storage import MarketStorage

    max_h = max(horizons)
    returns_end = (
        pd.Timestamp(end) + pd.Timedelta(days=int(max_h) + 10)
    ).strftime("%Y%m%d")

    with MarketStorage(read_only=True) as ms:
        market_df = ms.get_bars(symbols=symbols, start=start, end=returns_end)

    if market_df.empty:
        raise ValueError("No market data available for forward return calculation.")

    from backtest.factor.evaluation import _compute_forward_returns

    return _compute_forward_returns(market_df, horizons, ret_type)


def _residual_rank_icirs(
    residuals: pd.DataFrame,
    returns_df: pd.DataFrame,
    horizons: list[int],
) -> dict[int, dict]:
    """Per-date RankIC of residuals vs forward returns, then ICIR per horizon.

    Parameters
    ----------
    residuals : pd.DataFrame
        Columns ``[date, symbol, residual]``.
    returns_df : pd.DataFrame
        Columns ``[date, symbol, ret_1, ret_5, ...]``.
    horizons : list[int]
        Forward-return horizons to check, e.g. ``[1, 5, 20]``.

    Returns
    -------
    dict[int, dict]
        ``{horizon: {"icir", "ic_mean", "ic_std", "ic_positive_ratio", "ic_count"}}``.
    """
    from backtest.factor.evaluation import _compute_ic_stats, _rank_ic_series

    merged = residuals.merge(returns_df, on=["date", "symbol"], how="inner")

    results: dict[int, dict] = {}
    for h in horizons:
        ret_col = f"ret_{h}"
        if ret_col not in merged.columns:
            results[h] = {
                "icir": float("nan"),
                "ic_mean": float("nan"),
                "ic_std": float("nan"),
                "ic_positive_ratio": float("nan"),
                "ic_count": 0,
            }
            continue

        ic_series = merged.groupby("date").apply(
            lambda g: _rank_ic_series(g["residual"], g[ret_col]),
            include_groups=False,
        )
        results[h] = _compute_ic_stats(ic_series)

    return results


def _get_residual_icir_config() -> dict:
    """Read residual ICIR thresholds from config.yaml with fallback defaults."""
    try:
        from backtest.config_loader import get_section

        return get_section("thresholds", "admission", "residual_icir")
    except KeyError:
        return {
            "min_annual_icir": 0.05,
            "min_abs_ic_mean": 0.001,
            "horizons": [1, 5, 20],
            "ridge_alpha": 0.0,
        }


def residual_icir_check(
    factor_id: str,
    *,
    horizons: list[int] | None = None,
    threshold: float | None = None,
    ic_mean_threshold: float | None = None,
    alpha: float | None = None,
    ret_type: str = "open",
    start: str | None = None,
    end: str | None = None,
    factor_storage: FactorStorage | None = None,
    library: FactorLibrary | None = None,
) -> ResidualICIRResult:
    """Check if candidate adds incremental predictive power beyond ALL admitted factors.

    Steps:

    1. Load candidate from work DB.
    2. Load ALL admitted factors from library DB as regressors.
    3. If 0 regressors: trivial pass.
    4. Per-date Ridge regression -> residuals.
    5. Compute forward returns for configured horizons.
    6. Per-date RankIC of residuals vs returns per horizon -> ICIR.
    7. Annualize ICIR, pass if ANY horizon > threshold.

    Returns
    -------
    ResidualICIRResult
    """
    cfg = _get_residual_icir_config()
    if horizons is None:
        horizons = cfg.get("horizons", [1, 5, 20])
    if threshold is None:
        threshold = float(cfg.get("min_annual_icir", 0.05))
    if ic_mean_threshold is None:
        ic_mean_threshold = float(cfg.get("min_abs_ic_mean", 0.001))
    if alpha is None:
        alpha = float(cfg.get("ridge_alpha", 0.0))

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
                f"Candidate {factor_id} has no rows in the work DB. Backfill first."
            )

        reg_df = _load_all_admitted_regressors(
            factor_id,
            start=start or candidate["date"].min().strftime("%Y%m%d"),
            end=end or candidate["date"].max().strftime("%Y%m%d"),
            library=library,
        )
        reg_cols = [c for c in reg_df.columns if c not in ("date", "symbol")]
        n_regressors = len(reg_cols)

        if n_regressors == 0:
            return ResidualICIRResult(
                factor_id=factor_id,
                residual_rank_icirs={h: float("nan") for h in horizons},
                residual_rank_ic_means={h: float("nan") for h in horizons},
                residual_rank_ic_stds={h: float("nan") for h in horizons},
                residual_rank_ic_pos_ratios={h: float("nan") for h in horizons},
                annual_icirs={h: float("nan") for h in horizons},
                n_regressors=0,
                n_dates=0,
                n_obs_total=0,
                threshold=threshold,
                ic_mean_threshold=ic_mean_threshold,
                passed=True,
            )

        residuals = _per_date_ridge_residuals(
            candidate, reg_df, alpha=alpha
        )

        use_start = (
            start or candidate["date"].min().strftime("%Y%m%d")
        )
        use_end = (
            end or candidate["date"].max().strftime("%Y%m%d")
        )
        symbols = residuals["symbol"].unique().tolist()
        returns_df = _load_forward_returns_for_check(
            symbols=symbols,
            start=use_start,
            end=use_end,
            horizons=horizons,
            ret_type=ret_type,
        )

        icir_by_horizon = _residual_rank_icirs(residuals, returns_df, horizons)

        n_dates = max(
            (stats.get("ic_count", 0) for stats in icir_by_horizon.values()),
            default=0,
        )

        residual_rank_icirs: dict[int, float] = {}
        residual_rank_ic_means: dict[int, float] = {}
        residual_rank_ic_stds: dict[int, float] = {}
        residual_rank_ic_pos_ratios: dict[int, float] = {}
        annual_icirs: dict[int, float] = {}
        any_passed = False

        import math

        for h in horizons:
            stats = icir_by_horizon.get(h, {})
            raw_icir = stats.get("icir", float("nan"))
            residual_rank_icirs[h] = float(raw_icir) if not (isinstance(raw_icir, float) and math.isnan(raw_icir)) else float("nan")
            residual_rank_ic_means[h] = float(stats.get("ic_mean", float("nan")))
            residual_rank_ic_stds[h] = float(stats.get("ic_std", float("nan")))
            residual_rank_ic_pos_ratios[h] = float(stats.get("ic_positive_ratio", float("nan")))

            raw = residual_rank_icirs[h]
            if not math.isnan(raw):
                annual = raw * math.sqrt(252.0 / h)
            else:
                annual = float("nan")
            annual_icirs[h] = annual

            ic_mean = residual_rank_ic_means[h]
            if (not math.isnan(annual) and annual > threshold
                    and not math.isnan(ic_mean)
                    and abs(ic_mean) > ic_mean_threshold):
                any_passed = True

        return ResidualICIRResult(
            factor_id=factor_id,
            residual_rank_icirs=residual_rank_icirs,
            residual_rank_ic_means=residual_rank_ic_means,
            residual_rank_ic_stds=residual_rank_ic_stds,
            residual_rank_ic_pos_ratios=residual_rank_ic_pos_ratios,
            annual_icirs=annual_icirs,
            n_regressors=n_regressors,
            n_dates=int(n_dates),
            n_obs_total=int(len(residuals)),
            threshold=threshold,
            ic_mean_threshold=ic_mean_threshold,
            passed=any_passed,
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
    "ResidualICIRResult",
    "RidgeCheckError",
    "RidgeCheckResult",
    "StyleCloneRejectedError",
    "TIER_PURE_ALPHA",
    "TIER_REJECT",
    "TIER_SMART_BETA",
    "Tier",
    "residual_icir_check",
    "ridge_r2_check",
]
