"""Factor module: definition, computation, storage, offline evaluation, admission.

Two physical DuckDBs:

* ``factors.duckdb`` — work area used by ``backfill`` / ``compute`` /
  ``evaluation`` while researching new factors. Temporary.
* ``factor_library.duckdb`` — stable library, only written to by ``admit()``.
  This is the **only** source consulted by evaluation's cross-factor
  correlation check, so admission compares against stabilised peers — not
  whatever happens to be sitting in the work area.

Public API
----------
register(factor_id, *, name, category, data_sources, ...)
    Decorator to register a factor compute function.

compute_factor(factor_id, start_date, end_date, ...)
    Compute a single factor for a date range.

evaluate(factor_id, start, end, *, horizons, ret_type)
    Offline evaluation: IC / RankIC / ICIR / turnover / decay / group returns,
    plus cross-sectional rank correlation against admitted factors.

FactorStorage
    DuckDB wrapper for the work DB (factors.duckdb).
FactorLibrary
    DuckDB wrapper for the stable library (factor_library.duckdb).

admit / reject
    Promote a factor from work → library, or discard it. Both clear the
    work DB and update ``registry.json``.

rank, z_score
    Common operators for use inside factor compute functions.
"""

from backtest.factor.admission import (
    AdmissionAction,
    RECOMMENDED_THRESHOLDS,
    STATUS_ADMITTED,
    STATUS_REJECTED,
    admit,
    check_recommended_thresholds,
    get_admitted_factor_ids,
    get_pending_factor_ids,
    get_rejected_factor_ids,
    print_action,
    print_status,
    reject,
)
from backtest.factor.compute import compute_factor, compute_all
from backtest.factor.evaluation import evaluate, print_evaluation
from backtest.factor.registry import (
    get_factor_function,
    get_factor_meta,
    get_registry,
    list_factors,
    register,
)
from backtest.factor.storage import (
    FACTOR_LIBRARY_DB_PATH,
    FACTORS_WORK_DB_PATH,
    FactorLibrary,
    FactorStorage,
)
from backtest.factor.transforms import (
    abs_,
    cap_neutralize,
    cs_demean,
    cs_winsorize,
    cs_zscore,
    if_else,
    industry_neutralize,
    inverse,
    log,
    rank,
    sign,
    signed_power,
    sqrt,
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_covariance,
    ts_decay_exp,
    ts_decay_linear,
    ts_delta,
    ts_delay,
    ts_ir,
    ts_kurtosis,
    ts_max,
    ts_mean,
    ts_min,
    ts_pct_change,
    ts_product,
    ts_rank,
    ts_skewness,
    ts_std,
    ts_sum,
    z_score,
)

# Import top-level alphas package (private, gitignored) so its
# @register decorators run on package load — keeps the registry
# populated for backfill / evaluate / pipeline. If alphas/ is
# missing (e.g. fresh clone before user adds factors), skip silently.
try:
    import alphas  # noqa: F401, E402
except ImportError:
    pass

__all__ = [
    "register",
    "get_registry",
    "get_factor_meta",
    "get_factor_function",
    "list_factors",
    "admit",
    "reject",
    "AdmissionAction",
    "STATUS_ADMITTED",
    "STATUS_REJECTED",
    "RECOMMENDED_THRESHOLDS",
    "check_recommended_thresholds",
    "get_admitted_factor_ids",
    "get_pending_factor_ids",
    "get_rejected_factor_ids",
    "print_action",
    "print_status",
    "compute_factor",
    "compute_all",
    "evaluate",
    "print_evaluation",
    "FactorStorage",
    "FactorLibrary",
    "FACTORS_WORK_DB_PATH",
    "FACTOR_LIBRARY_DB_PATH",
    "rank",
    "z_score",
    "ts_rank",
    "ts_mean",
    "ts_std",
    "ts_sum",
    "ts_min",
    "ts_max",
    "ts_argmin",
    "ts_argmax",
    "ts_delta",
    "ts_delay",
    "ts_pct_change",
    "ts_product",
    "ts_skewness",
    "ts_kurtosis",
    "ts_ir",
    "ts_decay_linear",
    "ts_decay_exp",
    "ts_corr",
    "ts_covariance",
    "cs_zscore",
    "cs_demean",
    "cs_winsorize",
    "abs_",
    "sign",
    "log",
    "sqrt",
    "signed_power",
    "inverse",
    "if_else",
    "cap_neutralize",
    "industry_neutralize",
]
