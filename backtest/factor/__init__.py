"""Factor module: definition, computation, storage, and offline evaluation.

Public API
----------
register(factor_id, *, name, category, data_sources, ...)
    Decorator to register a factor compute function.

compute_factor(factor_id, start_date, end_date, ...)
    Compute a single factor for a date range.

evaluate(factor_id, start, end, *, horizons, ret_type)
    Offline evaluation: IC / RankIC / ICIR / turnover / decay / group returns,
    plus cross-sectional rank correlation against existing factors.

FactorStorage
    DuckDB wrapper for factors.duckdb (read / write / query).

rank, z_score
    Common operators for use inside factor compute functions.
"""

from backtest.factor.admission import (
    admit,
    get_admitted_factor_ids,
    get_pending_factor_ids,
    get_rejected_factor_ids,
    print_admission,
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
from backtest.factor.storage import FactorStorage
from backtest.factor.transforms import rank, z_score

__all__ = [
    "register",
    "get_registry",
    "get_factor_meta",
    "get_factor_function",
    "list_factors",
    "admit",
    "get_admitted_factor_ids",
    "get_pending_factor_ids",
    "get_rejected_factor_ids",
    "print_admission",
    "compute_factor",
    "compute_all",
    "evaluate",
    "print_evaluation",
    "FactorStorage",
    "rank",
    "z_score",
]
