"""Cleanup pipeline artifacts on rejection."""

from __future__ import annotations

import shutil
from pathlib import Path

from backtest.factor.admission import reject
from backtest.factor.storage import FactorStorage

from .state import PipelineState


def cleanup_on_rejection(state: PipelineState) -> None:
    """Delete all artifacts when a factor is rejected.

    1. Delete results/<factor_id>/ directory tree.
    2. Drop the factor column from the work DB.
    3. Mark the factor as rejected in the registry.
    """
    factor_id = state.factor_id
    results_dir = Path(state.config.results_root) / factor_id

    # 1. Delete results directory
    if results_dir.exists():
        shutil.rmtree(results_dir)
        print(f"  Cleaned up results: {results_dir}")

    # 2. Drop from work DB
    try:
        with FactorStorage() as fs:
            fs.delete_factor(factor_id)
            print(f"  Dropped from work DB: {factor_id}")
    except Exception as exc:
        print(f"  Warning: could not drop from work DB: {exc}")

    # 3. Mark rejected in registry
    try:
        reject(factor_id)
        print(f"  Marked rejected in registry: {factor_id}")
    except Exception as exc:
        print(f"  Warning: could not mark rejected: {exc}")
