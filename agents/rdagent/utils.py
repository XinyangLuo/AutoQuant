"""Shared helpers for the rdagent module."""

from __future__ import annotations

import shutil
from pathlib import Path


def cleanup_generated_factor(factor_id: str, generated_dir: Path | None = None) -> None:
    """Remove a generated factor code file and its result directory."""
    if generated_dir is None:
        generated_dir = Path(__file__).parent / "generated"
    file_path = generated_dir / f"{factor_id}.py"
    if file_path.exists():
        file_path.unlink()


def cleanup_run_results(factor_id: str, results_root: Path | str = "results/agent") -> None:
    """Remove all backtest results for a factor."""
    result_dir = Path(results_root) / factor_id
    if result_dir.exists():
        shutil.rmtree(result_dir)
