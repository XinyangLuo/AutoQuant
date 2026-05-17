"""Load BacktestResult parquet artifacts from disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


_DEFAULT_INITIAL_CASH = 1e8


@dataclass(frozen=True)
class BacktestArtifacts:
    """In-memory view of a saved BacktestResult directory.

    Only ``nav`` is required.  Detailed-mode results additionally populate
    ``positions`` / ``trades`` / ``metrics``; simple-mode results leave them
    as ``None``.
    """

    result_dir: Path
    nav: pd.DataFrame                       # date, nav, daily_return [, total_value, cash, position_value]
    positions: pd.DataFrame | None          # date, symbol, shares, market_value, weight, avg_cost
    trades: pd.DataFrame | None             # trade_date, symbol, direction, shares, price, amount, commission, reason
    metrics: pd.DataFrame | None            # daily portfolio statistics (turnover, exposure, …)
    metadata: dict
    initial_cash: float
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def n_days(self) -> int:
        return len(self.nav)

    @property
    def strategy_id(self) -> str:
        strat = self.metadata.get("strategy") or {}
        name = strat.get("name") if isinstance(strat, dict) else None
        return name or self.result_dir.name


def _read_parquet(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        return None
    return df if not df.empty else None


def _normalise_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Cast a date column to pandas datetime, without copying the whole frame."""
    if df is None or col not in df.columns:
        return df
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return df
    return df.assign(**{col: pd.to_datetime(df[col])})


def _resolve_initial_cash(metadata: dict, nav: pd.DataFrame) -> float:
    sim = metadata.get("simulation") or {}
    if isinstance(sim, dict) and sim.get("initial_cash"):
        return float(sim["initial_cash"])
    if metadata.get("initial_cash"):
        return float(metadata["initial_cash"])
    if "total_value" in nav.columns and "nav" in nav.columns:
        first_nav = float(nav["nav"].iloc[0])
        first_total = float(nav["total_value"].iloc[0])
        if first_nav > 0:
            return first_total / first_nav
    return _DEFAULT_INITIAL_CASH


def load_result(result_dir: str | Path) -> BacktestArtifacts:
    """Read the parquet files emitted by ``BacktestResult.save(...)``.

    Required: ``nav.parquet``.  Optional: ``positions.parquet``,
    ``trades.parquet``, ``metrics.parquet``, ``metadata.json``.

    Raises FileNotFoundError if the directory or ``nav.parquet`` is missing.
    """
    path = Path(result_dir)
    if not path.exists():
        raise FileNotFoundError(f"Result directory not found: {path}")

    nav_path = path / "nav.parquet"
    nav = _read_parquet(nav_path)
    if nav is None:
        raise FileNotFoundError(f"nav.parquet missing or empty: {nav_path}")

    nav = _normalise_date_col(nav, "date").sort_values("date").reset_index(drop=True)

    positions = _normalise_date_col(_read_parquet(path / "positions.parquet"), "date")
    trades = _normalise_date_col(_read_parquet(path / "trades.parquet"), "trade_date")
    metrics = _normalise_date_col(_read_parquet(path / "metrics.parquet"), "date")

    meta_path = path / "metadata.json"
    metadata: dict = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metadata = {}

    initial_cash = _resolve_initial_cash(metadata, nav)

    return BacktestArtifacts(
        result_dir=path,
        nav=nav,
        positions=positions,
        trades=trades,
        metrics=metrics,
        metadata=metadata,
        initial_cash=initial_cash,
        start=pd.Timestamp(nav["date"].iloc[0]),
        end=pd.Timestamp(nav["date"].iloc[-1]),
    )
