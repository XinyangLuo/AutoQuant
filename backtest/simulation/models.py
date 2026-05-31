from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class Trade:
    trade_date: Date
    symbol: str
    direction: Literal["buy", "sell", "short", "cover"]
    shares: int          # 正数，实际成交股数
    price: float         # 成交价格
    amount: float        # 成交金额 = shares * price
    commission: float    # 总费用（佣金+印花税+过户费）
    reason: str          # "normal", "limit_up_traded", "limit_down_traded",
                         # "limit_up_blocked", "limit_down_blocked", "suspended",
                         # "cash_insufficient", "delisted"


@dataclass
class Position:
    symbol: str
    shares: int          # 正=多仓, 负=空仓
    avg_cost: float      # 加权平均成本
    market_value: float  # 最新市值 = shares * close_price

    @property
    def is_long(self) -> bool:
        return self.shares > 0

    @property
    def is_short(self) -> bool:
        return self.shares < 0


@dataclass
class DailySnapshot:
    date: Date
    cash: float
    positions: dict[str, Position]  # symbol -> Position
    total_value: float              # cash + sum(market_value)
    nav: float                      # total_value / initial_cash


class BacktestResult:
    """聚合回测结果，提供 DataFrame 视图和保存功能。"""

    def __init__(
        self,
        nav_df: pd.DataFrame | None = None,
        trades: list[Trade] | None = None,
        snapshots: list[DailySnapshot] | None = None,
        metrics_df: pd.DataFrame | None = None,
        initial_cash: float = 1e8,
    ):
        self.nav_df = nav_df
        self.trades = trades or []
        self.snapshots = snapshots or []
        self.metrics_df = metrics_df
        self.initial_cash = initial_cash

    @property
    def positions_df(self) -> pd.DataFrame | None:
        """Long format: date, symbol, shares, market_value, weight, avg_cost"""
        if not self.snapshots:
            return None
        rows = []
        for snap in self.snapshots:
            for sym, pos in snap.positions.items():
                rows.append({
                    "date": snap.date,
                    "symbol": sym,
                    "shares": pos.shares,
                    "market_value": pos.market_value,
                    "weight": pos.market_value / snap.total_value if snap.total_value != 0 else 0,
                    "avg_cost": pos.avg_cost,
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["date", "symbol", "shares", "market_value", "weight", "avg_cost"]
        )

    @property
    def trades_df(self) -> pd.DataFrame | None:
        """Columns: trade_date, symbol, direction, shares, price, amount, commission, reason"""
        if not self.trades:
            return None
        return pd.DataFrame([
            {
                "trade_date": t.trade_date,
                "symbol": t.symbol,
                "direction": t.direction,
                "shares": t.shares,
                "price": t.price,
                "amount": t.amount,
                "commission": t.commission,
                "reason": t.reason,
            }
            for t in self.trades
        ])

    def save(self, output_dir: str, metadata: dict | None = None) -> None:
        """保存所有回测产出文件。"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        if self.nav_df is not None and not self.nav_df.empty:
            self.nav_df.to_parquet(path / "nav.parquet")
        positions = self.positions_df
        if positions is not None and not positions.empty:
            positions.to_parquet(path / "positions.parquet")
        trades = self.trades_df
        if trades is not None and not trades.empty:
            trades.to_parquet(path / "trades.parquet")
        if self.metrics_df is not None and not self.metrics_df.empty:
            self.metrics_df.to_parquet(path / "metrics.parquet")
        if metadata:
            with open(path / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    def summary(self) -> dict:
        """Flat dict of performance metrics.

        Thin shim around ``backtest.evaluation.metrics.compute_all_metrics`` —
        the evaluation module is the single source of truth for metric formulas.
        See ``backtest/evaluation/DESIGN.md`` for the full list of indicators.
        """
        if self.nav_df is None or self.nav_df.empty or len(self.nav_df) < 2:
            return {}
        # Local import keeps simulation -> evaluation a one-way edge.
        from backtest.evaluation.loader import BacktestArtifacts
        from backtest.evaluation.metrics import compute_all_metrics

        dates = pd.to_datetime(self.nav_df["date"])
        arts = BacktestArtifacts(
            result_dir=Path("."),
            nav=self.nav_df,
            positions=self.positions_df,
            trades=self.trades_df,
            metrics=self.metrics_df,
            metadata={},
            initial_cash=self.initial_cash,
            start=dates.min(),
            end=dates.max(),
        )

        # Load default benchmarks for excess metrics.
        from backtest.evaluation.benchmark import _BENCHMARK_ALIASES, load_benchmark
        bench_navs: dict[str, pd.Series] = {}
        start_str = arts.start.strftime("%Y%m%d")
        end_str = arts.end.strftime("%Y%m%d")
        for alias, code in _BENCHMARK_ALIASES.items():
            try:
                bench_navs[alias] = load_benchmark(code, start=start_str, end=end_str)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Failed to load benchmark {code} ({alias}): {exc}. "
                    f"Run `python -m backtest.data.backfill_indices --symbols {code}` to backfill.",
                    stacklevel=2,
                )

        return compute_all_metrics(arts, bench_nav=bench_navs.get("hs300"), bench_navs=bench_navs)


@dataclass
class DecileBacktestResult:
    """Result of a decile-layered backtest."""

    nav_df: pd.DataFrame
    decile_metrics: dict[int, dict]
    ls_metrics: dict
    monotonicity_score: float

    def save(self, output_dir: str, metadata: dict | None = None) -> None:
        """Persist nav and metrics to disk."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        if not self.nav_df.empty:
            self.nav_df.to_parquet(path / "nav.parquet", index=False)
        payload = {
            "decile_metrics": {str(k): v for k, v in self.decile_metrics.items()},
            "ls_metrics": self.ls_metrics,
            "monotonicity_score": self.monotonicity_score,
        }
        with open(path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        if metadata:
            with open(path / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
