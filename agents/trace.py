"""Trace management for factor iteration rounds.

Provides TraceRecord (aligned with the Trace JSONL Schema in
.claude/prompts/shared/output_formats.md) and TraceManager for
atomic read/write of per-run trace.jsonl files.

Usage::

    tm = TraceManager("results/run_001")
    record = TraceRecord.from_result_json(result_dict, round_num=1)
    tm.append(record)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _unique_tmp(path: Path) -> Path:
    """Generate a unique temporary file path to avoid cross-process collisions."""
    return path.parent / f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"


@dataclass
class TraceRecord:
    """Single-round trace record aligned with the Trace JSONL Schema."""

    round: int
    factor_id: str
    category: str
    data_sources: list[str]
    status: str
    failure_type: str | None
    error_signature: str | None
    diagnosis: str
    fix_strategy: str
    fix_level: str
    factor_change: str | None
    factor_params: dict[str, Any]
    strategy_params: dict[str, Any]
    code_summary: str
    tried_params: dict[str, Any]
    recommend_abandon: bool
    metrics: dict[str, Any]
    same_direction: bool
    new_hypothesis: str | None
    parent_round_id: int | None = None
    branch_id: str = "main"
    fork_reason: str | None = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "factor_id": self.factor_id,
            "category": self.category,
            "data_sources": self.data_sources,
            "status": self.status,
            "failure_type": self.failure_type,
            "error_signature": self.error_signature,
            "diagnosis": self.diagnosis,
            "fix_strategy": self.fix_strategy,
            "fix_level": self.fix_level,
            "factor_change": self.factor_change,
            "factor_params": self.factor_params,
            "strategy_params": self.strategy_params,
            "code_summary": self.code_summary,
            "tried_params": self.tried_params,
            "recommend_abandon": self.recommend_abandon,
            "metrics": self.metrics,
            "same_direction": self.same_direction,
            "new_hypothesis": self.new_hypothesis,
            "parent_round_id": self.parent_round_id,
            "branch_id": self.branch_id,
            "fork_reason": self.fork_reason,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceRecord:
        return cls(
            round=data["round"],
            factor_id=data["factor_id"],
            category=data.get("category", ""),
            data_sources=data.get("data_sources", []),
            status=data["status"],
            failure_type=data.get("failure_type"),
            error_signature=data.get("error_signature"),
            diagnosis=data.get("diagnosis", ""),
            fix_strategy=data.get("fix_strategy", ""),
            fix_level=data.get("fix_level", ""),
            factor_change=data.get("factor_change"),
            factor_params=data.get("factor_params", {}),
            strategy_params=data.get("strategy_params", {}),
            code_summary=data.get("code_summary", ""),
            tried_params=data.get("tried_params", {}),
            recommend_abandon=data.get("recommend_abandon", False),
            metrics=data.get("metrics", {}),
            same_direction=data.get("same_direction", True),
            new_hypothesis=data.get("new_hypothesis"),
            parent_round_id=data.get("parent_round_id"),
            branch_id=data.get("branch_id", "main"),
            fork_reason=data.get("fork_reason"),
            ts=data.get("ts", datetime.now(timezone.utc).isoformat()),
        )

    @staticmethod
    def from_result_json(
        result: dict[str, Any],
        *,
        round_num: int = 1,
        parent_round_id: int | None = None,
        branch_id: str = "main",
        rc_output: dict[str, Any] | None = None,
        code_summary: str = "",
        tried_params: dict[str, Any] | None = None,
        category: str = "",
        data_sources: list[str] | None = None,
    ) -> TraceRecord:
        """Build a TraceRecord from a result.json dict and optional RC output.

        Metrics mapping (deep-path extraction with safe fallback to None):
        - annual_icir       ← result["metrics"]["annual_icir"]
        - simple_sharpe     ← result["metrics"]["simple_sharpe"]
        - r2                ← result["experiment"]["step_results"]["step8"]["metrics"]["r2"]
        - max_existing_corr ← result["metrics"]["max_corr"]
        - residual_icir     ← result["metrics"]["residual_annual_icir"]
        """
        metrics_flat = result.get("metrics", {})
        experiment = result.get("experiment", {})
        step_results = experiment.get("step_results", {})
        step8 = step_results.get("step8", {})
        step8_metrics = step8.get("metrics", {}) if isinstance(step8, dict) else {}

        metrics = {
            "annual_icir": metrics_flat.get("annual_icir"),
            "simple_sharpe": metrics_flat.get("simple_sharpe"),
            "r2": step8_metrics.get("r2") if isinstance(step8_metrics, dict) else None,
            "max_existing_corr": metrics_flat.get("max_corr"),
            "residual_icir": metrics_flat.get("residual_annual_icir"),
        }

        error = result.get("error")
        error_signature = None
        if error:
            error_signature = error[:120] if len(error) > 120 else error

        # Prefer experiment-level category / data_sources when available
        cat = category or experiment.get("category", "")
        ds = data_sources if data_sources is not None else experiment.get("data_sources", [])

        # RC output fields (with safe defaults)
        rc = rc_output or {}
        return TraceRecord(
            round=round_num,
            factor_id=result.get("factor_id", ""),
            category=cat,
            data_sources=ds if isinstance(ds, list) else [],
            status=result.get("status", "fail"),
            failure_type=result.get("failure_type"),
            error_signature=error_signature,
            diagnosis=rc.get("diagnosis", ""),
            fix_strategy=rc.get("fix_strategy", ""),
            fix_level=rc.get("fix_level", ""),
            factor_change=rc.get("factor_change"),
            factor_params=rc.get("factor_params", {}),
            strategy_params=rc.get("strategy_params", {}),
            code_summary=code_summary,
            tried_params=tried_params if tried_params is not None else rc.get("factor_params", {}),
            recommend_abandon=rc.get("recommend_abandon", False),
            metrics=metrics,
            same_direction=rc.get("same_direction", True),
            new_hypothesis=rc.get("new_hypothesis"),
            parent_round_id=parent_round_id,
            branch_id=branch_id,
            fork_reason=None,
        )


class TraceManager:
    """Manage trace.jsonl in a single run directory."""

    TRACE_FILENAME = "trace.jsonl"

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.trace_path = self.run_dir / self.TRACE_FILENAME

    def read_all(self) -> list[dict[str, Any]]:
        """Read all trace records from disk."""
        if not self.trace_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(self.trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def get_max_round(self) -> int:
        """Return the maximum round number seen so far (0 if empty)."""
        records = self.read_all()
        if not records:
            return 0
        return max(r.get("round", 0) for r in records)

    def get_next_round(self) -> int:
        """Return the next round number (max + 1)."""
        return self.get_max_round() + 1

    def get_default_parent_round(self) -> int | None:
        """Default parent for linear iteration: max round so far.

        Returns ``None`` when no previous rounds exist.
        """
        max_r = self.get_max_round()
        return max_r if max_r > 0 else None

    def append(self, record: TraceRecord | dict[str, Any]) -> None:
        """Atomically append one record to trace.jsonl.

        Uses a process-unique temp file and atomic rename so concurrent
        writers (or readers) never see partially-written data.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = record.to_dict() if isinstance(record, TraceRecord) else dict(record)
        line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
        tmp = _unique_tmp(self.trace_path)
        # Copy existing content if file already exists, then append new line
        if self.trace_path.exists():
            with open(self.trace_path, "r", encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
                dst.write(src.read())
                dst.write(line)
        else:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(line)
        os.replace(str(tmp), str(self.trace_path))
