"""Knowledge Base automatic updater.

Provides KbUpdater for automated KB accumulation after Pass / Fail.
All writes are atomic (tmp → replace) and idempotent-friendly.

Usage::

    updater = KbUpdater()
    updater.update_on_pass(experiment)
    updater.update_on_fail(experiment, rc_output={...})
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .experiment import AutoQuantFactorExperiment

KB_DIR = Path(__file__).resolve().parent / "knowledge_base"


def _unique_tmp(path: Path) -> Path:
    """Generate a unique temporary file path to avoid cross-process collisions."""
    return path.parent / f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"


@dataclass
class KbUpdateSummary:
    """Summary of changes made by a KbUpdater call."""

    hypothesis_index_updated: bool = False
    successful_patterns_updated: bool = False
    anti_pattern_updated: bool = False
    failed_attempts_appended: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "hypothesis_index_updated": self.hypothesis_index_updated,
            "successful_patterns_updated": self.successful_patterns_updated,
            "anti_pattern_updated": self.anti_pattern_updated,
            "failed_attempts_appended": self.failed_attempts_appended,
        }


class KbUpdater:
    """Automated KB updater. No LLM logic — only structured data R/W."""

    def __init__(self, kb_dir: Path | str | None = None) -> None:
        self.kb_dir = Path(kb_dir) if kb_dir else KB_DIR
        self.kb_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entrypoints
    # ------------------------------------------------------------------

    def update_on_pass(
        self,
        experiment: AutoQuantFactorExperiment,
        trace_record: dict[str, Any] | None = None,
    ) -> KbUpdateSummary:
        """Called when a factor passes the pipeline.

        Updates successful_patterns and hypothesis_index.
        """
        summary = KbUpdateSummary()
        summary.successful_patterns_updated = self._update_successful_patterns(
            experiment, trace_record
        )
        summary.hypothesis_index_updated = self._update_hypothesis_index(experiment)
        return summary

    def update_on_fail(
        self,
        experiment: AutoQuantFactorExperiment,
        trace_record: dict[str, Any] | None = None,
        rc_output: dict[str, Any] | None = None,
    ) -> KbUpdateSummary:
        """Called when a factor fails the pipeline.

        Conditionally updates anti_patterns, appends failed_attempts,
        and updates hypothesis_index.
        """
        summary = KbUpdateSummary()
        if rc_output:
            summary.anti_pattern_updated = self._update_anti_patterns(rc_output)
        summary.hypothesis_index_updated = self._update_hypothesis_index(experiment)
        self._append_failed_attempts(experiment, trace_record, status="fail")
        summary.failed_attempts_appended = True
        return summary

    # ------------------------------------------------------------------
    # hypothesis_index.jsonl
    # ------------------------------------------------------------------

    def _update_hypothesis_index(
        self, experiment: AutoQuantFactorExperiment
    ) -> bool:
        """Upsert hypothesis_index.jsonl by factor_id.

        If factor_id exists → update status / best_icir / best_sharpe / ts.
        best_icir = max(existing, current annual_icir).
        best_sharpe = max(existing, current simple_sharpe).
        If not exists → append new record.
        """
        path = self.kb_dir / "hypothesis_index.jsonl"
        records = self._load_jsonl(path)

        # Extract current metrics
        eval_result = experiment.eval_result or {}
        simple_bt = experiment.simple_bt_metrics or {}
        current_icir = eval_result.get("annual_icir")
        if current_icir is None and experiment.step_results:
            sr3 = experiment.step_results.get("step3", {})
            current_icir = (sr3.get("metrics") or {}).get("annual_icir")
        current_sharpe = simple_bt.get("sharpe")

        fingerprint = self._extract_formula_fingerprint(experiment)
        category = experiment.category or ""
        data_sources = self._infer_data_sources(experiment)
        status = experiment.status or "pending"
        ts = datetime.now(timezone.utc).isoformat()

        existing_idx = None
        for i, r in enumerate(records):
            if r.get("factor_id") == experiment.factor_id:
                existing_idx = i
                break

        if existing_idx is not None:
            old = records[existing_idx]
            old_icir = old.get("best_icir")
            old_sharpe = old.get("best_sharpe")
            new_icir = (
                max(old_icir, current_icir)
                if old_icir is not None and current_icir is not None
                else (old_icir if old_icir is not None else current_icir)
            )
            new_sharpe = (
                max(old_sharpe, current_sharpe)
                if old_sharpe is not None and current_sharpe is not None
                else (old_sharpe if old_sharpe is not None else current_sharpe)
            )
            old.update(
                {
                    "status": status,
                    "best_icir": new_icir,
                    "best_sharpe": new_sharpe,
                    "ts": ts,
                }
            )
            if category:
                old["category"] = category
            if fingerprint:
                old["formula_fingerprint"] = fingerprint
        else:
            records.append(
                {
                    "factor_id": experiment.factor_id,
                    "category": category,
                    "formula_fingerprint": fingerprint or experiment.factor_id,
                    "data_sources": data_sources,
                    "status": status,
                    "best_icir": current_icir,
                    "best_sharpe": current_sharpe,
                    "ts": ts,
                }
            )

        self._save_jsonl(path, records)
        return True

    # ------------------------------------------------------------------
    # successful_patterns.json
    # ------------------------------------------------------------------

    def _update_successful_patterns(
        self,
        experiment: AutoQuantFactorExperiment,
        trace_record: dict[str, Any] | None = None,
    ) -> bool:
        """Append or update a successful pattern by factor_id (dedup key).

        Returns True if file was modified.
        """
        path = self.kb_dir / "successful_patterns.json"
        data: dict[str, list[dict[str, Any]]] = self._load_json(path)

        category = experiment.category or "general"
        if category not in data:
            data[category] = []

        # Check existing by factor_id
        existing_idx = None
        for i, p in enumerate(data[category]):
            if p.get("factor_id") == experiment.factor_id:
                existing_idx = i
                break

        # Build metrics
        eval_result = experiment.eval_result or {}
        simple_bt = experiment.simple_bt_metrics or {}
        annual_icir = eval_result.get("annual_icir")
        if annual_icir is None and experiment.step_results:
            annual_icir = (experiment.step_results.get("step3", {}).get("metrics") or {}).get(
                "annual_icir"
            )
        key_metrics: dict[str, Any] = {
            "annual_icir": annual_icir,
            "simple_sharpe": simple_bt.get("sharpe"),
        }
        # Strip None values
        key_metrics = {k: v for k, v in key_metrics.items() if v is not None}

        entry = {
            "factor_id": experiment.factor_id,
            "formula_pattern": self._extract_formula_fingerprint(experiment)
            or experiment.factor_id,
            "key_metrics": key_metrics,
            "why_it_works": "",
            "admission_date": str(date.today()),
        }

        if existing_idx is not None:
            data[category][existing_idx].update(entry)
        else:
            data[category].append(entry)

        self._save_json(path, data)
        return True

    # ------------------------------------------------------------------
    # anti_patterns.json
    # ------------------------------------------------------------------

    def _update_anti_patterns(self, rc_output: dict[str, Any]) -> bool:
        """Deduplicate-update anti_patterns.json by signature.

        Returns True if file was modified.
        """
        new_ap = rc_output.get("new_anti_pattern")
        if not new_ap:
            return False

        failure_type = rc_output.get("failure_type", "metrics_fail")
        pattern = new_ap.get("pattern", "")
        category = new_ap.get("category", "")
        signature = new_ap.get("signature", "")
        fix = new_ap.get("fix", "")

        if not signature:
            return False

        path = self.kb_dir / "anti_patterns.json"
        data: dict[str, list[dict[str, Any]]] = self._load_json(path)

        if failure_type not in data:
            data[failure_type] = []

        # Exact signature match
        matched_idx = None
        for i, ap in enumerate(data[failure_type]):
            if ap.get("signature") == signature:
                matched_idx = i
                break

        today = str(date.today())
        if matched_idx is not None:
            data[failure_type][matched_idx]["count"] = (
                data[failure_type][matched_idx].get("count", 1) + 1
            )
            data[failure_type][matched_idx]["last_seen"] = today
        else:
            data[failure_type].append(
                {
                    "pattern": pattern,
                    "category": category,
                    "signature": signature,
                    "fix": fix,
                    "count": 1,
                    "last_seen": today,
                }
            )

        self._save_json(path, data)
        return True

    # ------------------------------------------------------------------
    # failed_attempts.jsonl
    # ------------------------------------------------------------------

    def _append_failed_attempts(
        self,
        experiment: AutoQuantFactorExperiment,
        trace_record: dict[str, Any] | None = None,
        *,
        status: str,
    ) -> bool:
        """Append one line to failed_attempts.jsonl."""
        path = self.kb_dir / "failed_attempts.jsonl"

        # Extract best metrics
        eval_result = experiment.eval_result or {}
        simple_bt = experiment.simple_bt_metrics or {}
        annual_icir = eval_result.get("annual_icir")
        if annual_icir is None and experiment.step_results:
            annual_icir = (experiment.step_results.get("step3", {}).get("metrics") or {}).get(
                "annual_icir"
            )
        simple_sharpe = simple_bt.get("sharpe")

        code_summary = ""
        why_failed = ""
        failure_type = None
        if trace_record:
            code_summary = trace_record.get("code_summary", "")
            why_failed = trace_record.get("diagnosis", "")
            failure_type = trace_record.get("failure_type")
        if not code_summary and experiment.factor_code:
            code_summary = self._extract_formula_fingerprint(experiment) or ""
        if not why_failed and experiment.error:
            why_failed = experiment.error[:200]

        entry = {
            "factor_id": experiment.factor_id,
            "run_id": experiment.factor_id,
            "category": experiment.category or "",
            "data_sources": self._infer_data_sources(experiment),
            "status": status,
            "best_icir": annual_icir,
            "best_sharpe": simple_sharpe,
            "failure_type": failure_type,
            "code_summary": code_summary,
            "why_failed": why_failed,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        self._append_jsonl(path, entry)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path) -> Any:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        tmp = _unique_tmp(path)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(str(tmp), str(path))

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    @staticmethod
    def _save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        tmp = _unique_tmp(path)
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        os.replace(str(tmp), str(path))

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        tmp = _unique_tmp(path)
        # If file exists, copy existing content first
        if path.exists():
            with open(path, "r", encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
                dst.write(src.read())
                dst.write(line)
        else:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(line)
        os.replace(str(tmp), str(path))

    @staticmethod
    def _extract_formula_fingerprint(experiment: AutoQuantFactorExperiment) -> str:
        """Extract a short formula description from factor code.

        Tries: first meaningful line after @register, or first docstring line,
        or first non-empty non-import line. Falls back to empty string.
        """
        code = experiment.factor_code or ""
        if not code:
            return ""

        # Try to find a descriptive comment right after @register
        m = re.search(r"@register\([^)]*\)\s*(?:\n\s*['\"]{3}(.{3,200})['\"]{3})?", code)
        if m and m.group(1):
            return m.group(1).strip().replace("\n", " ")[:120]

        # Try first docstring
        m = re.search(r'"""(.{3,200})"""', code, re.DOTALL)
        if m:
            return m.group(1).strip().replace("\n", " ")[:120]

        # Try first non-trivial line of the function body
        lines = code.splitlines()
        in_func = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def "):
                in_func = True
                continue
            if in_func and stripped and not stripped.startswith("#"):
                # Skip decorators and common boilerplate
                if any(
                    stripped.startswith(p)
                    for p in ("@", "from ", "import ", "panel =", "return")
                ):
                    continue
                return stripped[:120]

        return ""

    @staticmethod
    def _infer_data_sources(experiment: AutoQuantFactorExperiment) -> list[str]:
        """Infer data sources from factor code heuristics."""
        code = experiment.factor_code or ""
        sources: list[str] = []
        if "market_storage" in code or "panel[" in code:
            # Default to market_daily unless financial columns are detected
            sources.append("market_daily")
        fina_cols = ["inc_", "bs_", "cf_", "income_q", "balancesheet_q", "cashflow_q"]
        if any(col in code for col in fina_cols):
            sources.extend(["income_q", "balancesheet_q", "cashflow_q"])
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result if result else ["market_daily"]
