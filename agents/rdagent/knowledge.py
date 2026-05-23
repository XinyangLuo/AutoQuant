"""A-share knowledge base — accumulates experience and supports retrieval."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core.knowledge_base import KnowledgeBase
from .core.proposal import Hypothesis
from .evaluator import QuantFeedback
from .experiment import AutoQuantFactorExperiment


@dataclass
class _ExperienceRecord:
    """Internal record for a single experiment outcome."""

    factor_id: str
    category: str
    keywords: list[str]
    hypothesis_text: str
    decision: bool
    rankicir: float
    ic_positive_ratio: float
    turnover: float
    max_corr: float
    simple_sharpe: float | None
    observation: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_ExperienceRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class AShareKnowledgeBase(KnowledgeBase):
    """Knowledge base for A-share quantitative research.

    Accumulates successful/failed factor patterns and provides:
    - Similar-case retrieval (by category + keywords)
    - SOTA (state-of-the-art) tracking
    - Initial domain knowledge (hard-coded)
    """

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = Path(__file__).parent / "kb.json"
        self.db_path = Path(db_path)
        self._records: list[_ExperienceRecord] = []
        self._sota: dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    # KnowledgeBase interface
    # ------------------------------------------------------------------

    def add_experience(
        self,
        experiment: AutoQuantFactorExperiment,
        feedback: QuantFeedback,
    ) -> None:
        """Record the outcome of one experiment."""
        record = _ExperienceRecord(
            factor_id=experiment.factor_id,
            category=experiment.category or "unknown",
            keywords=experiment.keywords or [],
            hypothesis_text=experiment.factor_code[:200] if experiment.factor_code else "",
            decision=feedback.decision,
            rankicir=feedback.rankicir,
            ic_positive_ratio=feedback.ic_positive_ratio,
            turnover=feedback.turnover,
            max_corr=feedback.max_corr,
            simple_sharpe=feedback.simple_sharpe,
            observation=feedback.observation,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._records.append(record)

        # Update SOTA
        self._update_sota(feedback)
        self.save()

    def retrieve_similar(
        self,
        hypothesis: Hypothesis,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve past experiences similar to the given hypothesis.

        Simple matching: category match + keyword overlap.
        Future: embedding-based retrieval.
        """
        scored: list[tuple[float, _ExperienceRecord]] = []
        target_kw = set(hypothesis.keywords)
        target_cat = hypothesis.category.lower()

        for rec in self._records:
            score = 0.0
            # Category match
            if rec.category.lower() == target_cat:
                score += 2.0
            # Keyword overlap
            rec_kw = set(rec.keywords)
            if target_kw and rec_kw:
                overlap = len(target_kw & rec_kw) / max(len(target_kw), len(rec_kw))
                score += overlap * 3.0
            # Decision bonus (learn from both successes and failures)
            if rec.decision:
                score += 0.5

            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec.to_dict() for _, rec in scored[:top_k]]

    def get_sota(self) -> dict[str, Any]:
        """Return current best-known performance."""
        return self._sota.copy()

    def save(self) -> None:
        """Persist to disk as JSON (atomic write)."""
        data = {
            "records": [r.to_dict() for r in self._records],
            "sota": self._sota,
            "initial_knowledge": self._initial_knowledge(),
        }
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.db_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(str(tmp_path), str(self.db_path))

    def load(self) -> None:
        """Load from disk."""
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.db_path.exists():
            self._records = []
            self._sota = {}
            return
        try:
            with self.db_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # Backup corrupt file and start fresh
            backup = self.db_path.with_suffix(
                f".corrupt.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
            )
            self.db_path.rename(backup)
            self._records = []
            self._sota = {}
            return
        self._records = [
            _ExperienceRecord.from_dict(r) for r in data.get("records", [])
        ]
        self._sota = data.get("sota", {})

    def _update_sota(self, feedback: QuantFeedback) -> None:
        """Update state-of-the-art if this feedback is better."""
        if not self._sota:
            self._sota = {
                "best_rankicir": float("-inf"),
                "best_simple_sharpe": float("-inf"),
                "best_factor_id": None,
            }

        if feedback.rankicir > self._sota.get("best_rankicir", float("-inf")):
            self._sota["best_rankicir"] = feedback.rankicir
            self._sota["best_factor_id"] = feedback.metrics.get("factor_id")

        if feedback.simple_sharpe is not None and feedback.simple_sharpe > self._sota.get(
            "best_simple_sharpe", float("-inf")
        ):
            self._sota["best_simple_sharpe"] = feedback.simple_sharpe

    def _initial_knowledge(self) -> dict[str, Any]:
        """Hard-coded domain knowledge for A-share quant research."""
        return {
            "trading_rules": {
                "t_plus_1": "A-shares settle T+1; cannot sell on the same day as purchase",
                "price_limits": "±10% normal, ±20% STAR Market (688/300/301)",
                "st_exclusion": "ST/*ST stocks must be excluded from universe",
                "ipo_cooldown": "Exclude IPO within 60 trading days",
            },
            "factor_categories": {
                "reversal": "Short-term mean reversion; works in high-vol regimes",
                "momentum": "Medium-term trend following; weaker in A-shares than US",
                "value": "P/E, P/B, EV/EBITDA; cyclical performance",
                "quality": "ROE stability, earnings quality; defensive",
                "growth": "Revenue/earnings growth; momentum-like",
                "liquidity": "Turnover, Amihud; often orthogonal to others",
                "volatility": "Realized vol; typically negatively priced",
            },
            "common_pitfalls": {
                "future_data": "Never use data not available at signal time (PIT isolation)",
                "survivorship_bias": "Exclude delisted stocks from historical analysis",
                "look_ahead": "Financial statements have announcement lag; use f_ann_date",
                "sector_neutrality": "Raw factors often reflect industry exposure; neutralize",
            },
            "operator_guidelines": {
                "rank": "Cross-sectional rank; robust but loses magnitude info",
                "z_score": "Standardizes across stocks; sensitive to outliers",
                "ts_mean": "Smoothing; longer windows = lower turnover",
                "ts_std": "Volatility measure; often inversely related to returns",
                "ts_corr": "Correlation with another series; useful for pairs",
            },
            "evaluation_thresholds": {
                "rankicir": "≥ 0.25 for candidate",
                "ic_positive_ratio": "≥ 52% for candidate",
                "turnover": "< 0.5 for candidate",
                "max_corr": "< 0.85 vs existing factors",
            },
        }
