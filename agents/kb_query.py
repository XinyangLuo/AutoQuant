"""
Knowledge Base 分层查询脚本。

负责 L1/L2 过滤和 L3 keyword 查询，供父进程 Bash 调用。
不直接操作 LLM context，只返回精简的 JSON 摘要。

Usage:
    python -m agents.kb_query --category volume_reversal --failure-type icir_fail --limit 3
    python -m agents.kb_query --check-duplicate --formula-fingerprint "rank(ts_std(turnover_rate, 20))" --category volume_reversal
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KB_DIR = Path(__file__).resolve().parent / "knowledge_base"


def _load_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s: %s", path, line[:200])
    return records


def query_anti_patterns(failure_type: str | None, category: str | None, limit: int = 3) -> list[dict]:
    """L2: 查询反模式，按 category 匹配 + count 排序截断。"""
    data = _load_json(KB_DIR / "anti_patterns.json")
    matches = []
    for ft, patterns in data.items():
        if failure_type and ft != failure_type:
            continue
        for p in patterns:
            if category and p.get("category") != category:
                continue
            matches.append({**p, "failure_type": ft})
    matches.sort(key=lambda x: x.get("count", 0), reverse=True)
    return matches[:limit]


def query_successful_patterns(category: str | None, limit: int = 3) -> list[dict]:
    """L2: 查询成功模式，按 ICIR 排序截断。"""
    data = _load_json(KB_DIR / "successful_patterns.json")
    matches = []
    for cat, patterns in data.items():
        if category and cat != category:
            continue
        for p in patterns:
            icir = p.get("key_metrics", {}).get("annual_icir") or 0.0
            matches.append({**p, "_category": cat, "_icir": icir})
    matches.sort(key=lambda x: x["_icir"], reverse=True)
    return [
        {
            "factor_id": m["factor_id"],
            "category": m["_category"],
            "formula_pattern": m.get("formula_pattern", ""),
            "key_metrics": m.get("key_metrics", {}),
            "why_it_works": m.get("why_it_works", ""),
        }
        for m in matches[:limit]
    ]


def query_failed_attempts(category: str | None, failure_type: str | None, limit: int = 5) -> list[dict]:
    """L3: 查询失败记录，keyword 过滤后截断。"""
    records = _load_jsonl(KB_DIR / "failed_attempts.jsonl")
    matches = []
    for r in records:
        if category and r.get("category") != category:
            continue
        if failure_type and r.get("failure_type") != failure_type:
            continue
        matches.append(r)
    matches.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return matches[:limit]


def query_hypothesis_index(category: str | None, formula_fingerprint: str | None, limit: int = 5) -> list[dict]:
    """L3: 查重查询，基于 keyword / category 过滤。"""
    records = _load_jsonl(KB_DIR / "hypothesis_index.jsonl")
    matches = []
    for r in records:
        if category and r.get("category") != category:
            continue
        if formula_fingerprint:
            fp = r.get("formula_fingerprint", "")
            # 简单 keyword 匹配：如果 fingerprint 包含相同的 transforms 名称
            tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", formula_fingerprint)
            match_count = sum(1 for t in tokens if t in fp)
            if match_count < max(1, len(tokens) // 2):
                continue
        matches.append(r)
    matches.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return matches[:limit]


def build_kb_summary(category: str | None, failure_type: str | None, limit: int = 3) -> dict[str, Any]:
    """组装 L1+L2 摘要，直接注入 prompt。"""
    anti_patterns = query_anti_patterns(failure_type, category, limit)
    successful_patterns = query_successful_patterns(category, limit)
    return {
        "anti_patterns": anti_patterns,
        "successful_patterns": successful_patterns,
        "sota": successful_patterns[0] if successful_patterns else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KB 分层查询")
    parser.add_argument("--category", type=str, default=None, help="因子 category")
    parser.add_argument("--failure-type", type=str, default=None, help="失败类型")
    parser.add_argument("--limit", type=int, default=3, help="返回条数上限")
    parser.add_argument("--check-duplicate", action="store_true", help="查重模式（查 hypothesis_index）")
    parser.add_argument("--formula-fingerprint", type=str, default=None, help="公式 fingerprint（查重模式用）")
    args = parser.parse_args()

    if args.check_duplicate:
        result = query_hypothesis_index(args.category, args.formula_fingerprint, args.limit)
        print(json.dumps({"duplicate_candidates": result}, ensure_ascii=False, indent=2))
    else:
        summary = build_kb_summary(args.category, args.failure_type, args.limit)
        # L3 失败记录只在需要时追加
        if args.failure_type:
            summary["recent_failures"] = query_failed_attempts(args.category, args.failure_type, args.limit)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
