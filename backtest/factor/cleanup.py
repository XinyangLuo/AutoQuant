#!/usr/bin/env python3
"""Clear temporary factor values from the **work** DB.

Use this when you want to drop a factor's research artefacts without
recording an admission decision. ``admit()`` and ``reject()`` both clear
work as a side effect, so reach for ``cleanup`` only when you're shelving
the factor for a re-attempt (registry status stays ``pending``).

Usage:
    python -m backtest.factor.cleanup f_001       # drop one factor
    python -m backtest.factor.cleanup --all       # wipe the entire work DB
    python -m backtest.factor.cleanup --orphans   # drop work rows that are
                                                  # already admitted in the library
"""

from __future__ import annotations

import argparse

from backtest.factor.admission import get_admitted_factor_ids
from backtest.factor.storage import FactorStorage


def cleanup_factor(factor_id: str) -> int:
    """Delete a factor's rows from the work DB. Returns rows deleted."""
    with FactorStorage() as work:
        return work.delete_factor(factor_id)


def cleanup_all() -> dict[str, int]:
    """Wipe every factor from the work DB. Returns ``{factor_id: rows}``."""
    with FactorStorage() as work:
        return work.delete_factors(sorted(work.get_existing_factor_ids()))


def cleanup_orphans() -> dict[str, int]:
    """Drop work rows for factors that are already admitted.

    These rows are leftovers from a partial admit (e.g. crash between the
    library write and the work clear). They're noise — admitted factors
    should live only in the library.
    """
    admitted = set(get_admitted_factor_ids())
    if not admitted:
        return {}
    with FactorStorage() as work:
        orphans = sorted(work.get_existing_factor_ids() & admitted)
        return work.delete_factors(orphans)


def main():
    parser = argparse.ArgumentParser(description="Clear work-DB factor data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("factor_id", nargs="?", default=None,
                       help="Single factor to clear")
    group.add_argument("--all", action="store_true",
                       help="Clear EVERY factor in the work DB")
    group.add_argument("--orphans", action="store_true",
                       help="Clear work rows for factors already admitted to the library")
    args = parser.parse_args()

    if args.all:
        cleared = cleanup_all()
        if not cleared:
            print("(work DB already empty)")
            return
        for fid, n in cleared.items():
            print(f"  {fid}: {n:,} rows")
        print(f"\nCleared {len(cleared)} factor(s), {sum(cleared.values()):,} rows total.")
        return

    if args.orphans:
        cleared = cleanup_orphans()
        if not cleared:
            print("(no orphan rows)")
            return
        for fid, n in cleared.items():
            print(f"  {fid}: {n:,} rows")
        return

    n = cleanup_factor(args.factor_id)
    print(f"  {args.factor_id}: cleared {n:,} rows")


if __name__ == "__main__":
    main()
