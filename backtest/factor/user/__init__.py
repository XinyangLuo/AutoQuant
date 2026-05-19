"""User-defined factors.

This is **the** place to write your own factor code. Conventions:

- One factor (or one tight family) per file.
- Each compute function carries the ``@register("f_xxx", ...)`` decorator
  from :mod:`backtest.factor.registry`.
- Import the new module here so the decorator runs at package import time
  and the registry is populated.

After adding a factor:

    # 1. let the registry pick it up + persist to disk
    python -c "from backtest.factor.registry import sync_registry; sync_registry()"

    # 2. compute values into the work DB
    python -m backtest.factor.backfill f_xxx

    # 3. run the full screening pipeline
    python scripts/run_factor_pipeline.py f_xxx \\
        --start 20160101 --end 20251231

    # 4. depending on results, admit or reject
    python -m backtest.factor.admission admit  f_xxx
    python -m backtest.factor.admission reject f_xxx
"""

from backtest.factor.user import reversal_zscore_combo  # noqa: F401
