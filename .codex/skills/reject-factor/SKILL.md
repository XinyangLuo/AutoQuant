---
name: reject-factor
description: Reject, clean, and archive failed AutoQuant factor experiments. Use when the user asks to reject-factor, reject a failed f_auto_ factor, clean pending factor data, mark a factor as rejected, archive results, or optionally remove generated factor code.
---

# Reject Factor

Use this project skill for the formal AutoQuant factor rejection workflow.

## Workflow

1. Read `AGENTS.md`, `agents/AGENTS.md`, and the detailed workflow in `references/workflow.md`.
2. If no `factor_id` is provided, list pending/rejected factors with `python -m backtest.factor.admission status`.
3. Confirm the `factor_id`, whether to keep generated code, and which results directory will be moved before destructive operations.
4. Run admission reject through `python -m backtest.factor.admission reject <factor_id>`.
5. Archive `results/<factor_id>/` under `results/rejected/` when present.
6. Check KB files for failed-attempt or anti-pattern records and report gaps.

## Reference

Load `references/workflow.md` when executing the workflow. It contains the full migrated `/reject-factor` procedure, cleanup steps, safety notes, and summary table.
