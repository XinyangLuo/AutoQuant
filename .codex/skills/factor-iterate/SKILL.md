---
name: factor-iterate
description: Run AutoQuant interactive A-share factor research and iteration. Use when the user asks to iterate, generate, repair, sweep, or evaluate a factor idea, mentions factor-iterate, provides a natural-language alpha hypothesis, or wants to continue from an agents/pdf_hypotheses hypothesis.md file.
---

# Factor Iterate

Use this project skill to execute the AutoQuant factor research loop inside this repository.

## Workflow

1. Read `AGENTS.md`, `agents/AGENTS.md`, and the detailed workflow in `references/workflow.md`.
2. For code changes, also read the relevant `DESIGN.md` before editing.
3. Use `conda activate AutoQuant` before Python commands.
4. Use `python -m agents.codex_cli` for schema, run, trace, KB update, and sweep commands.
5. Write generated factor code only under `alphas/exp/agent/<factor_id>/`.
6. Preserve the round trace in `results/<run_id>/trace.jsonl`.
7. Do not relax pipeline thresholds to pass a factor; improve the factor or strategy instead.

## Inputs

- Natural-language hypothesis, for example "成交额放量后短期反转，小盘股更强".
- Explicit formula or parameterized request.
- `--hypothesis agents/pdf_hypotheses/.../*_hypothesis.md`.

## Reference

Load `references/workflow.md` when executing the workflow. It contains the full migrated `/factor-iterate` procedure, run directory rules, RC handling, sweep fast path, trace schema, and pass/abandon cleanup.
