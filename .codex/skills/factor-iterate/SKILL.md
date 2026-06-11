---
name: factor-iterate
description: Run AutoQuant interactive A-share factor research and iteration. Use when the user asks to iterate, generate, repair, sweep, or evaluate a factor idea, provides a natural-language alpha hypothesis, or wants to continue from a PDF-derived hypothesis selection.
---

# Factor Iterate

Use this project skill to execute the AutoQuant factor research loop.

## Workflow

1. Read `AGENTS.md`, `agents/AGENTS.md`, `agents/FACTOR_CODE_GUIDE.md`, and `references/workflow.md`.
2. If the user provided a natural-language factor idea or explicit formula, start the iteration from that input.
3. If the user says to continue from a PDF result, a previous selection, "the latest", or a numbered hypothesis, discover candidates under `agents/pdf_hypotheses/` and present numbered menus instead of asking for a path.
4. When a hypothesis is selected, read the chosen `*_hypothesis.md`; prefer metadata from the sibling `manifest.json` when present.
5. Query schema before writing code with `conda activate AutoQuant && python -m agents.codex_cli schema --sources ...`.
6. Write factor code and config only under `alphas/exp/agent/<factor_id>/`.
7. Run the pipeline through `python -m agents.codex_cli run` or `sweep`, preserve trace in `results/<run_id>/trace.jsonl`, and update KB on pass/fail.
8. Stop when the factor passes, the maximum rounds are exhausted, or the repair direction is abandoned.

## Boundaries

- This skill may read PDF-derived hypothesis files, but it must not extract PDFs.
- Do not ask the user to hardcode `agents/pdf_hypotheses/...` paths unless they voluntarily provide one.
- Do not relax pipeline thresholds to pass a factor.
- Do not automatically admit a passed factor; leave admission as a separate human decision.

## Reference

`references/workflow.md` contains input resolution, hypothesis menus, per-round rules, pipeline commands, trace/KB handling, and pass/abandon cleanup.
