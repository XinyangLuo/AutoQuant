---
name: reject-factor
description: Reject, clean, and archive failed AutoQuant factor experiments. Use when the user asks to reject a failed factor, clean pending factor data, mark a factor as rejected, archive results, or remove generated factor code.
---

# Reject Factor

Use this project skill for the formal AutoQuant factor rejection workflow.

## Workflow

1. Read `AGENTS.md`, `agents/AGENTS.md`, and `references/workflow.md`.
2. If the user did not provide a `factor_id`, discover candidates from admission status, `alphas/exp/agent/`, and `results/`, then present a numbered menu.
3. After the user selects a factor, summarize the exact operations: admission reject, code deletion or retention, result archive target, and KB checks.
4. Ask for explicit confirmation before any destructive operation.
5. Run `conda activate AutoQuant && python -m backtest.factor.admission reject <factor_id> --notes "..."`
6. Delete `alphas/exp/agent/<factor_id>/` only when the user did not choose to keep code.
7. Move `results/<factor_id>/` and matching run traces to `results/rejected/` when present.
8. Check `agents/knowledge_base/failed_attempts.jsonl` and `agents/knowledge_base/anti_patterns.json`, then report any missing KB record.

## Boundaries

- This skill only rejects, cleans, archives, and checks KB state.
- It does not iterate factors and does not create hypotheses.
- All delete, move, and DB reject operations require confirmation.

## Reference

`references/workflow.md` contains candidate discovery, confirmation format, exact cleanup steps, safety checks, and final summary requirements.
