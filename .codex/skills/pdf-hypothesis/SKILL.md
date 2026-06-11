---
name: pdf-hypothesis
description: Extract, rank, and write AutoQuant factor hypotheses from Chinese A-share research PDFs. Use when the user asks to analyze research reports, derive factor ideas from PDFs, inspect research_papers, or create PDF-derived hypothesis files for later factor iteration.
---

# PDF Hypothesis

Use this project skill to turn research PDFs into structured AutoQuant factor hypotheses.

## Workflow

1. Read `AGENTS.md`, `agents/AGENTS.md`, and `references/workflow.md`.
2. If the user did not provide a PDF path, list `research_papers/*.pdf` and present a numbered menu. Wait for the user to choose.
3. Extract text with the configured `mcp-pdf` server when available. If unavailable, use a local PDF text extractor and record the fallback in `manifest.json`.
4. Enumerate all single-factor ideas before ranking or filtering.
5. Verify data availability with `conda activate AutoQuant && python -m agents.codex_cli schema --sources ...`.
6. Query `agents/knowledge_base/` through `agents.kb_query` when useful for duplicate, anti-pattern, or SOTA context.
7. Write one batch under `agents/pdf_hypotheses/<YYYYMMDD_HHMMSS_slug>/` with `manifest.json`, optional `extracted.md`, and selected `NN_<factor_slug>_hypothesis.md` files.
8. Show a numbered menu of generated hypothesis files. Tell the user they can continue later by saying which number to iterate, without copying paths.

## Boundaries

- This skill only extracts, ranks, reviews, and writes hypotheses.
- Do not run factor backtests or start factor iteration here.
- Do not write generated `.json` or `.md` files at the top level of `agents/pdf_hypotheses/`.

## Reference

`references/workflow.md` contains the full interaction contract, batch layout, manifest shape, ranking rules, and `hypothesis.md` template.
