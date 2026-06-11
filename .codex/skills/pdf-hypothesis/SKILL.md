---
name: pdf-hypothesis
description: Extract, rank, and write AutoQuant factor hypotheses from Chinese A-share research PDFs. Use when the user asks to analyze a research paper/report PDF, mentions pdf-hypothesis, wants factor ideas from research_papers, or asks to create agents/pdf_hypotheses hypothesis.md files for later factor iteration.
---

# PDF Hypothesis

Use this project skill to turn brokerage or academic research PDFs into structured AutoQuant factor hypotheses.

## Workflow

1. Read `AGENTS.md`, `agents/AGENTS.md`, and the detailed workflow in `references/workflow.md`.
2. If no PDF path is provided, list `research_papers/*.pdf` and ask the user to choose.
3. Extract PDF text through the configured `mcp-pdf` server when available.
4. Enumerate all single-factor ideas from the report before ranking or filtering.
5. Verify data availability with `python -m agents.codex_cli schema --sources ...`.
6. Save selected hypotheses under `agents/pdf_hypotheses/<slug>/`.
7. Do not directly start factor iteration from this skill; output the next `factor-iterate --hypothesis ...` step.

## Reference

Load `references/workflow.md` when executing the workflow. It contains the full migrated `/pdf-hypothesis` procedure, output JSON shape, hypothesis.md template, ranking rules, data mapping, and report-specific extraction guidance.
