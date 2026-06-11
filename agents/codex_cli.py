"""Codex helper CLI for single-factor iteration.

This module is a compatibility wrapper around ``agents.claude_cli``.  The
implementation remains there for now to avoid a large disruptive rename, while
new Codex-facing docs and commands can use ``python -m agents.codex_cli``.
"""

from __future__ import annotations

from .claude_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
