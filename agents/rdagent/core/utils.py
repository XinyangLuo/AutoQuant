"""Shared helpers for the rdagent module."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any


def render_prompt(template_path: Path | str, **kwargs: Any) -> str:
    """Render a markdown prompt template with Jinja2-style ``{{var}}`` substitution.

    This is a simple string replacement — no full Jinja2 dependency required.
    Warns if any placeholders remain unfilled.
    """
    path = Path(template_path)
    text = path.read_text(encoding="utf-8")
    for key, value in kwargs.items():
        placeholder = f"{{{{{key}}}}}"
        text = text.replace(placeholder, str(value))
    remaining = re.findall(r"\{\{(\w+)\}\}", text)
    if remaining:
        warnings.warn(f"Unfilled placeholders in prompt {path.name}: {remaining}")
    return text


def save_json(data: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path | str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
