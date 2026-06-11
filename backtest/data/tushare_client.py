"""Tushare Pro API client with retry and rate limiting."""

import os
import time
from pathlib import Path

import pandas as pd
import tushare as ts


def _find_project_root() -> Path:
    """Walk up from this file until we find the project root."""
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / ".env").exists() or (
            (p / "AGENTS.md").exists() and (p / "environment.yml").exists()
        ):
            return p
        p = p.parent
    raise RuntimeError("Project root not found")


_PROJECT_ROOT = _find_project_root()
_ENV_PATH = _PROJECT_ROOT / ".env"


def _load_token() -> str:
    env_token = os.getenv("TUSHARE_TOKEN")
    if env_token:
        return env_token
    if not _ENV_PATH.exists():
        return ""
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("TUSHARE_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise ValueError("TUSHARE_TOKEN not found in .env")


# Initialize global Tushare client once at import time
ts.set_token(_load_token())
pro = ts.pro_api()


def api_call(func, *args, sleep: float = 0.15, max_retries: int = 3, **kwargs):
    """Call a Tushare API function with retry and rate-limit sleep."""
    for attempt in range(1, max_retries + 1):
        try:
            result = func(*args, **kwargs)
            time.sleep(sleep)
            return result
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait = sleep * attempt * 2
            print(f"  API error (attempt {attempt}/{max_retries}): {exc}. Retry in {wait:.1f}s...")
            time.sleep(wait)
    return None


def fetch_and_transform(api_func, transform=None, **kwargs) -> pd.DataFrame:
    """Call api_func through api_call, return empty DF on None/empty, else apply transform."""
    df = api_call(api_func, **kwargs)
    if df is None or df.empty:
        return pd.DataFrame()
    return transform(df) if transform else df
