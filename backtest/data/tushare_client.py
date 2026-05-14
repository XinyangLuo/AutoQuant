"""Tushare Pro API client with retry and rate limiting."""

import time
from pathlib import Path

import tushare as ts


def _find_project_root() -> Path:
    """Walk up from this file until we find the project root (contains .env)."""
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / ".env").exists():
            return p
        p = p.parent
    raise RuntimeError("Project root not found")


_PROJECT_ROOT = _find_project_root()
_ENV_PATH = _PROJECT_ROOT / ".env"


def _load_token() -> str:
    if not _ENV_PATH.exists():
        raise FileNotFoundError(f"{_ENV_PATH} not found")
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
