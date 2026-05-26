"""Minute-level data storage: read/write parquet partitions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.data.tushare_client import _find_project_root


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_minute_bars(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    freq: str = "1min",
    columns: list[str] | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Read minute bars from parquet ``symbol/year`` partitions.

    Automatically stitches cross-year files.  Returns an empty DataFrame
    if no matching data.
    """
    if output_dir is None:
        output_dir = _find_project_root() / "data" / "minute"

    # Guard against empty-string dates
    if start_date is not None and not start_date:
        raise ValueError("start_date must not be an empty string")
    if end_date is not None and not end_date:
        raise ValueError("end_date must not be an empty string")

    freq_dir = output_dir / freq
    if not freq_dir.exists():
        return pd.DataFrame()

    if symbols is None:
        symbols = sorted(d.name for d in freq_dir.iterdir() if d.is_dir())
    else:
        symbols = [s for s in symbols if (freq_dir / s).exists()]

    if not symbols:
        return pd.DataFrame()

    start_year = int(start_date[:4]) if start_date else None
    end_year = int(end_date[:4]) if end_date else None

    # Determine which columns to read from parquet (must include date for filtering)
    read_cols = None
    if columns is not None:
        read_cols = list({*columns, "date"})

    pieces: list[pd.DataFrame] = []
    for symbol in symbols:
        symbol_dir = freq_dir / symbol
        for parquet_path in sorted(symbol_dir.glob("*.parquet")):
            year = int(parquet_path.stem)
            if start_year is not None and year < start_year:
                continue
            if end_year is not None and year > end_year:
                continue

            df = pd.read_parquet(parquet_path, columns=read_cols)
            if df.empty:
                continue

            if columns is not None:
                keep = [c for c in columns if c in df.columns]
                df = df[keep]

            pieces.append(df)

    if not pieces:
        return pd.DataFrame()

    result = pd.concat(pieces, ignore_index=True)

    # In-memory date filter
    if start_date is not None:
        result = result[result["date"] >= pd.to_datetime(start_date).date()]
    if end_date is not None:
        result = result[result["date"] <= pd.to_datetime(end_date).date()]

    sort_cols = [c for c in ["symbol", "date", "time"] if c in result.columns]
    if sort_cols:
        result = result.sort_values(sort_cols).reset_index(drop=True)
    return result
