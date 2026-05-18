"""Tests for FactorStorage and FactorLibrary."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.factor.storage import FactorLibrary, FactorStorage


@pytest.fixture
def sample_factors():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
        "symbol": ["A", "B", "A", "B"],
        "factor_id": ["f_001", "f_001", "f_001", "f_001"],
        "value": [1.0, 2.0, 1.1, 2.1],
    })


@pytest.fixture
def tmp_storage(tmp_path):
    """A FactorStorage backed by a temporary DuckDB file."""
    with FactorStorage(db_path=tmp_path / "test_factors.duckdb") as fs:
        yield fs


@pytest.fixture
def tmp_library(tmp_path):
    """A FactorLibrary backed by a temporary DuckDB file."""
    with FactorLibrary(db_path=tmp_path / "test_library.duckdb") as lib:
        yield lib


class TestInsertAndRead:
    def test_insert_and_get_factor(self, tmp_storage, sample_factors):
        tmp_storage.insert_factors(sample_factors)
        result = tmp_storage.get_factor("f_001", "20240101", "20240102")
        assert len(result) == 4
        assert list(result.columns) == ["date", "symbol", "value"]

    def test_get_factor_with_date_filter(self, tmp_storage, sample_factors):
        tmp_storage.insert_factors(sample_factors)
        result = tmp_storage.get_factor("f_001", "20240102", "20240102")
        assert len(result) == 2
        assert all(result["date"] == pd.Timestamp("2024-01-02"))

    def test_get_factor_panel(self, tmp_storage, sample_factors):
        tmp_storage.insert_factors(sample_factors)
        panel = tmp_storage.get_factor_panel(["f_001"], "20240101")
        assert len(panel) == 2
        assert "f_001" in panel.columns

    def test_upsert_overwrites(self, tmp_storage, sample_factors):
        tmp_storage.insert_factors(sample_factors)
        updated = sample_factors.copy()
        updated["value"] = [10.0, 20.0, 11.0, 21.0]
        tmp_storage.insert_factors(updated)

        result = tmp_storage.get_factor("f_001", "20240101", "20240101")
        assert result[result["symbol"] == "A"]["value"].iloc[0] == pytest.approx(10.0)


class TestStats:
    def test_get_max_date(self, tmp_storage, sample_factors):
        tmp_storage.insert_factors(sample_factors)
        assert tmp_storage.get_max_date("f_001") == "20240102"

    def test_get_max_date_empty(self, tmp_storage):
        assert tmp_storage.get_max_date("f_none") is None

    def test_get_factor_stats(self, tmp_storage, sample_factors):
        tmp_storage.insert_factors(sample_factors)
        stats = tmp_storage.get_factor_stats("f_001")
        assert stats["total_rows"] == 4
        assert stats["total_symbols"] == 2
        assert stats["min_date"] == "20240101"
        assert stats["max_date"] == "20240102"

    def test_get_existing_factor_ids(self, tmp_storage, sample_factors):
        tmp_storage.insert_factors(sample_factors)
        ids = tmp_storage.get_existing_factor_ids()
        assert ids == {"f_001"}


class TestEmptyInsert:
    def test_empty_dataframe_noop(self, tmp_storage):
        empty = pd.DataFrame(columns=["date", "symbol", "factor_id", "value"])
        tmp_storage.insert_factors(empty)
        assert tmp_storage.get_max_date("f_001") is None

    def test_missing_required_columns_raises(self, tmp_storage):
        bad_df = pd.DataFrame({"date": ["2024-01-01"], "symbol": ["A"]})
        with pytest.raises(ValueError, match="Missing required"):
            tmp_storage.insert_factors(bad_df)


class TestFactorLibrary:
    def test_delete_factor_disabled(self, tmp_library, sample_factors):
        tmp_library.insert_factors(sample_factors)
        with pytest.raises(NotImplementedError, match="append-only"):
            tmp_library.delete_factor("f_001")

    def test_promote_from_work_copies_rows(self, tmp_storage, tmp_library,
                                            sample_factors):
        tmp_storage.insert_factors(sample_factors)
        n = tmp_library.promote_from_work("f_001", tmp_storage)
        assert n == 4
        # Library now has the data.
        lib_rows = tmp_library.get_factor("f_001", "20240101", "20240102")
        assert len(lib_rows) == 4
        # Work still has the data — promote_from_work does NOT clear work.
        # That's a separate step taken by admit().
        work_rows = tmp_storage.get_factor("f_001", "20240101", "20240102")
        assert len(work_rows) == 4

    def test_promote_empty_returns_zero(self, tmp_storage, tmp_library):
        n = tmp_library.promote_from_work("f_missing", tmp_storage)
        assert n == 0

    def test_promote_then_clear_work_round_trip(self, tmp_storage, tmp_library,
                                                 sample_factors):
        """Mirror what admit() does end-to-end."""
        tmp_storage.insert_factors(sample_factors)
        tmp_library.promote_from_work("f_001", tmp_storage)
        cleared = tmp_storage.delete_factor("f_001")
        assert cleared == 4
        assert tmp_storage.get_factor("f_001").empty
        assert len(tmp_library.get_factor("f_001")) == 4
