"""Tests for the admission CLI (admit / reject / cleanup pathways)."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.factor.admission import admit, check_recommended_thresholds, reject
from backtest.factor.storage import FactorLibrary, FactorStorage


@pytest.fixture
def sample_rows():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
        "symbol": ["A"] * 3 + ["B"] * 3,
        "factor_id": ["f_test"] * 6,
        "value": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
    })


@pytest.fixture
def patched_storage(tmp_path, monkeypatch):
    """Point both DBs and the registry to tmp paths so the test is hermetic."""
    work_path = tmp_path / "work.duckdb"
    lib_path = tmp_path / "library.duckdb"
    monkeypatch.setattr("backtest.factor.storage.FACTORS_WORK_DB_PATH", work_path)
    monkeypatch.setattr("backtest.factor.storage.FACTOR_LIBRARY_DB_PATH", lib_path)
    # In-memory registry — admit/reject take an explicit dict so we don't need
    # to monkeypatch registry.json.
    yield {"work_path": work_path, "lib_path": lib_path}


def _seed_registry(factor_id="f_test"):
    return {
        factor_id: {
            "name": "test_factor",
            "category": "test",
            "data_sources": ["market_daily"],
            "description": "",
        }
    }


class TestAdmit:
    def test_promotes_data_clears_work_and_marks_admitted(
        self, patched_storage, sample_rows
    ):
        with FactorStorage() as work:
            work.insert_factors(sample_rows)
            assert len(work.get_factor("f_test")) == 6

        registry = _seed_registry()
        action = admit("f_test", notes="test", registry=registry)

        assert action.action == "admitted"
        assert action.rows_promoted == 6
        assert action.rows_cleared == 6

        # Work is now empty.
        with FactorStorage() as work:
            assert work.get_factor("f_test").empty
        # Library has the data.
        with FactorLibrary() as lib:
            assert len(lib.get_factor("f_test")) == 6

        meta = registry["f_test"]
        assert meta["status"] == "admitted"
        assert meta["admission"]["notes"] == "test"
        assert meta["admission_history"][-1]["action"] == "admitted"

    def test_admit_without_work_data_raises(self, patched_storage):
        registry = _seed_registry()
        with pytest.raises(ValueError, match="No data in work DB"):
            admit("f_test", registry=registry)

    def test_admit_unknown_factor_raises(self, patched_storage):
        with pytest.raises(KeyError):
            admit("f_unknown", registry={})


class TestReject:
    def test_clears_work_marks_rejected_does_not_touch_library(
        self, patched_storage, sample_rows
    ):
        with FactorStorage() as work:
            work.insert_factors(sample_rows)

        registry = _seed_registry()
        action = reject("f_test", notes="too noisy", registry=registry)

        assert action.action == "rejected"
        assert action.rows_promoted == 0
        assert action.rows_cleared == 6

        with FactorStorage() as work:
            assert work.get_factor("f_test").empty
        with FactorLibrary() as lib:
            assert lib.get_factor("f_test").empty

        assert registry["f_test"]["status"] == "rejected"

    def test_cannot_reject_already_admitted(self, patched_storage):
        registry = _seed_registry()
        registry["f_test"]["status"] = "admitted"
        with pytest.raises(ValueError, match="already admitted"):
            reject("f_test", registry=registry)


class TestThresholdCheck:
    def test_all_pass(self):
        m = {"rankicir": 0.30, "ic_positive_ratio": 0.55,
             "turnover": 0.40, "max_corr": 0.60}
        assert check_recommended_thresholds(m) == {
            "rankicir": True, "ic_positive_ratio": True,
            "turnover": True, "max_corr": True,
        }

    def test_fails_when_below_min(self):
        m = {"rankicir": 0.10, "ic_positive_ratio": 0.45,
             "turnover": 0.70, "max_corr": 0.90}
        assert check_recommended_thresholds(m) == {
            "rankicir": False, "ic_positive_ratio": False,
            "turnover": False, "max_corr": False,
        }

    def test_custom_threshold_override(self):
        m = {"rankicir": 0.20, "ic_positive_ratio": 0.55,
             "turnover": 0.40, "max_corr": 0.60}
        assert check_recommended_thresholds(m, {"min_rankicir": 0.15})["rankicir"]
        assert not check_recommended_thresholds(m, {"min_rankicir": 0.25})["rankicir"]
