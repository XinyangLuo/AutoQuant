"""Tests for the admission CLI (admit / reject / cleanup pathways).

After the wide-schema refactor, each factor has exactly one neutralization
variant (recorded in registry meta, not as a row dimension). Admission state
lives in the top-level ``status`` field; per-variant state is gone.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from backtest.factor.admission import (
    _discover_strategy_config,
    admit,
    check_recommended_thresholds,
    reject,
)
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
    yield {"work_path": work_path, "lib_path": lib_path}


def _seed_registry(factor_id="f_test"):
    return {
        factor_id: {
            "name": "test_factor",
            "category": "test",
            "data_sources": ["market_daily"],
            "description": "",
            "variant": "none",
            "frequency": "D",
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
        action = admit("f_test", notes="test", registry=registry,
                       skip_ridge_check=True)

        assert action.action == "admitted"
        assert action.rows_promoted == 6
        # delete_factor drops a column → 1.
        assert action.rows_cleared == 1

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
            admit("f_test", registry=registry, skip_ridge_check=True)

    def test_admit_unknown_factor_raises(self, patched_storage):
        with pytest.raises(KeyError):
            admit("f_unknown", registry={}, skip_ridge_check=True)


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
        assert action.rows_cleared == 1

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


class TestStrategyConfigCapture:
    """admit/reject can stamp strategy_config into the history entry."""

    def test_admit_records_strategy_config(self, patched_storage, sample_rows):
        with FactorStorage() as work:
            work.insert_factors(sample_rows)

        registry = _seed_registry()
        cfg = {"top_n": 50, "rebalance": "1W", "decay": 5}
        action = admit(
            "f_test", notes="test",
            registry=registry, strategy_config=cfg,
            skip_ridge_check=True,
        )

        assert action.action == "admitted"
        entry = registry["f_test"]["admission_history"][-1]
        assert entry["strategy_config"] == cfg

    def test_reject_records_strategy_config(self, patched_storage, sample_rows):
        with FactorStorage() as work:
            work.insert_factors(sample_rows)

        registry = _seed_registry()
        cfg = {"top_pct": 0.1, "rebalance": "1W", "decay": 5}
        reject("f_test", registry=registry, strategy_config=cfg)
        entry = registry["f_test"]["admission_history"][-1]
        assert entry["strategy_config"] == cfg

    def test_omitted_strategy_config_is_not_stored(self, patched_storage, sample_rows):
        """When omitted, the entry has no strategy_config key."""
        with FactorStorage() as work:
            work.insert_factors(sample_rows)

        registry = _seed_registry()
        admit("f_test", registry=registry, skip_ridge_check=True)
        entry = registry["f_test"]["admission_history"][-1]
        assert "strategy_config" not in entry


class TestDiscoverStrategyConfig:
    """_discover_strategy_config auto-reads results/<fid>/<tag>/pipeline.json."""

    def _make_pipeline_json(self, root, fid, tag, cfg):
        d = root / fid / tag
        d.mkdir(parents=True)
        (d / "pipeline.json").write_text(
            json.dumps({"strategy_config": cfg}), encoding="utf-8"
        )

    def test_single_tag_auto_picked(self, tmp_path):
        cfg = {"top_n": 50, "rebalance": "1W"}
        self._make_pipeline_json(tmp_path, "f_x", "top50_1w_d5", cfg)
        got = _discover_strategy_config("f_x", results_root=tmp_path)
        assert got == cfg

    def test_factor_eval_subdir_is_ignored(self, tmp_path):
        """factor_eval is not a tag — even if present, it shouldn't be picked."""
        cfg = {"top_n": 50}
        self._make_pipeline_json(tmp_path, "f_x", "top50_1w_d5", cfg)
        (tmp_path / "f_x" / "factor_eval").mkdir()
        got = _discover_strategy_config("f_x", results_root=tmp_path)
        assert got == cfg

    def test_multiple_tags_requires_explicit(self, tmp_path):
        self._make_pipeline_json(tmp_path, "f_x", "top50_1w_d5", {"a": 1})
        self._make_pipeline_json(tmp_path, "f_x", "top100_1w_d5", {"a": 2})
        with pytest.raises(ValueError, match="multiple pipeline.json"):
            _discover_strategy_config("f_x", results_root=tmp_path)

    def test_explicit_tag_reads_target(self, tmp_path):
        self._make_pipeline_json(tmp_path, "f_x", "top50_1w_d5", {"a": 1})
        self._make_pipeline_json(tmp_path, "f_x", "top100_1w_d5", {"a": 2})
        got = _discover_strategy_config(
            "f_x", results_root=tmp_path, tag="top100_1w_d5",
        )
        assert got == {"a": 2}

    def test_explicit_tag_missing_raises(self, tmp_path):
        self._make_pipeline_json(tmp_path, "f_x", "top50_1w_d5", {"a": 1})
        with pytest.raises(FileNotFoundError):
            _discover_strategy_config(
                "f_x", results_root=tmp_path, tag="nonexistent",
            )

    def test_factor_dir_missing_returns_none(self, tmp_path):
        # No pipeline run at all → None (manual admit allowed).
        got = _discover_strategy_config("f_unknown", results_root=tmp_path)
        assert got is None

    def test_no_tag_subdirs_returns_none(self, tmp_path):
        # factor dir exists but only contains factor_eval, no tag → None.
        (tmp_path / "f_x" / "factor_eval").mkdir(parents=True)
        got = _discover_strategy_config("f_x", results_root=tmp_path)
        assert got is None


class TestAdmitRidgeGate:
    """admit() runs ridge_r2_check and blocks the reject tier (unless force)."""

    @staticmethod
    def _seed_library_with_barra_l1(rng, dates, symbols, scale=1.0):
        """Insert all 6 Barra L1 regressor columns into the library DB."""
        import numpy as np
        from backtest.factor.admission_check import BARRA_L1_REGRESSORS
        n = len(dates) * len(symbols)
        rows = []
        for fid in BARRA_L1_REGRESSORS:
            rows.append(pd.DataFrame({
                "date": np.repeat(dates, len(symbols)),
                "symbol": np.tile(symbols, len(dates)),
                "factor_id": fid,
                "value": scale * rng.standard_normal(n),
            }))
        with FactorLibrary() as lib:
            for sub in rows:
                lib.insert_factors(sub, allow_unadmitted=True)

    def test_pure_alpha_stamps_tier_on_meta(self, patched_storage):
        import numpy as np
        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        symbols = [f"S{i:03d}" for i in range(25)]
        self._seed_library_with_barra_l1(rng, dates, symbols)

        # Candidate is pure noise → R² near 0 → pure_alpha tier.
        n = len(dates) * len(symbols)
        with FactorStorage() as work:
            work.insert_factors(pd.DataFrame({
                "date": np.repeat(dates, len(symbols)),
                "symbol": np.tile(symbols, len(dates)),
                "factor_id": "f_test",
                "value": rng.standard_normal(n),
            }))

        registry = _seed_registry()
        action = admit("f_test", registry=registry, skip_residual_icir_check=True)
        assert action.action == "admitted"

        meta = registry["f_test"]
        assert meta["tier"] == "pure_alpha"
        assert meta["r2"] < 0.10
        rc = meta["admission"]["ridge_check"]
        assert rc["tier"] == "pure_alpha"
        assert rc["n_obs"] == n

    def test_reject_tier_blocks_admit(self, patched_storage):
        import numpy as np
        from backtest.factor.admission_check import BARRA_L1_REGRESSORS
        rng = np.random.default_rng(1)
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        symbols = [f"S{i:03d}" for i in range(25)]
        self._seed_library_with_barra_l1(rng, dates, symbols)

        # Build candidate as a near-perfect sum of the regressors.
        n = len(dates) * len(symbols)
        rng2 = np.random.default_rng(1)  # reseed to reproduce same regressors
        regs = {fid: rng2.standard_normal(n) for fid in BARRA_L1_REGRESSORS}
        candidate = sum(regs.values()) + 0.001 * rng.standard_normal(n)

        with FactorStorage() as work:
            work.insert_factors(pd.DataFrame({
                "date": np.repeat(dates, len(symbols)),
                "symbol": np.tile(symbols, len(dates)),
                "factor_id": "f_test",
                "value": candidate,
            }))

        registry = _seed_registry()
        with pytest.raises(ValueError, match="blocked by ridge_r2_check"):
            admit("f_test", registry=registry)
        # Status untouched; factor still in work DB.
        assert registry["f_test"].get("status") not in {"admitted"}
        with FactorStorage() as work:
            assert not work.get_factor("f_test").empty

    def test_force_overrides_reject(self, patched_storage):
        import numpy as np
        from backtest.factor.admission_check import BARRA_L1_REGRESSORS
        rng = np.random.default_rng(2)
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        symbols = [f"S{i:03d}" for i in range(25)]
        self._seed_library_with_barra_l1(rng, dates, symbols)

        n = len(dates) * len(symbols)
        rng2 = np.random.default_rng(2)
        regs = {fid: rng2.standard_normal(n) for fid in BARRA_L1_REGRESSORS}
        candidate = sum(regs.values()) + 0.001 * rng.standard_normal(n)

        with FactorStorage() as work:
            work.insert_factors(pd.DataFrame({
                "date": np.repeat(dates, len(symbols)),
                "symbol": np.tile(symbols, len(dates)),
                "factor_id": "f_test",
                "value": candidate,
            }))

        registry = _seed_registry()
        action = admit("f_test", registry=registry, force=True, skip_residual_icir_check=True)
        assert action.action == "admitted"
        assert registry["f_test"]["tier"] == "reject"
        # force still records the verdict — caller chose to override knowingly.

    def test_bootstrap_categories_skip_ridge_check(self, patched_storage, sample_rows):
        """barra_l1 factors don't trigger the ridge gate (they ARE the regressors)."""
        with FactorStorage() as work:
            work.insert_factors(sample_rows)
        registry = _seed_registry()
        registry["f_test"]["category"] = "barra_l1"

        action = admit("f_test", registry=registry)
        assert action.action == "admitted"
        # No ridge check ran → no tier / r2 stamped.
        assert "tier" not in registry["f_test"]
        assert "ridge_check" not in registry["f_test"]["admission"]
