"""Tests for the admission CLI (admit / reject / cleanup pathways).

After §3 refactor, admission is per ``(factor_id, variant)``. These tests pin
that contract: state lives under ``variant_status`` / ``variant_admission_history``,
and the top-level ``status`` is a derived summary.
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
        "variant": ["raw"] * 6,
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
            # Single declared variant 'raw' so tests don't tangle with the default
            # 2-variant fan-out.
            "neutralizations": [{"industry": None, "cap": None}],
        }
    }


class TestAdmit:
    def test_promotes_data_clears_work_and_marks_admitted(
        self, patched_storage, sample_rows
    ):
        with FactorStorage() as work:
            work.insert_factors(sample_rows)
            assert len(work.get_factor("f_test", variant="raw")) == 6

        registry = _seed_registry()
        action = admit("f_test", variant="raw", notes="test", registry=registry)

        assert action.action == "admitted"
        assert action.variant == "raw"
        assert action.rows_promoted == 6
        assert action.rows_cleared == 6

        # Work is now empty for that variant.
        with FactorStorage() as work:
            assert work.get_factor("f_test", variant="raw").empty
        # Library has the data.
        with FactorLibrary() as lib:
            assert len(lib.get_factor("f_test", variant="raw")) == 6

        meta = registry["f_test"]
        assert meta["variant_status"]["raw"] == "admitted"
        # With a single declared variant, the derived top-level status matches.
        assert meta["status"] == "admitted"
        assert meta["admission"]["notes"] == "test"
        assert meta["variant_admission_history"]["raw"][-1]["action"] == "admitted"

    def test_admit_without_work_data_raises(self, patched_storage):
        registry = _seed_registry()
        with pytest.raises(ValueError, match="No data in work DB"):
            admit("f_test", variant="raw", registry=registry)

    def test_admit_unknown_factor_raises(self, patched_storage):
        with pytest.raises(KeyError):
            admit("f_unknown", variant="raw", registry={})


class TestReject:
    def test_clears_work_marks_rejected_does_not_touch_library(
        self, patched_storage, sample_rows
    ):
        with FactorStorage() as work:
            work.insert_factors(sample_rows)

        registry = _seed_registry()
        action = reject("f_test", variant="raw", notes="too noisy", registry=registry)

        assert action.action == "rejected"
        assert action.variant == "raw"
        assert action.rows_promoted == 0
        assert action.rows_cleared == 6

        with FactorStorage() as work:
            assert work.get_factor("f_test", variant="raw").empty
        with FactorLibrary() as lib:
            assert lib.get_factor("f_test", variant="raw").empty

        assert registry["f_test"]["variant_status"]["raw"] == "rejected"
        assert registry["f_test"]["status"] == "rejected"

    def test_cannot_reject_already_admitted(self, patched_storage):
        registry = _seed_registry()
        registry["f_test"]["variant_status"] = {"raw": "admitted"}
        with pytest.raises(ValueError, match="already admitted"):
            reject("f_test", variant="raw", registry=registry)


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
    """§5: admit/reject 把 strategy_config 写入 history entry,可手动传入或自动从 results 读。"""

    def test_admit_records_strategy_config(self, patched_storage, sample_rows):
        with FactorStorage() as work:
            work.insert_factors(sample_rows)

        registry = _seed_registry()
        cfg = {"top_n": 50, "rebalance": "1W", "decay": 5, "variant": "raw"}
        action = admit(
            "f_test", variant="raw", notes="test",
            registry=registry, strategy_config=cfg,
        )

        assert action.action == "admitted"
        entry = registry["f_test"]["variant_admission_history"]["raw"][-1]
        assert entry["strategy_config"] == cfg

    def test_reject_records_strategy_config(self, patched_storage, sample_rows):
        with FactorStorage() as work:
            work.insert_factors(sample_rows)

        registry = _seed_registry()
        cfg = {"top_pct": 0.1, "rebalance": "1W", "decay": 5}
        reject("f_test", variant="raw", registry=registry, strategy_config=cfg)
        entry = registry["f_test"]["variant_admission_history"]["raw"][-1]
        assert entry["strategy_config"] == cfg

    def test_omitted_strategy_config_is_not_stored(self, patched_storage, sample_rows):
        """Backward-compat: 不传则 entry 里没有 strategy_config 键。"""
        with FactorStorage() as work:
            work.insert_factors(sample_rows)

        registry = _seed_registry()
        admit("f_test", variant="raw", registry=registry)
        entry = registry["f_test"]["variant_admission_history"]["raw"][-1]
        assert "strategy_config" not in entry


class TestDiscoverStrategyConfig:
    """§5: _discover_strategy_config 自动从 results/<fid>/<variant>/<tag>/pipeline.json 读。"""

    def _make_pipeline_json(self, root, fid, variant, tag, cfg):
        d = root / fid / variant / tag
        d.mkdir(parents=True)
        (d / "pipeline.json").write_text(
            json.dumps({"strategy_config": cfg}), encoding="utf-8"
        )

    def test_single_tag_auto_picked(self, tmp_path):
        cfg = {"top_n": 50, "rebalance": "1W"}
        self._make_pipeline_json(tmp_path, "f_x", "raw", "top50_1w_d5", cfg)
        got = _discover_strategy_config("f_x", "raw", results_root=tmp_path)
        assert got == cfg

    def test_factor_eval_subdir_is_ignored(self, tmp_path):
        """factor_eval 不是 tag,即使存在也不该被误认为 pipeline.json 候选。"""
        cfg = {"top_n": 50}
        self._make_pipeline_json(tmp_path, "f_x", "raw", "top50_1w_d5", cfg)
        (tmp_path / "f_x" / "raw" / "factor_eval").mkdir()
        got = _discover_strategy_config("f_x", "raw", results_root=tmp_path)
        assert got == cfg

    def test_multiple_tags_requires_explicit(self, tmp_path):
        self._make_pipeline_json(tmp_path, "f_x", "raw", "top50_1w_d5", {"a": 1})
        self._make_pipeline_json(tmp_path, "f_x", "raw", "top100_1w_d5", {"a": 2})
        with pytest.raises(ValueError, match="multiple pipeline.json"):
            _discover_strategy_config("f_x", "raw", results_root=tmp_path)

    def test_explicit_tag_reads_target(self, tmp_path):
        self._make_pipeline_json(tmp_path, "f_x", "raw", "top50_1w_d5", {"a": 1})
        self._make_pipeline_json(tmp_path, "f_x", "raw", "top100_1w_d5", {"a": 2})
        got = _discover_strategy_config(
            "f_x", "raw", results_root=tmp_path, tag="top100_1w_d5",
        )
        assert got == {"a": 2}

    def test_explicit_tag_missing_raises(self, tmp_path):
        self._make_pipeline_json(tmp_path, "f_x", "raw", "top50_1w_d5", {"a": 1})
        with pytest.raises(FileNotFoundError):
            _discover_strategy_config(
                "f_x", "raw", results_root=tmp_path, tag="nonexistent",
            )

    def test_variant_dir_missing_returns_none(self, tmp_path):
        # 完全没跑 pipeline,目录不存在 → None(允许纯手动 admit)
        got = _discover_strategy_config("f_unknown", "raw", results_root=tmp_path)
        assert got is None

    def test_no_tag_subdirs_returns_none(self, tmp_path):
        # variant dir 存在但里面只有 factor_eval,没有 tag → None
        (tmp_path / "f_x" / "raw" / "factor_eval").mkdir(parents=True)
        got = _discover_strategy_config("f_x", "raw", results_root=tmp_path)
        assert got is None
