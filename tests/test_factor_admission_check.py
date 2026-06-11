"""Tests for ridge R² admission check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.factor.admission_check import (
    BARRA_L1_REGRESSORS,
    TIER_PURE_ALPHA,
    TIER_REJECT,
    TIER_SMART_BETA,
    _classify,
    _get_all_regressor_ids,
    _load_all_admitted_regressors,
    _pooled_r2,
    _ridge_fit,
    ridge_r2_check,
)


# Thresholds from config.yaml (single source of truth).
# Duplicated here so tests fail if config changes unintentionally.
R2_PURE_ALPHA_MAX = 0.2
R2_SMART_BETA_MAX = 0.7


@pytest.fixture(autouse=True)
def clean_registry():
    from backtest.factor import registry
    registry._REGISTRY_CACHE = {}
    registry._FACTOR_FUNCTIONS.clear()
    yield
    registry._REGISTRY_CACHE = {}
    registry._FACTOR_FUNCTIONS.clear()


class TestClassify:
    def test_pure_alpha_band(self):
        assert _classify(0.0) == TIER_PURE_ALPHA
        assert _classify(0.199) == TIER_PURE_ALPHA

    def test_smart_beta_band(self):
        assert _classify(0.20) == TIER_SMART_BETA
        assert _classify(0.699) == TIER_SMART_BETA

    def test_reject_band(self):
        assert _classify(0.70) == TIER_REJECT
        assert _classify(1.0) == TIER_REJECT

    def test_thresholds_match_config(self):
        from backtest.config_loader import get_section
        th = get_section("thresholds", "admission", "ridge_r2")
        assert th["pure_alpha_max"] == R2_PURE_ALPHA_MAX
        assert th["smart_beta_max"] == R2_SMART_BETA_MAX


class TestRidgeFit:
    def test_recovers_known_coefs(self):
        rng = np.random.default_rng(0)
        n, p = 500, 4
        X = rng.standard_normal((n, p))
        true_beta = np.array([1.0, -0.5, 2.0, 0.0])
        y = X @ true_beta + 0.05 * rng.standard_normal(n)
        beta, intercept = _ridge_fit(X, y, alpha=1e-6)
        np.testing.assert_allclose(beta, true_beta, atol=0.05)
        assert abs(intercept) < 0.05

    def test_alpha_shrinks_coefs(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 3))
        y = X @ np.array([3.0, 3.0, 3.0]) + 0.1 * rng.standard_normal(200)
        beta_small, _ = _ridge_fit(X, y, alpha=1e-6)
        beta_large, _ = _ridge_fit(X, y, alpha=1e6)
        assert np.linalg.norm(beta_large) < np.linalg.norm(beta_small)


class TestPooledR2:
    @staticmethod
    def _make_panel(n_days=60, n_symbols=40, n_regs=6, true_betas=None, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        symbols = [f"S{i:03d}" for i in range(n_symbols)]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

        reg_data = rng.standard_normal((len(index), n_regs))
        reg_cols = [f"r{i}" for i in range(n_regs)]
        regressors = pd.DataFrame(reg_data, index=index, columns=reg_cols).reset_index()

        if true_betas is None:
            true_betas = np.zeros(n_regs)
        noise = rng.standard_normal(len(index))
        y = reg_data @ true_betas + noise
        candidate = pd.DataFrame({
            "date": regressors["date"], "symbol": regressors["symbol"],
            "value": y,
        })
        return candidate, regressors

    def test_pure_noise_yields_pure_alpha(self):
        candidate, regressors = self._make_panel(true_betas=np.zeros(6), seed=42)
        r2, residual, keys = _pooled_r2(candidate, regressors, alpha=1.0)
        assert r2 < R2_PURE_ALPHA_MAX
        assert residual.shape[0] == keys.shape[0]
        assert keys.shape[1] == 2

    def test_pure_combination_yields_reject(self):
        candidate, regressors = self._make_panel(
            n_days=80, n_symbols=50,
            true_betas=np.array([2.0, -1.5, 1.0, 0.5, -0.5, 1.2]),
            seed=7,
        )
        rng = np.random.default_rng(7)
        reg_block = regressors.iloc[:, 2:].to_numpy()
        y = reg_block @ np.array([2.0, -1.5, 1.0, 0.5, -0.5, 1.2]) + 0.01 * rng.standard_normal(len(reg_block))
        candidate = candidate.assign(value=y)
        r2, _, _ = _pooled_r2(candidate, regressors, alpha=1.0)
        assert r2 >= R2_SMART_BETA_MAX

    def test_partial_signal_lands_in_smart_beta_band(self):
        candidate, regressors = self._make_panel(
            n_days=120, n_symbols=80,
            true_betas=np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
            seed=11,
        )
        r2, _, _ = _pooled_r2(candidate, regressors, alpha=1.0)
        assert R2_PURE_ALPHA_MAX <= r2 < R2_SMART_BETA_MAX

    def test_too_few_rows_raises(self):
        rng = np.random.default_rng(0)
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=1), [f"S{i}" for i in range(3)]],
            names=["date", "symbol"],
        )
        reg = pd.DataFrame(rng.standard_normal((3, 6)),
                           index=idx, columns=[f"r{i}" for i in range(6)]).reset_index()
        cand = pd.DataFrame({
            "date": reg["date"], "symbol": reg["symbol"],
            "value": rng.standard_normal(3),
        })
        with pytest.raises(ValueError, match="Too few overlapping rows"):
            _pooled_r2(cand, reg, alpha=1.0)


class TestRegressorLoading:
    def test_get_all_regressor_ids_uses_column_metadata_only(self):
        class FakeLibrary:
            def get_existing_factor_ids(self):
                return {"f_candidate", "f_a", "f_b"}

            def get_factors_long(self, *args, **kwargs):
                raise AssertionError("should not materialize all library factors")

        assert _get_all_regressor_ids("f_candidate", FakeLibrary()) == ["f_a", "f_b"]

    def test_load_all_admitted_regressors_reads_wide_once(self):
        expected = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "symbol": ["A", "B"],
            "f_a": [1.0, 2.0],
            "f_b": [3.0, 4.0],
        })

        class FakeLibrary:
            def get_existing_factor_ids(self):
                return {"f_candidate", "f_a", "f_b"}

            def get_factors_wide(self, factor_ids, start=None, end=None):
                assert factor_ids == ["f_a", "f_b"]
                assert start == "20240101"
                assert end == "20240131"
                return expected.copy()

            def get_factors_long(self, *args, **kwargs):
                raise AssertionError("should not load long factors and pivot")

        result = _load_all_admitted_regressors(
            "f_candidate",
            "20240101",
            "20240131",
            FakeLibrary(),
        )

        pd.testing.assert_frame_equal(result, expected)


class TestRidgeR2CheckIntegration:
    """End-to-end with real temp DuckDBs for FactorStorage + FactorLibrary."""

    def _seed_dbs(self, tmp_path, candidate_values, regressor_values, dates, symbols):
        from backtest.factor.storage import FactorLibrary, FactorStorage

        work_path = tmp_path / "factors.duckdb"
        lib_path = tmp_path / "factor_library.duckdb"

        with FactorStorage(db_path=work_path) as ws:
            ws.insert_factors(pd.DataFrame({
                "date": np.repeat(dates, len(symbols)),
                "symbol": np.tile(symbols, len(dates)),
                "factor_id": "f_alpha_candidate",
                "value": candidate_values,
            }))

        with FactorLibrary(db_path=lib_path) as lb:
            for reg_id, vals in regressor_values.items():
                lb.insert_factors(pd.DataFrame({
                    "date": np.repeat(dates, len(symbols)),
                    "symbol": np.tile(symbols, len(dates)),
                    "factor_id": reg_id,
                    "value": vals,
                }), allow_unadmitted=True)
        return work_path, lib_path

    def test_pure_noise_classified_pure_alpha(self, tmp_path):
        from backtest.factor.registry import register
        from backtest.factor.storage import FactorLibrary, FactorStorage

        @register(
            "f_alpha_candidate", name="alpha", category="test",
            data_sources=["market_daily"], variant="barra_ind_size", frequency="D",
        )
        def _f(panel):
            return panel.set_index(["date", "symbol"])["close"]

        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        symbols = [f"S{i:03d}" for i in range(30)]
        n = len(dates) * len(symbols)

        candidate = rng.standard_normal(n)
        regressors = {fid: rng.standard_normal(n) for fid in BARRA_L1_REGRESSORS}
        work_path, lib_path = self._seed_dbs(
            tmp_path, candidate, regressors, dates, symbols,
        )

        with FactorStorage(db_path=work_path) as ws, FactorLibrary(db_path=lib_path) as lb:
            result = ridge_r2_check(
                "f_alpha_candidate",
                factor_storage=ws, library=lb,
            )
        assert result.tier == TIER_PURE_ALPHA
        assert result.r2 < R2_PURE_ALPHA_MAX
        assert result.n_regressors == 6
        assert result.n_obs == n

    def test_style_clone_classified_reject(self, tmp_path):
        from backtest.factor.registry import register
        from backtest.factor.storage import FactorLibrary, FactorStorage

        @register(
            "f_alpha_candidate", name="alpha", category="test",
            data_sources=["market_daily"], variant="barra_ind_size", frequency="D",
        )
        def _f(panel):
            return panel.set_index(["date", "symbol"])["close"]

        rng = np.random.default_rng(2)
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        symbols = [f"S{i:03d}" for i in range(30)]

        regressors = {fid: rng.standard_normal(len(dates) * len(symbols))
                      for fid in BARRA_L1_REGRESSORS}
        candidate = (
            1.0 * regressors["f_barra_beta"]
            + 1.0 * regressors["f_barra_momentum"]
            + 1.0 * regressors["f_barra_value"]
            + 0.01 * rng.standard_normal(len(dates) * len(symbols))
        )
        work_path, lib_path = self._seed_dbs(
            tmp_path, candidate, regressors, dates, symbols,
        )

        with FactorStorage(db_path=work_path) as ws, FactorLibrary(db_path=lib_path) as lb:
            result = ridge_r2_check(
                "f_alpha_candidate",
                factor_storage=ws, library=lb,
            )
        assert result.tier == TIER_REJECT
        assert result.r2 >= R2_SMART_BETA_MAX

    def test_missing_regressor_raises(self, tmp_path):
        from backtest.factor.registry import register
        from backtest.factor.storage import FactorLibrary, FactorStorage

        @register(
            "f_alpha_candidate", name="alpha", category="test",
            data_sources=["market_daily"], variant="barra_ind_size", frequency="D",
        )
        def _f(panel):
            return panel.set_index(["date", "symbol"])["close"]

        rng = np.random.default_rng(3)
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        symbols = [f"S{i:03d}" for i in range(20)]
        n = len(dates) * len(symbols)

        work_path = tmp_path / "factors.duckdb"
        lib_path = tmp_path / "factor_library.duckdb"
        with FactorStorage(db_path=work_path) as ws:
            ws.insert_factors(pd.DataFrame({
                "date": np.repeat(dates, len(symbols)),
                "symbol": np.tile(symbols, len(dates)),
                "factor_id": "f_alpha_candidate",
                "value": rng.standard_normal(n),
            }))
        with FactorLibrary(db_path=lib_path):
            pass

        with FactorStorage(db_path=work_path) as ws, FactorLibrary(db_path=lib_path) as lb:
            with pytest.raises(ValueError, match="No admitted factors in library"):
                ridge_r2_check(
                    "f_alpha_candidate", factor_storage=ws, library=lb,
                )

    def test_empty_candidate_raises(self, tmp_path):
        from backtest.factor.registry import register
        from backtest.factor.storage import FactorLibrary, FactorStorage

        @register(
            "f_alpha_candidate", name="alpha", category="test",
            data_sources=["market_daily"], variant="barra_ind_size", frequency="D",
        )
        def _f(panel):
            return panel.set_index(["date", "symbol"])["close"]

        work_path = tmp_path / "factors.duckdb"
        lib_path = tmp_path / "factor_library.duckdb"
        with FactorStorage(db_path=work_path):
            pass
        with FactorLibrary(db_path=lib_path):
            pass

        with FactorStorage(db_path=work_path) as ws, FactorLibrary(db_path=lib_path) as lb:
            with pytest.raises(ValueError, match="no rows in the work DB"):
                ridge_r2_check(
                    "f_alpha_candidate", factor_storage=ws, library=lb,
                    regressors=BARRA_L1_REGRESSORS,
                )
