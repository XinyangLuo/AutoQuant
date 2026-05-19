"""Tests for factor registry."""

from __future__ import annotations

import json

import pytest

from backtest.factor import registry
from backtest.factor.registry import (
    get_factor_function,
    get_factor_meta,
    get_registry,
    list_factors,
    register,
    sync_registry,
    unregister,
)


@pytest.fixture(autouse=True)
def clean_registry(tmp_path, monkeypatch):
    """Clean in-memory registry **and** redirect ``_REGISTRY_PATH`` to a tmp
    file so any ``sync_registry()`` triggered by tests cannot clobber the
    real ``data/factor_library/registry.json`` on disk.
    """
    monkeypatch.setattr(registry, "_REGISTRY_PATH", tmp_path / "registry.json")
    # Clear in-memory state
    registry._REGISTRY_CACHE = {}
    registry._FACTOR_FUNCTIONS.clear()
    yield
    # Clean up after test
    registry._REGISTRY_CACHE = {}
    registry._FACTOR_FUNCTIONS.clear()


class TestRegister:
    def test_basic_registration(self):
        @register("f_test", name="test_factor", category="test", data_sources=["market_daily"])
        def test_factor(panel):
            return panel["close"]

        meta = get_factor_meta("f_test")
        assert meta["name"] == "test_factor"
        assert meta["category"] == "test"
        assert meta["data_sources"] == ["market_daily"]
        assert meta["func_name"] == "test_factor"

    def test_duplicate_id_different_func_raises(self):
        @register("f_dup", name="first", category="test", data_sources=["market_daily"])
        def first(panel):
            return panel["close"]

        with pytest.raises(ValueError, match="already registered"):
            @register("f_dup", name="second", category="test", data_sources=["market_daily"])
            def second(panel):
                return panel["open"]

    def test_duplicate_id_same_func_name_ok(self):
        """Re-registering with the same function name should be idempotent."""

        def idem(panel):
            return panel["close"]

        register("f_idem", name="idem", category="test", data_sources=["market_daily"])(idem)

        # Re-registering the same function object should not raise
        register("f_idem", name="idem", category="test", data_sources=["market_daily"])(idem)

    def test_parameters_preserved(self):
        @register(
            "f_params",
            name="param_factor",
            category="test",
            data_sources=["market_daily"],
            parameters={"window": 10, "threshold": 0.05},
        )
        def param_factor(panel, window=10, threshold=0.05):
            return panel["close"]

        meta = get_factor_meta("f_params")
        assert meta["parameters"]["window"] == 10
        assert meta["parameters"]["threshold"] == 0.05


class TestGetFactorFunction:
    def test_returns_registered_function(self):
        @register("f_func", name="func_test", category="test", data_sources=["market_daily"])
        def func_test(panel):
            return panel["close"]

        fn = get_factor_function("f_func")
        assert fn is func_test

    def test_unknown_factor_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_factor_function("f_nonexistent")


class TestListFactors:
    def test_list_all(self):
        @register("f_a", name="a", category="cat1", data_sources=["market_daily"])
        def a(panel):
            return panel["close"]

        @register("f_b", name="b", category="cat2", data_sources=["market_daily"])
        def b(panel):
            return panel["close"]

        factors = list_factors()
        assert len(factors) == 2
        ids = {f["factor_id"] for f in factors}
        assert ids == {"f_a", "f_b"}

    def test_filter_by_category(self):
        @register("f_c1", name="c1", category="momentum", data_sources=["market_daily"])
        def c1(panel):
            return panel["close"]

        @register("f_c2", name="c2", category="value", data_sources=["market_daily"])
        def c2(panel):
            return panel["close"]

        momentum = list_factors(category="momentum")
        assert len(momentum) == 1
        assert momentum[0]["factor_id"] == "f_c1"


class TestSyncRegistry:
    def test_persists_to_disk(self, tmp_path, monkeypatch):
        # Use a temp registry file
        test_path = tmp_path / "test_registry.json"
        monkeypatch.setattr(registry, "_REGISTRY_PATH", test_path)
        monkeypatch.setattr(registry, "_REGISTRY_CACHE", None)

        @register("f_disk", name="disk_test", category="test", data_sources=["market_daily"])
        def disk_test(panel):
            return panel["close"]

        sync_registry()

        # Verify file contents
        with open(test_path) as f:
            data = json.load(f)
        assert "f_disk" in data
        assert data["f_disk"]["name"] == "disk_test"


class TestUnregister:
    def test_removes_factor(self):
        @register("f_remove", name="remove", category="test", data_sources=["market_daily"])
        def remove(panel):
            return panel["close"]

        assert "f_remove" in get_registry()
        unregister("f_remove")
        assert "f_remove" not in get_registry()
