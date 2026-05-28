"""Tests for :mod:`backtest.factor.dag` — dependency graph + topological sort."""

import pytest
from backtest.factor.dag import (
    build_dependency_graph,
    get_admission_mode,
    get_depends_on,
    topological_sort,
)


class TestBuildDependencyGraph:
    def test_empty_registry(self):
        assert build_dependency_graph({}) == {}

    def test_factors_without_depends_on(self):
        registry = {
            "f_a": {"status": "admitted", "admission_mode": "raw"},
            "f_b": {"status": "pending"},
        }
        assert build_dependency_graph(registry) == {}

    def test_raw_mode_ignored_even_with_depends_on(self):
        registry = {
            "f_c": {"status": "admitted", "admission_mode": "raw",
                    "depends_on": ["f_a"]},
        }
        assert build_dependency_graph(registry) == {}

    def test_single_residual_factor(self):
        registry = {
            "f_c": {"status": "admitted", "admission_mode": "residual",
                    "depends_on": ["f_a", "f_b"]},
        }
        graph = build_dependency_graph(registry)
        assert graph == {"f_c": {"f_a", "f_b"}}

    def test_chain(self):
        registry = {
            "f_a": {"status": "admitted", "admission_mode": "raw"},
            "f_b": {"status": "admitted", "admission_mode": "residual",
                    "depends_on": ["f_a"]},
            "f_c": {"status": "admitted", "admission_mode": "residual",
                    "depends_on": ["f_b"]},
        }
        graph = build_dependency_graph(registry)
        assert graph == {"f_b": {"f_a"}, "f_c": {"f_b"}}


class TestTopologicalSort:
    def test_empty_list(self):
        assert topological_sort([], {}) == []

    def test_single_factor(self):
        assert topological_sort(["f_a"], {}) == ["f_a"]

    def test_no_dependencies(self):
        result = topological_sort(["f_a", "f_b"], {})
        assert set(result) == {"f_a", "f_b"}

    def test_simple_chain(self):
        registry = {
            "f_a": {"admission_mode": "raw"},
            "f_b": {"admission_mode": "residual", "depends_on": ["f_a"]},
        }
        result = topological_sort(["f_b", "f_a"], registry)
        assert result.index("f_a") < result.index("f_b")

    def test_diamond(self):
        registry = {
            "f_a": {"admission_mode": "raw"},
            "f_b": {"admission_mode": "residual", "depends_on": ["f_a"]},
            "f_c": {"admission_mode": "residual", "depends_on": ["f_a"]},
            "f_d": {"admission_mode": "residual", "depends_on": ["f_b", "f_c"]},
        }
        result = topological_sort(["f_d", "f_c", "f_b", "f_a"], registry)
        assert result.index("f_a") < result.index("f_b")
        assert result.index("f_a") < result.index("f_c")
        assert result.index("f_b") < result.index("f_d")
        assert result.index("f_c") < result.index("f_d")

    def test_cycle_raises(self):
        registry = {
            "f_a": {"admission_mode": "residual", "depends_on": ["f_b"]},
            "f_b": {"admission_mode": "residual", "depends_on": ["f_a"]},
        }
        with pytest.raises(ValueError, match="[Cc]ycle"):
            topological_sort(["f_a", "f_b"], registry)

    def test_factors_not_in_request_ignored(self):
        """Factors in registry but not in factor_ids don't affect sort."""
        registry = {
            "f_a": {"admission_mode": "residual", "depends_on": ["f_z"]},
        }
        result = topological_sort(["f_a"], registry)
        assert result == ["f_a"]

    def test_filters_to_requested_ids_only(self):
        registry = {
            "f_a": {"admission_mode": "raw"},
            "f_b": {"admission_mode": "residual", "depends_on": ["f_a"]},
        }
        result = topological_sort(["f_b"], registry)
        assert result == ["f_b"]


class TestGetAdmissionMode:
    def test_residual(self):
        assert get_admission_mode("f_x", {"f_x": {"admission_mode": "residual"}}) == "residual"

    def test_raw(self):
        assert get_admission_mode("f_x", {"f_x": {"admission_mode": "raw"}}) == "raw"

    def test_missing(self):
        assert get_admission_mode("f_x", {"f_x": {}}) is None
        assert get_admission_mode("f_x", {}) is None


class TestGetDependsOn:
    def test_present(self):
        assert get_depends_on("f_x", {"f_x": {"depends_on": ["f_a", "f_b"]}}) == ["f_a", "f_b"]

    def test_missing(self):
        assert get_depends_on("f_x", {"f_x": {}}) == []
        assert get_depends_on("f_x", {}) == []
