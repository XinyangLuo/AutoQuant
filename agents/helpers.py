"""Shared helpers for factor code validation and manipulation."""

from __future__ import annotations

import ast
import re

# Exact names exported by ``backtest.factor.transforms`` (keep in sync).
_VALID_TRANSFORMS: set[str] = {
    "abs_", "cap_neutralize", "cs_demean", "cs_mad_winsorize",
    "cs_ols_residualize", "cs_winsorize", "cs_zscore", "if_else",
    "industry_median_fill", "industry_neutralize", "inverse", "log",
    "rank", "sign", "signed_power", "single_quarter", "sqrt",
    "ts_argmax", "ts_argmin", "ts_corr", "ts_covariance", "ts_decay_exp",
    "ts_decay_linear", "ts_delay", "ts_delta", "ts_ir", "ts_kurtosis",
    "ts_max", "ts_mean", "ts_min", "ts_pct_change", "ts_product",
    "ts_rank", "ts_skewness", "ts_std", "ts_sum", "ttm", "yoy", "z_score",
}


def validate_python_code(code: str) -> None:
    """Raise SyntaxError if *code* is not valid Python."""
    ast.parse(code)


def validate_transforms_imports(code: str) -> None:
    """Check that every name imported from ``backtest.factor.transforms`` exists.

    Raises ``ValueError`` with a descriptive message if an unknown operator is
    referenced (e.g. LLM hallucinated ``cs_rank``).
    """
    tree = ast.parse(code)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "backtest.factor.transforms":
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    if name == "*":
                        continue
                    if name not in _VALID_TRANSFORMS:
                        bad.append(name)
    if bad:
        raise ValueError(
            f"Unknown transform(s) imported: {bad}. "
            f"Valid names: {sorted(_VALID_TRANSFORMS)}"
        )


def force_register_factor_id(code: str, factor_id: str) -> str:
    """Rewrite the ``@register(...)`` decorator in *code* to use *factor_id*.

    - If the function already has ``@register("old_id")``, replace the arg.
    - If it has ``@register`` bare, insert the arg.
    - If it has no ``@register`` at all, prepend one.
    """
    tree = ast.parse(code)
    modified = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for i, dec in enumerate(node.decorator_list):
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "register":
                has_factor_id_kw = any(kw.arg == "factor_id" for kw in dec.keywords)
                if has_factor_id_kw:
                    for kw in dec.keywords:
                        if kw.arg == "factor_id":
                            kw.value = ast.Constant(value=factor_id)
                elif dec.args:
                    dec.args[0] = ast.Constant(value=factor_id)
                else:
                    dec.args.insert(0, ast.Constant(value=factor_id))
                modified = True
                break
            if isinstance(dec, ast.Name) and dec.id == "register":
                node.decorator_list[i] = ast.Call(
                    func=ast.Name(id="register", ctx=ast.Load()),
                    args=[ast.Constant(value=factor_id)],
                    keywords=[],
                )
                modified = True
                break
        if modified:
            break

    if not modified:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node.decorator_list.insert(
                    0,
                    ast.Call(
                        func=ast.Name(id="register", ctx=ast.Load()),
                        args=[ast.Constant(value=factor_id)],
                        keywords=[],
                    ),
                )
                modified = True
                break

    if not modified:
        raise ValueError("No factor function found to decorate with @register")

    return ast.unparse(ast.fix_missing_locations(tree))
