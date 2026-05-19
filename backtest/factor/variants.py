"""因子变体(neutralization variants)命名规范与解析工具。

每个因子可以以多种"中性化变体"存在,落到 ``factors_daily`` 表的 ``variant`` 列上。
一个变体由 ``(industry, cap)`` 二元组决定:

| spec key  | spec 写法(声明用)        | variant 名(存储用) | 含义 |
|-----------|---------------------------|---------------------|------|
| industry  | ``None``                  | ``none``            | 不做行业中性化 |
|           | ``"SW-L1"``               | ``swl1``            | 申万一级,组内 zscore |
|           | ``"SW-L2"``               | ``swl2``            | 申万二级,组内 zscore |
| cap       | ``None``                  | ``none``            | 不做市值中性化 |
|           | ``"circ_mv-q5"``          | ``capq5``           | 流通市值 5 分位组,组内 zscore |
|           | ``"circ_mv-q10"``         | ``capq10``          | 流通市值 10 分位组 |
|           | ``"total_mv-q5"``         | ``totalq5``         | 总市值 5 分位组 |
|           | ``"total_mv-q10"``        | ``totalq10``        | 总市值 10 分位组 |

variant 字符串规范: ``"<industry>_<cap>"``,例:``"swl1_capq5"`` / ``"swl2_none"``。
特例: ``"raw"`` 等价于 ``"none_none"``;读写时统一标准化为 ``"raw"``。

合法组合数: 3 行业 × 5 市值 = 15。**不预先全算**,只算 registry 声明的子集。
"""

from __future__ import annotations

from typing import Literal


# ---------------------------------------------------------------------------
# 命名映射
# ---------------------------------------------------------------------------

IndustrySpec = Literal[None, "SW-L1", "SW-L2"]
CapSpec = Literal[None, "circ_mv-q5", "circ_mv-q10", "total_mv-q5", "total_mv-q10"]

_INDUSTRY_SPEC_TO_TOKEN: dict[IndustrySpec, str] = {
    None: "none",
    "SW-L1": "swl1",
    "SW-L2": "swl2",
}

_CAP_SPEC_TO_TOKEN: dict[CapSpec, str] = {
    None: "none",
    "circ_mv-q5": "capq5",
    "circ_mv-q10": "capq10",
    "total_mv-q5": "totalq5",
    "total_mv-q10": "totalq10",
}

_INDUSTRY_TOKEN_TO_SPEC = {v: k for k, v in _INDUSTRY_SPEC_TO_TOKEN.items()}
_CAP_TOKEN_TO_SPEC = {v: k for k, v in _CAP_SPEC_TO_TOKEN.items()}

VALID_INDUSTRY_SPECS = tuple(_INDUSTRY_SPEC_TO_TOKEN.keys())
VALID_CAP_SPECS = tuple(_CAP_SPEC_TO_TOKEN.keys())

# 默认变体: 每个因子默认至少这两个 variant 必算
DEFAULT_NEUTRALIZATIONS: list[dict[str, object]] = [
    {"industry": None, "cap": None},               # → "raw"
    {"industry": "SW-L2", "cap": "circ_mv-q5"},    # → "swl2_capq5",推荐基线
]

# 通用 baseline,主要用作评测/相关性比较的默认 variant
BASELINE_VARIANT: str = "swl2_capq5"
RAW_VARIANT: str = "raw"


# ---------------------------------------------------------------------------
# 编解码
# ---------------------------------------------------------------------------

def variant_name(industry: IndustrySpec, cap: CapSpec) -> str:
    """``(industry, cap)`` → variant 字符串。``(None, None)`` → ``"raw"``。"""
    if industry not in _INDUSTRY_SPEC_TO_TOKEN:
        raise ValueError(
            f"Unknown industry spec: {industry!r}. "
            f"Valid: {VALID_INDUSTRY_SPECS}"
        )
    if cap not in _CAP_SPEC_TO_TOKEN:
        raise ValueError(
            f"Unknown cap spec: {cap!r}. Valid: {VALID_CAP_SPECS}"
        )
    if industry is None and cap is None:
        return RAW_VARIANT
    return f"{_INDUSTRY_SPEC_TO_TOKEN[industry]}_{_CAP_SPEC_TO_TOKEN[cap]}"


def parse_variant(variant: str) -> tuple[IndustrySpec, CapSpec]:
    """variant 字符串 → ``(industry, cap)``。

    ``"raw"`` → ``(None, None)``;``"swl1_capq5"`` → ``("SW-L1", "circ_mv-q5")``。
    """
    if variant == RAW_VARIANT:
        return (None, None)
    if "_" not in variant:
        raise ValueError(
            f"Variant '{variant}' has no '_' separator. "
            f"Expected '<industry>_<cap>' or 'raw'."
        )
    ind_tok, cap_tok = variant.split("_", 1)
    if ind_tok not in _INDUSTRY_TOKEN_TO_SPEC:
        raise ValueError(
            f"Unknown industry token '{ind_tok}' in variant '{variant}'. "
            f"Valid: {sorted(_INDUSTRY_TOKEN_TO_SPEC.keys())}"
        )
    if cap_tok not in _CAP_TOKEN_TO_SPEC:
        raise ValueError(
            f"Unknown cap token '{cap_tok}' in variant '{variant}'. "
            f"Valid: {sorted(_CAP_TOKEN_TO_SPEC.keys())}"
        )
    return (_INDUSTRY_TOKEN_TO_SPEC[ind_tok], _CAP_TOKEN_TO_SPEC[cap_tok])


def canonicalize_variant(variant: str) -> str:
    """把 ``"none_none"`` 等价归一为 ``"raw"``。其他情况原样返回。"""
    if variant == "none_none":
        return RAW_VARIANT
    return variant


def normalize_neutralizations(
    raw: list[dict] | None,
) -> list[dict]:
    """把 registry 里的 ``neutralizations`` 列表做规范化,确保每个 dict 含
    ``industry`` 和 ``cap`` 两个键,并验证 spec 合法。

    ``None`` 输入返回 :data:`DEFAULT_NEUTRALIZATIONS` 的副本。
    重复的变体(同 ``variant_name``)只保留首次出现。
    """
    if raw is None:
        return [dict(d) for d in DEFAULT_NEUTRALIZATIONS]

    seen: set[str] = set()
    out: list[dict] = []
    for d in raw:
        if not isinstance(d, dict):
            raise TypeError(
                f"neutralizations entry must be dict, got {type(d).__name__}"
            )
        industry = d.get("industry")
        cap = d.get("cap")
        name = variant_name(industry, cap)
        if name in seen:
            continue
        seen.add(name)
        out.append({"industry": industry, "cap": cap})
    return out


def expand_variant_names(neutralizations: list[dict] | None) -> list[str]:
    """声明列表 → variant 名列表(顺序与声明一致,去重)。"""
    return [
        variant_name(d["industry"], d["cap"])
        for d in normalize_neutralizations(neutralizations)
    ]


__all__ = [
    "BASELINE_VARIANT",
    "RAW_VARIANT",
    "DEFAULT_NEUTRALIZATIONS",
    "VALID_INDUSTRY_SPECS",
    "VALID_CAP_SPECS",
    "variant_name",
    "parse_variant",
    "canonicalize_variant",
    "normalize_neutralizations",
    "expand_variant_names",
]
