"""Test residual_icir_check with a synthetic candidate based on an admitted factor."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.factor.storage import FactorLibrary, FactorStorage
from backtest.factor.admission_check import residual_icir_check
from backtest.factor.registry import _load_registry, _save_registry

FACTOR_ID = "f_test_residual_icir"
ORIGINAL = "f_barra_value"

print("=" * 60)
print("残差 ICIR 增量信息检查 — 测试")
print("=" * 60)

# 1. Read f_turnover_vol from library
with FactorLibrary() as lib:
    orig = lib.get_factor(ORIGINAL)

if orig.empty:
    print(f"ERROR: {ORIGINAL} has no data in library")
    exit(1)

print(f"\n原始因子 {ORIGINAL}: {len(orig)} 行, "
      f"{orig['symbol'].nunique()} 只股票, "
      f"{orig['date'].nunique()} 个交易日")

# 2. Create a synthetic candidate: original + tiny noise
# Since f_turnover_vol IS in the library, regressing against it should
# produce near-zero residual -> near-zero residual ICIR -> FAIL
rng = np.random.default_rng(42)
noise = rng.standard_normal(len(orig)) * 0.01

candidate = orig.rename(columns={"value": "value_orig"}).copy()
candidate["value"] = candidate["value_orig"] + noise  # nearly identical
candidate["factor_id"] = FACTOR_ID
candidate = candidate[["date", "symbol", "factor_id", "value"]]

print(f"\n候选因子 {FACTOR_ID}: {ORIGINAL} + N(0, 0.01²) 噪声")
print(f"  (近乎相同的因子 -> 预期残差 ICIR ~ 0 -> 不通过)")

# 3. Insert into work DB
with FactorStorage() as work:
    work.insert_factors(candidate)
print(f"  已写入 work DB")

# 4. Register (temporary)
reg = _load_registry()
reg[FACTOR_ID] = {
    "name": "Test Residual ICIR",
    "category": "test",
    "variant": "barra_ind_size",
    "frequency": "D",
    "status": "pending",
}
_save_registry(reg)

# 5. Run residual ICIR check
print(f"\n--- 运行 residual_icir_check ---")
try:
    result = residual_icir_check(FACTOR_ID)
    print(f"\n结果:")
    print(f"  passed:        {result.passed}")
    print(f"  n_regressors:  {result.n_regressors}")
    print(f"  n_dates:       {result.n_dates}")
    print(f"  n_obs_total:   {result.n_obs_total}")
    print(f"  threshold:     {result.threshold}")
    print(f"\n  各周期残差 ICIR:")
    for h in sorted(result.annual_icirs.keys()):
        raw = result.residual_rank_icirs[h]
        annual = result.annual_icirs[h]
        rmean = result.residual_rank_ic_means[h]
        print(f"    {h:>2}D:  raw_icir={raw:+.4f}  annual_icir={annual:+.4f}  ic_mean={rmean:+.4f}")

    if not result.passed:
        print(f"\n  -> 预期之内：因子几乎与已入库因子重复，无增量信息，不通过")
    else:
        print(f"\n  -> 意外：通过了，需排查")
except Exception as exc:
    print(f"\nERROR: {type(exc).__name__}: {exc}")

# 6. Cleanup
print(f"\n--- 清理 ---")
with FactorStorage() as work:
    work.delete_factor(FACTOR_ID)
reg.pop(FACTOR_ID, None)
_save_registry(reg)
print(f"  已清理 {FACTOR_ID}")
print(f"\n{'=' * 60}")
print(f"测试完成")
