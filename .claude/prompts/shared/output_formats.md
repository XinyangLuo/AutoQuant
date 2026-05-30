# Shared: Output Format Templates

> 标准化 JSON schema 模板，注入各 subagent 的 prompt 中强制结构化输出。
> Copy 对应 schema block 到 prompt 的 `# Output Format` 部分。

## Hypothesis JSON Schema (HG)

```json
{
  "hypothesis_text": "string: 自然语言假设描述",
  "category": "string: factor taxonomy 分类",
  "data_sources": ["string array: 数据源列表"],
  "construction_logic": "string: 分步骤构造逻辑",
  "expected_icir": "float: 预期年化 ICIR",
  "alignment_score": "float 0-1: 假设与数据/市场逻辑的一致性",
  "impact_score": "float 0-1: 预期 alpha 强度",
  "novelty_score": "float 0-1: 与已有因子的差异化程度",
  "feasibility_score": "float 0-1: 数据可用性和实现难度",
  "risk_reward_score": "float 0-1: 风险调整后的收益预期"
}
```

## Diagnosis JSON Schema (RC)

```json
{
  "failure_type": "string: result.json 中的 failure_type",
  "diagnosis": "string: 根因分析，一段话",
  "fix_strategy": "string: 具体修复建议",
  "fix_level": "factor | strategy_only | both | retry",
  "factor_params": {"key": "value"},
  "strategy_params": {"decay": 5, "rebalance": "1D", "top_k": 100, "variant": "none"},
  "same_direction": true,
  "recommend_abandon": false,
  "new_anti_pattern": null
}
```

- `fix_level`: "factor"=改因子代码, "strategy_only"=只改 config.yaml, "both"=两个都改, "retry"=原样重试
- `factor_params`: 窗口/horizon/variant/公式参数等
- `strategy_params`: decay(1-30), rebalance("1D"/"5D"/"1W"/"2W"/"1M"/"EOM"), top_k(50-500), variant("none"/"barra_ind_size"/"barra_l3")
- `new_anti_pattern`: {"pattern": "...", "category": "...", "signature": "...", "fix": "..."} 或 null

## Trace Record Schema

```json
{
  "round": 1,
  "factor_id": "f_auto_YYYYMMDD_NNN",
  "category": "string",
  "data_sources": ["string array"],
  "status": "pass|fail|error",
  "failure_type": "string|null",
  "error_signature": "string|null",
  "diagnosis": "string",
  "fix_strategy": "string",
  "fix_level": "factor|strategy_only|both|retry",
  "factor_params": {},
  "strategy_params": {},
  "code_summary": "string: 公式+构造简述",
  "tried_params": {},
  "recommend_abandon": false,
  "metrics": {
    "annual_icir": null,
    "simple_sharpe": null,
    "r2": null,
    "max_existing_corr": null,
    "residual_icir": null
  },
  "same_direction": true,
  "parent_round": null,
  "branch_id": "main"
}
```
