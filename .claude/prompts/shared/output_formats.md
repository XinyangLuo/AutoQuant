# Shared: Output Format Schemas

> 标准 JSON schema，注入各 subagent 的 prompt 末尾。

---

## Hypothesis JSON Schema

Used by HG output, HO input/output.

```json
{
  "hypothesis_text": "string — One-sentence testable hypothesis",
  "category": "string — Factor category (momentum_reversal, volume_reversal, volatility, value, growth, quality, fund_flow, technical, sentiment, composite)",
  "data_sources": ["string — e.g., market_daily, income_q"],
  "formula_draft": "string — AutoQuant transforms style formula",
  "construction_logic": ["string — Step-by-step construction (3-5 steps)"],
  "parameters": {
    "window": "int",
    "long_window": "int (optional)",
    "variant": "string — none | barra_l3 | barra_ind_size",
    "additional_params": "any (optional)"
  },
  "suggested_config": {
    "pipeline": {
      "default_decay": "int",
      "default_rebalance": "string — 1D | 5D | 1W | 2W | 1M | EOM",
      "default_top_k": "int",
      "ret_type": "string — open | close"
    },
    "strategy": {
      "universe": {
        "exclude_st": "bool",
        "exclude_new_ipo_days": "int",
        "include_cyb": "bool",
        "include_kcb": "bool",
        "include_bse": "bool",
        "min_market_cap": "int",
        "min_avg_amount": "int"
      }
    },
    "simulation": {
      "initial_cash": "int",
      "commission_rate": "float",
      "stamp_duty_rate": "float",
      "allow_short": "bool"
    }
  },
  "self_assessment": {
    "alignment_score": "float 0.0-1.0",
    "impact_score": "float 0.0-1.0",
    "novelty_score": "float 0.0-1.0",
    "feasibility_score": "float 0.0-1.0",
    "risk_reward_score": "float 0.0-1.0"
  },
  "expected_icir": "float",
  "rationale": "string — 2-3 sentences explaining why this should work"
}
```

---

## HO Review JSON Schema

Used by HO output.

```json
{
  "optimized_hypothesis": {
    "formula_draft": "string — improved formula or same as input",
    "parameters": {
      "window": "int",
      "variant": "string"
    },
    "suggested_config": {
      "decay": "int",
      "rebalance": "string",
      "top_k": "int"
    },
    "construction_logic": ["string — improved steps"]
  },
  "ho_review": {
    "duplicate_risk": "string — low | medium | high",
    "similar_factors": [
      {
        "factor_id": "string",
        "similarity": "string — high | medium",
        "note": "string"
      }
    ],
    "anti_pattern_warnings": [
      {
        "pattern": "string",
        "severity": "string — high | medium | low",
        "suggestion": "string"
      }
    ],
    "param_suggestions": {
      "param_name": "string — suggested value or range"
    },
    "data_availability": "string — full | partial | missing",
    "missing_columns": ["string"],
    "logic_issues": ["string"],
    "overall_risk": "string — low | medium | high",
    "recommendation": "string — proceed | revise | abandon"
  }
}
```

---

## Diagnosis JSON Schema

Used by RC output.

```json
{
  "failure_type": "string — code_error | schema_error | coverage_fail | neutralization_fail | icir_fail | monotonicity_fail | config_error | backtest_fail | ridge_fail | residual_fail | execution_error | metrics_fail",
  "diagnosis": "string — Root cause analysis (2-3 sentences)",
  "fix_strategy": "string — Specific fix recommendation",
  "fix_level": "string — factor | strategy_only | both | retry",
  "factor_change": "string — params | formula (only when fix_level includes factor)",
  "factor_params": {
    "param_name": "value"
  },
  "strategy_params": {
    "decay": "int (optional)",
    "rebalance": "string (optional)",
    "top_k": "int (optional)"
  },
  "same_direction": "bool",
  "recommend_abandon": "bool",
  "new_hypothesis": "string | null — Concrete alternative hypothesis if same_direction=false",
  "new_anti_pattern": {
    "pattern": "string",
    "category": "string",
    "signature": "string — machine-matchable pattern signature",
    "fix": "string"
  }
}
```

`new_anti_pattern` may be `null` if no new generalizable pattern was discovered.
```

---

## Trace JSONL Schema

Each line is one round.

```json
{
  "round": "int",
  "parent_round_id": "int — parent round number (round-1 for linear, fork target for branch)",
  "branch_id": "string — main | main_fork_xxx | explore_xxx",
  "fork_reason": "string | null",
  "factor_id": "string",
  "category": "string",
  "data_sources": ["string"],
  "status": "string — pass | fail | error",
  "failure_type": "string | null",
  "error_signature": "string | null — First 120 chars of error",
  "diagnosis": "string — From RC output",
  "fix_strategy": "string — From RC output",
  "fix_level": "string",
  "factor_change": "string | null",
  "factor_params": {"param": "value"},
  "strategy_params": {"param": "value"},
  "code_summary": "string — 20-word formula description",
  "tried_params": {"param": "value"},
  "recommend_abandon": "bool",
  "metrics": {
    "annual_icir": "float | null",
    "simple_sharpe": "float | null",
    "r2": "float | null",
    "max_existing_corr": "float | null",
    "residual_icir": "float | null"
  },
  "same_direction": "bool",
  "new_hypothesis": "string | null — If RC suggested new direction",
  "ts": "string — ISO timestamp"
}
```

---

## Hypothesis Index Entry Schema

Used by `agents/knowledge_base/hypothesis_index.jsonl`.

```json
{
  "factor_id": "string",
  "category": "string",
  "formula_fingerprint": "string — Short formula description for quick comparison",
  "data_sources": ["string"],
  "status": "string — pass | fail | pending",
  "best_icir": "float",
  "best_sharpe": "float",
  "ts": "string — ISO timestamp"
}
```
