# Shared: Context Section Templates

> 标准上下文块，按顺序注入 prompt。
> 每个 block 用大标题 `# Section Name` 分隔，帮助 LLM 区分信息类型。
>
> **关键设计**：这些 section 不是"全部注入"，而是由父进程根据 **条件注入规则** 按需选择。

---

## Section Ordering (Recommended)

```
# Role
[copy from role.md]

# Scenario Description
[当前问题域：A-share 量化因子迭代]

# Current Hypothesis
[来自 HG/HO 输出的 hypothesis JSON（FC/RC 用）]

# Current SOTA / Best Attempt
[同 category 的最佳实现：factor_id, formula, key_metrics]

# Previous Experiments and Feedbacks
[trace 摘要：按条件注入规则选择 last N 轮或全量]

# Trend Analysis
[连续同方向轮数、指标改善趋势]

# Identified Challenges
[从失败实验提取的未解决问题]

# Knowledge Base Query Results
[anti_patterns 匹配结果 + successful_patterns 参考 — 已过滤，不丢原始 JSON]

# Diff from Previous Round
[修复场景专用：代码 diff]

# Your Task
[具体指令]

# Output Format
[copy from output_formats.md]
```

---

## 条件注入规则（核心）

父进程在调用 subagent 前，根据 `failure_type` 选择 section 组合：

| failure_type | 必须注入 | 可选注入 | 省略 |
|---|---|---|---|
| `code_error` / `schema_error` | `# Previous Experiments (Last 1)` + `# Diff from Previous Round` | schema 参考 | `# Current SOTA`, `# Trend Analysis`, 更早历史 |
| `coverage_fail` | `# Previous Experiments (Last 1)` + data_sources 列表 | schema 全量 | `# Current SOTA` |
| `icir_fail` | `# Trend Analysis` (最近 3 轮) + `# Current SOTA` | `# Knowledge Base Query Results` (anti_patterns) | 更早历史 |
| `monotonicity_fail` | `# Previous Experiments (Last 2)` + construction_logic | `# Current SOTA` | 更早历史 |
| `backtest_fail` | `# Trend Analysis` (simple/detailed metrics + 策略参数历史) + `# Current SOTA` | `# Knowledge Base Query Results` | 无关 category |
| `neutralization_fail` | `# Previous Experiments (Last 1)` + variant 历史 | `# Current SOTA` | 更早历史 |
| `config_error` | `# Previous Experiments (Last 1)` + 当前 config | schema 参考 | `# Current SOTA`, `# Trend Analysis` |
| `metrics_fail` | `# Trend Analysis` (最近 3 轮) + 具体失败指标 | `# Knowledge Base Query Results` (anti_patterns) | 无关 category |
| `ridge_fail` | max_existing_factor 信息 + `# Knowledge Base Query Results` (相关成功模式) | 该对手的构造方式 | `# Trend Analysis` |
| `residual_fail` | `# Previous Experiments (Last 1)` + 残差指标 | `# Current SOTA` | 其他 |
| `execution_error` | `# Previous Experiments (Last 1)` + error | retry 指令 | 一切历史 |
| **HG 调用** | `# Current SOTA` + `# Knowledge Base Query Results` (successful_patterns) + schema | `# Identified Challenges` | trace |
| **HO 调用** | `# Knowledge Base Query Results` (full L2) + schema + `# Current SOTA` | — | trace |

**原则**：不要让 subagent 自己解析大文件。父进程做过滤和摘要，只把精炼后的信息注入 prompt。

---

## Section Templates

### `# Current SOTA for Category: {category}`

```
# Current SOTA for Category: {category}

Best Factor: {factor_id}
Formula Pattern: {formula_pattern}
Key Metrics:
  - Annual ICIR: {annual_icir}
  - Simple Sharpe: {simple_sharpe}
  - R² (Ridge): {r2}

Your new implementation must either:
- Beat SOTA on ICIR by >0.2, OR
- Achieve comparable ICIR with lower correlation to existing factors
```

---

### `# Previous Experiments (Last N Rounds)`

```
# Previous Experiments (Last {N} Rounds)

Round {N-2}: {status} | {failure_type} | ICIR={icir} | Sharpe={sharpe}
  Diagnosis: {diagnosis}
  Fix: {fix_strategy}
  Params: {tried_params}

Round {N-1}: {status} | {failure_type} | ICIR={icir} | Sharpe={sharpe}
  Diagnosis: {diagnosis}
  Fix: {fix_strategy}
  Params: {tried_params}

Round {N}: {status} | {failure_type} | ICIR={icir} | Sharpe={sharpe}
  Diagnosis: {diagnosis}
  Fix: {fix_strategy}
  Params: {tried_params}
```

**变体**：
- `N=1`：code_error / schema_error / execution_error / residual_fail
- `N=2`：monotonicity_fail / coverage_fail
- `N=3`：icir_fail / backtest_fail
- `N=0`（省略）：首次 round 或 HO 调用

---

### `# Trend Analysis`

```
# Trend Analysis

- Consecutive same-direction rounds: {X}
- Metrics improvement: {Y/N}
  - ICIR trend: {up | down | flat}
  - Sharpe trend: {up | down | flat}
- Strategy params tried: {list of (decay, rebalance, top_k) combinations}
- Factor params tried: {list of (window, horizon, variant) combinations}

Recommendation: {continue_same_direction | change_params | change_formula | abandon}
```

---

### `# Diff from Previous Round`

```
# Changes from Round {N-1}

```diff
{line_diff_between_previous_and_current_code}
```

Focus your changes ONLY on the diff lines above. Do not modify other parts of the code.
```

---

### `# Knowledge Base Query Results`

```
# Knowledge Base Query Results

## Anti-Pattern Warnings (matched by failure_type + category)
{For each matched anti_pattern (max 3):
- Pattern: {pattern}
  Category: {category}
  Signature: {signature}
  Fix: {fix}
  Seen: {count} times, last {last_seen}
}

## Successful Patterns Reference (same category, top by ICIR)
{For each successful_pattern (max 3):
- Factor: {factor_id}
  Formula: {formula_pattern}
  Metrics: ICIR={annual_icir}, Sharpe={simple_sharpe}
  Why it works: {why_it_works}
}

## Recent Failed Attempts (same category, last 5)
{For each failed_attempt (max 5):
- Factor: {factor_id}
  Failure: {failure_type}
  Best ICIR: {best_icir}
  Why failed: {why_failed}
}
```

**注意**：这是父进程过滤后的摘要，不是原始 JSON。原始文件不直接注入 prompt。

---

### `# Identified Challenges`

```
# Key Learnings and Unresolved Challenges

1. [{category}] {challenge_description} — from {factor_id} round {N}
2. [{category}] {challenge_description} — from {factor_id} round {N}
```

---

### `# Your Task`

每个 subagent 的具体任务指令，定义在各自的 prompt 文件中。

---

### `# Output Format`

引用 `shared/output_formats.md` 中对应的 JSON schema。
