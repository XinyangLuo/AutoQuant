# Shared: Context Section Templates

> 标准上下文块，按顺序注入 prompt。
> 每个 block 用大标题 `# Section Name` 分隔，帮助 LLM 区分信息类型。

## Section Ordering (Recommended)

```
# Role
[copy from role.md]

# Scenario Description
[当前问题域：A-share 量化因子迭代]

# Current SOTA / Best Attempt
[同 category 的最佳实现：factor_id, formula, key_metrics]

# Previous Experiments and Feedbacks
[trace.jsonl 摘要：最近 3 轮的 failure_type, metrics, diagnosis]

# Identified Challenges
[从失败实验提取的未解决问题：Key Learnings and Unresolved Challenges]

# Knowledge Base Query Results
[anti_patterns 匹配结果 + successful_patterns 参考]

# Your Task
[具体指令]

# Output Format
[copy from output_formats.md]
```

## SOTA Block Template

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

## Trace Summary Block Template

```
# Previous Experiments (Last 3 Rounds)

Round {N-2}: {status} | {failure_type} | ICIR={icir} | Sharpe={sharpe}
  Diagnosis: {diagnosis}
  Fix: {fix_strategy}

Round {N-1}: {status} | {failure_type} | ICIR={icir} | Sharpe={sharpe}
  Diagnosis: {diagnosis}
  Fix: {fix_strategy}

Round {N}: {status} | {failure_type} | ICIR={icir} | Sharpe={sharpe}
  Diagnosis: {diagnosis}
  Fix: {fix_strategy}

# Trend Analysis
[连续同方向轮数: X | 指标改善: Y/N | 建议: ...]
```

## Diff Block Template (Repair Scenarios)

```
# Changes from Round {N-1}

```diff
[line diff between previous and current code]
```

Focus your changes ONLY on the diff lines above. Do not modify other parts of the code.
```

## Challenges Block Template

```
# Key Learnings and Unresolved Challenges

1. [{category}] {challenge_description} — from {factor_id} round {N}
2. [{category}] {challenge_description} — from {factor_id} round {N}
```
