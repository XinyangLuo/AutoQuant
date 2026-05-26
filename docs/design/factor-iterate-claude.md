# Factor Iterate — 纯 Claude Code 驱动方案

## 1. 目标

用户在 Claude Code 中输入一个自然语言因子猜测，Claude 自动完成「代码生成 → 回测 → 分析 → 修复 → 再回测」的迭代循环，直到因子达标或达到轮数上限。

**不需要维护额外的 Python agent 框架**，Claude 直接操作文件和命令行工具。

## 2. 架构

```
用户 (Claude Code)
   |
   v
/factor-iterate "成交额放量后短期反转"
   |
   v
Claude Agent（主循环）
   |
   +-- Write 因子代码 → alphas/exp/agent/<factor_id>/factor.py
   +-- Bash 跑单轮流水线 → claude_cli.py
   +-- Read 结果 JSON
   +-- 分析 PASS/FAIL
   +-- 读 trace.jsonl 避免重复
   +-- 决策：修复 / 调参 / 换方向 / 停止
   |
   v
Trace 文件（唯一持久化状态）
   results/agent/runs/<run_id>/trace.jsonl
```

## 3. 核心设计决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 决策层 | Claude 自己分析 | 纯 Claude 方案，无额外 Python 框架 |
| 执行层 | 复用现有 backtest 流水线 | 不重复造轮子 |
| 状态持久化 | JSONL trace 文件 | 简单、可读、Claude 每轮读取 |
| 代码生成 | Claude 直接 Write 文件 | 不需要 H2E 转换层 |
| 列名注入 | Python schema resolver + Claude prompt | 避免 hallucinate |
| 同方向迭代 | Claude prompt 规则约束 | 代码错误时禁止换方向 |

## 4. 文件结构

```
.claude/
  commands/
    factor-iterate.md          # slash command 入口

agents/
  claude_cli.py                # 单轮流水线 CLI（输出结构化 JSON）

results/agent/runs/
  <run_id>/
    hypothesis.md              # 用户原始猜测
    trace.jsonl                # 迭代历史（Claude 读/写）
    round_001/
      factor.py                # 因子代码
      result.json              # 回测结果
    round_002/
      ...
```

## 5. 单轮流水线 CLI

`agents/claude_cli.py run <factor_id> --run-dir <dir>` 执行：

1. `backtest.factor.compute.compute_factor()` → backfill
2. `backtest.factor.evaluate()` → IC/RankIC/turnover/corr
3. `SingleFactorStrategy + SimpleSimulator` → simple BT
4. `DetailedSimulator` → detailed BT（conditional）
5. 输出 `result.json`：

```json
{
  "factor_id": "f_auto_001",
  "status": "pass|fail|error",
  "error": "NameError: abs_",
  "metrics": {
    "rankicir": 0.32,
    "ic_positive_ratio": 0.65,
    "turnover": 0.28,
    "max_corr": 0.15,
    "simple_sharpe": 1.1,
    "simple_mdd": -0.12
  }
}
```

## 6. Trace 格式

`trace.jsonl`（每轮 append 一行）：

```json
{
  "round": 1,
  "factor_id": "f_auto_run001_001",
  "status": "fail",
  "failure_type": "code_error",
  "error": "NameError: abs_",
  "diagnosis": "abs_ 不在 transforms 中，应使用 abs",
  "fix_strategy": "替换 abs_ 为 abs",
  "code_summary": "momentum * volume spike, 20-day window",
  "tried_params": {"horizon": 5, "top_pct": 0.1, "window": 20},
  "metrics": {},
  "same_direction": true
}
```

Claude 每轮开始前读取全部 trace，做决策时检查：
- 是否重复过相同的 error？
- 是否试过相同的参数组合？
- 同方向已连续失败几轮？

## 7. 迭代流程

```
Round 1:
  1. Claude 根据 hypothesis 生成代码
  2. Write → alphas/exp/agent/f_auto_001/factor.py
  3. Bash → claude_cli.py run f_auto_001
  4. Read → result.json
  5. 分析：FAIL (code_error)
  6. 写 trace.jsonl（第 1 行）
  7. 决策：same_direction=true，修复代码

Round 2:
  1. Claude 读 trace.jsonl（知道 Round 1 的失败原因）
  2. 生成修复后的代码
  3. Write → alphas/exp/agent/f_auto_002/factor.py
  4. Bash → claude_cli.py run f_auto_002
  5. 分析：FAIL (weak_signal, RankICIR=0.15)
  6. 写 trace.jsonl（第 2 行）
  7. 决策：same_direction=true，调参 horizon 5→10

Round 3:
  1. Claude 读 trace（知道已试过 horizon=5，代码已修复）
  2. 生成调参后的代码
  3. ... → PASS (RankICIR=0.32, Sharpe=1.1)
  4. 写 trace（第 3 行，status=pass）
  5. 报告候选因子，结束
```

## 8. Claude 决策规则

写在 `.claude/commands/factor-iterate.md` 中的核心规则：

**失败分类**：
- `code_error`：SyntaxError、NameError、TypeError 等
- `schema_error`：KeyError（列名不存在）
- `weak_signal`：RankICIR 低于阈值
- `high_turnover`：换手高于阈值
- `high_corr`：与已有因子相关性过高

**修复策略**：
- `code_error` / `schema_error` → **必须 same_direction=true**，只修复代码
- `weak_signal` → same_direction=true，调 horizon/window/构造；若已调 3 轮仍不达标，允许换方向
- `high_turnover` → same_direction=true，加 decay/smooth
- `high_corr` → same_direction=true，换构造方式

**停止条件**：
- 因子通过全部阈值
- 达到 max_rounds（默认 10）
- 用户输入 "停止"
- 连续 3 轮 same_direction 失败且无明显进展

## 9. 列名精准注入

在 `claude_cli.py` 中提供 schema 查询命令：

```bash
python -m agents.claude_cli schema --sources market_daily,income_q
```

输出可用列名列表。Claude 在生成代码前先查 schema，避免 hallucinate。

常见映射表（内置在 CLI 中）：
- `buy_sm` → `mf_buy_sm_amount`
- `ts_zscore` → `z_score`
- `cs_rank` → `rank`

## 10. 与原有 Python Agent 的关系

原 RD-Agent Python agent 循环已移除。Factor Iterate (Claude Code) 现在是唯一的 agent 模式。

| 触发方式 | `/factor-iterate "..."` |
| 决策层 | Claude Code 直接分析 |
| 执行层 | `python -m agents.claude_cli run` |
| 输出 | 候选因子 + trace.jsonl |

## 11. 验证计划

1. **单元测试**：用已知 bug 的因子（`abs_` 未定义）验证 Claude 能正确分类为 code_error 并修复
2. **集成测试**：跑 3-round 完整链路，验证 trace.jsonl 累积、结果正确
3. **手动验证**：在 Claude Code 中执行 `/factor-iterate "test reversal on volume"`

## 12. 风险与回退

| 风险 | 缓解 |
|---|---|
| Claude 决策不一致 | trace.jsonl 提供明确历史，减少上下文漂移 |
| 会话中断 | trace.jsonl 已落盘，可手动恢复 |
| 死循环 | max_rounds + 连续 3 轮失败强制停止 |
| 列名 hallucinate | schema 查询 + 常见映射表 |
| 代码质量差 | 每轮代码都通过 `ast.parse()` 语法校验 |
