# Multi-Agent 自动因子挖掘 — 实施计划

> **核心理念**：用 Claude Code subagent（Agent tool，非 Python agent）做决策，Python 层只管执行。先跑通最小闭环，再逐步加复杂度。

## 1. 目标

用 Claude Code 的 subagent 能力，自动化「生成假设 → 写因子代码 → 执行 pipeline → 诊断失败 → 修复重试」的因子迭代循环，并积累跨 run 知识。

## 2. 架构总览（演进终态）

```
Claude Code 父进程
  ├── /factor-iterate    增强版单方向迭代（Phase 1）
  ├── /factor-explore    多方向并行探索（Phase 3+）
  └── /library-audit     库健康审计（Phase 4+）

Subagent 类型（2 个，非 5 个）：
  ├── Factor Coder (FC)   写代码 + 执行 claude_cli run
  └── Result Critic (RC)  诊断失败 + 查 KB + 决定 repair/abandon

Knowledge Base（3 个文件，非 8 个）：
  agents/knowledge_base/
    ├── anti_patterns.json      反模式库
    ├── successful_patterns.json 成功模式库
    └── failed_attempts.jsonl          失败实验记录（仅失败，用于学习错误建模）
```

**方向选择 + 假设生成**不拆分独立 agent，由父进程（Claude Code 对话）直接完成。审计能力推迟到有 >10 个 admitted factor 后再建。

## 3. Phase 1：最小可行骨架 ✅（已完成，2026-05-30）

### 3.1 范围

**只做一件事**：增强现有 `/factor-iterate`，在修复环节引入 RC subagent + KB 查询。

**不做的事**（留给后续 Phase）：
- 不建 scheduler.py / orchestrator.py / knowledge.py
- 不加 bandit 方向选择
- 不并行探索
- 不加 claude_cli 新子命令
- 不写 subagent 系统 prompt 文件（先用内联 prompt）

**实际完成情况**：
- ✅ 端到端验证：3 轮 vol_reversal 放弃 → RC 正确诊断 backtest_fail + neutralization_fail
- ✅ 成功案例：`-ts_std(turnover_rate, 20)` 1 轮 pass → ICIR=2.62, Sharpe=0.989, 残差入库
- ✅ KB bootstrap：successful_patterns 含 10 个 admitted 因子，anti_patterns 含 1 条反模式
- ✅ FC 增强：config.yaml 自动生成、价格复权/ST/涨跌停/财务季度/成交量单位 5 条数据 pitfall
- ✅ Pipeline report 对 agent 路径可见（result.json.report_path + round dir 副本）
- ✅ StepThresholds 补全 detailed max_drawdown + max_annual_turnover

### 3.2 改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `agents/knowledge_base/` | **新建目录** | 含 `anti_patterns.json`、`successful_patterns.json`、`failed_attempts.jsonl` 的空 schema 文件 |
| `.claude/commands/factor-iterate.md` | **改** | Round loop 中：FC 写完代码 → RC subagent 诊断 → 根据 RC 输出决定 repair/abandon/pass |
| `agents/claude_cli.py` | 不改 | 继续用 `schema` + `run` |
| `agents/runner.py` 等 | 不改 | 继续用现有执行层 |

### 3.3 Round Loop（增强后）

```
每个 round:
  1. [FC] 生成/修复因子代码 → Write factor.py
  2. [FC] Bash: claude_cli run → 等返回
  3. [FC] Read result.json
  4. [父进程] 如果 pass → 结束循环（写入 successful_patterns）
  5. [父进程] 如果 fail → 启动 RC subagent 诊断
  6. [RC]   Read result.json + trace.jsonl + anti_patterns.json
  7. [RC]   输出诊断 JSON（failure_type / fix_strategy / same_direction / recommend_abandon）
  8. [父进程] 追加 trace.jsonl
  9. [父进程] 根据 RC 输出：repair → 回到 step 1 / abandon → 结束
```

### 3.4 RC Subagent 调用方式

通过 Claude Code 的 `Agent` 工具，每次 fail 后启动一个一次性 subagent：

```
Agent tool params:
  description: "诊断因子失败原因并给出修复建议"
  subagent_type: "general-purpose"
  prompt: |
    你是 Result Critic，负责诊断量化因子 pipeline 失败的原因。

    ## Context
    - 原始假设: {hypothesis_summary}
    - 本轮 round: {round_num} / {max_rounds}
    - 当前 factor_id: {factor_id}
    - 之前尝试: {trace_summary}

    ## 输入文件
    - result.json: Read results/{factor_id}/{strategy}/result.json
    - trace.jsonl: Read results/{run_id}/trace.jsonl
    - 反模式库: Read agents/knowledge_base/anti_patterns.json
    - 成功模式库: Read agents/knowledge_base/successful_patterns.json

    ## 任务
    1. 读取 result.json，确认 failure_type 和关键指标
    2. 查询反模式库：是否有匹配当前 failure_type + category 的已知模式？
    3. 查询成功模式库：同 category 的 SOTA 指标是多少？
    4. 输出诊断 JSON（见 OutputFormat）

    ## OutputFormat
    {
      "failure_type": "...",
      "diagnosis": "根因分析，一段话",
      "fix_strategy": "具体修复建议",
      "same_direction": true/false,
      "repair_params": {...},
      "recommend_abandon": true/false,
      "new_anti_pattern": null 或 {新增的反模式记录}
    }

    ## Decision Rules
    - code_error / schema_error → same_direction=true，只修代码
    - coverage_fail → same_direction=true，改数据源或放宽条件
    - icir_fail → 先查反模式：同 category 的历史 icir_fail 怎么修的？如果反模式库无匹配 → same_direction=true 尝试不同窗口/horizon
    - ridge_fail → 查 max_existing_corr：如果 >0.85 且是已有 alpha → recommend_abandon=true；如果是 Barra L1 → same_direction=true 换构造方式
    - residual_fail → recommend_abandon=true（无增量信息）
    - 连续 3 轮同方向无改善 → recommend_abandon=true
```

### 3.5 KB 文件 Schema

**`anti_patterns.json`** — 反模式库（手动 bootstrap，后续自动积累）：
```json
{
  "icir_fail": [
    {
      "pattern": "窗口过长导致反转因子失效",
      "category": "volume_reversal",
      "signature": "volume_window > 20",
      "fix": "缩窗到 5-10 天",
      "count": 1,
      "last_seen": "2026-05-30"
    }
  ],
  "ridge_fail": [
    {
      "pattern": "和 Barra 动量强相关",
      "category": "momentum",
      "signature": "max_corr_with Barra_momentum > 0.9",
      "fix": "换中间 horizon 或改用残差动量",
      "count": 1,
      "last_seen": "2026-05-30"
    }
  ]
}
```

**`successful_patterns.json`** — 成功模式库：
```json
{
  "volume_reversal": [
    {
      "factor_id": "f_vol_rev_5d",
      "formula_pattern": "rank(ts_mean(amount, 5)) * (-1) * rank(ret_5d)",
      "key_metrics": {"annual_icir": 1.55, "simple_sharpe": 0.95},
      "why_it_works": "零售投资者过度反应导致短期反转，放量确认参与度",
      "admission_date": "2026-05-15"
    }
  ]
}
```

**`failed_attempts.jsonl`** — 仅失败记录（append-only）：
```json
{"factor_id": "f_auto_xxx", "run_id": "...", "category": "momentum", "data_sources": ["market_daily"], "status": "fail", "best_icir": 1.55, "best_sharpe": 0.45, "failure_type": "backtest_fail", "code_summary": "公式简述", "why_failed": "根因一句话", "ts": "2026-05-30T10:00:00"}
```

### 3.6 验证结果 ✅

1. ✅ **已知 pass 因子变体**：`-ts_std(turnover_rate, 20)` 1 轮 pass，RC 未触发（符合预期）
2. ✅ **已知 fail 因子**：vol_reversal 3 轮放弃，RC 正确诊断 backtest_fail + neutralization_fail，barra_l3 死胡同识别准确
3. ✅ **KB 积累**：anti_patterns 1 条（variant_switch），successful_patterns 10 条（7 Barra L1 + 3 user alphas），failed_attempts 1 条

### 3.7 不做的事（明确排除）

- ❌ 不新增 Python 模块（knowledge.py / scheduler.py / orchestrator.py）
- ❌ 不加 CLI 子命令（kb-query / run-index / admit-correlations）
- ❌ 不写 subagent 系统 prompt 文件（RC prompt 内联在 slash command 里）
- ❌ 不做并行探索
- ❌ 不做库审计

## 4. Phase 2：KB 积累 + 自动引导（待定）

**触发条件**：Phase 1 跑通 ≥20 次迭代，KB 有 ≥10 条反模式、≥3 条成功模式。

- 父进程在 framing 阶段自动查 KB：反模式 → 避免已知坑；成功模式 → 参考公式模板
- 将 RC 的修复建议从「一次性 subagent」升级为「可追溯的 KB 查询 + 内联决策」
- 考虑把 RC 的内联 prompt 抽到 `.claude/prompts/result_critic.md`

## 5. Phase 3：多方向并行探索（待定）

**触发条件**：Phase 2 稳定运行，单方向成功率（最终 pass）> 20%。

- 父进程手动指定 2 个方向（如「volume_reversal + fund_flow」）
- 每个方向独立跑 `/factor-iterate`（不同 run dir、不同 factor_id）
- 通过 Claude Code 的 `run_in_background` 并行
- 验证 work DB 并发安全
- 评估 token 消耗是否可控

## 6. Phase 4+：自动化 + 审计（远期）

**触发条件**：Phase 3 验证通过，admitted factor > 10。

- 库审计（冗余检测 / 覆盖缺口 / 性能衰减）
- 自动触发探索（audit → 发现覆盖缺口 → 自动启动该方向迭代）
- Bandit 方向选择（如果方向数 > 5 且手动选择不可持续）

## 7. 设计原则（贯穿所有 Phase）

1. **Claude Code subagent 做决策，Python 做执行**：不建独立的 agent 循环
2. **文件是最简单的 IPC**：所有 agent 间通信走 JSON/JSONL 文件
3. **先有数据再建系统**：每个 Phase 的触发条件都是上一个 Phase 的产出数据，不基于猜测建架构
4. **向内联倾斜**：prompt、决策逻辑优先内联在 slash command 里；只有复用 ≥3 次时才抽到独立文件
5. **不超过 3 个 subagent 类型**：在当前阶段，2 个（FC + RC）就够了
