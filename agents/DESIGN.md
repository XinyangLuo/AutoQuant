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

Subagent 类型（3 个，演进中）：
  ├── HypothesisGen (HG)  生成假设 + 5 维自评（alignment/impact/novelty/feasibility/risk-reward）
  ├── Factor Coder (FC)   写代码 + 执行 claude_cli run
  └── Result Critic (RC)  诊断失败 + 查 KB + 决定 repair/abandon

Knowledge Base（3 文件 → 混合检索）：
  Phase 1~2: JSON 文件（轻量）
    agents/knowledge_base/
      ├── anti_patterns.json
      ├── successful_patterns.json
      └── failed_attempts.jsonl
  Phase 3+: 向量检索（DuckDB vss）
    └── factors_pending.duckdb 中 embedding + HNSW 索引

Trace（线性 → DAG）：
  Phase 1: 线性 trace.jsonl（append-only）
  Phase 2+: DAG trace（支持 parent_round_id / branch_id，允许多分支并行）
```

**方向选择**由父进程直接完成（不拆独立 agent）；**假设生成**在 Phase 1 后期拆出 HG subagent。审计能力推迟到有 >10 个 admitted factor 后再建。

## 3. Phase 1：最小可行骨架（当前）

### 3.1 范围

**只做一件事**：增强现有 `/factor-iterate`，在修复环节引入 RC subagent + KB 查询。

**不做的事**（留给后续 Phase）：
- 不建 scheduler.py / orchestrator.py / knowledge.py
- 不加 bandit 方向选择
- 不并行探索
- 不加 claude_cli 新子命令
- 不写 subagent 系统 prompt 文件（先用内联 prompt）

### 3.2 改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `agents/knowledge_base/` | **新建目录** | 含 `anti_patterns.json`、`successful_patterns.json`、`failed_attempts.jsonl` 的空 schema 文件 |
| `.claude/prompts/` | **新建目录** | Prompt 模板骨架：`shared/`（role/output_format/context）+ `factor_coder.md` + `result_critic.md` + `hypothesis_gen.md` |
| `.claude/commands/factor-iterate.md` | **改** | Round loop 中：FC 写完代码 → RC subagent 诊断 → 根据 RC 输出决定 repair/abandon/pass；后续 P1.A.0 将内联 prompt 迁移到 `.claude/prompts/` |
| `agents/claude_cli.py` | 不改 | 继续用 `schema` + `run` |
| `agents/runner.py` 等 | 不改 | 继续用现有执行层 |

### 3.3 Prompt 模板系统（Phase 1 增强）

参考 RD-Agent 的 Jinja2 YAML 共享块思路，AutoQuant 用更轻量的 Markdown 文件级复用。

目录结构：

```
.claude/prompts/
├── shared/
│   ├── role.md              # FC/RC/HG 角色定义模板
│   ├── output_formats.md    # JSON schema 模板（hypothesis / diagnosis / trace）
│   └── context_sections.md  # 标准上下文块（SOTA / Trace / Challenges / Diff）
├── factor_coder.md          # FC system prompt（从 factor-iterate.md §4-5 抽取）
├── result_critic.md         # RC system prompt（从 factor-iterate.md §9 抽取）
└── hypothesis_gen.md        # HG prompt（新增）
```

设计模式（参考 RD-Agent）：
- **Role Anchoring**：每个 system prompt 开头固定 persona（"You are a quantitative factor researcher..."）
- **Section Labeling**：用大标题明确每个上下文块（`# Scenario Description` / `# Current SOTA` / `# Previous Experiments` / `# Identified Challenges`）
- **SOTA 锚定**：每次生成都强制对比当前 category 的最佳实现（ICIR/Sharpe/公式）
- **结构化输出**：用 Pydantic-style JSON schema 注入 prompt，要求纯 JSON 返回
- **Diff 注入**：repair 场景下给 diff 而非完整代码（P2.A.3 实施）

> **实施说明**：P1.A.0 把现有 `factor-iterate.md` 内联 prompt 逐步迁移到独立文件；迁移期间两者并存，验证稳定后再删除内联版本。

### 3.4 Hypothesis Generation（Phase 1 后期）

在 FC 编码之前增加显式假设生成步骤，避免"直接写代码"的盲目性。

**Round Loop 更新后**：

```
每个 round:
  0. [HG] 生成/修正假设 → 输出 hypothesis JSON（Round 1 新建，后续 round 可复用）
     父进程审核假设合理性（alignment/impact/novelty/feasibility/risk-reward）
  1. [FC] 根据假设生成/修复因子代码 → Write factor.py
  2. [FC] Bash: claude_cli run → 等返回
  3. [FC] Read result.json
  4. [父进程] 如果 pass → 结束循环（写入 successful_patterns）
  5. [父进程] 如果 fail → 启动 RC subagent 诊断
  6. [RC]   Read result.json + trace.jsonl + anti_patterns.json
  7. [RC]   输出诊断 JSON（failure_type / fix_strategy / same_direction / recommend_abandon）
  8. [父进程] 追加 trace.jsonl
  9. [父进程] 根据 RC 输出：repair → 回到 step 1 / abandon → 结束
```

**Hypothesis JSON schema**：

```json
{
  "hypothesis_text": "成交额放量后 5 天短期反转，小盘股（circ_mv 后 30%）更强",
  "category": "volume_reversal",
  "data_sources": ["market_daily"],
  "construction_logic": "1) 5 日成交额 zscore > 1 视为放量; 2) 5 日收益 rank 取负; 3) 小盘股加权",
  "expected_icir": 1.2,
  "alignment_score": 0.8,
  "impact_score": 0.7,
  "novelty_score": 0.6,
  "feasibility_score": 0.9,
  "risk_reward_score": 0.7
}
```

父进程审核规则：
- `feasibility_score < 0.5` → 要求 HG 重写
- `novelty_score < 0.3` 且 successful_patterns 中已有高度相似模式 → 要求换方向
- `alignment_score < 0.5`（假设与 category 不匹配）→ 要求修正 category 或重新表述

### 3.5 Round Loop（增强后）

> **注意**：3.5 为原 3.3 Round Loop 的更新版，已整合 HG 步骤（见 3.4）。RC Subagent 调用方式和 Decision Rules 保持不变，详见 3.6。

```
每个 round:
  0. [HG/父进程] 假设生成与审核（Round 1 必须，后续可选修正）
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

### 3.6 RC Subagent 调用方式

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
    - result.json: Read results/<factor_id>/<strategy>/result.json
    - trace.jsonl: Read results/<run_id>/trace.jsonl
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
      "fix_level": "factor|strategy_only|both|retry",
      "factor_params": {},
      "strategy_params": {},
      "same_direction": true/false,
      "recommend_abandon": true/false,
      "new_anti_pattern": null 或 {新增的反模式记录}
    }

    ## Decision Rules
    - code_error / schema_error → fix_level="factor"，只修代码
    - coverage_fail → fix_level="factor"，改数据源或放宽条件
    - icir_fail → fix_level="factor"，先查反模式：有匹配→采用其 fix；无匹配→改窗口/horizon
    - monotonicity_fail → fix_level="factor"，加二次过滤或改构造方式
    - config_error → fix_level="strategy_only"
    - backtest_fail → 按差距分级：Sharpe≥70%阈值且ICIR达标→fix_level="strategy_only"；否则→fix_level="factor"。另可参考相对基准超额指标（vs HS300/CSI500/CSI1000）判断因子是否因风格暴露而失败
    - ridge_fail → 查 max_existing_corr：>0.85且用户alpha→recommend_abandon=true；Barra L1→fix_level="factor"换构造方式
    - residual_fail → recommend_abandon=true（无增量信息）
    - execution_error → fix_level="retry"
    - 连续 3 轮同方向无改善 → recommend_abandon=true
```

### 3.7 KB 文件 Schema

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
      "key_metrics": {"annual_icir": 1.55, "simple_sharpe": 0.95, "excess_sharpe_hs300": 0.72, "excess_sharpe_csi500": 0.68, "excess_sharpe_csi1000": 0.55},
      "why_it_works": "零售投资者过度反应导致短期反转，放量确认参与度",
      "admission_date": "2026-05-15"
    }
  ]
}
```

**`failed_attempts.jsonl`** — 每行一个 run（append-only）：
```json
{"factor_id": "f_auto_xxx", "run_id": "...", "category": "momentum", "data_sources": ["market_daily"], "status": "pass", "rounds": 3, "best_icir": 1.55, "best_sharpe": 0.95, "ts": "2026-05-30T10:00:00"}
```

### 3.8 验证标准

Phase 1 做完后，用以下场景验证：

1. **已知 pass 因子变体**：故意引入一个 code_error，验证 RC 能正确诊断并给出 fix，下一轮修复成功
2. **已知 fail 因子**：用历史上确定不过的因子，验证 RC 能正确 recommend_abandon
3. **KB 积累**：连续跑 3 个迭代后，检查 `failed_attempts.jsonl` 和 `anti_patterns.json` 是否正确写入

### 3.9 不做的事（明确排除）

- ❌ 不新增 Python 模块（knowledge.py / scheduler.py / orchestrator.py）
- ❌ 不加 CLI 子命令（kb-query / run-index / admit-correlations）
- ❌ ~~不写 subagent 系统 prompt 文件~~ → **已改为**：Prompt 模板建 `.claude/prompts/` 骨架，但完整 prompt 迁移留在 P1.A.0
- ❌ 不做并行探索
- ❌ 不做库审计

## 4. Phase 2：KB 积累 + 自动引导 + 工程化增强（待定）

**触发条件**：Phase 1 跑通 ≥20 次迭代，KB 有 ≥10 条反模式、≥3 条成功模式。

### 4.1 KB 自动引导

- 父进程在 framing 阶段自动查 KB：反模式 → 避免已知坑；成功模式 → 参考公式模板
- 将 RC 的修复建议从「一次性 subagent」升级为「可追溯的 KB 查询 + 内联决策」
- RC prompt 已抽到 `.claude/prompts/result_critic.md`（P1.A.0 完成）

### 4.2 Diff 注入（P2.A.3）

Repair 场景下，FC subagent 的 prompt 中不再给出完整代码，而是给出**上一轮代码 → 本轮目标代码的 diff**：

```
# Changes from Round 2
```diff
- ts_mean(amount, 20)
+ ts_mean(amount, 10)
- rank(ret_20d)
+ rank(ret_5d)
```
```

省 token（diff 通常 < 完整代码的 20%），同时强制 FC 聚焦修改点而非重写全部。

实现：父进程维护 `sota_files`（当前最佳尝试的文件内容），与当前 `files` 做 line diff，注入 prompt。

### 4.3 QuantFeedback 多层拆分（P2.A.2）

当前单层 `QuantFeedback` 拆为三层，对应 RD-Agent 的 Execution → Evaluation → Hypothesis Feedback：

| 层级 | 来源 | 内容 | 消费方 |
|------|------|------|--------|
| `execution` | runner.py / claude_cli | 运行报错、traceback、DuckDB 锁、OOM | RC（判断 fix_level="retry"） |
| `evaluation` | PipelineState step1~step10 | 各 step pass/fail、ICIR、Sharpe、R²、max_corr | RC（判断 failure_type + fix 策略） |
| `hypothesis` | HG / 父进程 | 假设的 alignment/impact/novelty/feasibility/risk-reward | HG 自评 + 父进程审核 |

`AutoQuantFactorEvaluator` 负责汇总三层为结构化 `QuantFeedback`，RC subagent 消费 evaluation 层。

### 4.4 Workspace Checkpoint / Rollback（P3.A.2，可提前到 Phase 2 后期）

引入轻量 `FactorWorkspace`：

```python
@dataclass
class FactorWorkspace:
    files: dict[str, str]  # factor.py + config.yaml
    def checkpoint(self, run_dir: Path, round: int): ...
    def rollback(self, run_dir: Path, round: int): ...
```

- `checkpoint`：zip 打包当前文件到 `results/{run_id}/checkpoints/{round:03d}.zip`
- `rollback`：从 checkpoint 恢复，用于"改坏了需要回退到上一轮"

参考 RD-Agent `FBWorkspace` 的 `create_ws_ckp` / `recover_ws_ckp`，但不做 Docker 隔离（本地 conda 执行是 feature 而非 bug）。

### 4.5 研报 PDF → Hypothesis（P2.A.4）

通过 PDF-MCP server 让 Agent 读取券商/学术研报 PDF，从研报论点中自动提取量化因子灵感并生成 hypothesis。

**架构约束**：不依赖模型原生多模态能力。MCP server 负责 PDF 文本提取（`pdfplumber` / `pymupdf` / `pdfminer`），LLM 消费的是**提取后的纯文本**（含表格 markdown），不传 PDF 二进制或页面截图给模型。

**工作流（hypothesis.md 中介模式）**：

```
/pdf-hypothesis
    │
    ├── 读 PDF → 穷举因子 → 按 Sharpe/IR 排序 → 排名表
    ├── 用户选因子
    └── 输出: agents/pdf_hypotheses/<slug>/<factor_name>_hypothesis.md

用户审阅 hypothesis.md（可手动修改公式/参数/config）

/factor-iterate --hypothesis <path>
    │
    ├── 解析 hypothesis.md → FC 编码 → run → RC 诊断 → repair/abandon
    └── 因子代码 → alphas/exp/agent/<factor_id>/factor.py
```

**关键设计决策**：

- `/pdf-hypothesis` 和 `/factor-iterate` 不直接相互调用。`hypothesis.md` 是两者之间的**唯一契约**。
- hypothesis.md 让用户有机会在跑 pipeline 之前审阅和修改公式/参数/config，避免盲目执行。
- 因子源码始终写入 `alphas/exp/agent/<factor_id>/`，不在 `results/` 下写源码。

**选定方案：`mcp-pdf`（rsp2k）**

| 维度 | 详情 |
|------|------|
| **安装** | `pip install mcp-pdf`（已加入 `environment.yml`） |
| **底层库** | PyMuPDF → pdfplumber → pypdf **自动 fallback** |
| **关键工具** | `pdf_to_markdown`（正文+表格→markdown）、`extract_text`、`extract_tables`、`get_metadata` |
| **中文支持** | ✅ PyMuPDF + pdfplumber 原生支持 |
| **表格** | ⭐⭐⭐⭐⭐ 三引擎（Camelot/pdfplumber/Tabula），研报数据表格无压力 |
| **许可证** | 开源免费 |

**Claude Code 配置**（项目根 `.mcp.json`，已创建）：

```json
{
  "mcpServers": {
    "pdf": {
      "command": "conda",
      "args": ["run", "-n", "AutoQuant", "mcp-pdf"]
    }
  }
}
```

配置后，Agent 可通过 `pdf_to_markdown` 工具读取研报，获取纯文本 markdown（含表格），LLM 只消费文本、不传 PDF 二进制。

**与标准 HG（§3.4）的关系**：

| 维度 | 标准 HG | PDF → HG（本功能） |
|------|---------|---------------------|
| 输入来源 | 用户自然语言描述 / trace 上下文 | 研报 PDF 文本（via mcp-pdf） |
| 触发方式 | 每个 `/factor-iterate` round 1 必须执行 | 通过 `/pdf-hypothesis <path>` 独立触发 |
| 输出格式 | 同 §3.4 hypothesis JSON schema | 同左，但可批量生成多条（一篇研报可能包含多个独立论点） |
| 审核门槛 | `feasibility < 0.5` → 重写 | 额外检查：论点是否有足够数据支撑？是否与 KB 成功/失败模式冲突？ |
| Prompt 模板 | `.claude/prompts/hypothesis_gen.md` | `.claude/prompts/pdf_hypothesis.md` |

**典型使用场景**：

```text
# 第一步：从研报提取因子
/pdf-hypothesis research_papers/华泰多因子系列之四.pdf
# → 排名表 → 选因子 → 生成 hypothesis.md

# 第二步：审阅 hypothesis.md（可手动修改公式/参数）

# 第三步：执行迭代
/factor-iterate --hypothesis agents/pdf_hypotheses/<slug>/xxx_hypothesis.md
```

## 5. Phase 3：多方向并行探索（待定）

**前置条件**：P2.A.1 Trace DAG 结构已完成（`parent_round_id` + `branch_id` 字段写入）。

**触发条件**：Phase 2 稳定运行，单方向成功率（最终 pass）> 20%。

### 5.1 并行策略

- 父进程手动指定 2 个方向（如「volume_reversal + fund_flow」）
- 每个方向独立跑 `/factor-iterate`（不同 run dir、不同 factor_id、不同 `branch_id`）
- 通过 Claude Code 的 `run_in_background` 并行
- 验证 work DB 并发安全
- 评估 token 消耗是否可控

### 5.2 DAG Trace 支持

Trace 从线性升级为 DAG：

```json
{"round": 1, "branch_id": "main", "parent_round": null, ...}
{"round": 2, "branch_id": "main", "parent_round": 1, ...}
{"round": 1, "branch_id": "fund_flow", "parent_round": 1, ...}  // fork from main round 1
```

- `branch_id` 标识并行分支（`main` / `fund_flow` / `momentum` 等）
- `parent_round` 指向分叉的父节点（`null` 表示该分支起点）
- SOTA 查找按 `branch_id` 隔离（各分支独立进化），但 `successful_patterns` 全局共享
- Phase 1 即可写入 DAG 字段（`branch_id` 固定为 `"main"`，`parent_round` 为上一 round），Phase 3 才真正利用分支能力

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
5. **不超过 3 个 subagent 类型**：当前演进至 3 个（HG + FC + RC），不再增加
6. **Prompt 优先结构化模板化**：参考 RD-Agent 的 Jinja2 YAML 共享块模式，用轻量 Markdown 文件级复用实现（Role Anchoring + Section Labeling + SOTA 锚定 + 结构化 JSON schema）。prompt 质量是迭代成功率的上限。

## 8. RD-Agent 对比参考（设计决策上下文）

> 本节记录与 RD-Agent 的功能模块对比结论，作为后续设计决策的上下文。**不照搬 RD-Agent**，只取其适合 AutoQuant 的思路。

### 8.1 差距总览

| 维度 | RD-Agent（成熟框架） | AutoQuant 当前状态 | 应对策略 |
|------|----------------------|-------------------|----------|
| Core Abstractions | Scenario → Task → Experiment → Workspace → Developer → Evaluator → Trace (DAG) | 只有简单 `AutoQuantFactorExperiment` + `QuantFeedback` + 线性 `trace.jsonl` | 渐进引入：HG（假设层）→ DAG Trace → Workspace |
| Prompt 工程 | Jinja2 YAML 模板 + `share.yaml` 共享块 + 层级覆盖 | 全部内联在 `factor-iterate.md` | 轻量 Markdown 文件级复用（P1.A.0） |
| 提案流水线 | HypothesisGen → Hypothesis2Experiment → ExpGen → Experiment2Feedback，含自批判 + 加权选择 | 父进程直接决策，无显式假设层 | 先加 HG 步骤（P1.A.0b），后续再评估是否需要 Hypothesis2Experiment |
| Feedback 层级 | Execution → Evaluation → Hypothesis（5 维评分） | 只有 PipelineState pass/fail + 指标 | 拆三层（P2.A.2），Hypothesis 层简化版 |
| 知识管理 | 向量库 + 知识图谱 + CoSTEER 三路 RAG | 3 个 JSON 文件 | Phase 2 先 JSON + 简单匹配；Phase 3+ DuckDB vss |
| Workspace | `FBWorkspace`（文件字典 → Docker），zip checkpoint | 直接写磁盘 | 轻量 `FactorWorkspace`（P3.A.2），不做 Docker |
| 并行探索 | DAG Trace 分支 + asyncio + FileLock | 单方向串行 | Phase 3 启用，`branch_id` 字段 Phase 2 即写入 |
| 执行隔离 | Docker 容器 | 本地 conda | **不跟**：本地执行访问 DuckDB/数据更快，是 feature |

### 8.2 立即可借鉴的 Prompt 模式

来自 RD-Agent，已融入 §3.3 Prompt 模板系统设计：

1. **Role Anchoring**：system prompt 开头固定 persona
2. **分块上下文**：`# Scenario` / `# SOTA` / `# Previous Experiments` / `# Challenges`
3. **SOTA 对比**：每次生成强制对比当前 category 最佳实现
4. **Diff 注入**：repair 场景给 diff 而非完整代码
5. **问题积累**：从失败实验提取 "Key Learnings and Unresolved Challenges"
6. **结构化输出**：Pydantic-style JSON schema 注入 prompt

### 8.3 明确不跟的 RD-Agent 特性

| 特性 | 不跟原因 |
|------|----------|
| Docker 隔离 | AutoQuant 需要直接访问本地 DuckDB 和大量历史数据，Docker 增加复杂度且无收益 |
| 独立 Python agent 循环 | Claude Code subagent 模式更轻量，Python 层只做执行 |
| Jinja2 YAML 模板系统 | Markdown 文件级复用够用，YAML + Jinja2 增加学习成本 |
| 完整的 Hypothesis2Experiment + ExpPlanner | 对单因子迭代过重；父进程直接审核 hypothesis 更轻量 |
| Bandit 方向选择 | 方向数 < 5 时手动选择更可控；等方向数 > 5 再评估（Phase 4+）|
