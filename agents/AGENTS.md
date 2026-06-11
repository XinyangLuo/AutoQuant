# Agent 投研系统

## 1. 定位

**Codex 直接驱动**的 A 股因子迭代研究系统。不再维护独立的 Python agent 循环（原 RD-Agent），由 Codex 本身承担决策、代码生成、结果分析和迭代逻辑。

闭环流程：

```
研报 PDF (research_papers/)
    |
    v
/pdf-hypothesis                 ← P2.A.4：MCP 文本提取 → 因子穷举 → hypothesis.md
    |
    v
用户审阅 hypothesis.md
    |
    v
/factor-iterate --hypothesis   ← 原有入口，也可直接自然语言
    |
    v
Codex（决策层）
    |-- [FC] 根据 hypothesis.md / 自然语言 生成/修复因子代码
    |-- Write 到 alphas/exp/agent/<factor_id>/factor.py
    |
    v
python -m agents.codex_cli run <factor_id>   （执行层）
    |-- compute → backfill → evaluate → simple BT → detailed BT
    |-- 输出 result.json
    |
    v
Codex（分析层）
    |-- Read result.json
    |-- 写 trace.jsonl
    |-- [RC] 诊断失败 → 查 KB → 输出 repair/abandon 指令
    |-- 决策：修复 / 调参 / 换方向 / 停止
```

**阅读网页/PDF 等能力**通过 Codex 的 MCP tools / skills 扩展，不在 Python 层实现。PDF 阅读走 `mcp-pdf`（pdfplumber / pymupdf 文本提取），不依赖模型原生多模态。详见 `DESIGN.md` §4.5。

## 2. 使用方式

### 交互式因子研究

```
/factor-iterate 成交额放量后短期反转，尤其在小盘股里更强
/factor-iterate max_rounds=5 data_sources=market_daily,income_q 低估值盈利改善动量
```

详见 `.codex/commands/factor-iterate.md`。

### 命令行工具

```bash
conda activate AutoQuant

# 查询数据 schema（Codex 生成代码前先查可用列名）
python -m agents.codex_cli schema --sources market_daily
python -m agents.codex_cli schema --sources market_daily,income_q

# 单轮执行（传入因子 ID 和因子文件；--run-dir 可选，默认输出到 results/<factor_id>/）
python -m agents.codex_cli run f_auto_001 \
    --factor-file results/<factor_id>/factor.py

# 自动追加 trace（从 result.json 构建并写入 trace.jsonl）
python -m agents.codex_cli trace-append --run-dir results/<run_id>/ \
    --result results/<factor_id>/<strategy>/result.json \
    --round 1 --category volume_reversal

# 自动更新 KB（Pass / Fail 时调用）
python -m agents.codex_cli kb-update \
    --result results/<factor_id>/<strategy>/result.json \
    --status pass

# 多 universe 策略参数扫描（top10% 选股，自动检测量价/基本面因子类型）
python -m agents.codex_cli sweep f_auto_001 \
    --factor-file alphas/exp/agent/f_auto_001/factor.py

# 帮助
python -m agents.codex_cli --help
```

## 3. 目录结构

```
agents/
├── __init__.py               # 空（保持）
├── AGENTS.md                 # 本文
├── DESIGN.md                 # Multi-Agent 自动因子挖掘实施计划
├── claude_cli.py             # CLI 实现主体（codex_cli.py 的兼容后端）
├── codex_cli.py              # 单轮执行 CLI 入口（schema + run）
├── config.py                 # AgentConfig：阈值从 config.yaml 读取
├── experiment.py             # AutoQuantFactorExperiment dataclass
├── evaluator.py              # QuantFeedback + AutoQuantFactorEvaluator
├── runner.py                 # AutoQuantFactorRunner：对接 backtest 流水线
├── schema.py                 # 数据 schema 查询（列名、别名映射）
├── helpers.py                # 工具函数（代码校验、@register 注入）
├── trace.py                  # trace.jsonl 读写
├── kb_query.py               # KB 分层查询
├── kb_update.py              # KB 自动更新
├── sweep.py                  # 多 universe 参数扫描
├── FACTOR_CODE_GUIDE.md      # LLM 因子代码参考手册
├── knowledge_base/           # Agent 知识库（跨 run 本地持久化，gitignore）
│   ├── anti_patterns.json    #   失败模式 → 修复建议
│   ├── successful_patterns.json  #   成功模式 → SOTA 基准
│   └── failed_attempts.jsonl #   失败实验索引（append-only）
└── pdf_hypotheses/           # PDF→hypothesis 中间产物（gitignore）
    └── <slug>/               #   每次提取一个子目录
        └── *_hypothesis.md

.codex/
├── commands/
│   ├── factor-iterate.md     # /factor-iterate 命令
│   └── pdf-hypothesis.md     # /pdf-hypothesis 命令（P2.A.4）
├── prompts/
│   ├── shared/
│   │   ├── role.md           # FC/RC/HG 角色定义
│   │   ├── output_formats.md # JSON schema 模板
│   │   └── context_sections.md
│   ├── factor_coder.md       # FC system prompt
│   ├── result_critic.md      # RC system prompt
│   ├── hypothesis_gen.md     # HG prompt
│   └── pdf_hypothesis.md     # PDF→hypothesis 分析 prompt（P2.A.4）
└── settings.json
```

**不再包含**：独立 Python agent 循环和 LLM API 调用。这些现在由 Codex 本身处理；Python 侧只保留 CLI、trace/KB 文件工具和回测执行适配。

**Prompt 模板**：已迁移到 `.codex/prompts/`（Markdown 文件级复用），参考 RD-Agent 的 Jinja2 YAML 共享块思路，但保持轻量。

## 4. 执行层模块

### `codex_cli.py` — CLI 入口

五个子命令：

- `schema --sources`：输出指定数据源在 `panel` 中的可用列名（JSON）
- `run <factor_id> --run-dir --factor-file`：运行完整 step1~step10 流水线，输出 `result.json`。超额收益基准由 universe 决定（默认沪深300）
  - `--run-dir` 模式下自动追加 `trace.jsonl`（需配合 `--round`/`--parent-round`/`--branch-id`）
  - `--auto-kb-update` 自动更新 KB 文件
  - `--feedback-format` 控制 feedback 输出：`flat` / `layered`（默认） / `relevant`
- `sweep <factor_id> --factor-file`：多 universe 策略参数扫描（见下方 § sweep 章节）
- `trace-append --run-dir --result`：从 `result.json` 构建并追加 `trace.jsonl` 记录
- `kb-update --result --status`：根据运行结果自动更新 KB 四文件

### `sweep.py` — 多 Universe 策略参数扫描

对已通过 step1~step4 的因子，在四大宽基指数 universe 下自动扫描策略参数组合：

- **Universe**：沪深300 / 中证500 / 中证1000 / 中证2000（串行，避免 DB 争用）
- **选股**：统一 top 10%（`top_pct=0.1`）
- **参数网格**（按因子类型自动选择）：
  - 量价因子：decay ∈ {5, 10, 15} × rebalance ∈ {1D, 5D}（6 组合）
  - 基本面因子：decay ∈ {5} × rebalance ∈ {1M, 3M}（2 组合）
- **并行**：universe 之间串行，同一 universe 内 strategy 组合并行
- **输出**：每个 universe 选出最优 strategy，最终生成 `cross_universe.json` 跨 universe 比较

目录结构：
```
results/{factor_id}/
  hs300/
    factor_eval/
    top10pct_1D_d5/
      simple/ detailed/ plots/ pipeline_report.md
    top10pct_5D_d10/
      ...
  csi500/ ...
  csi1000/ ...
  csi2000/ ...
  cross_universe.json
```

通过的因子自动写入 `results/candidates/<factor_id>/`，等待人工 review 后 admit。

### `trace.py` — Trace 管理

`TraceManager` 管理单 run 目录下的 `trace.jsonl` 读写，支持：
- `read_all()` / `get_next_round()` / `get_default_parent_round()`
- `append(record)`：原子追加（tmp → replace）
- `TraceRecord.from_result_json()`：从 `result.json` 深层路径提取 metrics（r² / max_existing_corr / residual_icir）

### `runner.py` — 流水线执行器

`AutoQuantFactorRunner` 是 agent 与回测流水线的适配层：

1. 写因子代码到磁盘 → import 触发 `@register`
2. `compute_factor()` + `apply_variant_pipeline()` 中性化 → work DB
3. 调用 `backtest.pipeline.steps` 中的 step1~step10 函数依次执行门控流水线
4. 收集 `PipelineState` 结果写入 experiment（step_results, eval_result, bt_metrics, ridge_result, residual_icir_result）

阈值和策略默认值全部从 `backtest.pipeline.config.PipelineConfig` / `StepThresholds` 读取，agent 不重复定义。

### `experiment.py` — 实验数据类

`AutoQuantFactorExperiment` 跟踪一个因子的完整生命周期：factor_id、代码、评测结果、回测指标、状态。

### `evaluator.py` — 评估器

`AutoQuantFactorEvaluator` 汇总 experiment 中各 step 的 pass/fail 状态，生成结构化 `QuantFeedback`。

**三层拆分**：
- `ExecutionFeedback` — 代码/配置/覆盖度错误（`code_error`, `schema_error`, `coverage_fail`, `config_error`）
- `EvaluationFeedback` — 统计/回测/评估指标（`icir_fail`, `backtest_fail`, `ridge_fail`, `residual_fail` 等）
- `HypothesisFeedback` — 方向/构造/数据源元信息

`get_relevant_layer(failure_type)` 按失败类型返回对应层，供 RC prompt 选择性注入，减少 token 消耗。

### `kb_update.py` — KB 自动更新器

`KbUpdater` 提供 Pass/Fail 后的自动 KB 更新：
- `update_on_pass()` → upsert `hypothesis_index.jsonl` + 更新 `successful_patterns.json`
- `update_on_fail()` → 条件更新 `anti_patterns.json`（`signature` 去重，`count++`）+ 追加 `failed_attempts.jsonl` + upsert `hypothesis_index.jsonl`

所有写操作原子（tmp → replace），`hypothesis_index` 和 `successful_patterns` 按 `factor_id` 去重，`anti_patterns` 按 `signature` 去重。

### `config.py` — 配置

`AgentConfig` 从 `config.yaml` 读取 agent 特有字段（start_date, end_date）。所有流水线阈值和策略默认值由 `backtest.pipeline.config.PipelineConfig` / `StepThresholds` 管理，agent 不重复定义。

### `schema.py` — Schema 查询

提供 `get_panel_columns_for_data_sources()` + `COLUMN_ALIASES` 映射表／Codex 在生成因子代码前调用 `codex_cli schema` 获取真实列名，避免 hallucinate。

### `helpers.py` — 工具函数

- `validate_python_code()` — ast.parse 语法校验
- `validate_transforms_imports()` — 检查 `backtest.factor.transforms` 中的导入是否合法
- `force_register_factor_id()` — AST 重写 `@register` 装饰器中的 factor_id

## 5. 与回测系统的集成

Agent 层不重复实现任何回测逻辑，全部委托给 `backtest/`：

| 边界 | 消费方 | 提供方 |
|------|--------|--------|
| 因子计算 + 中性化 | `runner.py` | `backtest.factor.compute` |
| 流水线步骤 step1~step10 | `runner.py` | `backtest.pipeline.steps` |
| 阈值 + 策略默认值 | `runner.py` | `backtest.pipeline.config.PipelineConfig` |
| 因子准入 | Codex 建议 → 人工 | `backtest.factor.admission` |

## 6. 编码约定

- 与根目录 `AGENTS.md` §9 一致
- Agent 特有：因子 ID 前缀 `f_auto_`（Codex 生成）vs `f_`（人工）
- 因子代码遵循 `FACTOR_CODE_GUIDE.md` 规范

## 7. Multi-Agent 自动因子挖掘（Codex Subagent 模式）

> 本章为摘要；详细机制以 [`agents/DESIGN.md`](DESIGN.md) 和 `.codex/commands/` 下的命令文件为准。

### 7.1 核心思路

用 Codex 的 **Codex subagent（subagent）** 做决策，Python 层只管执行。不建独立的 agent 循环，不新增 Python 模块（Phase 1 零新代码）。

**与旧版设计的关键区别**：
- ~~5 个专用 subagent~~ → **2 个**（Factor Coder + Result Critic）
- ~~8 个 KB 文件~~ → **3 个**（anti_patterns + successful_patterns + failed_attempts）
- ~~新增 knowledge.py / scheduler.py / orchestrator.py~~ → **Phase 1 不写任何新 Python**
- ~~bandit 方向选择~~ → 方向选择由父进程（Codex 对话）直接完成
- ~~codex_cli kb-query / run-index 子命令~~ → 用 Read + jq 代替

### 7.2 Phase 1（当前）：增强 `/factor-iterate` + KB 驱动修复

```
/ factor-iterate "成交额放量后短期反转，小盘股更强"
  │
  ├─ Round 1..N:
  │   ├─ [Factor Coder]  生成/修复代码 → codex_cli run
  │   ├─ [Result Critic] 诊断 fail → 查 KB 反模式 → 输出 repair/abandon 指令
  │   └─ [父进程] 追加 trace.jsonl，更新 KB
  │
  └─ Pass → 因子写入 candidates/，KB 写入 successful_patterns
     Abandon → KB 写入 anti_patterns
```

**改动清单（Phase 1）**：

| 文件 | 操作 | 说明 |
|------|------|------|
| `agents/knowledge_base/` | 新建 | 3 个 JSON 文件（空 schema + 手动 bootstrap） |
| `.codex/commands/factor-iterate.md` | 改 | 集成 RC subagent + KB 查询 |
| `agents/codex_cli.py` | 不改 | |
| `agents/` 其他模块 | 不改 | |

后续 Phase（并行探索、自动审计、bandit 调度）的触发条件、范围、设计细节见 [`DESIGN.md`](DESIGN.md)。
