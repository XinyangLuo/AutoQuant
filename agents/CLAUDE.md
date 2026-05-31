# Agent 投研系统

## 1. 定位

**Claude Code 直接驱动**的 A 股因子迭代研究系统。不再维护独立的 Python agent 循环（原 RD-Agent），由 Claude Code 本身承担决策、代码生成、结果分析和迭代逻辑。

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
Claude Code（决策层）
    |-- [FC] 根据 hypothesis.md / 自然语言 生成/修复因子代码
    |-- Write 到 alphas/exp/agent/<factor_id>/factor.py
    |
    v
python -m agents.claude_cli run <factor_id>   （执行层）
    |-- compute → backfill → evaluate → simple BT → detailed BT
    |-- 输出 result.json
    |
    v
Claude Code（分析层）
    |-- Read result.json
    |-- 写 trace.jsonl
    |-- [RC] 诊断失败 → 查 KB → 输出 repair/abandon 指令
    |-- 决策：修复 / 调参 / 换方向 / 停止
```

**阅读网页/PDF 等能力**通过 Claude Code 的 MCP tools / skills 扩展，不在 Python 层实现。PDF 阅读走 `mcp-pdf`（pdfplumber / pymupdf 文本提取），不依赖模型原生多模态。详见 `DESIGN.md` §4.5。

## 2. 使用方式

### 交互式因子研究

```
/factor-iterate 成交额放量后短期反转，尤其在小盘股里更强
/factor-iterate max_rounds=5 data_sources=market_daily,income_q 低估值盈利改善动量
```

详见 `.claude/commands/factor-iterate.md`。

### 命令行工具

```bash
conda activate AutoQuant

# 查询数据 schema（Claude 生成代码前先查可用列名）
python -m agents.claude_cli schema --sources market_daily
python -m agents.claude_cli schema --sources market_daily,income_q

# 单轮执行（传入因子 ID 和因子文件；--run-dir 可选，默认输出到 results/<factor_id>/）
python -m agents.claude_cli run f_auto_001 \
    --factor-file results/<factor_id>/factor.py

# 帮助
python -m agents.claude_cli --help
```

## 3. 目录结构

```
agents/
├── __init__.py               # 空（保持）
├── CLAUDE.md                 # 本文
├── DESIGN.md                 # Multi-Agent 自动因子挖掘实施计划
├── PLAN.md                   # 演进路线（Phase 1→4）
├── claude_cli.py             # 单轮执行 CLI 入口（schema + run）
├── config.py                 # AgentConfig：阈值从 config.yaml 读取
├── experiment.py             # AutoQuantFactorExperiment dataclass
├── evaluator.py              # QuantFeedback + AutoQuantFactorEvaluator
├── runner.py                 # AutoQuantFactorRunner：对接 backtest 流水线
├── schema.py                 # 数据 schema 查询（列名、别名映射）
├── helpers.py                # 工具函数（代码校验、@register 注入）
├── FACTOR_CODE_GUIDE.md      # LLM 因子代码参考手册
├── knowledge_base/           # Agent 知识库（跨 run 持久化，git 追踪）
│   ├── anti_patterns.json    #   失败模式 → 修复建议
│   ├── successful_patterns.json  #   成功模式 → SOTA 基准
│   └── failed_attempts.jsonl #   失败实验索引（append-only）
└── pdf_hypotheses/           # PDF→hypothesis 中间产物（gitignore）
    └── <slug>/               #   每次提取一个子目录
        └── *_hypothesis.md

.claude/
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

**不再包含**：Python agent 循环、LLM API 调用、knowledge base Python 模块。这些现在由 Claude Code 本身处理。

**Prompt 模板**：已迁移到 `.claude/prompts/`（Markdown 文件级复用），参考 RD-Agent 的 Jinja2 YAML 共享块思路，但保持轻量。

## 4. 执行层模块

### `claude_cli.py` — CLI 入口

两个子命令：

- `schema --sources`：输出指定数据源在 `panel` 中的可用列名（JSON）
- `run <factor_id> --run-dir --factor-file`：运行完整 step1~step10 流水线，输出 `result.json`（含自动计算的 HS300/CSI500/CSI1000 超额指标）

通过的因子自动写入 `results/candidates/<factor_id>/`，等待人工 review 后 admit。

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

`AutoQuantFactorEvaluator` 汇总 experiment 中各 step 的 pass/fail 状态，生成结构化 `QuantFeedback`（decision / observation / suggestion / metrics / failed_step），供 Claude 分析。

### `config.py` — 配置

`AgentConfig` 从 `config.yaml` 读取 agent 特有字段（start_date, end_date）。所有流水线阈值和策略默认值由 `backtest.pipeline.config.PipelineConfig` / `StepThresholds` 管理，agent 不重复定义。

### `schema.py` — Schema 查询

提供 `get_panel_columns_for_data_sources()` + `COLUMN_ALIASES` 映射表／Claude 在生成因子代码前调用 `claude_cli schema` 获取真实列名，避免 hallucinate。

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
| 因子准入 | Claude Code 建议 → 人工 | `backtest.factor.admission` |

## 6. 编码约定

- 与根目录 `CLAUDE.md` §9 一致
- Agent 特有：因子 ID 前缀 `f_auto_`（Claude 生成）vs `f_`（人工）
- 因子代码遵循 `FACTOR_CODE_GUIDE.md` 规范

## 7. Multi-Agent 自动因子挖掘（Claude Code Subagent 模式）

> **实施计划**：[`agents/PLAN.md`](PLAN.md)。本章为摘要，完整演进路线以 PLAN.md 为准。

### 7.1 核心思路

用 Claude Code 的 **Agent tool（subagent）** 做决策，Python 层只管执行。不建独立的 agent 循环，不新增 Python 模块（Phase 1 零新代码）。

**与旧版设计的关键区别**：
- ~~5 个专用 subagent~~ → **2 个**（Factor Coder + Result Critic）
- ~~8 个 KB 文件~~ → **3 个**（anti_patterns + successful_patterns + failed_attempts）
- ~~新增 knowledge.py / scheduler.py / orchestrator.py~~ → **Phase 1 不写任何新 Python**
- ~~bandit 方向选择~~ → 方向选择由父进程（Claude Code 对话）直接完成
- ~~claude_cli kb-query / run-index 子命令~~ → 用 Read + jq 代替

### 7.2 Phase 1（当前）：增强 `/factor-iterate` + KB 驱动修复

```
/ factor-iterate "成交额放量后短期反转，小盘股更强"
  │
  ├─ Round 1..N:
  │   ├─ [Factor Coder]  生成/修复代码 → claude_cli run
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
| `.claude/commands/factor-iterate.md` | 改 | 集成 RC subagent + KB 查询 |
| `agents/claude_cli.py` | 不改 | |
| `agents/` 其他模块 | 不改 | |

后续 Phase（并行探索、自动审计、bandit 调度）的触发条件、范围、设计细节见 [`PLAN.md`](PLAN.md)。
