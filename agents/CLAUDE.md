# Agent 投研系统

## 1. 定位

**Claude Code 直接驱动**的 A 股因子迭代研究系统。不再维护独立的 Python agent 循环（原 RD-Agent），由 Claude Code 本身承担决策、代码生成、结果分析和迭代逻辑。

闭环流程：

```
用户 (Claude Code 对话 / /factor-iterate)
    |
    v
Claude Code（决策层）
    |-- 生成/修复因子代码
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
    |-- 决策：修复 / 调参 / 换方向 / 停止
```

**阅读网页/PDF 等能力**通过 Claude Code 的 MCP tools / skills 扩展，不在 Python 层实现。

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

# 单轮执行（传入因子 ID 和因子文件）
python -m agents.claude_cli run f_auto_001 \
    --run-dir results/agent/runs/my_run/round_001 \
    --factor-file results/agent/runs/my_run/round_001/factor.py

# 帮助
python -m agents.claude_cli --help
```

## 3. 目录结构

```
agents/
├── __init__.py               # 空（保持）
├── CLAUDE.md                 # 本文
├── claude_cli.py             # 单轮执行 CLI 入口（schema + run）
├── config.py                 # AgentConfig：阈值从 config.yaml 读取
├── experiment.py             # AutoQuantFactorExperiment dataclass
├── evaluator.py              # QuantFeedback + AutoQuantFactorEvaluator
├── runner.py                 # AutoQuantFactorRunner：对接 backtest 流水线
├── schema.py                 # 数据 schema 查询（列名、别名映射）
├── helpers.py                # 工具函数（代码校验、@register 注入）
└── FACTOR_CODE_GUIDE.md      # LLM 因子代码参考手册
```

**不再包含**：Python agent 循环、LLM API 调用、hypothesis 生成、knowledge base、prompt 模板。这些现在由 Claude Code 本身处理。

## 4. 执行层模块

### `claude_cli.py` — CLI 入口

两个子命令：

- `schema --sources`：输出指定数据源在 `panel` 中的可用列名（JSON）
- `run <factor_id> --run-dir --factor-file`：运行完整 step1~step10 流水线，输出 `result.json`

通过的因子自动写入 `results/agent/candidates/<factor_id>/`，等待人工 review 后 admit。

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
