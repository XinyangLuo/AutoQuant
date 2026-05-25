# Agent 投研系统

本文件给 Claude Code 在 `agents/` 目录工作时提供导航。子模块细节走各模块 `DESIGN.md`。

## 1. 定位

基于 RD-Agent（Microsoft Research）核心抽象的 A 股自动因子挖掘系统。

闭环流程：

```
HypothesisGen (LLM) → Hypothesis2Experiment (code gen)
                            ↓
                  AutoQuantFactorRunner (backfill → eval → BT)
                            ↓
                  AutoQuantFactorEvaluator (metrics → Feedback)
                            ↓
                  Trace / KnowledgeBase (history → next round)
```

人工介入点：候选因子生成审核报告后，由人最终决定是否 `admit` 到因子库。

## 2. 环境与命令

使用 conda 环境 `AutoQuant`（Python 3.11.15）。

```bash
# Agent 因子研究主循环（日期从 config.yaml agent.start_date/end_date 读取）
python -m agents.rdagent.run run \
    --max-rounds 10 \
    --output-dir results/agent/run_001

# 指定种子假设（跳过 Round-1 LLM 生成）
python -m agents.rdagent.run run --seed "20-day momentum x volume spike"

# 查看候选列表
python -m agents.rdagent.run list-candidates results/agent/run_001

# Admit / Reject 候选因子
python -m agents.rdagent.run admit f_auto_xxx --run-dir results/agent/run_001
python -m agents.rdagent.run reject f_auto_xxx --reason "turnover too high"
```

依赖环境变量：

```
DEEPSEEK_API_KEY=...      # DeepSeek API（OpenAI-compatible）
DEEPSEEK_BASE_URL=...     # 可选，默认 https://api.deepseek.com
```

## 3. 目录结构

```
agents/
├── rdagent/                    # RD-Agent × AutoQuant 集成（代码已落地）
│   ├── core/                   # RD-Agent 核心抽象（零外部依赖）
│   │   ├── scenario.py         # Scenario ABC
│   │   ├── proposal.py         # Hypothesis, HypothesisGen, Hypothesis2Experiment
│   │   ├── experiment.py       # Experiment ABC
│   │   ├── evaluation.py       # Evaluator ABC, Feedback dataclass
│   │   ├── evolving_framework.py  # Trace, EvolvingStrategy
│   │   ├── knowledge_base.py   # KnowledgeBase ABC
│   │   └── utils.py            # render_prompt, save_json, load_json
│   ├── scenario.py             # AShareQuantScenario（A 股场景实现）
│   ├── experiment.py           # AutoQuantFactorExperiment
│   ├── runner.py               # AutoQuantFactorRunner（对接 backtest 流水线）
│   ├── evaluator.py            # AutoQuantFactorEvaluator → QuantFeedback
│   ├── hypothesis.py           # AutoQuantFactorHypothesisGen + Hypothesis2Experiment
│   ├── experiment.py           # AutoQuantFactorExperiment
│   ├── knowledge.py            # AShareKnowledgeBase（经验积累 + 相似检索）
│   ├── config.py               # AgentConfig（阈值统一从 config.yaml 读取）
│   ├── run.py                  # 主循环 + CLI
│   ├── utils.py                # cleanup_generated_factor 等
│   ├── prompts/                # LLM Prompt 模板（markdown）
│   └── DESIGN.md               # 详细设计文档
└── CLAUDE.md                   # 本文
```

## 4. 模块接口契约

| 边界 | 提供方 | 消费方 | 形式 |
|---|---|---|---|
| 场景描述 | `AShareQuantScenario` | HypothesisGen / H2E | Prompt 上下文（schema / rules / thresholds） |
| 因子注册 | `hypothesis.py` (`_inject_factor_id`) | `backtest.factor.registry` | Python 代码文件 + `@register` |
| 因子计算 | `backtest.factor.compute` | `AutoQuantFactorRunner` | `compute_factor()` + `apply_variant_pipeline()` |
| 因子评估 | `backtest.factor.evaluation` | `AutoQuantFactorRunner` | `evaluate()` → rankicir / IC+ / turnover |
| 策略回测 | `backtest.simulation` | `AutoQuantFactorRunner` | `SingleFactorStrategy` + `Simple/DetailedSimulator` |
| 评测指标 | `backtest.evaluation` | `AutoQuantFactorEvaluator` | `evaluate(result_dir)` → metrics dict |
| 因子准入 | `backtest.factor.admission` | `run.py` CLI | `admit()` / `reject()` |
| 历史反馈 | `Trace` | `HypothesisGen.gen()` | `trace.hist[-5:]` 最近 5 轮结果 |
| 知识检索 | `AShareKnowledgeBase` | `HypothesisGen` | `get_sota()` + `retrieve_similar()` |

## 5. 与回测系统的集成

Agent 模块**不重复实现**任何回测逻辑，全部委托给 `backtest/`：

- `runner.py` → `backtest.factor.compute.compute_factor()` + `apply_variant_pipeline()`
- `runner.py` → `backtest.factor.evaluation.evaluate()`
- `runner.py` → `backtest.strategy.SingleFactorStrategy`
- `runner.py` → `backtest.simulation.SimpleSimulator` / `DetailedSimulator`
- `runner.py` → `backtest.evaluation.evaluate()`
- `evaluator.py` → `backtest.factor.admission.RECOMMENDED_THRESHOLDS`
- `config.py` → `backtest.config_loader.get_section()`

Agent 的阈值（RankICIR / IC+ / turnover / Sharpe）与回测系统的 `admission` 和 `pipeline` 配置**单一来源**，通过 `config.yaml` 统一读取。

## 6. 编码约定

- 与根目录 `CLAUDE.md` §9 一致
- Agent 特有：`factor_id` 前缀 `f_auto_`（AI 生成） vs `f_`（人工）
- Prompt 模板用 markdown，变量占位符 `{{var_name}}`
