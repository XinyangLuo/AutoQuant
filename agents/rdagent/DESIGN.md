# RD-Agent × AutoQuant 集成设计

> **状态：代码已落地。**
> `core/` / `scenario.py` / `runner.py` / `evaluator.py` / `hypothesis.py` / `knowledge.py` / `run.py` / `config.py` 均已完成实现。
> 实施顺序见文末「实施顺序」节；总体优先级在根目录 `TODO.md` P1。

## 定位

让 RD-Agent（Microsoft Research 的迭代式 AI Agent 框架）调用 AutoQuant 的完整流水线，自动发现、评估、迭代优化 A 股因子，并将有效因子沉淀到 AutoQuant 因子库。

## 架构

```
RD-Agent Core Loop                              AutoQuant Backend
+------------------+                            +------------------------+
| HypothesisGen    | --(Hypothesis)-->          |                        |
|   (LLM-based)    |                            |  AShareQuantScenario   |
+------------------+                            |  (prompt context)      |
         |                                      +------------------------+
         v                                                   |
+------------------+                                         v
| Hypothesis2Exp   | --(FactorExperiment)-->    +------------------------+
|   (code gen)     |                            |  AutoQuantFactorRunner |
+------------------+                            |  (backfill→eval→BT)    |
         |                                      +------------------------+
         v                                                   |
+------------------+                                         v
|   Runner         | --(execute)-->            +------------------------+
|   (orchestrator) |                            |  AutoQuantFeedback     |
+------------------+                            |  (metrics→structured)  |
         |                                      +------------------------+
         v
+------------------+
|   Trace / KB     | --(history)--> back to HypothesisGen
+------------------+
```

## 已确认的设计决策

| 决策项 | 选择 |
|--------|------|
| RD-Agent 安装方式 | **核心抽象直接复制** — 仅复制 `rdagent/core/` 的抽象基类到 `agents/rdagent/core/`，零外部依赖 |
| 迭代速度 | **分阶段执行** — 早期只做 factor_eval + simple_backtest（秒级），达标后再跑 detailed backtest |
| 准入策略 | **半自动** — RD-Agent 自动 admit 达标的因子，同时生成审核报告供事后检查 |

## 目录结构

```
agents/
├── __init__.py                     # (existing)
└── rdagent/
    ├── __init__.py
    ├── DESIGN.md                   # 本文档
    ├── core/                       # 从 RD-Agent 复制的核心抽象
    │   ├── __init__.py
    │   ├── scenario.py             # Scenario ABC
    │   ├── proposal.py             # Hypothesis, HypothesisGen, Hypothesis2Experiment
    │   ├── experiment.py           # Experiment ABC
    │   ├── evaluation.py           # Evaluator ABC, Feedback
    │   ├── evolving_framework.py   # EvolvingStrategy, Trace
    │   ├── knowledge_base.py       # KnowledgeBase ABC
    │   └── utils.py                # Common helpers
    ├── scenario.py                 # AShareQuantScenario (具体实现)
    ├── experiment.py               # AutoQuantFactorExperiment
    ├── runner.py                   # AutoQuantFactorRunner
    ├── evaluator.py                # AutoQuantFactorEvaluator
    ├── hypothesis.py               # AutoQuantFactorHypothesisGen + Hypothesis2Experiment
    ├── knowledge.py                # AShareKnowledgeBase
    ├── utils.py                    # Shared helpers
    ├── run.py                      # CLI entry point
    └── prompts/                    # Prompt templates
        ├── hypothesis_gen.system.md
        ├── hypothesis_gen.user.md
        ├── hypothesis2experiment.system.md
        ├── hypothesis2experiment.user.md
        └── scenario_desc.md
```

## 核心类设计

### 1. AShareQuantScenario (`scenario.py`)

继承 `rdagent.core.scenario.Scenario`，描述 A 股量化场景：
- 数据 schema（market_daily 字段列表）
- A 股规则（T+1、涨跌停、ST/IPO 排除）
- 评估标准（RankICIR ≥ 0.25、IC+ ≥ 52%、换手 < 0.5）
- 因子分类体系（reversal、momentum、value、quality 等）
- 中性化选项（raw、swl2_capq5）

**复用 AutoQuant**：
- `backtest/factor/variants.py:BASELINE_VARIANT`
- `backtest/factor/admission.py:RECOMMENDED_THRESHOLDS`
- `backtest/data/storage.py:MarketStorage.get_bars()` 探测 schema

### 2. AutoQuantFactorExperiment (`experiment.py`)

继承 `rdagent.core.experiment.Experiment`：

```python
@dataclass
class AutoQuantFactorExperiment(Experiment):
    factor_id: str
    factor_code: str              # Python code with @register decorator
    factor_file_path: Path        # agents/rdagent/generated/f_{uuid}.py
    eval_result: EvaluationResult | None
    simple_bt_metrics: dict | None
    detailed_bt_metrics: dict | None
    pipeline_report_path: Path | None
```

### 3. AutoQuantFactorRunner (`runner.py`)

执行 AutoQuant 完整流水线：

```
1. Write code to disk → import to trigger @register
2. Backfill: compute_factor() + neutralize → work DB
3. Factor evaluation: evaluate() → IC/RankIC/turnover/corr
4. Simple backtest: SingleFactorStrategy + SimpleSimulator
5. Detailed backtest: SingleFactorStrategy + DetailedSimulator (only if step 3 passes)
6. Collect all metrics into experiment
```

**分阶段执行**：步骤 3+4 每轮必跑（秒级）；步骤 5 只在 RankICIR ≥ 0.25 且 Sharpe ≥ 0.5 时跑（分钟级）。

**复用 AutoQuant**：
- `backtest.factor.compute.compute_factor()`
- `backtest.factor.evaluate()`
- `backtest.strategy.SingleFactorStrategy`
- `backtest.simulation.SimpleSimulator / DetailedSimulator`
- `backtest.evaluation.evaluate()`

### 4. AutoQuantFactorEvaluator (`evaluator.py`)

将 AutoQuant 指标转为 RD-Agent `Feedback`：

```python
@dataclass
class QuantFeedback(Feedback):
    decision: bool                # 是否达标
    rankicir: float
    ic_positive_ratio: float
    turnover: float
    max_corr: float
    simple_sharpe: float
    simple_mdd: float
    detailed_sharpe: float | None
    cost_drag: float | None
    monotonicity: float | None
    observation: str              # LLM 可读总结
    suggestion: str               # 下一轮改进建议
```

### 5. AutoQuantFactorHypothesisGen (`hypothesis.py`)

继承 `rdagent.core.proposal.HypothesisGen`，生成因子假设。

Prompt 注入内容：
1. 数据 schema（market_daily 字段）
2. A 股规则（T+1、涨跌停、ST/IPO 排除）
3. 评估标准（RankICIR、IC+、换手门槛）
4. 可用算子（rank、z_score、ts_mean、ts_rank、ts_std、cap_neutralize、industry_neutralize）
5. 历史反馈（上一轮结果和失败原因）
6. RAG（KnowledgeBase 检索相似案例）
7. SOTA 追踪（当前最佳因子表现）

### 6. AutoQuantFactorHypothesis2Experiment (`hypothesis.py`)

将 Hypothesis 转为 AutoQuant 兼容的因子代码。

**输出格式**：生成带 `@register` 装饰器的 Python 函数：

```python
@register(
    "f_auto_001",
    name="momentum_volume_spike",
    category="momentum",
    data_sources=["market_daily"],
    description="20-day momentum × volume spike deviation",
    parameters={"ret_window": 20, "vol_window": 20},
)
def momentum_volume_spike(panel: pd.DataFrame, ...) -> pd.Series:
    ...
```

### 7. AShareKnowledgeBase (`knowledge.py`)

积累 A 股量化领域知识：
- 成功/失败因子模式
- 各分类在不同市场环境的表现
- 常见数据陷阱（未来信息、幸存者偏差、PIT）
- 算子效果评估

初始知识库包含：交易规则、因子分类、常见陷阱、算子指南、评估门槛。

## 迭代循环 (`run.py`)

```python
for round in range(max_rounds):
    hypothesis = hypothesis_gen.gen(trace=trace)
    experiment = h2e.convert(hypothesis, trace)
    experiment = runner.run(experiment)       # 执行 AutoQuant 流水线
    feedback = evaluator.evaluate(experiment) # 评估并生成反馈
    trace.hist.append((experiment, feedback))
    knowledge_base.add_experience(experiment, feedback)
    save_checkpoint(trace, knowledge_base, round)
    if feedback.decision and feedback.simple_sharpe > 1.0:
        break

# 事后审核报告
best = select_best_factors(trace, top_k=5)
for exp in best:
    auto_admit_with_report(exp)  # admit + 生成审核报告
```

## 与 AutoQuant 的接口契约

| 边界 | RD-Agent 侧 | AutoQuant 侧 | 形式 |
|------|-------------|--------------|------|
| 因子注册 | 生成带 `@register` 的代码 | `backtest.factor.registry.register` 装饰器 | Python 代码文件 |
| 因子计算 | 调用 Runner | `compute_factor()` + `apply_neutralizations()` | Python API |
| 因子评估 | 调用 Runner | `evaluate()` → `EvaluationResult` | Python API |
| 策略信号 | 调用 Runner | `SingleFactorStrategy.run()` | Python API |
| 回测执行 | 调用 Runner | `SimpleSimulator.run()` / `DetailedSimulator.run()` | Python API |
| 结果评估 | 调用 Evaluator | `backtest.evaluation.evaluate()` | Python API |
| 因子准入 | 半自动 admit | `admit()` / `reject()` | Python API |

## 去重与隔离

- **相关性检查**：evaluate() 的 `corr_top_k` 只与 **library DB** 中已有因子比较，不与同轮其他 AI 生成因子比较（防止 inbreeding）
- **因子 ID 命名**：`f_auto_{batch}_{seq}`，与人工因子 `f_###` / `f_rev_##` 区分
- **Work DB 清理**：每轮 reject 的因子自动清理 work DB，admit 的因子迁移到 library DB

## 实施顺序

1. **Phase 1**：复制 `rdagent/core/` 抽象基类 → 创建目录结构
2. **Phase 2**：实现 `AShareQuantScenario` + Prompt 模板
3. **Phase 3**：实现 `AutoQuantFactorExperiment` + `AutoQuantFactorRunner`
4. **Phase 4**：实现 `AutoQuantFactorEvaluator`（指标转换）
5. **Phase 5**：实现 `HypothesisGen` + `Hypothesis2Experiment`
6. **Phase 6**：实现 `AShareKnowledgeBase`
7. **Phase 7**：实现主循环 `run.py` + 集成测试

## 复用清单

| AutoQuant 模块 | 复用点 |
|----------------|--------|
| `backtest.factor.registry` | `@register` 装饰器、因子注册 |
| `backtest.factor.compute` | `compute_factor()`、中性化 |
| `backtest.factor.evaluate` | `evaluate()`、IC/RankIC/ICIR |
| `backtest.factor.admission` | `admit()`/`reject()`、门槛检查 |
| `backtest.factor.variants` | `BASELINE_VARIANT`、`expand_variant_names()` |
| `backtest.strategy` | `StrategyConfig`、`SingleFactorStrategy` |
| `backtest.simulation` | `SimpleSimulator`、`DetailedSimulator` |
| `backtest.evaluation` | `evaluate()`、指标计算 |
| `backtest.data.storage` | `MarketStorage`、数据获取 |
| `scripts/run_factor_pipeline.py` | 参考流水线编排逻辑 |

## DeepSeek API
使用DeepSeek V4 Pro作为Agent的底层模型，API在.env文件中使用
    # Please install OpenAI SDK first: `pip3 install openai`
    import os
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )

    print(response.choices[0].message.content)
