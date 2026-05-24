# AutoQuant Agent 因子投研系统

基于 DeepSeek LLM 的自动化 A 股因子挖掘 Agent。循环执行：**假设生成 → 代码生成 → 回测流水线 → 评估反馈 → 迭代优化**。

## 前置条件

1. **conda 环境**（`AutoQuant`，Python 3.11）
2. **数据库**：`market.duckdb` 已有日行情数据
3. **API Key**：`.env` 中配置 `DEEPSEEK_API_KEY`

```bash
# .env
DEEPSEEK_API_KEY=sk-...
# DEEPSEEK_BASE_URL=https://api.deepseek.com   # 可选，自定义端点
```

## 快速开始

### 1. 安装依赖

```bash
conda activate AutoQuant
pip install openai
```

### 2. 运行 Agent 循环

```bash
python -m agents.rdagent.run run \
    --max-rounds 10 \
    --start 20200101 \
    --end 20231231 \
    --output-dir results/agent/run_001
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-rounds` | 10 | 最多迭代几轮 |
| `--start` | 20160101 | 回测起始日期 |
| `--end` | 20231231 | 回测结束日期 |
| `--output-dir` | results/agent/run_001 | 输出目录 |
| `--min-rankicir` | 0.25 | 候选门槛：RankICIR |
| `--min-ic-pos` | 0.52 | 候选门槛：IC+ 比例 |
| `--max-turnover` | 0.5 | 候选门槛：最大换手 |
| `--min-sharpe` | 0.8 | 候选门槛：Simple Sharpe |

### 3. 查看候选因子

```bash
# Markdown 报告
python -m agents.rdagent.run list-candidates results/agent/run_001

# 或读文件
cat results/agent/run_001/candidates.md
```

### 4. 人工审核后准入

```bash
# 准入（迁移到 library DB）
python -m agents.rdagent.run admit f_auto_xxx --run-dir results/agent/run_001

# 或拒绝
python -m agents.rdagent.run reject f_auto_xxx --reason "与现有因子相关性过高"
```

## 工作流程

```
Round 1        Round 2        ...
   |              |
   v              v
LLM 生成假设   LLM 生成新假设（基于上一轮反馈）
   |              |
   v              v
LLM 生成代码   LLM 生成新代码
   |              |
   v              v
Runner 执行    Runner 执行
(compute + evaluate + backtest)
   |              |
   v              v
Evaluator 评估  Evaluator 评估
   |              |
   v              v
Trace/KB 记录  Trace/KB 记录（积累 + 检索相似案例）
```

每轮产物保存在 `--output-dir`：

```
results/agent/run_001/
├── candidates.md          # 候选因子审核报告
├── run_metadata.json      # 运行元数据
├── kb.json               # 知识库（经验积累）
└── checkpoints/          # 每轮 checkpoint
    ├── trace_round_001.json
    ├── candidates_round_001.json
    ...
```

## 候选标准

Agent 自动标记为 **candidate** 的因子需同时满足：

| 指标 | 门槛 | 说明 |
|------|------|------|
| RankICIR | >= 0.25 | 风险调整预测能力 |
| IC+ ratio | >= 52% | 正 IC 天数占比 |
| Turnover | < 0.5 | 日频换手（越低越稳定）|
| Max corr | < 0.85 | 与现有因子最大相关性 |
| Simple Sharpe | >= 0.8 | 简单回测夏普比 |

达到 **high bar**（Sharpe > 1.0）时自动提前终止。

## 准入策略

**半自动**：Agent 生成候选列表和审核报告，人工最终确认后执行 `admit`。

```bash
# 1. 看报告
cat results/agent/run_001/candidates.md

# 2. 确认准入
python -m agents.rdagent.run admit f_auto_003

# 3. 验证
python -m backtest.factor.admission status
```

## 文件结构

```
agents/rdagent/
├── README.md                 # 本文档
├── DESIGN.md                 # 详细设计文档
├── core/                     # RD-Agent 抽象基类
│   ├── scenario.py           # Scenario ABC
│   ├── proposal.py           # Hypothesis + HypothesisGen ABC
│   ├── experiment.py         # Experiment ABC
│   ├── evaluation.py         # Evaluator + Feedback ABC
│   ├── evolving_framework.py # Trace + EvolvingStrategy
│   ├── knowledge_base.py     # KnowledgeBase ABC
│   └── utils.py              # 共享辅助
├── scenario.py               # AShareQuantScenario（A 股场景）
├── experiment.py             # AutoQuantFactorExperiment
├── runner.py                 # AutoQuantFactorRunner（回测流水线）
├── evaluator.py              # AutoQuantFactorEvaluator
├── hypothesis.py             # HypothesisGen + Hypothesis2Experiment
├── knowledge.py              # AShareKnowledgeBase
├── run.py                    # CLI 入口 + 主循环
├── utils.py                  # 清理辅助
├── generated/                # LLM 生成的因子代码（gitignored）
└── prompts/                  # Prompt 模板
    ├── scenario_desc.md
    ├── hypothesis_gen.system.md
    ├── hypothesis_gen.user.md
    ├── hypothesis2experiment.system.md
    └── hypothesis2experiment.user.md
```

## 提示词模板

Prompt 模板在 `prompts/` 目录下，可手动编辑以调整 LLM 行为：

- `scenario_desc.md` — A 股场景描述（数据 schema、交易规则、算子列表）
- `hypothesis_gen.*.md` — 假设生成（system + user）
- `hypothesis2experiment.*.md` — 代码生成（system + user）

修改后下次运行自动生效，无需重启。
