# CLAUDE.md

本文件给 Claude Code (claude.ai/code) 在本仓库工作时提供导航。子模块的细节走 `DESIGN.md`；本文只覆盖项目骨架 + 跨模块约定。

## 1. 项目概览

AutoQuant 是面向 A 股的「量化研究 → 信号 → 推送」一体化系统，三大模块：

- **回测系统** (`backtest/`)：数据下载 → 因子生成 → 策略组合 → 回测执行 → 评测，完整流水线
- **Agent 投研系统** (`agents/`)：基于 RD-Agent 框架的自动因子挖掘（代码已落地，见 `agents/rdagent/`）
- **交易模块** (`trading/`)：第一阶段只做策略信号推送 + 本地仓位跟踪，不直连券商

数据流：

```
研报/论文 → Agent (Reader/IdeaMiner/Researcher/Curator)
                    ↓                              ↑
              回测 (Data → Factor → Strategy → Sim → Eval)
                    ↓
              交易 (Signals → Push → Positions)
```

## 2. 环境与命令

使用 conda 环境 `AutoQuant`（Python 3.11.15）。所有 Python 命令前先激活：

```bash
conda activate AutoQuant
```

环境初始化：`conda env create -f environment.yml`。`pyproject.toml` 暂未启用。

`.env` 字段：

```
TUSHARE_TOKEN=...        # Tushare Pro API（已配置）
ANTHROPIC_API_KEY=...    # Claude（TBD）
DEEPSEEK_API_KEY=...     # DeepSeek API（Agent 投研系统）
WECHAT_WEBHOOK=...       # 企微推送（TBD）
FEISHU_WEBHOOK=...       # 飞书推送（TBD）
```

常用入口（更多见 [`backtest/PIPELINE.md`](backtest/PIPELINE.md)）：

```bash
# 数据
python -m backtest.data.cold_start                       # 一键全量初始化
python -m backtest.data.update_daily                     # 日更（所有表）

# 因子 → 回测 → 入库
python -m backtest.factor.backfill f_xxx                 # 回填因子到 work DB
python -m backtest.pipeline run-all f_xxx --start 20210101 --end 20241231  # step1~step9 全自动
python -m backtest.factor.admission admit f_xxx          # 人工 admit
python -m backtest.factor.update                         # 日更 admitted 因子

# 评测
python -m backtest.evaluation <result_dir>               # 8 子图 + summary

# Agent 因子挖掘
python -m agents.rdagent.run run --max-rounds 10         # 跑 Agent 研究循环
python -m agents.rdagent.run list-candidates <run_dir>   # 查看候选列表

# 测试
pytest tests/
```

## 3. 目录结构（当前真实状态）

```
AutoQuant/
├── backtest/                # 回测系统（已落地，主战场）
│   ├── data/                # 数据下载/缓存/更新 + DESIGN.md
│   │   ├── backfill/        # 各表初始化脚本
│   │   ├── fetcher/         # Tushare 接口封装
│   │   ├── realtime/        # xtquant L1 行情（macOS 走 Wine）
│   │   ├── cold_start.py    # 一键全量
│   │   ├── update_daily.py  # 日更入口
│   │   ├── storage.py       # MarketStorage（DuckDB 封装）
│   │   └── trade_calendar.py
│   ├── factor/              # 因子定义/计算/评测/入库 + DESIGN.md
│   │   ├── builtin/barra/   # Barra 7 个 L1 风险因子（结构件）
│   │   ├── compute.py / backfill.py / update.py
│   │   ├── evaluation.py / admission.py / cleanup.py
│   │   ├── storage.py       # FactorStorage(work) + FactorLibrary(library)
│   │   ├── transforms.py    # 算子库（rank/zscore/ts_*/cs_*/industry_*）
│   │   └── variants.py      # variant 标签：none / barra_l3 / barra_ind_size
│   ├── strategy/            # 因子 → target_weight + DESIGN.md
│   ├── simulation/          # Simple/Detailed 双轨回测引擎 + DESIGN.md
│   ├── evaluation/          # 从 parquet 反推策略质量 + DESIGN.md
│   ├── pipeline/            # step1~step9 因子挖掘门控流水线 + DESIGN.md
│   ├── PIPELINE.md          # 端到端使用手册（最重要的文档）
│   └── CLAUDE.md            # 回测系统总览
├── alphas/                  # 私有 alpha 代码（gitignored）
├── agents/                  # Agent 投研系统
│   ├── rdagent/             # RD-Agent 集成（代码已落地）
│   │   ├── core/            # Scenario/Hypothesis/Experiment/Evaluator ABC
│   │   ├── scenario.py      # AShareQuantScenario
│   │   ├── runner.py        # AutoQuantFactorRunner（对接 backtest 流水线）
│   │   ├── evaluator.py     # AutoQuantFactorEvaluator
│   │   ├── hypothesis.py    # LLM 假设生成 + 代码生成
│   │   ├── experiment.py    # AutoQuantFactorExperiment
│   │   ├── knowledge.py     # AShareKnowledgeBase
│   │   ├── config.py        # AgentConfig（阈值与 admission 单一来源）
│   │   ├── run.py           # 主循环 + CLI
│   │   ├── prompts/         # LLM Prompt 模板
│   │   └── DESIGN.md        # 详细设计
│   └── CLAUDE.md            # Agent 系统总览
├── tests/                   # pytest 套件
├── scripts/                 # 一次性脚本（数据修复/bulk 重跑）
├── trading/                 # 交易模块骨架（待 fill）
├── data/                    # 数据根
│   ├── duckdb/              # market.duckdb / factors_pending.duckdb / factor_library.duckdb
│   ├── factor_library/      # registry.json（因子元数据）
│   └── minute/              # 分钟级 parquet（预留）
├── results/                 # 回测产出、研究档案
├── notebooks/               # 探索
├── environment.yml
├── CLAUDE.md                # 本文
└── TODO.md                  # P0~P4 工单池
```

## 4. 数据存储（已敲定）

DuckDB，三个物理库 + 八张表：

| DB | 表 | 主键 | 说明 |
|---|---|---|---|
| `data/duckdb/market.duckdb` | `market_daily` | `(date, symbol)` | 日行情，回测主用 |
| | `income_q` / `balancesheet_q` / `cashflow_q` | `(symbol, end_date, f_ann_date, update_flag, report_type)` | Tushare 原始三表，物理保留所有版本，查询时 `get_fina_snapshot(D)` 按 `f_ann_date <= D` + QUALIFY 取 PIT 快照 |
| | `dividends` | `(symbol, end_date)` | 仅 `div_proc='实施'` |
| | `index_daily` | `(date, symbol)` | 6 大宽基指数 |
| | `index_members` | `(index_code, symbol, trade_date)` | 月度成分股权重 densify 到每个交易日；默认 4 大宽基（HS300/CSI500/CSI1000/CSI2000） |
| | `sw_industry` | `(symbol, level, industry_code, in_date)` | SW2021 行业归属历史，L1/L2 |
| | `trade_calendar` | `(date)` | 交易日历 |
| `data/duckdb/factors_pending.duckdb` | `factors_daily` | `(date, symbol)` | **work DB**：研究中/未 admit 因子。宽表，每个 factor_id 一列。`FactorStorage` 读写 |
| `data/duckdb/factor_library.duckdb` | `factors_daily` | `(date, symbol)` | **library DB**：admitted 因子。同 schema。`FactorLibrary` 读写，强制 admission invariant |

详见 [`backtest/data/DESIGN.md`](backtest/data/DESIGN.md) 与 [`backtest/factor/DESIGN.md`](backtest/factor/DESIGN.md)。

## 5. 回测系统

入口文档：[`backtest/CLAUDE.md`](backtest/CLAUDE.md) + [`backtest/PIPELINE.md`](backtest/PIPELINE.md)。

各子模块设计：[`data/DESIGN.md`](backtest/data/DESIGN.md) · [`factor/DESIGN.md`](backtest/factor/DESIGN.md) · [`strategy/DESIGN.md`](backtest/strategy/DESIGN.md) · [`simulation/DESIGN.md`](backtest/simulation/DESIGN.md) · [`evaluation/DESIGN.md`](backtest/evaluation/DESIGN.md) · [`pipeline/DESIGN.md`](backtest/pipeline/DESIGN.md)。

## 6. Agent 投研系统

总览：[`agents/CLAUDE.md`](agents/CLAUDE.md)。设计文档：[`agents/rdagent/DESIGN.md`](agents/rdagent/DESIGN.md)。

代码已落地：`agents/rdagent/` 包含完整的 hypothesis → code gen → backfill → evaluate → backtest → feedback 闭环实现。基于 RD-Agent (Microsoft) 的核心抽象（`rdagent/core/` 的 ABC 层，零外部依赖）。

复用回测系统现成 API：`compute_factor`、`evaluate`、`SingleFactorStrategy`、`Simple/DetailedSimulator`、`admission.admit`。

## 7. 交易模块（第一阶段）

仅做信号推送 + 仓位跟踪，不直连券商：

- **信号** (`trading/signals/`)：策略 → 标准化 JSON/parquet
- **推送** (`trading/push/`)：监听信号目录，渲染为可读消息 → 企微/飞书/Server酱/邮件（渠道 TBD）
- **仓位** (`trading/positions/`)：本地 YAML 持仓表，CLI 录入

第二阶段（远期）：对接 QMT / Ptrade / easytrader。

## 8. 模块接口契约

| 边界 | 提供方 | 消费方 | 形式 |
|---|---|---|---|
| 数据访问 | `backtest.data` | factor/strategy/agent | `MarketStorage` Python API |
| 因子注册/计算 | `backtest.factor` | strategy/agent | `@register` 装饰器 + `compute_factor()` |
| 目标持仓 | `backtest.strategy` | `backtest.simulation` | `DataFrame[date, symbol, target_weight]` |
| 交易日志 | `backtest.simulation` | `backtest.evaluation` | parquet（nav/positions/trades/metrics） |
| 标准化信号 | strategy | `trading.push` | JSON/parquet（schema TBD） |
| 持仓回写 | `trading.positions` | backtest（实盘对比） | YAML/CSV |

接口要稳定，内部实现允许重构。

## 9. 编码约定

- Python 3.11；`from __future__ import annotations` + 类型注解
- PEP 8；lint/format 选型 TBD
- 测试：pytest（`tests/`）
- 命名：因子函数 `snake_case`，策略类 `PascalCase`，模块小写下划线
- 注释：中文允许，标识符必须英文
- Commit message：`feat / fix / refactor / docs / test / chore / perf` 前缀，**一律英文**

## 10. Claude Code 协作提示

- **环境**：任何 Python 命令前先 `conda activate AutoQuant`
- **数据获取**：取行情/财务/资金/板块数据时优先用 `tushare-data` skill
- **耗时任务**：回测、批量抓取、批量因子计算用 `run_in_background: true` 起背景任务
- **改动前**：先读对应模块的 `DESIGN.md`，确认接口契约（§8）
- **文档分层**：`CLAUDE.md` 只在项目根 + 大子项目根两级（当前是 `./CLAUDE.md` + `backtest/CLAUDE.md`）；更深的模块用 `DESIGN.md`，避免自动加载文件过多
- **新模块骨架**：每个新模块至少 `__init__.py` + `DESIGN.md`
- **Plan 完成后**：把设计沉淀到对应 `DESIGN.md`；不要在 `CLAUDE.md` 里塞细节
- **TODO**：按 P0~P4 分级；完成的项**直接删除**（不留划线），全部清空后整份 `TODO.md` 也删

## 11. 待决事项

- 推送渠道选型（企微 / 飞书 / Server酱 / 邮件）
- Agent 文档解析方案（unstructured / PyMuPDF / Claude 多模态）
- 向量检索引入时机（取决于因子库规模）

详细工单池：[TODO.md](TODO.md)。
