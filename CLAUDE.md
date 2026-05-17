# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. 项目概览

AutoQuant 是一个面向 A 股的「量化研究 → 信号 → 推送」一体化系统。三大模块分工：

- **回测系统**：数据下载、因子生成、策略组合、回测执行、结果分析的完整流水线
- **Agent 投研系统**：基于 Claude Agent SDK 的多 agent 协作，自动阅读研报/论文/博客找灵感，调用回测系统验证,沉淀到因子/策略库
- **交易模块**：第一阶段只做策略信号推送 + 本地仓位跟踪，不直连券商

整体数据流（文字版）：

```
研报/论文/博客 → Agent (Reader → Idea Miner → Researcher → Curator)
                       ↓                                ↑
                  回测系统 (Data → Factors → Strategy → Engine → Analysis)
                       ↓
                  交易模块 (Signals → Push / Positions)
```

## 2. 环境与命令

使用 conda 环境 `AutoQuant`（Python 3.11.15）。所有 Python 命令必须先 `conda activate AutoQuant`。

```bash
conda activate AutoQuant
```

**首次初始化**（待 `environment.yml` 落地后）：

```bash
conda env create -f environment.yml
conda activate AutoQuant
pip install -e .            # 等 pyproject.toml 落地后
```

注意：当前 conda 环境 `AutoQuant` 实际尚未在本机创建，`environment.yml` 也尚未编写。首次开工要先建立这些。

**常用命令占位**（待对应模块实现后补具体命令）：

- 运行回测：`python -m autoquant.backtest.engine <strategy>`
- 启动 Agent 研究循环：`python -m autoquant.agents.researcher`
- 跑测试：`pytest`
- Lint：`ruff check . && black --check .`

**`.env` 字段**：

```
TUSHARE_TOKEN=...        # Tushare Pro API（已配置）
ANTHROPIC_API_KEY=...    # Claude Agent SDK
WECHAT_WEBHOOK=...       # 企微推送（TBD）
FEISHU_WEBHOOK=...       # 飞书推送（TBD）
```

## 3. 目录结构（提案）

> 目前仓库只有 `CLAUDE.md` / `.env` / `.claude/`。下列结构是目标布局，尚未落地。

```
AutoQuant/
├── autoquant/                # 主 Python 包
│   ├── backtest/             # 回测系统
│   │   ├── data/             # 数据下载、缓存、增量更新
│   │   ├── factors/          # 因子定义与计算
│   │   ├── strategy/         # 策略组合与选股逻辑
│   │   ├── engine/           # 回测引擎
│   │   └── evaluation/       # 评测：指标 + 净值/回撤/月度图（前称 analysis）
│   ├── agents/               # Claude Agent SDK 投研子 agent
│   │   ├── readers/          # 研报/论文/博客抓取与摘要
│   │   ├── researchers/      # 因子/组合发现
│   │   └── curator/          # 评估、入库、检索
│   └── trading/              # 交易模块
│       ├── signals/          # 策略 → 标准化信号
│       ├── push/             # 推送渠道(企微/飞书/邮件)
│       └── positions/        # 仓位跟踪
├── data/
│   ├── duckdb/               # DuckDB：主表 market_daily（小而稳）+ 因子表 factors_daily（长表）
│   ├── minute/               # 分钟级数据（parquet，预留）
│   └── factor_library/       # 因子代码与元数据（registry，因子值在 factors_daily）
├── results/                  # 回测产出、研究档案、agent 记忆
├── notebooks/                # 探索与可视化
├── scripts/                  # 一次性脚本（数据修复、bulk 重跑等）
├── tests/
├── environment.yml
├── pyproject.toml
└── CLAUDE.md
```

## 4. 回测系统

详见 [`backtest/CLAUDE.md`](backtest/CLAUDE.md)。简要概览：

- **数据模块**：行情/基本面数据下载、缓存、增量更新。详见 [`backtest/data/DESIGN.md`](backtest/data/DESIGN.md)。
- **因子模块**：定义、计算、登记、静态评估（IC/RankIC/ICIR）。因子值写入 `factors_daily` 长表；稳定因子可晋升回 `market_daily`。详见 [`backtest/factor/DESIGN.md`](backtest/factor/DESIGN.md)。
- **策略模块**：因子组合 + 选股/择时 + 风控 → **每日目标持仓**（与引擎解耦）。详见 [`backtest/strategy/DESIGN.md`](backtest/strategy/DESIGN.md)。
- **回测引擎**：目标持仓 + 成本模型 → 成交序列 + 净值曲线。日频、T+1、A 股规则。详见 [`backtest/simulation/DESIGN.md`](backtest/simulation/DESIGN.md)。
- **评测模块**：从 simulation 落盘的 parquet 反推策略质量。收益 / 风险 / 胜率 / 交易 / 持仓指标 + 8 子图大图 + JSON/CSV。详见 [`backtest/evaluation/DESIGN.md`](backtest/evaluation/DESIGN.md)。

## 5. Agent 投研系统

### 5.1 底座

**已敲定使用 Claude Agent SDK（Python）**。每个子 agent 都是一个独立 SDK agent，通过 orchestrator 或直接 tool-call 编排。

### 5.2 子 agent 分工

- **Reader**：抓取并摘要研报 PDF、arXiv 论文、券商研究公众号、雪球/知乎博客；输出结构化摘要（核心论点、可能可量化的信号）
- **Idea Miner**：从大量摘要中挑出"看起来可以量化"的因子/策略灵感，转成结构化的因子假设（数据需求、计算公式雏形、预期方向）
- **Researcher**：把因子假设转成代码，调用回测系统 Python API，完成 "写因子 → 跑回测 → 读结果" 闭环，做多轮调参
- **Curator**：根据回测指标做通过/不通过判定，把通过的因子/策略写入因子库与研究档案，并生成可检索的标签；负责"因子晋升"（长表 → 主表列）的执行

### 5.3 工具接口（agent 可调用）

- `register_factor(name, code, metadata)`
- `run_backtest(strategy_config) -> result_id`
- `get_backtest_result(result_id) -> 指标字典`
- `search_factor_library(query) -> 候选因子列表`
- `promote_factor(factor_name)` —— 把因子从 `factors_daily` 晋升为 `market_daily` 的一列

### 5.4 持久化

- 因子代码 + 元数据：`data/factor_library/`
- 因子值：DuckDB `factors_daily`（长表）
- 研究档案：`results/research/<date>/<topic>/`（原始摘要、假设、回测结果、最终判定）
- Agent 记忆：按 Claude Agent SDK 约定持久化（含 prompt cache）

**技术候选（TBD）**：

- PDF/文档解析：unstructured / PyMuPDF / Claude 多模态（PDF 直传）
- 抓取：feedparser（RSS）/ Playwright（动态页）/ httpx + bs4（静态页）
- 向量检索（如因子库规模上升）：chroma / lancedb

## 6. 交易模块

### 6.1 第一阶段（已敲定）

**仅做信号推送 + 仓位跟踪**，不直连券商，风险最低。

### 6.2 信号流转（`autoquant/trading/signals/`）

- 策略 → 标准化信号文件（JSON 或 parquet）
- 信号 schema：`asof_date / symbol / target_weight / action(buy/sell/hold) / reason / strategy_id`
- 每个交易日盘前/盘后批量生成
- 信号目录：`results/signals/<strategy_id>/<date>.json`

### 6.3 推送（`autoquant/trading/push/`）

- 监听信号目录，把当日信号渲染成可读消息推送出去
- 推送内容：策略名、调仓建议、风险提示、回测最近表现
- **推送渠道（TBD）**：企业微信机器人 / 飞书 webhook / Server酱 / 邮件

### 6.4 仓位跟踪（`autoquant/trading/positions/`）

- 手动维护本地持仓表：`data/positions/<account>.yaml`
- 字段：账户、标的、数量、成本、买入日、所属策略
- 提供 CLI 录入/编辑入口，避免裸改 YAML
- 后续可扩展：读券商对账单 CSV，自动对账

### 6.5 第二阶段路线（仅记录，暂不实现）

对接 QMT / Ptrade / easytrader 等通道，把"推送"演进为"半自动下单"。

## 7. 模块协作 / 接口契约

| 边界 | 提供方 | 消费方 | 形式 |
|---|---|---|---|
| 因子注册 / 回测调用 / 晋升 | 回测系统 | Agent 投研系统 | Python tool 接口（§5.3） |
| 标准化信号 | 回测系统 + 策略 | 交易模块（推送） | JSON/parquet（§6.2） |
| 持仓回写 | 交易模块 | 回测系统（实盘对比） | YAML/CSV（§6.4） |
| 数据访问 | 数据模块 | 因子/策略/Agent | Python API（详见 backtest/data/DESIGN.md） |

上表里的边界要尽量稳定；内部实现允许重构。

## 8. 编码与协作约定

- Python 3.11；默认开类型注解（`from __future__ import annotations` + `typing`）
- 风格：PEP 8；Lint/Format 候选 ruff + black（TBD）
- 测试：pytest；至少覆盖数据接口的 schema、回测引擎的核心成交逻辑
- 命名：
  - 因子函数 `snake_case`（如 `momentum_20d`）
  - 策略类 `PascalCase`（如 `MomentumLongShort`）
  - 模块/包小写下划线
- 注释：中文允许，但标识符必须英文
- Commit message：`feat / fix / refactor / docs / test / chore` 前缀

## 9. Claude Code 协作提示

- **环境**：任何 Python 命令前先 `conda activate AutoQuant`
- **数据**：取行情/财务/资金/板块数据时优先调用 `tushare-data` skill
- **耗时任务**：回测、批量抓取、批量因子计算用 `run_in_background: true` 起背景任务
- **改动**：动手前先读对应模块小节，确认接口契约（§7）
- **Agent 实现**：写 Claude Agent SDK 相关代码时使用 `claude-api` skill 的最佳实践（含 prompt caching）
- **新模块骨架**：每个 `autoquant/<module>/<sub>/` 至少有 `__init__.py` + `README.md`
- **设计文档**：plan 完成后写对应模块的 `DESIGN.md`（如 `backtest/data/DESIGN.md`）；`CLAUDE.md` 仅保留项目根 + 大子项目根两级，更深目录一律用 `DESIGN.md`，避免自动加载文件过多拖慢启动

## 10. 待决事项（TBD 清单）

- [ ] 回测引擎选型（自研 / vectorbt / backtrader / qlib）
- [ ] 推送渠道选型（企微 / 飞书 / Server酱 / 邮件）
- [ ] 文档解析方案（unstructured / PyMuPDF / Claude 多模态）
- [ ] 网页抓取方案（feedparser / Playwright / httpx+bs4）
- [ ] 是否需要向量检索（取决于因子库规模）
- [ ] `environment.yml` 实际依赖清单（tushare、pandas、pyarrow、duckdb、anthropic、claude-agent-sdk、…）
- [ ] 是否启用 `pyproject.toml`（建议是，便于 `pip install -e .`）

**已敲定**：DuckDB 六表设计——`market_daily`（日行情，主键 `(date, symbol)`，回测主用）+ `factors_daily`（因子长表 `(date, symbol, factor_name, value)`，研究主用，稳定因子可晋升回 `market_daily`）+ `income_q` / `balancesheet_q` / `cashflow_q`（Tushare 原始三表各自独立物理表，主键 `(symbol, end_date, f_ann_date, update_flag)`，物理保留所有版本，查询时由 `get_fina_snapshot(D)` 分别做 `f_ann_date <= D` + `QUALIFY` 取最新可见版本后 outer-join，正确处理业绩修正及约 1% 的三表独立修正 case）+ `dividends`（分红事件，主键 `(symbol, end_date)`，仅 `div_proc='实施'`）；parquet（分钟级、回测产出）；Claude Agent SDK；交易模块第一阶段仅信号推送 + 仓位跟踪；静态因子评估 IC/RankIC/ICIR；绩效核心指标 Sharpe/年化/波动/最大回撤。
