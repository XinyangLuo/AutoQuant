# 回测系统

## 定位

数据 → 因子 → 策略 → 引擎 → 分析的完整流水线。

回测系统同时作为 **Agent 投研系统的执行后端**：Agent 生成的因子代码通过 `backtest.factor.compute` 计算、通过 `backtest.factor.evaluation` 静态评估、通过 `backtest.simulation` 回测验证。回测模块不感知 Agent 层，仅提供稳定 API。详见 [`agents/CLAUDE.md`](../agents/CLAUDE.md)。

## 数据流

```
market_daily / income_q / balancesheet_q / cashflow_q
    ↓
因子模块（因子定义、计算、登记、静态评估）
    ↓
策略模块（因子组合 + 选股/择时 + 风控 → 每日目标持仓）
    ↓
回测引擎（目标持仓 + 成本模型 → 成交序列 + 净值曲线）
    ↓
评测模块（绩效指标、可视化；归因为 roadmap）
```

## 模块交互契约

| 流向 | 提供方 | 消费方 | 形式 | 要点 |
|---|---|---|---|---|
| 原始数据 | 数据模块 | 因子/策略/引擎 | Python API (`get_panel` / `get_bars` / `get_fina_snapshot`) | 数据模块不感知上层逻辑 |
| 因子宽表 | 因子模块 | 策略模块 | DataFrame `(date, symbol, f1, f2, ...)` | 策略只读因子值，不做计算 |
| 目标持仓 | 策略模块 | 回测引擎 | DataFrame `(date, symbol, target_weight)` | 策略不关心成交细节 |
| 交易日志 | 回测引擎 | 评测模块 | `trades.parquet` / `positions.parquet` / `nav.parquet` / `metrics.parquet` | 评测纯消费，不修改 |

## 各子模块一句话定位

- **数据模块**：把外部数据拉到本地，建立可重放、可增量更新的数据池。详见 [`backtest/data/DESIGN.md`](data/DESIGN.md)。
- **因子模块**：定义、计算、登记、静态评估因子。双库设计——研究中的新因子写 `factors_pending.duckdb`（work，临时）；人工 `admit` 后迁移到 `factor_library.duckdb`（library，稳定）。详见 [`backtest/factor/DESIGN.md`](factor/DESIGN.md)。
- **策略模块**：把因子组合成可执行的策略，**只输出每日目标持仓**。详见 [`backtest/strategy/DESIGN.md`](strategy/DESIGN.md)。
- **回测引擎**：双轨回测（简单/详细）。把策略目标持仓 → 净值曲线。日频，A 股规则。详见 [`backtest/simulation/DESIGN.md`](simulation/DESIGN.md)。
- **评测模块**：从 simulation 落盘的 parquet 反推策略质量。收益/风险/胜率/交易/持仓指标 + 净值/回撤/月度热力图 + 可选基准对比。全项目指标计算单一真理源（`BacktestResult.summary()` 已退化为薄封装）。详见 [`backtest/evaluation/DESIGN.md`](evaluation/DESIGN.md)。

## 策略模块与引擎的交互细节

### 策略输出格式

```python
signals: pd.DataFrame = strategy.run(start_date, end_date)
# columns: [date, symbol, target_weight]
# - date: 目标持仓生效日（已含 delay=1，即 T日信号 → T+1日生效）
# - symbol: 股票代码
# - target_weight: 目标权重（可正可负，sum 不一定为 1）
```

### 双轨回测接口

#### 简单回测（SimpleSimulator）

- **用途**：因子研究、参数扫描、大批量快速验证
- **价格**：复权价格（`adj_close = close * adj_factor`）
- **假设**：不模拟金额，无滑点、无手续费、无涨跌停、无停牌、无分红
- **输出**：仅 `nav` 曲线

```python
from backtest.simulation import SimpleSimulator, SimulationConfig

sim = SimpleSimulator(SimulationConfig(initial_cash=1e8))
result = sim.run(signals, market_data)
# result.nav_df: columns [date, nav, daily_return]
```

#### 详细回测（DetailedSimulator）

- **用途**：策略实盘前验证，精细化模拟
- **价格**：实际价格（非复权），分红送转改变股数/现金
- **假设**：手续费、涨跌停、停牌、分红送转、板块差异化交易单位
- **模式**：`o2o`（默认，开盘价成交）/ `c2c`（收盘价成交）
- **输出**：`nav` + `positions` + `trades`

```python
from backtest.simulation import DetailedSimulator, SimulationConfig

sim = DetailedSimulator(SimulationConfig(
    initial_cash=1e8,
    commission_rate=0.0003,
    price_type="o2o",
))
result = sim.run(signals, market_data, dividends_data)
# result.nav_df      : columns [date, nav, daily_return, total_value, cash, position_value]
# result.positions_df: long format [date, symbol, shares, market_value, weight, avg_cost]
# result.trades_df   : columns [trade_date, symbol, direction, shares, price, amount, commission, reason]
```

### 关键设计原则

- **策略与引擎解耦**：策略只产出"理想持仓"，引擎负责"能不能成交"
- **涨停跌停属于未来信息**：策略计算时**不知道**明天哪只涨停，因此策略层**不处理**涨跌停过滤；引擎在执行日根据当日行情判断是否可交易
- **Delay = 1**：T日收盘后根据T日已知信息（因子值、行情）计算信号，产出 T+1 日目标持仓。这是 A 股 T+1 的最小可行延迟

## 关键设计决策

- **回测引擎与策略解耦**：策略只产出目标持仓，引擎负责成交模拟（停牌、涨跌停、成本、复权）
- **因子双库 + 人工 admission**：研究中的新因子值写 `factors_pending.duckdb`（work, 临时）。看完三层评测（factor eval + simple BT + detailed BT）后人工 `admit`，把数据迁移到 `factor_library.duckdb`（library, 稳定）并清空 work；`reject` 则只清 work、不写 library。`FactorLibrary.insert_factors` 拒绝写未 admit 因子（`allow_unadmitted=True` 旁路供 `promote_from_work` 和测试用），把"work-only / library-only"从约定升级为强制 invariant。Evaluation 的"与现有因子相关性"只读 library，避免临时数据互相污染。整体使用方式见 [`backtest/PIPELINE.md`](PIPELINE.md)。
- **财务数据未来信息隔离（PIT）**：`income_q` / `balancesheet_q` / `cashflow_q` 三张物理表各自保留所有版本（原始 + 修正）；查询时 `get_fina_snapshot(D)` 对每张表分别按 `f_ann_date <= D` 过滤 + `QUALIFY ROW_NUMBER()` 取最新可见版本，再 outer-join 成 wide DataFrame，正确处理业绩修正（restatement）及约 1% 的三表独立修正 case。详见 [`backtest/data/DESIGN.md`](data/DESIGN.md) 的 PIT 章节
- **评测指标单一来源**：所有 Sharpe / 回撤 / 换手 / 胜率等指标只在 `backtest/evaluation/metrics.py` 实现。`BacktestResult.summary()` 与 `scripts/` 中的 `compute_metrics` 都委托到此，杜绝公式漂移
