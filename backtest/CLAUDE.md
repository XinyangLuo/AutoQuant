# 回测系统

## 定位

数据 → 因子 → 策略 → 引擎 → 分析的完整流水线。

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
分析模块（绩效指标、归因、可视化）
```

## 模块交互契约

| 流向 | 提供方 | 消费方 | 形式 | 要点 |
|---|---|---|---|---|
| 原始数据 | 数据模块 | 因子/策略/引擎 | Python API (`get_panel` / `get_bars` / `get_fina_snapshot`) | 数据模块不感知上层逻辑 |
| 因子宽表 | 因子模块 | 策略模块 | DataFrame `(date, symbol, f1, f2, ...)` | 策略只读因子值，不做计算 |
| 目标持仓 | 策略模块 | 回测引擎 | DataFrame `(date, symbol, target_weight)` | 策略不关心成交细节 |
| 交易日志 | 回测引擎 | 分析模块 | `trades.parquet` / `positions.parquet` / `nav.parquet` | 分析纯消费，不修改 |

## 各子模块一句话定位

- **数据模块**：把外部数据拉到本地，建立可重放、可增量更新的数据池。详见 [`backtest/data/CLAUDE.md`](data/CLAUDE.md)。
- **因子模块**：定义、计算、登记、静态评估因子。因子值写入 `factors_daily` 长表；稳定因子可"晋升"回 `market_daily`。详见 [`backtest/factor/CLAUDE.md`](factor/CLAUDE.md)。
- **策略模块**：把因子组合成可执行的策略，**只输出每日目标持仓**。详见 [`backtest/strategy/CLAUDE.md`](strategy/CLAUDE.md)。
- **回测引擎**：把策略目标持仓 + 成本模型 → 成交序列与净值曲线。日频先行，T+1，A 股规则。
- **分析模块**：从回测产出反推策略质量。绩效指标 + 归因 + 可视化。

## 策略模块与引擎的交互细节

### 策略输出格式

```python
signals: pd.DataFrame = strategy.run(start_date, end_date)
# columns: [date, symbol, target_weight]
# - date: 目标持仓生效日（已含 delay=1，即 T日信号 → T+1日生效）
# - symbol: 股票代码
# - target_weight: 目标权重（可正可负，sum 不一定为 1）
```

### 引擎输入与处理

```python
engine.run_backtest(
    signals=signals,
    start_date="20200101",
    end_date="20241231",
    initial_cash=1e8,
    commission_rate=0.0003,
    slippage=0.001,
)
```

引擎内部逐日处理：
1. 按 `date` groupby，逐日读取目标持仓
2. 对每只股票检查当日是否可交易：
   - **停牌**：不可买也不可卖
   - **涨停**（`close >= limit_up` 且前日非涨停）：不可买入
   - **跌停**（`close <= limit_down` 且前日非跌停）：不可卖出
3. 不可交易的标的从目标持仓中剔除
4. 剩余可交易标的重新归一化权重
5. 按当日收盘价（或 vwap）成交，记录交易日志
6. T+1 日持仓生效（A股 T+1 规则）
7. 输出 `trades.parquet` / `positions.parquet` / `nav.parquet`

### 关键设计原则

- **策略与引擎解耦**：策略只产出"理想持仓"，引擎负责"能不能成交"
- **涨停跌停属于未来信息**：策略计算时**不知道**明天哪只涨停，因此策略层**不处理**涨跌停过滤；引擎在执行日根据当日行情判断是否可交易
- **Delay = 1**：T日收盘后根据T日已知信息（因子值、行情）计算信号，产出 T+1 日目标持仓。这是 A 股 T+1 的最小可行延迟

## 关键设计决策

- **回测引擎与策略解耦**：策略只产出目标持仓，引擎负责成交模拟（停牌、涨跌停、成本、复权）
- **因子晋升机制**：`factors_daily` 中被验证稳定的因子，可晋升为 `market_daily` 的一列，加速常用路径
- **财务数据未来信息隔离（PIT）**：`income_q` / `balancesheet_q` / `cashflow_q` 三张物理表各自保留所有版本（原始 + 修正）；查询时 `get_fina_snapshot(D)` 对每张表分别按 `f_ann_date <= D` 过滤 + `QUALIFY ROW_NUMBER()` 取最新可见版本，再 outer-join 成 wide DataFrame，正确处理业绩修正（restatement）及约 1% 的三表独立修正 case。详见 [`backtest/data/CLAUDE.md`](data/CLAUDE.md) 的 PIT 章节
