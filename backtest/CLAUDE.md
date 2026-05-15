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
- **因子模块**：定义、计算、登记、静态评估因子。因子值写入 `factors_daily` 长表；稳定因子可"晋升"回 `market_daily`。
- **策略模块**：把因子组合成可执行的策略，**只输出每日目标持仓**。
- **回测引擎**：把策略目标持仓 + 成本模型 → 真实成交序列与净值曲线。日频先行，T+1，A 股规则。
- **分析模块**：从回测产出反推策略质量。绩效指标 + 归因 + 可视化。

## 关键设计决策

- **回测引擎与策略解耦**：策略只产出目标持仓，引擎负责成交模拟（停牌、涨跌停、成本、复权）
- **因子晋升机制**：`factors_daily` 中被验证稳定的因子，可晋升为 `market_daily` 的一列，加速常用路径
- **财务数据未来信息隔离（PIT）**：`income_q` / `balancesheet_q` / `cashflow_q` 三张物理表各自保留所有版本（原始 + 修正）；查询时 `get_fina_snapshot(D)` 对每张表分别按 `f_ann_date <= D` 过滤 + `QUALIFY ROW_NUMBER()` 取最新可见版本，再 outer-join 成 wide DataFrame，正确处理业绩修正（restatement）及约 1% 的三表独立修正 case。详见 [`backtest/data/CLAUDE.md`](data/CLAUDE.md) 的 PIT 章节
