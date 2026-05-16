# 策略模块

## 定位

因子值 → 每日目标持仓（`DataFrame(date, symbol, target_weight)`）。

策略模块**只产出目标持仓**，不关心执行细节（停牌、涨跌停、成本、复权、T+1）。这些由回测引擎处理。

## 目录结构

```
backtest/strategy/
├── __init__.py              # 导出核心 API
├── config.py                # 策略配置定义、YAML/JSON 加载与验证
├── base.py                  # 抽象策略基类 StrategyBase
├── universe.py              # Universe 筛选器
├── weight.py                # 权重分配器
├── neutralize.py            # 中性化（行业/市值）
├── signals.py               # 信号格式化（策略输出 → 引擎输入）
├── strategies/
│   ├── __init__.py
│   ├── single_factor.py     # 单因子策略：topK / long-short / 分层
│   └── multi_factor.py      # 多因子组合策略
```

## 核心概念

### 配置系统（YAML）

```yaml
strategy:
  name: "momentum_top20"
  type: "single_factor_topk"
  rebalance_freq: "1W"          # 1D / 1W / 2W / 1M / EOM
  delay: 1                      # T日收盘计算 → T+delay日执行

universe:
  exclude_st: true
  exclude_new_ipo_days: 252
  include_cyb: true
  include_kcb: false
  index_members: "000300.SH"    # null = 全市场
  min_market_cap: 5e8
  min_avg_amount: 1e7

factors:
  - id: "f_001"
    direction: "desc"
    weight: 1.0

selection:
  method: "topk"                # topk / long_short / decile
  top_k: 20
  bottom_k: 20

weighting:
  method: "equal"               # equal / market_cap / factor_value

neutralize:
  industry: false
  industry_method: "group_rank"
  market_cap: false
```

Python 加载：
```python
from backtest.strategy import StrategyConfig
config = StrategyConfig.from_yaml("strategy_config.yaml")
config.validate()
```

### 策略基类

```python
from backtest.strategy import StrategyBase, StrategyConfig

class MyStrategy(StrategyBase):
    def generate_signals(self, factor_panel, market_panel, rebalance_dates):
        # factor_panel: DataFrame [date, symbol, f_001, f_002, ...]
        # market_panel: DataFrame [date, symbol, close, circ_mv, ...]
        # rebalance_dates: list of YYYYMMDD strings
        # return: DataFrame [date, symbol, target_weight]
        ...

strategy = MyStrategy(config)
signals = strategy.run("20200101", "20241231")
```

### Universe 筛选

`UniverseFilter` 在选股前过滤可交易标的：

1. **ST/*ST 过滤**：`is_st == 0`
2. **新股过滤**：`list_date` 到当前日期的交易日数 ≥ 配置值（calendar days 近似）
3. **板块过滤**：创业板（30xxxx）、科创板（68xxxx）可选开关
4. **指数成分股过滤**：查 `index_members` 表（需 data 模块扩展）
5. **流动性过滤**：最小流通市值、最小20日平均成交额

### 选股方式

| 方式 | 说明 | 权重特点 |
|---|---|---|
| **topk** | 选因子排序前 K 只做多 | 正权重，sum ≈ 1.0 |
| **long_short** | 前 K 只做多 + 后 K 只做空 | 多头 sum ≈ 0.5，空头 sum ≈ -0.5 |
| **decile** | 分10组，每组独立等权 | 每组 sum = 1.0，用于单调性分析 |

### 权重分配

- **等权（equal）**：默认，每只股票 1/K
- **市值加权（market_cap）**：按流通市值加权
- **因子值加权（factor_value）**：按因子值绝对值加权

### 中性化

- **市值中性化**：对因子值做 `log(circ_mv)` 的截面回归，取残差。去除因子中的市值暴露，避免策略实质是押注大小盘。
- **行业中性化**（待行业数据）：
  - `group_rank`：每组内做 percentile rank，组间可比
  - `group_zscore`：每组内做 z-score
  - `group_topk`：每组内独立选 topK，保证各行业均衡暴露

### 再平衡频率

- `1D`：日频
- `1W`：每周第一个交易日
- `2W`：双周第一个交易日
- `1M`：每月第一个交易日
- `EOM`：每月最后一个交易日

再平衡日期由 `backtest/data/trade_calendar.py` 生成。

### Delay = 1（A 股 T+1）

策略在 T 日收盘后计算信号，产出的是 **T+1 日的目标持仓**。体现在输出 DataFrame 的 `date` 列上：
- `run()` 内部先按 `rebalance_dates` 计算信号
- 然后 `_apply_delay()` 将日期 forward shift `delay` 个交易日

**关键契约**：
- 策略层只负责 "T+1 日应该持有什么"
- 引擎层负责 "T+1 日开盘能不能买到"（停牌、涨跌停过滤）

### 涨停过滤

策略模块**不处理**涨停/停牌/跌停。这些属于执行层面的不可交易信息，由 engine 负责：
1. 检查当日开盘是否涨停（买入不可执行）/ 跌停（卖出不可执行）/ 停牌
2. 不可交易的标的从目标持仓中剔除
3. 剩余可交易标的重新归一化权重

策略层的 `UniverseFilter` 会做基础流动性过滤（最小市值、最小成交额），目的是减少无效计算，**不替代** engine 的精细过滤。

### 多因子组合

```python
# 多因子配置示例
factors:
  - id: "f_001"
    direction: "desc"
    weight: 1.0
  - id: "f_002"
    direction: "asc"
    weight: 0.5

combine_method: "zscore_equal"   # zscore_equal / ic_weighted / icir_weighted
```

组合方式：
- **zscore_equal**：每个因子先截面 zscore，再按配置 weight 加权求和
- **ic_weighted**：运行时调用 `evaluate()` 计算过去 252 交易日的滚动 IC，按 IC 均值加权
- **icir_weighted**：同上，按 ICIR 加权

## 使用方式

### Python API

```python
from backtest.strategy import (
    StrategyConfig, SingleFactorStrategy, format_signals
)

# 从 YAML 加载配置
config = StrategyConfig.from_yaml("strategy_config.yaml")
config.validate()

# 创建策略并运行
strategy = SingleFactorStrategy(config)
signals = strategy.run("20200101", "20241231")
# signals: DataFrame [date, symbol, target_weight]

# 格式化后传给引擎
formatted = format_signals(signals)
# engine.run_backtest(formatted, ...)
```

### 程序化配置（无需 YAML）

```python
from backtest.strategy import (
    StrategyConfig, UniverseConfig, FactorConfig,
    SelectionConfig, WeightingConfig, SingleFactorStrategy
)

config = StrategyConfig(
    name="momentum_top20",
    strategy_type="single_factor_topk",
    rebalance_freq="1W",
    universe=UniverseConfig(exclude_st=True, include_kcb=False),
    factors=[FactorConfig(id="f_001", direction="desc")],
    selection=SelectionConfig(method="topk", top_k=20),
    weighting=WeightingConfig(method="equal"),
)

strategy = SingleFactorStrategy(config)
signals = strategy.run("20200101", "20241231")
```

## 与上下游的交互

### 上游（因子模块）

```python
FactorStorage.get_factor(factor_id, start, end)       # 单因子时序
FactorStorage.get_factor_panel(factor_ids, date)      # 多因子宽截面（pivot）
```

`base.py` 的 `_load_factor_panel()` 将多个因子时序合并为宽表 `(date, symbol, f_001, f_002, ...)`。

### 下游（回测引擎）

```python
# 策略 → 引擎
signals_df = strategy.run(start, end)
# columns: [date, symbol, target_weight]
# date 是生效日（已含 delay），不是信号计算日

# 引擎内部：
# 1. 按 date groupby，逐日处理
# 2. 检查每只股票当日是否可交易（非停牌、非涨停买入/非跌停卖出）
# 3. 不可交易标的剔除，剩余标的重新归一化权重
# 4. 按收盘价（或 vwap）成交，计算实际持仓变化
# 5. 输出 trades.parquet / positions.parquet / nav.parquet
```

## Data 模块扩展需求

策略模块依赖以下 data 模块新增功能：

| 需求 | 优先级 | 说明 |
|---|---|---|
| 指数成分股 | 高 | `index_members` 表：`symbol, index_code, trade_date, weight`。Tushare `pro.index_weight`。支持 000300.SH / 000905.SH / 000852.SH / 932000.SH / 399006.SZ |
| 申万行业分类 | 高 | `sw_industry` 表：`symbol, industry_code, industry_name, level, trade_date`。Tushare `pro.index_classify` + `pro.index_member` |
| 20日平均成交额 | 中 | 策略层用 `get_bars()` 自行计算，或 market_daily 预计算 |

## 待实现 / 预留

- [ ] 行业中性化：待 `sw_industry` 表落地后，在 `neutralize.py` 中接入真实行业数据
- [ ] 指数成分股过滤：待 `index_members` 表落地后，在 `universe.py` 中接入
- [ ] 回测引擎：接收 `signals` DataFrame，模拟成交，输出净值曲线
- [ ] 分析模块：从 engine 产出计算 Sharpe / 年化 / 回撤 / 归因
- [ ] CLI 入口：`python -m backtest.strategy.run --config strategy_config.yaml`
