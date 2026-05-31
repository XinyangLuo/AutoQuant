# 回测引擎（Simulation）

## 定位

把策略产出的「每日目标持仓」模拟成「净值曲线 + 交易日志 + 持仓记录」。

双轨设计：
- **SimpleSimulator**：向量化快速回测，用复权价格。用于因子研究、参数扫描。
- **DetailedSimulator**：逐日事件驱动回测，用实际价格，模拟 broker 交易。用于策略实盘前验证。

## 目录结构

```
backtest/simulation/
├── __init__.py         # 导出公共 API
├── models.py           # Trade, Position, DailySnapshot, BacktestResult
├── config.py           # SimulationConfig
├── simple.py           # SimpleSimulator
├── detailed.py         # DetailedSimulator
├── executor.py         # OrderExecutor（涨跌停、停牌、手续费）
├── dividends.py        # DividendHandler（分红送转）
└── utils.py            # 板块识别、交易单位取整、停牌检测
```

## 核心概念

### 输入格式

```python
signals: pd.DataFrame
# columns: date (datetime), symbol (str), target_weight (float)
# date = 持仓生效日（已由 strategy 应用 delay=1）
# target_weight 可正可负
```

### 价格体系

| 回测类型 | 价格 | 分红处理 |
|---|---|---|
| SimpleSimulator | `adj_close = close * adj_factor` | 不复权，忽略分红 |
| DetailedSimulator | 实际价格（非复权） | ex_date 送转股调整股数，pay_date 现金分红到账 |

### 交易单位（按板块）

| 板块 | 代码特征 | 规则 |
|---|---|---|
| 科创板 | 688xxx.SH | 200 股起，超过按 1 股递增 |
| 北交所 | 8xxxxx.BJ / 4xxxxx.BJ | 100 股起，超过按 1 股递增 |
| 主板/创业板 | 60xxx/00xxx/30xxx | 100 股整数倍 |

**规则**：目标股数 < 最低单位 → 跳过不买。

### 涨跌停规则

**o2o（默认，open-to-open）**：
- buy/cover：涨停开盘且 low ≈ limit_up → 不能成交；low < limit_up（盘中打开）→ 可用 limit_up 成交
- sell/short：跌停开盘且 high ≈ limit_down → 不能成交；high > limit_down（盘中打开）→ 可用 limit_down 成交

**c2c（close-to-close）**：
- buy/cover：close ≈ limit_up → 不能成交
- sell/short：close ≈ limit_down → 不能成交

### 停牌与退市

- **停牌**：Tushare 停牌股票无 daily 数据。数据中某日期无某股票 → 该日停牌。保持持仓，不交易。
- **退市**：通过 `stock_basic` 表 `list_status='D'` + `delist_date` 判断。回测结束时退市持仓清零。

### 手续费

- **佣金**：双向，`max(amount * rate, 5元)`，默认 rate=0.0003
- **印花税**：仅卖出/short，rate=0.001
- **过户费**：双向，rate=0.00002

## 类设计

### SimulationConfig

```python
@dataclass
class SimulationConfig:
    initial_cash: float = 100_000_000.0
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.001
    transfer_fee_rate: float = 0.00002
    price_type: Literal["o2o", "c2c"] = "o2o"
    allow_short: bool = True
    benchmark: str | None = None   # 主基准（用于 evaluate() 主基准对比；summary() 自动加载 HS300/CSI500/CSI1000）
```

### SimpleSimulator

```python
class SimpleSimulator:
    def run(self, signals, market_data) -> BacktestResult
    # 向量化：weight × adj_close.pct_change()，cumprod 得净值
```

### DetailedSimulator

```python
class DetailedSimulator:
    def run(self, signals, market_data, dividends_data=None) -> BacktestResult
    # 逐日循环：分红 → 更新市值 → 调仓信号 → 执行订单 → 记录快照
```

### OrderExecutor

```python
class OrderExecutor:
    def can_trade(self, symbol, direction, bar) -> tuple[bool, float | None, str]
    # 返回 (能否交易, 成交价格, 原因)

    def calculate_cost(self, amount, direction) -> float
    # 佣金 + 印花税 + 过户费
```

### DividendHandler

```python
class DividendHandler:
    def apply(self, date, portfolio, dividends) -> list[dict]
    # ex_date: 送转股调整股数
    # pay_date: 现金分红增加现金
```

### BacktestResult

```python
class BacktestResult:
    nav_df: pd.DataFrame        # date, nav, daily_return, ...
    positions_df: pd.DataFrame  # date, symbol, shares, market_value, ...
    trades_df: pd.DataFrame     # trade_date, symbol, direction, shares, price, ...

    def save(self, output_dir) -> None
    def summary(self) -> dict   # 总收益, 年化, Sharpe, 最大回撤 + 自动加载 HS300/CSI500/CSI1000 超额指标
```

## 与上下游的交互

### 上游（策略模块）

```python
signals = strategy.run(start, end)        # DataFrame [date, symbol, target_weight]
formatted = format_signals(signals)        # 标准化
```

### 上游（数据模块）

```python
MarketStorage.get_bars(symbols, start, end, columns=[...])  # 行情
MarketStorage.get_dividends(symbols, start, end)            # 分红
MarketStorage.get_stock_status(symbol, date)                # L/D/S 状态
```

### 下游（分析模块）

```python
result = sim.run(signals, market_data)
result.save("results/<factor_id>/<variant>/<tag>/detailed/")
# analysis_module.load(result) -> Sharpe, 归因, 可视化
```

## Data 模块扩展需求

| 需求 | 说明 |
|---|---|
| stock_basic 表 | 存储上市/退市/板块状态 |
| name_changes 表 | 存储更名记录 |
| get_dividends() | 查询分红送转事件 |
| get_stock_status() | 查询某股票某日状态（L=上市, D=退市, S=停牌） |

## TODO

- [ ] T+1 交割制度（当前假设当日可完成全部调仓）
- [ ] Benchmark 支持（数据模块需先支持指数行情）
- [ ] 详细评测（归因、分层等）留给 analysis 模块

---

# P0 实施计划

## P0-3: 交易日历表（simulation 模块部分）

### 结论：**引擎无需改动**

`detailed.py:194-277` 已按"信号即真相"驱动：

```python
dates = sorted(set(bar_by_date.keys()))   # 遍历所有交易日
for date_str in dates:
    sig_df = signal_by_date.get(date_str)
    if sig_df is not None:                # 仅当当天有信号时调仓
        _rebalance(...)
```

策略层把 `rebalance_freq='1M'` 转换为「仅在每月首个交易日有 target_weight 行」的 signals DataFrame，引擎自动只在那些日期触发 `_rebalance`，其余交易日只跑分红 + 净值更新。SimpleSimulator 同理。

### 契约（文档化即可，无代码变更）
- **策略产出的 signals.date 列**是真相：哪天没行就不调仓
- **引擎不感知 rebalance_freq**：无论日频/周频/月频，引擎逻辑完全一致
- 7d13ad8 commit 修过的 prior-day NAV 计算（rebalance 前用上一日收盘市值定 size，rebalance 后再用今日收盘更新）在所有频率下都成立

### 完成标准
- [ ] 跑一个 `freq='1M'` 的回测，确认 `trades.parquet` 仅在每月首个交易日有行
- [ ] 同因子同 universe，对比 `freq='1D'` vs `freq='1M'`，确认换手率显著下降（应该 ~1/20）
- [ ] 把上述「契约」一段并入正文「关键设计原则」节，删除本 P0 节
