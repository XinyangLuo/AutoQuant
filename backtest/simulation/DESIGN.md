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
├── models.py           # Trade, Position, DailySnapshot, BacktestResult, DecileBacktestResult
├── config.py           # SimulationConfig
├── simple.py           # SimpleSimulator
├── detailed.py         # DetailedSimulator
├── decile.py           # DecileSimulator + plot_decile_backtest
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

## 关键设计原则

- **信号即真相**：策略产出的 `signals.date` 列是引擎的唯一输入。哪天没有行，引擎就不调仓。引擎不感知 `rebalance_freq`——日频/周频/月频对引擎逻辑完全一致。
- **Prior-day NAV 计算**：rebalance 前先以上一日收盘市值确定仓位 size，rebalance 完成后再用当日收盘更新市值。该逻辑在所有频率下均成立。
- **双轨并行**：`SimpleSimulator` 用于因子研究/参数扫描（快），`DetailedSimulator` 用于策略实盘前验证（真），`DecileSimulator` 用于因子单调性分析（分层）。

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
    # 向量化：长表 forward return 与 signals 按 (date, symbol) 对齐，
    # 按日聚合 weight × return 后 cumprod 得净值；不构造 date × symbol dense pivot

    def run_batch(self, signals, market_data, strategy_col="combo_tag") -> dict[str, BacktestResult]
    # 参数 sweep 快路径：signals 在普通 [date, symbol, target_weight] 外增加策略维度列，
    # 一次计算 forward return，按 (strategy_col, date) 稀疏聚合；不构造 strategy × date × symbol dense cube
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

### DecileSimulator

```python
class DecileSimulator:
    def run(self, factor_df, market_data) -> DecileBacktestResult
    # 向量化：每日按因子值分 10 组 → 每组等权 → 组内 adj_price 计算日收益 → cumprod 得 NAV
    # delay=1 内置：T 日因子 → T+1 日开盘/收盘交易 → T+2 日收益
    # 同时计算 long-short 组合（最大 decile - 最小 decile）和单调性得分（Spearman）
```

输入 `factor_df: [date, symbol, value]`；`market_data: [date, symbol, close, open, adj_factor]`。

`DecileBacktestResult` 包含：
- `nav_df`: `[date, d0_nav, ..., d9_nav, ls_nav]`
- `decile_metrics`: 每组 `compute_single_nav_metrics()` 结果
- `ls_metrics`: long-short 组合指标
- `monotonicity_score`: 年化收益与 decile 排名的 Spearman 相关系数

`plot_decile_backtest(result, output_path)` 绘制 2-panel 图：
- 上：10 条 decile NAV 曲线（log 轴，RdYlGn 色板，D1/D10 加粗）
- 下：long-short NAV + 单调性得分标题

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
- [x] Benchmark 支持（`index_daily` 表 + `backtest.data.backfill.indices` 已落地）
- [ ] 详细评测（归因、分层等）留给 analysis 模块

---
