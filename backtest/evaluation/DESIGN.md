# 评测模块（Evaluation）

## 定位

读 `BacktestResult.save()` 落盘的 parquet 四件套，输出指标表 + 图表 + JSON/CSV。是「从回测产出反推策略质量」的单一入口，**全项目指标计算只在此一处实现**——`BacktestResult.summary()` 与 `scripts/` 中的 `compute_metrics()` 都退化为薄封装。

本模块只做：净值/收益/风险/胜率/交易/持仓六组指标 + 可选的基准对比。**本期不做归因**（个股、板块、行业贡献分解留作 roadmap）。

## 目录结构

```
backtest/evaluation/
├── __init__.py          # 导出公共 API
├── __main__.py          # python -m backtest.evaluation 入口
├── cli.py               # argparse + stdout + 落盘编排
├── loader.py            # BacktestArtifacts dataclass + load_result()
├── metrics.py           # 所有指标的纯函数实现（单一真理源）
├── benchmark.py         # 指数 NAV 加载 + beta/alpha/IR/超额曲线
├── plot.py              # 8 子图大图，Agg 后端
├── report.py            # EvaluationReport + evaluate() + render_table()
└── DESIGN.md            # 本文件
```

## 输入数据契约

读 `BacktestResult.save(output_dir)`（详见 `backtest/simulation/DESIGN.md`）落盘的：

| 文件 | 必须 | 列 |
|---|---|---|
| `nav.parquet` | ✅ | `date, nav, daily_return[, total_value, cash, position_value]` |
| `positions.parquet` | 仅 detailed | `date, symbol, shares, market_value, weight, avg_cost`（长表） |
| `trades.parquet` | 仅 detailed | `trade_date, symbol, direction, shares, price, amount, commission, reason` |
| `metrics.parquet` | 仅 detailed | 日组合统计（`turnover, position_count, cash_ratio, gross_exposure, herfindahl, top5_weight, …`） |
| `metadata.json` | 可选 | `simulation.initial_cash` 等元数据；缺失时 `initial_cash` 默认 1e8（或反推） |

SimpleSimulator 只产 `nav.parquet`；模块自动降级，trading/holdings 段表格显示 `n/a`，相应子图绘"no detailed metrics"。

## 核心数据结构

```python
# loader.py
@dataclass(frozen=True)
class BacktestArtifacts:
    result_dir: Path
    nav: pd.DataFrame                   # 必有
    positions: pd.DataFrame | None
    trades: pd.DataFrame | None
    metrics: pd.DataFrame | None        # 日组合统计
    metadata: dict
    initial_cash: float
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def strategy_id(self) -> str        # metadata.strategy.name 或 result_dir.name


# report.py
@dataclass
class EvaluationReport:
    artifacts: BacktestArtifacts
    benchmark_code: str | None
    bench_nav: pd.Series | None         # 归一到 1.0，按 nav.date reindex+ffill
    metrics: dict                       # 扁平字典，summary.csv 一行
    monthly_returns: pd.DataFrame       # year × month
    yearly_returns: pd.Series
    drawdown: pd.DataFrame              # (date, drawdown)
    rolling_sharpe: pd.Series           # 90 日
    reason_histogram: pd.Series | None  # trades.reason.value_counts()
    bench_metrics: dict
    excess_curve: pd.Series | None

    def to_json(self) -> dict           # summary.json
    def to_dataframe(self) -> pd.DataFrame  # summary.csv（单行）
    def render_markdown(self) -> str
```

## 公共 API

```python
from backtest.evaluation import (
    evaluate,            # 主入口
    render_table,        # 表格渲染
    load_result,         # 仅读 parquet
    compute_all_metrics, # 纯函数计算所有扁平指标
    EvaluationReport,
    BacktestArtifacts,
)

report = evaluate(
    result_dir="results/<factor_id>/<variant>/<tag>/detailed",
    benchmark=None,           # 可选，e.g. "000300.SH"
    plot=True,                # 默认 True
    output_dir=None,         # 默认与 result_dir 同
    rf=0.0,                  # 年化无风险利率
    rolling_sharpe_window=90,
)
print(render_table(report))
```

`evaluate()` 始终落盘 `summary.json` + `summary.csv`；`plot=True` 时另出 `report.png`。

## 指标公式

`nav` 为 NAV 曲线（起点归一到 1.0），`r = nav.pct_change().dropna()`，`n = len(r)`。

### 收益（`compute_return_metrics`）

| 指标 | 公式 |
|---|---|
| `total_return` | `nav[-1] / nav[0] - 1` |
| `annual_return` | **`(nav[-1]/nav[0]) ** (252/n) - 1`**（几何年化；取代旧 `(1+r.mean())**252-1` 的 bug） |
| `annual_volatility` | `r.std(ddof=1) * sqrt(252)` |
| `best_day` / `worst_day` | `r.max()` / `r.min()` |
| `best_month` / `worst_month` | `((1+r).resample('ME').prod()-1).max()` / `.min()` |
| `skewness` / `kurtosis` | `r.skew()` / `r.kurt()`（pandas 默认 Fisher，excess kurtosis） |

附带：`compute_monthly_return_matrix(nav)`（year × month pivot）、`compute_yearly_returns(nav)`（按日历年）。

### 风险（`compute_risk_metrics`）

| 指标 | 公式 |
|---|---|
| `max_drawdown` | `(nav / nav.cummax() - 1).min()` |
| `max_drawdown_start` / `max_drawdown_end` | `mdd_end = idxmin(dd_series)`；`mdd_start = nav[:mdd_end].idxmax()` |
| `recovery_days` | 第一个 `t > mdd_end` 满足 `nav[t] >= nav[mdd_start]` 的 `(t - mdd_end).days`；未恢复返 `None` |
| `avg_drawdown` | `dd_series[dd_series < 0].mean()` |
| `var_95` | `numpy.percentile(r, 5)`（历史法，5 分位） |
| `cvar_95` | `r[r <= var_95].mean()` |

附带：`compute_drawdown_series(nav)`（长表）、`compute_rolling_sharpe(nav, window=90)`。

### 风险调整（`compute_risk_adjusted`）

| 指标 | 公式 |
|---|---|
| `sharpe` | `(annual_return - rf) / annual_volatility` |
| `sortino` | `(annual_return - rf) / (r[r<0].std(ddof=1) * sqrt(252))` |
| `calmar` | `annual_return / abs(max_drawdown)` |
| `information_ratio` | `excess.mean() / excess.std(ddof=1) * sqrt(252)`，仅在有 benchmark 时计算 |

### 胜率（`compute_winrate_metrics`）

| 指标 | 公式 |
|---|---|
| `daily_win_rate` | `(r > 0).mean()` |
| `monthly_win_rate` | `(monthly_ret > 0).mean()` |
| `yearly_win_rate` | `(yearly_ret > 0).mean()` |
| `profit_loss_ratio` | `r[r>0].mean() / abs(r[r<0].mean())` |

### 交易（`compute_trading_stats(trades, metrics_df, initial_cash)`）

- `total_trades = len(trades)`
- `total_commission = trades.commission.sum()`
- `total_stamp_duty = metrics_df.stamp_duty.sum()`（fallback：trades 中 sell/short 的 `amount * 0.001`）
- `total_transfer_fee = metrics_df.transfer_fee.sum()`
- `total_fees`（三者之和）/ `fees_pct_of_initial = total_fees / initial_cash`
- `avg_daily_turnover = metrics_df.turnover.mean()`（已是双边）/ `annual_turnover = avg_daily_turnover * 252`
- `reason_histogram = trades.reason.value_counts()`（保留在 `EvaluationReport`，不进扁平 dict）

### 持仓（`compute_holdings_stats(metrics_df)`）

对 `position_count, long_count, short_count, cash_ratio, gross_exposure, net_exposure, herfindahl, top5_weight, top10_weight` 求 `mean()`，前缀 `avg_`。

### 基准对比（`compute_benchmark_metrics(strat_nav, bench_nav)`）

仅在 `evaluate(..., benchmark=code)` 给出指数代码时计算：

- `bench_nav` 按 `strat_nav.date` reindex + ffill，再 `/ bench_nav.iloc[0]` 归一
- `cum_excess = (1+excess_daily).cumprod() - 1`；`annual_excess = excess_daily.mean() * 252`
- `tracking_error = excess_daily.std(ddof=1) * sqrt(252)`
- `information_ratio = annual_excess / tracking_error`
- `beta, alpha_daily = numpy.polyfit(r_bench, r_strat, 1)`；`alpha_annual = alpha_daily * 252`
- `excess_max_drawdown = excess_dd.min()`（在 `1 + cum_excess` 上做回撤）
- `corr = numpy.corrcoef(r_strat, r_bench)[0,1]`

### 顶层合并

`compute_all_metrics(artifacts, bench_nav=None, rf=0.0) -> dict`：合并以上六组指标到一个扁平 dict。
DataFrame 形态的附属产出（monthly_returns / yearly_returns / drawdown / rolling_sharpe / excess_curve / reason_histogram）由 `evaluate()` 独立存到 `EvaluationReport`。

## 绘图

`matplotlib.use("Agg")`，`figsize=(16, 32)`，`dpi=150`，8 个垂直 subplot，英文标签。色板与 `tight_layout(rect=[0,0,1,0.985])` / `bbox_inches="tight"` 复刻 `backtest/factor/evaluation.py` 风格。

| # | 内容 | 备注 |
|---|---|---|
| 1 | Strategy NAV (+ benchmark NAV 叠加) | steelblue 主曲线，darkorange 叠加；标题带 total/ann return |
| 2 | Drawdown underwater | red fill_between；标题带 max_dd 与起止日期 |
| 3 | Monthly return heatmap (year × month) | `RdYlGn` cmap，0 居中；单元格标 `+X.X%` |
| 4 | Yearly returns bar | 正绿负红；柱顶标 `+X.X%` |
| 5 | Position count + cash ratio | twin axes：左 position_count(steelblue) / 右 cash_ratio(gray dashed) |
| 6 | Daily turnover | 紫色 alpha=0.6 + 20 日滚动均线（黑色） |
| 7 | Daily return histogram | hist 60 bins + 正态曲线红虚线 + var/cvar 垂线 |
| 8 | Rolling Sharpe (90d) | teal 实线 + axhline(mean) |

SimpleSimulator 容错：`metrics_df is None` 时第 5、6 子图绘 "no detailed metrics" 文本占位，其余子图照常。

## CLI

```bash
python -m backtest.evaluation <result_dir> \
    [--benchmark 000300.SH] [--no-plot] [--rf 0.0] [--output-dir DIR] [--rolling-window 90]
```

行为：
1. `evaluate(result_dir, benchmark=..., plot=..., rf=...)` 拿到 `EvaluationReport`
2. `print(render_table(report))` 控制台输出 markdown 表（分组：Return / Risk-Adjusted / Risk / Win Rate / Trading / Holdings / Benchmark / Yearly / Trade Reasons）
3. 落盘 `summary.json` + `summary.csv` + `report.png`（无 `--no-plot` 时）

## 基准指数数据通道（side-quest）

当前 `backtest/data/storage.py` 没有指数表。本期顺手把通道补上：

### `backtest/data/storage.py` 新增

```python
INDEX_DAILY_COLUMNS = [
    "date", "symbol",
    "open", "high", "low", "close",
    "pre_close", "change", "pct_chg",
    "volume", "amount",
]
INDEX_DAILY_SCHEMA = """
CREATE TABLE IF NOT EXISTS index_daily (
    date DATE, symbol VARCHAR,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    pre_close DOUBLE, change DOUBLE, pct_chg DOUBLE,
    volume DOUBLE, amount DOUBLE,
    PRIMARY KEY (date, symbol)
)
"""
```

`_init_tables()` 注册；`MarketStorage` 加：

- `insert_index_daily(df)` — 按 `(date, symbol)` upsert
- `get_index_bars(symbols, start=None, end=None) -> pd.DataFrame` — 与 `get_bars()` 同风格
- `get_max_index_date(symbol) -> str | None` — 增量回补

### `backtest/data/index_fetcher.py`（新）

包一层 `pro.index_daily(ts_code, start_date, end_date)`，列重命名 `trade_date→date / ts_code→symbol / vol→volume`，date 转 `datetime.date`。

### `backtest/data/backfill_indices.py`（新，模仿 `backfill_daily_basic.py`）

- 默认指数：`["000300.SH", "000905.SH", "000852.SH", "000001.SH", "399006.SZ"]`
- 对每个 symbol：`get_max_index_date()` → 从下一交易日开始 fetch → `insert_index_daily()`
- CLI：`python -m backtest.data.backfill_indices [--symbols A,B]`（日期范围由脚本内部根据已有数据自动决定）

### `evaluation.benchmark.load_benchmark(code, start, end)`

```python
with MarketStorage() as ms:
    df = ms.get_index_bars([code], start=start, end=end)
nav = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float).sort_index()
return nav / nav.iloc[0]
```

## 与上下游的交互

### 上游（simulation 产出）

```python
from backtest.simulation import DetailedSimulator
result = sim.run(signals, market_data, dividends_data)
result.save("results/<factor_id>/<variant>/<tag>/detailed/")
```

### 上游（数据模块）—— 仅当有 benchmark 时

```python
MarketStorage.get_index_bars([code], start, end)   # 新增方法
```

### 下游（scripts / agent）

- `scripts/backtest_f_rev_05.py`：删 `compute_metrics()`，改 `evaluate(...) + print(render_table(...))`
- Agent 投研系统：`get_backtest_result(result_id)` 工具底层调 `evaluate()` 拿扁平 dict

## `BacktestResult.summary()` 退化为薄封装

`backtest/simulation/models.py:126-179` 改为：

```python
def summary(self) -> dict:
    if self.nav_df is None or self.nav_df.empty or len(self.nav_df) < 2:
        return {}
    from backtest.evaluation.loader import BacktestArtifacts
    from backtest.evaluation.metrics import compute_all_metrics
    from pathlib import Path
    arts = BacktestArtifacts(
        result_dir=Path("."),
        nav=self.nav_df,
        positions=self.positions_df,
        trades=self.trades_df,
        metrics=self.metrics_df,
        metadata={},
        initial_cash=self.initial_cash,
        start=pd.to_datetime(self.nav_df["date"]).min(),
        end=pd.to_datetime(self.nav_df["date"]).max(),
    )
    return compute_all_metrics(arts, bench_nav=None)
```

**已知行为变化**：`annual_return` 不再是 `(1+r.mean())**252-1`（算术），改为 `(nav_end/nav_start)**(252/n)-1`（几何）。在 ~1 年回测中两者数量级差距明显，是修 bug。其他指标向后兼容。

## 设计原则

- **只读**：模块只消费 simulation 产出的 parquet，绝不修改原文件
- **纯函数**：所有 `compute_*` 函数无副作用，便于单测
- **单一指标源**：`BacktestResult.summary()` 与 scripts 都委托到本模块，避免公式漂移
- **降级容错**：SimpleSimulator 只产 `nav.parquet`，本模块自动跳过 trading/holdings 段
- **不做归因**：归因要拿到 returns × symbol 矩阵 + 板块/行业映射，依赖 `backtest/data/sw_industry`（未落地）。Roadmap

## TODO / 路线图

- [ ] 个股贡献 Top/Bottom 10（基于 `positions × daily_return`，无需新表）
- [ ] 板块/行业归因（等 `backtest/data/sw_industry` 落地）
- [ ] 多策略横向对比表（输入多个 `result_dir`）
- [ ] 滚动窗口对比（in-sample vs out-of-sample）
- [ ] Brinson 归因（行业表 + 基准成分股，依赖 `index_members`）
