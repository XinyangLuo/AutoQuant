# 因子模块

## 定位

因子定义、计算、存储、离线评测。输入来自数据模块（market_daily / financial_statements_q），输出到 `factors.duckdb` 的 `factors_daily` 长表。因子模块**不直接参与回测**——回测只消费 strategy 产出的权重。

## 目录结构

```
backtest/factor/
├── __init__.py              # 导出核心 API
├── registry.py              # 因子注册/查询/元数据管理
├── compute.py               # 因子计算引擎（批量窗口 + PIT隔离）
├── storage.py               # FactorStorage: DuckDB factors.duckdb 读写
├── transforms.py            # 通用算子: rank() 截面归一化、z_score() 时序标准化
├── backfill.py              # 统一回补 CLI
├── update.py                # 统一增量更新 CLI
├── evaluation.py            # 离线评测: IC/RankIC/ICIR/turnover/decay/group_return + 与现有因子相关性
└── builtin/
    ├── __init__.py
    └── momentum.py          # 示例因子: 20日动量 (f_001)
```

## 核心概念

### 因子命名

- **编号命名**：`f_001`, `f_002`, ... 作为稳定唯一键
- **语义别名**：注册时同时记录 `name`（如 `momentum_20d`），便于人类阅读
- **注册表**：`data/factor_library/registry.json` 持久化元数据

### 因子定义

```python
from backtest.factor import register

@register(
    "f_001",
    name="momentum_20d",
    category="momentum",
    data_sources=["market_daily"],
    description="20日收益率动量因子",
    parameters={"window": 20},
)
def momentum_20d(panel: pd.DataFrame) -> pd.Series:
    """
    panel: get_bars() 返回的宽 DataFrame，含 'close' 等列
    返回: MultiIndex (date, symbol) 的 Series
    """
    return panel["close"] / panel["close"].shift(20) - 1
```

### 数据隔离（严禁未来函数）

- **行情因子**：`get_bars(end_date=date)` 天然只返回 `<= date` 的数据
- **财务因子**：必须通过 `get_fina_snapshot(as_of_date=date)`，已封装 `f_ann_date <= date` + `QUALIFY ROW_NUMBER()`
- **核心原则**：因子计算函数**不直接访问数据库**，统一由 `compute.py` 注入 panel 数据

### 通用算子（transforms.py）

为减少重复代码、统一规范，常用变换在 `backtest.factor.transforms` 中提供：

```python
from backtest.factor import rank, z_score

# 截面归一化到 [0, 1]，参考 WorldQuant BRAIN 的 rank()
# 公式: (rank - 1) / (N - 1)，ties 取平均秩
ranked = rank(raw_series)             # 升序，最大值 → 1
ranked_desc = rank(raw_series, ascending=False)

# 时序 z-score，每个 symbol 按 window 滚动
z = z_score(raw_series, window=60)
z_lenient = z_score(raw_series, window=60, min_periods=20)
```

输入与输出均为 MultiIndex `(date, symbol)` 的 `pd.Series`——即因子函数的标准返回类型。在因子函数末尾调用即可，例如：

```python
@register("f_010", name="momentum_rank_20d", ...)
def momentum_rank_20d(panel, window=20):
    df = panel[["date", "symbol", "close"]].copy()
    df = df.sort_values(["symbol", "date"])
    df["raw"] = df.groupby("symbol")["close"].pct_change(window)
    return rank(df.set_index(["date", "symbol"])["raw"])
```

### 存储

- **独立 DuckDB**：`data/duckdb/factors.duckdb`（与 `market.duckdb` 物理分离）
- **Schema**：`(date DATE, symbol VARCHAR, factor_id VARCHAR, value DOUBLE, ann_date VARCHAR, f_ann_date VARCHAR)`
- `ann_date`/`f_ann_date` 仅用于财务因子溯源，非财务因子留空

## 使用方式

### 回补

```bash
# 所有因子全量回补
python -m backtest.factor.backfill --all

# 单个因子回补
python -m backtest.factor.backfill f_001

# 测试模式（最近10个交易日）
python -m backtest.factor.backfill f_001 --test-days 10
```

### 增量更新

```bash
python -m backtest.factor.update
```

### 离线评测

```bash
# Close-to-Close 收益率（默认）
python -m backtest.factor.evaluation f_001 --start 20240101 --end 20241231

# Open-to-Open 收益率
python -m backtest.factor.evaluation f_001 --start 20240101 --end 20241231 --ret-type open

# 自定义 horizon
python -m backtest.factor.evaluation f_001 --start 20240101 --end 20241231 --horizons 1,3,5,10
```

### Python API

```python
from backtest.factor import compute_factor, evaluate, FactorStorage

# 计算因子
df = compute_factor("f_001", "20240101", "20241231")

# 写入存储
with FactorStorage() as fs:
    fs.insert_factors(df)

# 评测
result = evaluate("f_001", "20240101", "20241231", ret_type="close")
print(result.summary())
```

## 评测指标

| 指标 | 定义 |
|---|---|
| **IC** | Pearson 相关系数 `corr(factor_t, ret_{t+h})` |
| **RankIC** | Spearman 秩相关系数 |
| **ICIR** | `mean(IC) / std(IC) * sqrt(252)` |
| **IC>0 占比** | 正 IC 日占比 |
| **t-stat** | IC 序列的 t 统计量 |
| **Turnover** | 相邻两期因子排名的换手率 |
| **Decay** | 不同 horizon 的 IC 衰减曲线 |
| **分组收益** | 按因子值分10组，检验单调性 |
| **与现有因子相关性** | 逐日截面 RankIC（与 `factors_daily` 中每个其他因子配对），按日均值排序输出 top-K；用 `result.max_corr()` 取最大值，避免相似因子重复入库 |

`evaluate(...)` 默认计算所有现有因子的相关性，可用 `include_corr=False` 关闭；CLI 同步提供 `--no-corr` / `--corr-top-k N`。

## 与数据模块的交互

| 消费方 | 提供方 | 函数 |
|---|---|---|
| compute.py | data/storage.py | `get_bars()`, `get_panel()`, `get_fina_snapshot()` |
| backfill.py | data/storage.py | `get_max_date()` (market_daily 边界) |
| evaluation.py | data/storage.py | `get_bars()` (计算未来收益率) |
| evaluation.py | factor/storage.py | `get_factor()` (读取因子值) |
