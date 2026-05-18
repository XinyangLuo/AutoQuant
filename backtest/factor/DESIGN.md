# 因子模块

## 定位

因子定义、计算、存储、离线评测、入库决策。

**两个物理 DuckDB**：

- `data/duckdb/factors.duckdb` —— **工作区**。`backfill` / `compute` / `evaluation` 期间临时数据落在这里。研究中的新因子的"住所"，临时。
- `data/duckdb/factor_library.duckdb` —— **稳定库**。只有 `admit()` 会写入。Evaluation 的"与现有因子相关性"检查只读这个库，避免临时数据相互污染。

因子模块**不直接参与回测**——回测只消费 strategy 产出的权重。

## 目录结构

```
backtest/factor/
├── __init__.py              # 导出核心 API
├── registry.py              # 因子注册 / 元数据 (registry.json)
├── compute.py               # 因子计算引擎 (PIT 隔离 + 批量窗口)
├── storage.py               # FactorStorage (work) + FactorLibrary (library)
├── transforms.py            # 通用算子: rank() / z_score()
├── backfill.py              # 把因子值算到 work DB
├── update.py                # 增量更新 library DB (已 admitted 因子)
├── evaluation.py            # 离线评测: IC/RankIC/ICIR/turnover/decay/corr
├── admission.py             # admit / reject / status —— 看完报告后人工触发
├── cleanup.py               # 清 work DB 临时数据
└── builtin/
    ├── __init__.py
    └── ...                  # 内置因子定义
```

## 核心概念

### 因子命名

- **编号命名**：`f_001`, `f_002`, ... 作为稳定唯一键
- **语义别名**：注册时同时记录 `name`（如 `momentum_20d`），便于人类阅读
- **注册表**：`data/factor_library/registry.json` 持久化元数据 + status

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

```python
from backtest.factor import rank, z_score

# 截面归一化到 [0, 1]，参考 WorldQuant BRAIN 的 rank()
ranked = rank(raw_series)             # 升序，最大值 → 1
ranked_desc = rank(raw_series, ascending=False)

# 时序 z-score，每个 symbol 按 window 滚动
z = z_score(raw_series, window=60)
z_lenient = z_score(raw_series, window=60, min_periods=20)
```

输入与输出均为 MultiIndex `(date, symbol)` 的 `pd.Series`。

### 双库存储

```
┌─────────────────────────────────┐
│ data/duckdb/factors.duckdb      │  ← FactorStorage (work)
│                                 │     · backfill / compute 写入
│  factors_daily                  │     · evaluation 读取
│  (date, symbol, factor_id,      │     · admit 后清空
│   value, ann_date, f_ann_date)  │     · 没有 status='admitted' 的概念
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ data/duckdb/factor_library.duckdb │  ← FactorLibrary (library)
│                                 │     · 只有 admit() 写入
│  factors_daily                  │     · evaluation 的 corr 比较读这里
│  (same schema)                  │     · update 增量维护
│                                 │     · delete_factor() 被禁用 (append-only)
└─────────────────────────────────┘
```

**Schema 完全一致**，区别只在数据生命周期：work 是研究 churn，library 是已稳定的事实。

`ann_date` / `f_ann_date` 仅用于财务因子溯源，非财务因子留空。

## 使用方式

### 1. 回填（写 work DB）

```bash
# 单因子回填到 work
python -m backtest.factor.backfill f_001

# 测试模式（最近 60 个交易日）
python -m backtest.factor.backfill f_001 --test-days 60

# 所有 pending 因子（未 admit、未 reject）批量回填到 work
python -m backtest.factor.backfill --pending
```

### 2. 离线评测（读 work + library）

```bash
# 因子指标 + 与 library 因子的相关性
python -m backtest.factor.evaluation f_001 --start 20210101 --end 20241231 --plot
```

输出末尾会打印对照 `RECOMMENDED_THRESHOLDS` 的 4 项检查（informational only，不 gate）：

```
--- Reference thresholds (primary_horizon=20, informational only) ---
  RankICIR      = +0.3140  (>= 0.25)  OK
  IC+ ratio     =  55.20%  (>= 52%)   OK
  Turnover      =  0.4220  (<  0.5)   OK
  Max |corr|    =  0.7800  (<  0.85)  OK
  → All reference thresholds met. Run a backtest and decide on `admit`.
```

### 3. 完整 pipeline driver（推荐）

```bash
python scripts/run_factor_pipeline.py f_001 \
    --start 20210101 --end 20241231 \
    --top-n 50 --rebalance 1W --decay 5 \
    --direction desc --benchmark 000300.SH
```

输出到 `results/<factor_id>/{factor_eval, simple, detailed}/`。

### 4. 入库 / 拒绝（人工触发）

看完三层报告后，人工运行：

```bash
# 把 factor 从 work DB 迁移到 library DB，更新 registry status=admitted
python -m backtest.factor.admission admit  f_001 --notes "Sharpe 1.45, IR 0.92 vs 000300"

# 清掉 work DB 的临时数据，更新 status=rejected
python -m backtest.factor.admission reject f_001 --notes "RankICIR 仅 0.18"

# 查看所有因子状态
python -m backtest.factor.admission status
python -m backtest.factor.admission status f_001
```

### 5. 临时数据清理

```bash
# 单因子清空 work，不改 status（保持 pending）
python -m backtest.factor.cleanup f_001

# 清空整个 work DB
python -m backtest.factor.cleanup --all

# 清掉 work 中已经 admit 到 library 的孤儿数据（崩溃恢复用）
python -m backtest.factor.cleanup --orphans
```

### 6. 增量更新（library DB）

```bash
python -m backtest.factor.update    # 把 admitted 因子追平到 market_daily 最新一天
```

只更新 library 库的 admitted 因子；从不写 work。

### Python API

```python
from backtest.factor import (
    compute_factor, evaluate, FactorStorage, FactorLibrary,
    admit, reject, get_admitted_factor_ids,
)

# 计算因子并写 work
df = compute_factor("f_001", "20210101", "20241231")
with FactorStorage() as fs:
    fs.insert_factors(df)

# 评测
result = evaluate("f_001", "20210101", "20241231", ret_type="open")
print(result.summary())
print(result.threshold_metrics(20))

# 看完回测人工决定后:
admit("f_001", notes="Sharpe 1.45")    # work → library, status=admitted
# 或
reject("f_001", notes="ICIR 不达标")   # 清 work, status=rejected
```

## 评测指标

| 指标 | 定义 |
|---|---|
| **IC** | Pearson 相关系数 `corr(factor_t, ret_{t+h})` |
| **RankIC** | Spearman 秩相关系数 |
| **ICIR** | `mean(IC) / std(IC)` |
| **IC>0 占比** | 正 IC 日占比 |
| **t-stat** | IC 序列的 t 统计量 |
| **Turnover** | 相邻两期因子排名的换手率 |
| **Decay** | 不同 horizon 的 IC 衰减曲线 |
| **分组收益** | 按因子值分10组，检验单调性 |
| **与现有因子相关性** | 与 **library DB** 中每个因子的逐日截面 RankIC，按日均值排序输出 top-K |

`evaluate(...)` 默认计算所有 library 因子的相关性，可用 `include_corr=False` 关闭；CLI 同步提供 `--no-corr` / `--corr-top-k N`。

## 参考阈值（不强制 gate）

```python
RECOMMENDED_THRESHOLDS = {
    "min_rankicir": 0.25,
    "min_ic_positive_ratio": 0.52,
    "max_turnover": 0.5,
    "max_corr": 0.85,
    "primary_horizon": 20,
    "ret_type": "open",
    "exclude_limit_up": True,
}
```

`check_recommended_thresholds(metrics)` 返回 `{check: bool}` dict，仅用于评测打印和决策辅助。`admit()` 不做检查——是否入库由人类看完三层报告自行决定。

## 与数据模块的交互

| 消费方 | 提供方 | 函数 |
|---|---|---|
| compute.py | data/storage.py | `get_bars()`, `get_panel()`, `get_fina_snapshot()` |
| backfill.py | data/storage.py | `get_max_date()` (market_daily 边界) |
| evaluation.py | data/storage.py | `get_bars()` (计算未来收益率) |
| evaluation.py | factor/storage.py | `FactorStorage.get_factor()` + `FactorLibrary.get_factors_long()` |

## 端到端流程

```
            ┌─────────────────┐
            │ 1. @register    │
            │    定义因子代码 │
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │ 2. backfill     │
            │    写 work DB   │
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │ 3. evaluate     │
            │    读 work DB   │
            │    + library corr│
            └────────┬────────┘
                     ▼
            ┌─────────────────────┐
            │ 4. run_pipeline     │
            │    simple + detailed│
            └────────┬────────────┘
                     ▼
                 [人工判断]
                 ┌────┴────┐
                 ▼         ▼
            ┌────────┐ ┌──────────┐
            │ admit  │ │  reject  │
            │ work→lib│ │  清 work │
            │ 清 work │ │          │
            └────────┘ └──────────┘
```
