# 因子模块

## 定位

因子定义、计算、存储、离线评测、入库决策。

**两个物理 DuckDB**：

- `data/duckdb/factors_pending.duckdb` —— **工作区**。`backfill` / `compute` / `evaluation` 期间临时数据落在这里。研究中的新因子的"住所"，临时。任何代码都能写。
- `data/duckdb/factor_library.duckdb` —— **稳定库**。只有 `admit()` 会写入。`FactorLibrary.insert_factors` 强制要求 `factor_id` 在 registry 中 `status='admitted'`，否则抛 `PermissionError`（`allow_unadmitted=True` 旁路仅供 `promote_from_work` / 测试 seeding）。Evaluation 的"与现有因子相关性"检查只读这个库，避免临时数据相互污染。

因子模块**不直接参与回测**——回测只消费 strategy 产出的权重。

## 目录结构

```
backtest/factor/
├── __init__.py              # 导出核心 API + 在末尾自动 import 顶层 alphas/
├── registry.py              # 因子注册 / 元数据 (registry.json)
├── compute.py               # 因子计算引擎 (PIT 隔离 + 批量窗口)
├── storage.py               # FactorStorage (work) + FactorLibrary (library)
├── transforms.py            # 通用算子: rank() / z_score()
├── backfill.py              # 把因子值算到 work DB
├── update.py                # 增量更新 library DB (已 admitted 因子)
├── evaluation.py            # 离线评测: IC/RankIC/ICIR/turnover/decay/corr
├── admission.py             # admit / reject / status —— 看完报告后人工触发
├── cleanup.py               # 清 work DB 临时数据
└── builtin/                 # 引擎自带的结构性因子（Barra 风险模型）
    ├── __init__.py
    └── barra/               # Barra 风格因子
```

**私有 alpha 代码**位于仓库顶层 `/alphas/`（已 gitignored，不公开到远端）。`backtest/factor/__init__.py` 在末尾会 `try: import alphas` 触发 `@register` 注册。`alphas/` 与 `backtest/factor/builtin/` 的边界：

| | builtin/barra | alphas/ |
|---|---|---|
| 性质 | 引擎结构件，中性化必用 | 候选 alpha，可增可减 |
| 追踪 | 入 git | 不入 git |
| factor_id 前缀 | `f_barra_*` | 自由（建议 `f_*` 编号） |

## 核心概念

### 因子命名

- **编号命名**：`f_001`, `f_002`, ... 作为稳定唯一键
- **语义别名**：注册时同时记录 `name`（如 `momentum_20d`），便于人类阅读
- **注册表**：`data/factor_library/registry.json` 持久化元数据 + status

### 中性化变体（variant）

每个因子在 registry 中记录**一个** `variant` 标签 —— 它描述了 compute → apply pipeline 用的是哪条后处理路径。值已经 baked-in，存 `factors_daily` 只占一列。

| variant | pipeline | 用途 |
|---|---|---|
| `none` | 直通：compute 返回什么就存什么 | Barra L1 composite（内部对每个 L3 分量自己跑过 L3 pipeline 再等权平均） |
| `barra_l3` | MAD winsorize → SW-L1 行业中位数填充 → cs_zscore | 风格暴露因子。Barra L1 composite 内部即时调用，**不**留库 |
| `barra_ind_size` | 上面三步 + 截面 OLS 对 (industry dummies + Size_z) 取残差 → 再 cs_zscore | **用户 alpha 默认** —— 剥离行业 + 市值风格暴露，剩纯 alpha |

`DEFAULT_VARIANT = "barra_ind_size"`，未显式声明的 `@register` 因子自动走这条路径。Size_z 来自 `f_barra_size`（已 admitted 到 library），因此 `f_barra_size` 是 alpha pipeline 的硬依赖。

要切换 variant：重跑 `backfill`，列上的值被覆盖即可（PK 是 `(date, symbol)`，没有 variant 维度）。

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
    # variant 默认 "barra_ind_size"，alpha 因子无需声明。
    # 风格因子（Barra L1 composite）显式声明 variant="none"。
)
def momentum_20d(panel: pd.DataFrame) -> pd.Series:
    """
    panel: get_bars() 返回的宽 DataFrame，含 'close' 等列
    返回: MultiIndex (date, symbol) 的 Series（raw 值；中性化 pipeline 由
          apply_variant_pipeline 在 backfill 后做，不是 compute 函数自己做）
    """
    return panel["close"] / panel["close"].shift(20) - 1
```

### 数据隔离（严禁未来函数）

- **行情因子**：`get_bars(end_date=date)` 天然只返回 `<= date` 的数据
- **财务因子**：必须通过 `get_fina_snapshot(as_of_date=date)`，已封装 `f_ann_date <= date` + `QUALIFY ROW_NUMBER()`
- **核心原则**：因子计算函数**不直接访问数据库**，统一由 `compute.py` 注入 panel 数据

### 通用算子（transforms.py）

所有算子输入 / 输出均为 MultiIndex `(date, symbol)` 的 `pd.Series`。按用途分为三类：截面归一化、时序变换、中性化。

```python
from backtest.factor import (
    rank, z_score,           # 截面 + 时序
    ts_rank, ts_mean, ts_std,  # 纯时序滚动
)
from backtest.factor.transforms import industry_neutralize, cap_neutralize
```

#### 截面算子

| 算子 | 功能 | 输出范围 | 典型用法 |
|---|---|---|---|
| `rank(s, ascending=True)` | 每日截面排名，归一化到 `[0, 1]` | `[0, 1]` | `rank(s)` 消除量纲，便于多因子组合 |

```python
ranked = rank(raw_series)              # 升序，最大值 → 1
ranked_desc = rank(raw_series, ascending=False)  # 降序，最大值 → 0
```

#### 时序算子

| 算子 | 功能 | 输出范围 | 参数 |
|---|---|---|---|
| `ts_rank(s, window)` | 每个 symbol 在滚动窗口内的排名，缩放到 `[-1, 1]` | `[-1, 1]` | `window`, `min_periods` |
| `ts_mean(s, window)` | 滚动均值 | 原值域 | `window`, `min_periods` |
| `ts_std(s, window)` | 滚动标准差 (ddof=1) | `≥ 0` | `window`, `min_periods` |
| `z_score(s, window)` | 滚动 z-score：`(x - μ) / σ` | 无界 | `window`, `min_periods` |

```python
# ts_rank: 过去 20 日内的排名，-1=窗口最小，1=窗口最大
tr = ts_rank(raw_series, window=20)

# ts_mean / ts_std: 过去 60 日均值与波动
m60 = ts_mean(raw_series, window=60)
v60 = ts_std(raw_series, window=60)

# z_score: 过去 60 日 z-score（与 ts_rank 不同：z_score 用标准化值，ts_rank 用秩次）
z60 = z_score(raw_series, window=60, min_periods=20)
```

**`ts_rank` 与 `rank` 的区别**：
- `rank` 是**截面**的（同一天不同股票之间比），输出 `[0, 1]`
- `ts_rank` 是**时序**的（同一股票不同日期之间比），输出 `[-1, 1]`

#### 中性化算子

`industry_neutralize` / `cap_neutralize` 是历史算子，**当前 pipeline 已不用**——`barra_l3` / `barra_ind_size` 直接走 `cs_mad_winsorize` + `industry_median_fill` + `cs_zscore` + `cs_ols_residualize`，逻辑都在 `compute.apply_variant_pipeline` 里。这两个算子保留作为低阶工具，写非标准 pipeline 时仍可用：

```python
ind_neutral = industry_neutralize(raw_series, industry_panel)
cap_neutral = cap_neutralize(raw_series, cap_panel, cap_field='circ_mv', quantiles=5)
```

### 双库存储

```
┌──────────────────────────────────────────────────────┐
│ data/duckdb/factors_pending.duckdb                   │  ← FactorStorage (work)
│                                                      │     · backfill / compute 写入
│  factors_daily                                       │     · evaluation 读取
│  (date DATE, symbol VARCHAR,                         │     · admit() 后该列被迁移走
│   f_xxx_1 DOUBLE, f_xxx_2 DOUBLE, ...)              │     · 任意代码可写
│  PK (date, symbol)                                   │
│  每个 factor_id 一列。加因子 = ALTER TABLE ADD COLUMN│
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ data/duckdb/factor_library.duckdb                    │  ← FactorLibrary (library)
│                                                      │     · 只有 admit() 写入
│  factors_daily (same schema)                         │     · FactorLibrary.insert_factors
│                                                      │       强制 status='admitted' 不变量
│                                                      │     · evaluation 的 corr 比较读这里
│                                                      │     · update 增量维护
│                                                      │     · delete_factor 被禁用 (append-only)
└──────────────────────────────────────────────────────┘
```

**Schema 完全一致**（宽表，PK `(date, symbol)`，每个 factor_id 占一列），区别只在数据生命周期：work 是研究 churn，library 是已稳定的事实。PIT 元数据（`ann_date` / `f_ann_date`）不在因子表里——隔离由 `compute.py` 调 `get_fina_snapshot(D)` 时上游保证，审计追溯走 market.duckdb 的财报表。

## 使用方式

### 1. 回填（写 work DB）

```bash
# 单因子回填
python -m backtest.factor.backfill f_001

# 测试模式（最近 60 个交易日）
python -m backtest.factor.backfill f_001 --test-days 60

# 所有 pending 因子（未 admit、未 reject）批量回填到 work
python -m backtest.factor.backfill --pending
```

### 2. 离线评测（读 work + library）

```bash
# 拿入库的因子值算 IC/RankIC/turnover/corr
python -m backtest.factor.evaluation f_001 --start 20210101 --end 20241231 --plot
```

corr 比较只读 library DB —— 候选因子拿自己的 pipeline 输出去比已 admitted 的稳定因子。

输出末尾会打印对照 `RECOMMENDED_THRESHOLDS` 的 4 项检查（informational only，不 gate）：

```
--- Reference thresholds (primary_horizon=20, informational only) ---
  RankICIR      = +0.3140  (>= 0.25)  OK
  IC+ ratio     =  55.20%  (>= 52%)   OK
  Turnover      =  0.4220  (<  0.5)   OK
  Max |corr|    =  0.7800  (<  0.85)  OK
  → All reference thresholds met. Run a backtest and decide on `admit`.
```

### 3. Pipeline driver（推荐，step1~step9 带淘汰门控）

```bash
python -m backtest.pipeline init f_001 \
    --start 20210101 --end 20241231 --frequency D

python -m backtest.pipeline run-all f_001
```

逐 step 执行，任一 step 失败即停并清理产物。详见 `pipeline/DESIGN.md`。

### 4. 旧版 pipeline driver（无淘汰门控）

```bash
python scripts/run_factor_pipeline.py f_001 \
    --start 20210101 --end 20241231 \
    --direction desc --benchmark 000300.SH
```

输出到 `results/<factor_id>/<tag>/{factor_eval, simple, detailed}/`。新版 pipeline 已覆盖此场景，旧脚本保留兼容。

### 5. 入库 / 拒绝（人工触发）

看完三层报告后，人工运行。`admit` / `reject` 按因子整体操作（一个 factor_id 对应 registry 中的一个 variant 标签）。

```bash
# admit：把因子从 work → library，清 work，registry 标 admitted
python -m backtest.factor.admission admit f_001 \
    --notes "Sharpe 1.45, IR 0.92 vs 000300"

# reject：清 work，registry 标 rejected（不动 library）
python -m backtest.factor.admission reject f_001 --notes "RankICIR 仅 0.18"

# 查看所有因子状态
python -m backtest.factor.admission status
python -m backtest.factor.admission status f_001
```

### 6. 临时数据清理

```bash
# 单因子清空 work，不改 status（保持 pending）
python -m backtest.factor.cleanup f_001

# 清空整个 work DB
python -m backtest.factor.cleanup --all

# 清掉 work 中已经 admit 到 library 的孤儿数据（崩溃恢复用）
python -m backtest.factor.cleanup --orphans
```

### 7. 增量更新（library DB）

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
from backtest.factor.compute import apply_variant_pipeline

# 1. 计算 raw 因子值
raw_df = compute_factor("f_001", "20210101", "20241231")

# 2. 应用 registry 声明的 variant pipeline（barra_ind_size / barra_l3 / none）
processed_df = apply_variant_pipeline(raw_df, "f_001")

# 3. 写 work
with FactorStorage() as fs:
    fs.insert_factors(processed_df)

# 4. 评测
result = evaluate("f_001", "20210101", "20241231", ret_type="open")
print(result.summary())
print(result.threshold_metrics(20))

# 5. 看完回测人工决定后
admit("f_001", notes="Sharpe 1.45")
# 或
reject("f_001", notes="RankICIR 偏低")
```

## 评测指标

`evaluate(factor_id)` 直接拉入库的因子值（先 work 后 library，按 admission 状态选），所有指标都基于这份值算 —— 不在 evaluation 层做二次中性化。

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
| **与现有因子相关性** | 与 **library DB** 中已 admitted 因子的逐日截面 RankIC，按日均值排序输出 top-K |

CLI 提供 `--corr-top-k N`（0 表示跳过 corr 检查）。

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

