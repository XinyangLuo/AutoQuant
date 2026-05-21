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
└── builtin/                 # 引擎自带的结构性因子（如 Barra 风险模型）
    ├── __init__.py
    └── barra/               # Barra 风格因子（详见 P0-2 实施计划）
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

每个因子在 registry 中声明若干「中性化变体」`(industry, cap)`，backfill 时按声明做 fan-out，各 variant **并存** 于 `factors_daily`（PK 加 `variant`）。

variant 是因子的一部分 —— 不同 variant 是不同的「因子样貌」，评测、admission、策略消费都按 variant 独立进行，不跨 variant 转换。

| 维度 | 取值 | variant token |
|---|---|---|
| industry | `None` / `"SW-L1"` / `"SW-L2"` | `none` / `swl1` / `swl2` |
| cap | `None` / `"circ_mv-q5"` / `"circ_mv-q10"` / `"total_mv-q5"` / `"total_mv-q10"` | `none` / `capq5` / `capq10` / `totalq5` / `totalq10` |

variant 字符串 = `"<industry>_<cap>"`，例：`swl2_capq5` / `swl1_totalq10`。特例：`"raw"` ≡ `"none_none"`。

合法组合数：3 × 5 = **15**。`@register` 通过 `neutralizations=[...]` 声明子集，**默认 2 个**：

```python
[
    {"industry": None,    "cap": None},          # → "raw"
    {"industry": "SW-L2", "cap": "circ_mv-q5"},  # → "swl2_capq5"，baseline
]
```

`BASELINE_VARIANT = "swl2_capq5"`（申万 2 级 + 流通市值 5 等分）是 evaluation / strategy 默认消费的口径。要加变体就改 registry 重跑 backfill，不会预先把 15 种都算。

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
    neutralizations=[
        {"industry": None,    "cap": None},
        {"industry": "SW-L2", "cap": "circ_mv-q5"},
    ],  # 可省，省略时用默认 2 变体
)
def momentum_20d(panel: pd.DataFrame) -> pd.Series:
    """
    panel: get_bars() 返回的宽 DataFrame，含 'close' 等列
    返回: MultiIndex (date, symbol) 的 Series（raw 值，中性化由 backfill 做）
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

由 `backfill.fan_out` 内部调用，用户通常**不直接调用** —— 在 `@register(neutralizations=[...])` 中声明即可。

```python
ind_neutral = industry_neutralize(raw_series, industry_panel)
cap_neutral = cap_neutralize(raw_series, cap_panel, cap_field='circ_mv', quantiles=5)
```

### 双库存储

```
┌──────────────────────────────────────────────────────┐
│ data/duckdb/factors.duckdb                           │  ← FactorStorage (work)
│                                                      │     · backfill / compute 写入（各 variant 并存）
│  factors_daily                                       │     · evaluation 读取
│  (date, symbol, factor_id, variant,                  │     · admit(variant) 后清空该 variant 行
│   value, ann_date, f_ann_date)                       │     · 没有 status='admitted' 的概念
│  PK (date, symbol, factor_id, variant)               │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ data/duckdb/factor_library.duckdb                    │  ← FactorLibrary (library)
│                                                      │     · 只有 admit(variant) 写入
│  factors_daily (same schema)                         │     · evaluation 的 corr 比较读这里
│                                                      │     · update 增量维护
│                                                      │     · delete_factor() 被禁用 (append-only)
└──────────────────────────────────────────────────────┘
```

**Schema 完全一致**，区别只在数据生命周期：work 是研究 churn，library 是已稳定的事实。

`ann_date` / `f_ann_date` 仅用于财务因子溯源，非财务因子留空。

## 使用方式

### 1. 回填（写 work DB）

```bash
# 单因子回填所有声明的变体（默认 raw + swl2_capq5）
python -m backtest.factor.backfill f_001

# 测试模式（最近 60 个交易日）
python -m backtest.factor.backfill f_001 --test-days 60

# 所有 pending 因子（任一 variant 未 admit、未 reject）批量回填到 work
python -m backtest.factor.backfill --pending
```

### 2. 离线评测（读 work + library）

```bash
# 默认评测 baseline variant（swl2_capq5）：拿入库的值算 IC/RankIC/turnover/corr
python -m backtest.factor.evaluation f_001 --start 20210101 --end 20241231 --plot

# 评测其他 variant，例如 raw
python -m backtest.factor.evaluation f_001 --start 20210101 --end 20241231 --variant raw
```

corr 比较在 **同 variant 内** 进行（候选与 library 因子都各自读 variant 的入库值）—— 因为入库的值就是该 variant 自己的「纯净因子值」，跨 variant 不互比。

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
    --direction desc --benchmark 000300.SH
```

输出到 `results/<factor_id>/<variant>/{factor_eval, <tag>/{simple, detailed}}/`。

### 4. 入库 / 拒绝（人工触发）

看完三层报告后，人工运行。`admit` / `reject` 的单位是 **(factor_id, variant)** —— 同一因子的不同 variant 可以独立 admit 或 reject。

```bash
# admit baseline variant（默认 swl2_capq5）：把该 variant 从 work → library
python -m backtest.factor.admission admit f_001 \
    --notes "Sharpe 1.45, IR 0.92 vs 000300"

# 也可以 admit 别的 variant，例如 raw
python -m backtest.factor.admission admit f_001 --variant raw \
    --notes "raw 信号在小盘 universe 更强"

# 清掉某 variant 的 work 数据，标记 variant_status[swl2_capq5]=rejected
python -m backtest.factor.admission reject f_001 --notes "RankICIR 仅 0.18"

# 查看所有因子状态（按 variant 展开）
python -m backtest.factor.admission status
python -m backtest.factor.admission status f_001
```

`registry.json` 中状态以 `variant_status: {variant: 'admitted' | 'rejected'}` 嵌套字典存储；顶层 `status` 字段作为汇总（所有 variant 都 admitted → `admitted`，全 rejected → `rejected`，混合 → `mixed`，其余 → `pending`）。

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
from backtest.factor.compute import apply_neutralizations
from backtest.factor.variants import BASELINE_VARIANT

# 1. 计算 raw 因子值
raw_df = compute_factor("f_001", "20210101", "20241231")
# 2. fan-out 到所有声明的 variant
all_variants_df = apply_neutralizations(raw_df, "f_001")
# 3. 写 work
with FactorStorage() as fs:
    fs.insert_factors(all_variants_df)

# 评测 baseline variant
result = evaluate("f_001", "20210101", "20241231",
                  variant=BASELINE_VARIANT, ret_type="open")
print(result.summary())
print(result.threshold_metrics(20))

# 看完回测人工决定后:
admit("f_001", variant=BASELINE_VARIANT, notes="Sharpe 1.45")
# 或
reject("f_001", variant="raw", notes="raw 口径 IC 偏低")
```

## 评测指标

`evaluate(factor_id, variant=...)` 直接拉入库的该 variant 因子值，所有指标都基于这份值算 —— 不在 evaluation 层做二次中性化。

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
| **与现有因子相关性** | 与 **library DB** 中**同 variant** 因子的逐日截面 RankIC，按日均值排序输出 top-K。同 variant 内比 —— 因为入库的值已是该 variant 自己的「纯净因子值」，跨 variant 不互比 |

CLI 同步提供 `--variant <name>` / `--corr-top-k N`（0 表示跳过 corr 检查）。

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

---

# P0 实施计划

> 本节为待落地任务的执行说明。完成对应任务后从这里删除并把内容合并到上面正文。

## P0-1: 常用因子算子库扩展

### 目标
按 [WorldQuant Operators](https://platform.worldquantbrain.com/learn/operators) 一线常用算子补全 `transforms.py`，避免每次写因子代码重复构造样板。

### 落地位置
- 主文件: `backtest/factor/transforms.py`（已有 `_ts_roll` 私有辅助、`rank/z_score/ts_rank/ts_mean/ts_std` 7 个算子 + `industry_neutralize/cap_neutralize`）
- 公开 API 导出: `backtest/factor/__init__.py`
- 测试: 新建 `tests/factor/test_transforms.py`（如不存在）

### 接口约定
**所有算子**：输入/输出均为 MultiIndex `(date, symbol)` 的 `pd.Series`；多变量算子（`ts_corr` / `ts_cov`）输入两个 Series；NaN 跳过；`min_periods` 默认 = `window`。

### 待补算子（按类别）

**时序滚动**（复用 `_ts_roll`）：

| 算子 | 签名 | 说明 |
|---|---|---|
| `ts_sum(s, window, min_periods=None)` | 滚动和 | |
| `ts_max(s, window, min_periods=None)` | 滚动最大 | |
| `ts_min(s, window, min_periods=None)` | 滚动最小 | |
| `ts_argmax(s, window, min_periods=None)` | 最大值距今天数（0=今天），归一化到 `[0,1]` | 用 `rolling.apply(lambda x: x.argmax()/(len(x)-1))` |
| `ts_argmin(s, window, min_periods=None)` | 最小值距今天数 | 同上 |
| `ts_corr(s1, s2, window, min_periods=None)` | 滚动 Pearson 相关 | 按 symbol 分组对齐两 Series 后 `rolling.corr` |
| `ts_cov(s1, s2, window, min_periods=None)` | 滚动协方差 | |
| `delay(s, d=1)` | 按 symbol 时序后移 `d` 期 | `groupby('symbol').shift(d)` |
| `delta(s, d=1)` | `s - delay(s, d)` | |
| `decay_linear(s, window)` | 线性递减加权滚动均值（权重 `window,window-1,...,1`） | |

**截面**（按 date groupby + apply）：

| 算子 | 签名 | 说明 |
|---|---|---|
| `cs_zscore(s)` | 每日截面 z-score（均值 0，方差 1） | **注意**：现有 `z_score(s, window)` 是**时序**滚动；新增 `cs_zscore` 是**截面**单日。Barra 中性化必用 |
| `cs_winsorize(s, k=3, mad_const=1.4826)` | 每日截面 MAD 去极值，超出 `median ± k·mad_const·MAD` 的 clip 到边界 | Barra 三级因子标准预处理 |
| `cs_scale(s)` | 每日截面 L1 归一化：`s / s.abs().sum()` | |

**元素级**：

| 算子 | 签名 | 说明 |
|---|---|---|
| `signed_power(s, e)` | `sign(s) * |s|**e` | |
| `sign_log(s)` | `sign(s) * log(1 + |s|)` | 压缩长尾 |

### 命名整理（背向兼容）
- 现有 `z_score` 是时序滚动，名字与截面 z-score 冲突。**新增 `ts_zscore` 作为别名**（同实现），同时引入 `cs_zscore`。`z_score` 保留为 `ts_zscore` 别名，标 `DeprecationWarning`。
- 现有 `industry_neutralize` 内部按 industry 分组做的就是 `cs_zscore`，可在实现后调用新算子做内部清理（不必本次完成）。

### 完成标准
- [ ] 算子全部实现，签名注释含示例
- [ ] `__init__.py` 全部导出
- [ ] 每个算子至少一条单元测试覆盖普通情况 + NaN 情况
- [ ] 在 `alphas/reversal_extensions.py` 任选一个因子用新算子改写，确认值与原实现一致（dry run）

---

## P0-2: Barra 风险模型

### 目标
1. 清空旧的实验性因子（`f_rev_*`、`f_*` 旧 alpha），重建正式 Barra 因子库
2. 注册 11 个 Barra 三级因子 + 7 个一级合成
3. 把 PLAN.md §2.2 的统一中性化 pipeline 取代当前的多 variant fan-out (`industry_neutralize` × `cap_neutralize`)
4. 入库时新增 PLAN.md §2.3 的 Ridge 检查

### Variant 体系重构

| 旧 | 新 |
|---|---|
| 默认 2 variant: `raw` + `swl2_capq5`；可声明 15 种 | **简化为 2 variant**: `raw`（原始值）+ `neutral`（统一中性化残差） |
| `BASELINE_VARIANT = "swl2_capq5"` | `BASELINE_VARIANT = "neutral"` |
| `variants.py` 枚举 industry × cap | `variants.py` 只保留 `RAW = "raw"`、`NEUTRAL = "neutral"` |

**Barra 因子例外**：Size 因子自身不被中性化（它是中性化的回归变量），Industry 也是回归变量。所以 11 个 Barra 三级因子默认只跑 `raw` variant；7 个一级 Barra 因子也只 `raw`（合成自三级 raw）。

### 11 个三级因子（落地位置）

新建目录 `backtest/factor/builtin/barra/`：

```
backtest/factor/builtin/
├── __init__.py
└── barra/
    ├── __init__.py     # 导入各 .py 触发 @register
    ├── size.py         # LNCAP
    ├── beta.py         # BETA (WLS, 252日, 半衰期63日, vs 000300.SH)
    ├── momentum.py     # RSTR (简化版: 过去252日累计log return, 滞后11日)
    ├── value.py        # BTOP / ETOP / DTOP
    ├── quality.py      # ROA / GP / AGRO
    ├── liquidity.py    # STOM (过去21日 Σ amount/circ_mv 的 log)
    ├── growth.py       # EGRO (5年 EPS 对时间回归斜率 / mean(|EPS|))
    └── composite.py    # 7 个一级合成因子
```

注册命名规范：
- 三级: `f_barra_{l1}_{l3}`，如 `f_barra_size_lncap` / `f_barra_value_btop` / `f_barra_quality_agro`
- 一级合成: `f_barra_{l1}`，如 `f_barra_value`（= mean(`f_barra_value_btop`, `f_barra_value_etop`, `f_barra_value_dtop`)）
- `category = "barra_{l1}"`

**单 variant 实现要点**：
- Size/Beta/Momentum/Liquidity/Growth: 一级 = 二级 = 三级，注册时直接把三级因子的 `factor_id` 复用为一级名（或额外注册一个等价 alias）
- Value: 7 个 factor_id（3 三级 + 1 一级；二级层不必单独建 factor_id，合成时直接从三级跳到一级即可）
- Quality: 7 个（3 三级 + 1 一级）
- Beta: 用 `get_index_bars(['000300.SH'])` 取基准日收益 → 按 symbol 滚动 252 日 WLS，权重 `0.5^((T-t)/63)`。考虑用 `numpy.linalg.lstsq` + 权重对角矩阵；性能瓶颈在 `groupby('symbol').rolling(252).apply`
- AGRO: `get_fina_snapshot(D)` 取 5 年内每个季度的 `bs_total_assets`，对 `end_date` 做线性回归取斜率
- EGRO: 同上但用 `inc_basic_eps`

### 三级因子合成流程（PLAN.md §2.1 计算规则）

每个三级因子在 register 中显式声明 `compute_pipeline=["mad_winsorize", "industry_median_fill", "cs_zscore"]`，由 `compute.py` 统一应用：

1. 算出 raw 三级因子值（symbol × date）
2. `cs_winsorize(raw, k=3)` — MAD 去极值，超出 `median ± 3×1.4826×MAD` clip 到边界
3. 申万一级行业截面中位数填充缺失值 — 依赖 `MarketStorage.get_industry_panel_range(start, end, level='L1')`
4. `cs_zscore` — 截面 z-score
5. 写入 `factors_daily` 作为该三级因子的 `raw` variant

二级 → 一级合成：等权平均所属三级因子的 z-score 值，再次 `cs_zscore` 标准化。落到 `factors_daily` 作为一级因子的 `raw` variant。

### 统一中性化 pipeline（PLAN.md §2.2，已落地）

**对非 Barra 候选 alpha**，`compute.py::apply_variant_pipeline` 在 `variant="barra_ind_size"` 分支执行：

```
1. MAD winsorize (cs_mad_winsorize, k=3)
2. SW-L1 行业中位数填充
3. cs_zscore
4. 截面 OLS:  factor ~ intercept + 行业 dummies (drop_first) + Size_z
   - Size_z 直接读 factor_storage 中已 barra_l3 流水线处理过的
     f_barra_size_lncap(已 z-score)
5. 取残差,再次 cs_zscore
```

实施要点：
- OLS 用 `cs_ols_residualize`（`transforms.py`）—— 单次 groupby('date') 循环，
  行业 dummies 在循环外通过 `Categorical` 一次性编码，每日按 codes 切片做
  identity-style block，避免 `pd.get_dummies` 的 N+1 开销
- `np.linalg.lstsq` 闭式解；OLS 残差对设计矩阵的列**精确正交**（数值~1e-10）
- Size_z 从 `factor_storage.get_factor("f_barra_size_lncap")` 直接读
  —— Commit 2 已落地 11 个 L3 + 7 个 L1，所以这一列保证存在
- 行业哑变量从 `get_industry_panel_range(..., level='L1')` 取（31 个行业 one-hot
  → drop_first 共 30 列）
- 三级 Barra 因子（`f_barra_value_btop` 等）使用 `variant="barra_l3"` 走自身
  pipeline（无 OLS）—— 不在这里处理；这里只服务 user 注册的候选 alpha
- 输出写入宽表 `factors_daily`，列名 = `factor_id`，覆盖更新

### 旧数据清理

执行顺序（任务开工第一步）：
```bash
# 1. work DB 全清
python -m backtest.factor.cleanup --all
# 2. library DB 中旧因子全清
# 旧 factor_id 列表见 registry.json，主要是 f_rev_* 与 f_001~f_xxx 中的实验性条目
python -m backtest.factor.admission reject <factor_id> --all-variants  # 或直接 SQL DELETE
# 3. 旧 alpha 代码本地归档或删除（alphas/ 已 gitignored，仅本地操作）
mv alphas/reversal_extensions.py alphas/_archive/  # 或直接 rm
mv alphas/reversal_zscore_combo.py alphas/_archive/
# 4. registry.json 中相应条目删除
```
注意 schema 不需要改动 —— `factors_daily` 表结构通用，新 Barra 因子直接 INSERT 即可。

### Ridge 入库检查（PLAN.md §2.3）

新增 `backtest/factor/admission_check.py`（或并入 `admission.py`）：

```python
def ridge_r2_check(factor_id: str, variant: str = 'neutral',
                   start: str | None = None, end: str | None = None) -> dict:
    """
    Pool candidate factor values + 6 个一级 Barra 因子（除 Size 和 Industry,
    即 Beta / Momentum / Value / Quality / Liquidity / Growth）的 raw variant,
    做 Ridge regression: candidate ~ 6 Barra.
    Returns: {"r2": float, "tier": "pure_alpha"|"smart_beta"|"edge_smart_beta"|"reject",
              "residual_icir": float | None}
    """
```

R² 分层（PLAN.md §4 step8）：
- `< 0.10`: pure_alpha
- `[0.10, 0.50)`: smart_beta
- `[0.50, 0.80)`: edge_smart_beta — 需对 Ridge 残差再算 ICIR，日频 > 1.0 / 月频 > 0.8 才保留
- `>= 0.80`: reject

`admit()` 在写入 library 时同时把 `tier` 和 `r2` 写入 `registry.json` 的 meta。

### 完成标准
- [ ] 11 个三级 + 7 个一级 Barra 因子注册并可回填
- [ ] `backfill --pending` 跑通整组 Barra 因子，回填 work + admit 到 library
- [ ] 新中性化 pipeline 替换旧 `apply_neutralizations`；旧 fan-out 代码移除（保留 `industry_neutralize`/`cap_neutralize` 算子，仅作为底层工具不再被默认 pipeline 调用）
- [ ] `BASELINE_VARIANT = "neutral"`，evaluation/strategy 默认消费 neutral
- [ ] Ridge R² 检查实现并接入 `admit` 流程
- [ ] 一篇 sanity-check 笔记：选两只代表股票（如 600519.SH / 300750.SZ）打印 Barra 7 个一级因子近 1 年值，确认量级合理（z-score 后基本在 [-3, 3]）

### 与 PLAN.md §4 因子挖掘 pipeline（P1）的衔接
P0-2 落地后，PLAN.md §4 的 step2（"中性化后与 size/industry corr < 0.05"）才能验证。step8（Ridge R²）的实现就是本节的 `ridge_r2_check`，可直接复用。
