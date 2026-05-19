# 因子筛选 Pipeline

回测系统端到端操作手册：从"灵感"经过定义 → 回填 → 离线评测 → 策略 → 简单回测 → 详细回测，**人工**决定 admit / reject，沉淀为可复用的库内因子。

> 各子模块的内部设计见各自的 `DESIGN.md`：
> [`factor/DESIGN.md`](factor/DESIGN.md) · [`strategy/DESIGN.md`](strategy/DESIGN.md) · [`simulation/DESIGN.md`](simulation/DESIGN.md) · [`evaluation/DESIGN.md`](evaluation/DESIGN.md) · [`data/DESIGN.md`](data/DESIGN.md)
> 本文档专注于"怎么跑一整套筛选流程、参数怎么调、指标怎么读、什么样的值代表好/坏因子、什么时候 admit"。

---

## 0. 总览

### 两个物理 DuckDB（关键设计）

```
data/duckdb/factors.duckdb         ← 工作区 (FactorStorage)
                                     · backfill / evaluation / 回测 写读这里
                                     · 临时数据，admit 或 cleanup 时清空

data/duckdb/factor_library.duckdb  ← 稳定库 (FactorLibrary)
                                     · 只有 admit() 写入
                                     · evaluation 的 "与现有因子相关性" 只读这里
                                     · update 增量维护
                                     · append-only：禁用 delete_factor
```

**意义**：新因子的研究数据永远不会污染稳定库；与稳定库的相关性比较永远是"对照已稳定的对手"。

### 七个阶段（含可选十段分层）

```
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ 1. 定义因子  │ │ 2. 回填(work)│ │ 3. 离线评测  │ │ 4. 策略组装  │
│  @register   │→│  backfill    │→│  factor.eval │→│  Strategy    │
└──────────────┘ └──────────────┘ └──────────────┘ │   Config     │
       │                            │               └──────┬───────┘
       │                            │ (可选 --decile)        ▼
       │                            ▼              ┌──────────────┐
       │                      ┌──────────────┐    │ 5. 简单回测  │
       │                      │ 十段分层回测 │    │  Simple      │
       │                      │  Decile BT   │    └──────┬───────┘
       │                      └──────────────┘           ▼
       │                                         ┌──────────────┐
       │                                         │ 6. 详细回测  │
       │                                         │  Detailed    │
       │                                         └──────┬───────┘
       │                                                ▼
       │                                         ┌──────────────┐
       │                                         │ 7. 回测评测  │
       │                                         │  eval.eval   │
       │                                         └──────┬───────┘
       │                                                ▼
       │                                         ┌──────────────┐
       └────────────────────────────────────────→│ 8. 人工决策  │
                                                 │ admit/reject │
                                                 └──────────────┘
```

### 端到端命令一览

```bash
conda activate AutoQuant

# 1. 写因子代码（手工） + 装饰器注册
$EDITOR backtest/factor/user/my_factor.py

# 2. 回填因子值到 work DB
python -m backtest.factor.backfill f_xxx                 # 全历史
python -m backtest.factor.backfill f_xxx --test-days 60  # 调试只跑近 60 天

# 3-7. 一条命令跑完三层评测，输出到 results/<factor_id>/<variant>/{factor_eval, <tag>/{simple,detailed}}
python scripts/run_factor_pipeline.py f_xxx \
    --start 20210101 --end 20241231 \
    --direction desc --benchmark 000300.SH

# 显式指定绝对数量 + 周度换仓 (旧默认)
python scripts/run_factor_pipeline.py f_xxx \
    --start 20210101 --end 20241231 \
    --top-n 50 --rebalance 1W --decay 5 \
    --direction desc --benchmark 000300.SH

# 跳过十段分层回测
python scripts/run_factor_pipeline.py f_xxx --no-decile

# 8. 看完三层报告后人工决定（独立命令，不会自动触发）
python -m backtest.factor.admission admit  f_xxx --notes "Sharpe 1.45"   # work→library
python -m backtest.factor.admission reject f_xxx --notes "RankICIR 不达标"

# 维护命令
python -m backtest.factor.admission status               # 看所有因子状态
python -m backtest.factor.cleanup f_xxx                  # 清 work，不改 status
python -m backtest.factor.update                         # 增量更新 library 中 admitted 因子
```

### 关键约束

- **Delay = 1**：T 日已知信息 → T+1 日生效。所有阶段（评测的前瞻收益、策略目标持仓、引擎成交日）都遵守。
- **PIT 隔离**：行情靠 `get_bars(end_date=D)` 天然限制，财务靠 `get_fina_snapshot(D)` 的 `f_ann_date <= D` + QUALIFY。
- **涨跌停**：策略层不过滤、由引擎在成交日判定。**离线评测**默认 `--exclude-limit-up` 丢掉无法成交的 IC 样本。
- **入库门槛不强制**：`RECOMMENDED_THRESHOLDS` 只用于参考输出，`admit()` 不做 gate；是否入库由人类看完三层报告决定。

---

## Stage 1：构建因子（表达式 + Python 代码）

### 1.1 注册装饰器

```python
# backtest/factor/user/my_factor.py
from backtest.factor.registry import register
from backtest.factor.transforms import rank, z_score

@register(
    "f_101",                         # 必填，全局唯一稳定 ID
    name="my_idea",                  # 语义别名
    category="reversal",             # 任意分类标签
    data_sources=["market_daily"],   # 决定 compute.py 注入什么 panel
    description="20 日反转 × 20 日换手率 (z-score 标准化后相乘)",
    parameters={"ret_window": 20, "turnover_window": 20, "z_window": 60},
)
def my_idea(panel, ret_window=20, turnover_window=20, z_window=60):
    df = panel[["date", "symbol", "close", "turnover_rate"]].copy()
    df["adj_close"] = df["close"] * panel["adj_factor"]
    df = df.sort_values(["symbol", "date"])

    df["ret_n"]      = df.groupby("symbol")["adj_close"].pct_change(ret_window)
    df["turnover_n"] = df.groupby("symbol")["turnover_rate"].transform(
        lambda x: x.rolling(turnover_window, min_periods=turnover_window // 2).mean()
    )

    idx = df.set_index(["date", "symbol"]).index
    ret_neg = (-df["ret_n"]).set_axis(idx)
    to_mean = df["turnover_n"].set_axis(idx)
    return z_score(ret_neg, z_window) * z_score(to_mean, z_window)
```

### 1.2 输入 panel 的约定

`data_sources` 决定 `compute.py` 注入什么 DataFrame：

| data_sources | panel 内容 |
|---|---|
| `["market_daily"]` | 行情宽表（含 close/open/turnover_rate/circ_mv 等所有 `market_daily` 列） |
| `["income_q"]` / `["balancesheet_q"]` / `["cashflow_q"]` | 财务 PIT 快照，列名加 `inc_/bs_/cf_` 前缀 |
| 两者混合 | 按日 outer-join 后注入 |

**因子函数禁止直接访问数据库**——PIT 安全的底线。

### 1.3 通用算子

| 算子 | 语义 | 何时用 |
|---|---|---|
| `rank(s)` | 截面 percentile rank，输出 `[0, 1]` | 让因子尺度统一、抗厚尾 |
| `rank(s, ascending=False)` | 反向 | direction 不想配成 `asc` 时 |
| `z_score(s, window)` | 时序 z-score（每 symbol 按 `window` 滚动） | 时序异常值消除、做时变标准化 |
| `ts_rank(s, window)` | 时序排名，窗口内缩放到 `[-1, 1]` | 当前值在历史上的位置；`-1`=窗口最小，`1`=窗口最大 |
| `ts_mean(s, window)` | 时序滚动均值 | 平滑因子值 |
| `ts_std(s, window)` | 时序滚动标准差 | 衡量因子值的时序波动 |

混用很常见：`z_score`（时序标准化）→ `rank`（截面归一化）的两步走最稳。

### 1.4 命名建议

- **f_xxx**：稳定 ID，按 category 留段（`f_mom_*` 动量、`f_rev_*` 反转、`f_fin_*` 财务、`f_qual_*` 质量、`f_vol_*` 波动率……）
- **name**：snake_case 语义名
- **parameters**：把窗口长度都暴露出来，便于参数扫描

---

## Stage 2：回填因子值（写 work DB）

### 2.1 命令

```bash
python -m backtest.factor.backfill f_101                 # 全历史 → work DB
python -m backtest.factor.backfill f_101 --test-days 60  # 调试
python -m backtest.factor.backfill --pending             # 所有 pending 因子批量
```

### 2.2 增量更新（library DB）

`update` 命令**只动 library 库的 admitted 因子**——work 库的临时数据不在它的管辖范围。

```bash
python -m backtest.factor.update    # 把 library 里的 admitted 因子追平到最新交易日
```

### 2.3 注意事项

- **窗口因子需前置 lookback**：`parameters={"window": 20}` 决定 compute.py 多取 `int(window * 1.5)` 个日历日。多窗口时填**最大值**。
- **写入策略**：upsert，同一天同一因子重复回填会覆盖。
- **空数据排查**：跑出来的 DataFrame 是空 → lookback 不够 / 数据列对不齐 / `set_index` 漏字段。先 `--test-days 5` + `df.shape` 排查。

---

## Stage 3：离线评测（IC / RankIC / ICIR / corr）

### 3.1 命令

```bash
python -m backtest.factor.evaluation f_101 \
    --start 20210101 --end 20241231 \
    --horizons 1,5,10,20,60 \
    --ret-type open \
    --corr-top-k 5 \
    --plot --plot-horizon 20
```

Python API：

```python
from backtest.factor import evaluate, print_evaluation
res = evaluate("f_101", "20210101", "20241231",
               horizons=[1, 5, 10, 20, 60], ret_type="open")
print_evaluation(res)             # 末尾会打 RECOMMENDED_THRESHOLDS 对照
print(res.summary())              # 每 horizon 一行的指标表
print(res.threshold_metrics(20))  # 4 项 admission 参考指标
```

### 3.2 评测 config / 命令行参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--start` / `--end` | 必填 | 至少 2 年（>500 个交易日）以让 IC 序列统计稳定 |
| `--horizons` | `1,5,10,20,60` | 前瞻 N 个交易日的累计收益；同时给出 decay 曲线 |
| `--ret-type` | `open` | `close`：收盘到收盘。`open`：**T+1 开盘到 T+1+h 开盘**，贴近 A 股 T+1 |
| `--corr-top-k` | `5` | 与 **library 库** 中已 admitted 因子的相关性 Top-K；`0` 跳过 |
| `--no-exclude-limit-up` | off | 默认排除涨停无法成交的样本 |
| `--all` | off | 一次跑所有注册因子；输出对比表 |
| `--plot` / `--plot-horizon` | off, 20 | 单因子模式下保存日频 IC/RankIC + 累计 IC/RankIC 四图到 `results/<factor_id>/<variant>/factor_eval/<factor_id>_<h>d.png` |
| `--decile` | off | 跑十段分层回测，产出 10 组 NAV 曲线 + 多空对冲图 |

### 3.3 输出指标怎么看

`EvaluationResult.summary()` 返回每个 horizon 一行的表：

```
 horizon  IC_mean  IC_std   ICIR   IC_tstat  IC_pos_ratio  RankIC_mean  RankIC_std  RankICIR  ...
       1   0.012    0.090   0.13     2.5       0.541       0.018         0.085      0.21
       5   0.025    0.115   0.22     4.2       0.555       0.035         0.108      0.32
      20   0.041    0.140   0.29     5.5       0.572       0.058         0.130      0.45
      60   0.018    0.155   0.12     2.3       0.530       0.025         0.150      0.17
```

#### 单指标解读速查

| 指标 | 公式 | 看什么 |
|---|---|---|
| **IC_mean** | `mean(corr(factor_t, ret_{t+h}))` | 因子与未来收益线性相关性日均值；**正值 = 正向预测** |
| **IC_std** | IC 序列波动 | 越小越稳；超过 `\|IC_mean\|` 4~5 倍说明噪音大 |
| **ICIR** | `IC_mean / IC_std` | **核心指标**：信号信噪比 |
| **IC_tstat** | `IC_mean / (IC_std/√n)` | IC 显著性；\|t\| > 2 才算统计显著 |
| **IC_pos_ratio** | 正 IC 日占比 | > 0.55 算稳健 |
| **RankIC_mean** | Spearman 秩相关 | 对极端值不敏感，**A 股更看 RankIC** |
| **RankICIR** | `RankIC_mean / RankIC_std` | **入库参考主门槛**（RECOMMENDED 默认 ≥ 0.25） |

#### 参考阈值（CLI 末尾自动打印）

跑完 `evaluation` 会在末尾看到对照 `RECOMMENDED_THRESHOLDS` 的检查：

```
--- Reference thresholds (primary_horizon=20, informational only) ---
  RankICIR      = +0.3140  (>= 0.25)  OK
  IC+ ratio     =  55.20%  (>= 52%)   OK
  Turnover      =  0.4220  (<  0.5)   OK
  Max |corr|    =  0.7800  (<  0.85)  OK
  → All reference thresholds met. Run a backtest and decide on `admit`.
```

**注意**：仅供参考。`admit()` **不会自动 gate**——是否入库由人类看完 simple/detailed 回测后决定。阈值只是给"这个因子值不值得跑回测"提供一个粗筛。

#### 好/中/差因子的经验区间（A 股日频，h=20）

| 档位 | RankICIR | RankIC_mean | IC+ 比例 | Turnover | 备注 |
|---|---|---|---|---|---|
| 顶级（罕见） | > 0.7 | > 0.10 | > 0.62 | < 0.30 | 老牌因子在样本外几乎不可能；做出来先怀疑数据泄露 |
| **优秀** | 0.40 ~ 0.70 | 0.05 ~ 0.10 | 0.56 ~ 0.62 | < 0.40 | 多数年份能稳定贡献 alpha |
| 合格 | 0.25 ~ 0.40 | 0.03 ~ 0.05 | 0.52 ~ 0.56 | < 0.50 | **达到 RECOMMENDED**：能考虑入库但建议组合使用 |
| 边缘 | 0.15 ~ 0.25 | 0.01 ~ 0.03 | 0.50 ~ 0.52 | 0.50 ~ 0.70 | 单独用风险高；可入多因子作辅助 |
| 失败 | < 0.15 / 负 | ≈ 0 / 负 | < 0.50 | > 0.70 | 直接弃用，或反向看看 |

#### 多 horizon 对比（decay 曲线）

- **理想**：单峰、缓慢衰减。峰值在 5~20d 是日/周频再平衡的甜点。
- **病态**：
  - 1d ICIR 极高、20d/60d 同样高 → 警惕未来信息泄露
  - 所有 horizon 全 < 0.1 → 因子无效
  - 1d 负、20d 突变正 → 噪音夹反转效应，需中性化或剔除短期反转

#### 换手 Turnover

| Turnover | 解读 |
|---|---|
| < 0.10 | 几乎静态，多见于估值/质量类 |
| 0.10 ~ 0.30 | 中低换手，月频再平衡可保 alpha |
| 0.30 ~ 0.50 | 高换手，**周频/日频 + 成本控制**才能落地 |
| > 0.50 | 噪声多 / 日间波动；RECOMMENDED 默认 fail |

#### 分组收益（group_returns）

10 分位前瞻收益均值表，看 **top vs bottom 的 spread + 单调性**。第 0 组到第 9 组应当单调变化。中间出现"驼峰"或"V 型"说明因子主要靠极端值生效，鲁棒性差。

> 与 `--decile` 十段分层回测的区别：`group_returns` 是**静态截面均值**（每组只算一次前瞻收益的平均），不产时间序列；`--decile` 是**动态净值曲线**（每期重新分组、等权、 cumprod），能看回撤和稳定性。

#### 与现有（library）因子相关性

```
factor_id  corr   n_dates
f_001     +0.85    600   ← 危险！与现有动量因子高度相似
f_rev_03  -0.34    600
```

- **RECOMMENDED 默认 |corr| < 0.85**。超过即视为重复因子，建议拒。
- 即便低于 0.85，超过 0.7 也应当反思相对边际信息。
- 比较只对 **library 库中已 admitted** 的因子做，**不与其他 pending 因子比对**（避免临时数据互相污染）。

### 3.4 十段分层回测（`--decile`）

十段分层是离线评测的**可选补充**，不依赖策略信号，直接用因子值将 universe 股票每期分 10 组（`pd.qcut`），每组等权持有，追踪 10 条净值曲线 + 多空对冲（D10 - D1）。

| 指标 | 看什么 |
|---|---|
| **Monotonicity** | 年化收益与分位排名的相关系数；**> 0.5 强单调**，0.2~0.5 弱单调，< 0 反向或无序 |
| **D10 vs D1 年化收益** | 头尾差距越大，因子区分度越好；理想：D10 最高、D1 最低（或反向因子则相反） |
| **LS Sharpe** | 多空对冲的年化 Sharpe；> 0.5 说明对冲后仍有稳定 alpha |
| **LS MaxDD** | 多空对冲的回撤；> -20% 说明对冲比较干净 |

> **注意**：十段分层和策略回测（simple/detailed）是**独立的验证维度**。一个因子可以有高 IC 但分层不单调（如 `z(-ret) * z(turnover)` 这类四象限因子），也可以分层单调但 IC 一般（非线性关系）。两者互补。

命令：

```bash
python scripts/run_factor_pipeline.py f_xxx --decile
# 或单独跑（不跑 simple/detailed）
python -m backtest.factor.evaluation f_xxx --start 20210101 --end 20241231 --decile
```

输出到 `results/<factor_id>/<variant>/decile_backtest/<factor_id>_<variant>_decile.png`：

| 面板 | 看什么 |
|---|---|
| 上：10 条 NAV 曲线（log 轴） | D1~D10 是否单调排列；RdYlGn 色谱，D1 红、D10 绿 |
| 下：Long-Short NAV | D10 - D1 的累计净值；> 1 且持续上升 = 多空有效 |

### 3.5 画图（`--plot`）

输出到 `results/<factor_id>/<variant>/factor_eval/<factor_id>_<h>d.png`，四个面板：

| 面板 | 看什么 |
|---|---|
| 日 IC 柱状 | 多数日子方向是否一致；红柱占比直观反映 IC+_ratio |
| 日 RankIC 柱状 | 同上，对极值不敏感 |
| 累计 IC 曲线 | 单调上升 = 持续有效；平台/掉头 = 风格切换风险 |
| 累计 RankIC | 平滑度优于 IC，看长期 alpha 累积 |

---

## Stage 4：策略组装（中性化 + Decay + 选股）

策略把因子值变成 `(date, symbol, target_weight)`。所有可调项汇总在 `StrategyConfig`。

### 4.1 完整 YAML 范例

```yaml
name: "my_idea_top10pct_daily"

strategy:
  type: "single_factor_topk"
  rebalance_freq: "1D"
  delay: 1

universe:
  exclude_st: true
  exclude_new_ipo_days: 252
  include_cyb: true
  include_kcb: false
  index_members: null               # 例 "000300.SH"
  min_market_cap: 500000000.0       # 5 亿流通市值
  min_avg_amount: 10000000.0        # 1000 万 20 日均成交额

factors:
  - id: "f_101"
    direction: "desc"               # desc = 因子值大的好
    weight: 1.0

selection:
  method: "topk"
  top_pct: 0.1
  bottom_pct: 0.1

weighting:
  method: "equal"

neutralize:
  industry: false
  industry_method: "group_rank"
  market_cap: true                  # 几乎总要开

decay: 5                            # 线性衰减加权 5 日

backtest:
  start_date: "20210101"
  end_date:   "20241231"
  benchmark:  "000300.SH"
```

### 4.2 关键字段详解

#### `rebalance_freq`

| 频率 | 何时合适 |
|---|---|
| `1D` | 高换手因子（turnover > 0.3）；成本敏感 |
| `1D` | **默认**，适合多数 alpha 因子 |
| `2W` | 中低换手（turnover 0.1~0.3） |
| `1M` | 估值/质量等慢变因子 |
| `EOM` | 月末再平衡（避开月初波动 / 配合月报披露） |

#### `selection.method`

| 方法 | 输出特征 | 何时用 |
|---|---|---|
| `topk` | 多头 top_k 只，sum = 1 | A 股主流，**生产 80% 用这个** |
| `long_short` | 多 top_k + 空 bottom_k | 研究 alpha 纯度（A 股做空成本高，仅评估） |
| `decile` | 10 组各 sum = 1 | 验证单调性、做分层归因 |

#### `weighting.method`

| 方法 | 行为 | 注意 |
|---|---|---|
| `equal` | 1/K 等权 | 默认，最简单可比 |
| `market_cap` | 按 `circ_mv` 加权 | 减少小盘集中，但拖低收益 |
| `factor_value` | 按 \|因子值\| 加权 | 让强信号股票暴露更大 |

#### `neutralize`

| 选项 | 何时启用 |
|---|---|
| `market_cap: true` | **几乎总要开**，避免实质押注小盘 |
| `industry: true` | 想要"行业内 alpha"，不被行业 beta 拖累（待 `sw_industry` 表落地） |
| `industry_method: group_rank` | 简单稳健；推荐默认 |
| `industry_method: group_demean` | 保留因子原尺度 |
| `industry_method: group_zscore` | 行业内尺度统一 |

#### `decay`

线性加权平滑：`decay(x, n) = (n·x[T] + (n-1)·x[T-1] + ... + 1·x[T-n+1]) / (n·(n+1)/2)`

| decay | 效果 |
|---|---|
| `None` | 原始信号 |
| `3` | 弱平滑 |
| `5` | **常用默认**，明显降换手 ~30%，IC 损失轻微 |
| `10` | 强平滑；不要再叠加月频再平衡 |

#### `universe.min_market_cap` / `min_avg_amount`

**生产默认**：

```yaml
min_market_cap: 500000000      # 5 亿流通市值
min_avg_amount: 10000000       # 20 日均成交额 1000 万
```

研究阶段可拉宽，生产/详细回测必须收紧。

### 4.3 Python 实例化

```python
from backtest.strategy import (
    StrategyConfig, UniverseConfig, FactorConfig,
    SelectionConfig, WeightingConfig, NeutralizeConfig, BacktestConfig,
    SingleFactorStrategy,
)

config = StrategyConfig(
    name="my_idea_top10pct_daily",
    strategy_type="single_factor_topk",
    rebalance_freq="1D", delay=1,
    universe=UniverseConfig(min_market_cap=5e8, min_avg_amount=1e7),
    factors=[FactorConfig(id="f_101", direction="desc")],
    selection=SelectionConfig(method="topk", top_pct=0.1),
    weighting=WeightingConfig(method="equal"),
    neutralize=NeutralizeConfig(market_cap=True),
    decay=5,
    backtest=BacktestConfig(start_date="20210101", end_date="20241231"),
)
config.validate()

strategy = SingleFactorStrategy(config)
signals = strategy.run(config.backtest.start_date, config.backtest.end_date)
```

---

## Stage 5：简单回测（SimpleSimulator）

- **复权价** `close * adj_factor`
- **无成本**：手续费 / 印花税 / 涨跌停 / 停牌 / 分红 **全部忽略**
- **向量化**：极快，适合大规模参数扫描
- **只产 `nav.parquet`**

```python
from backtest.simulation import SimpleSimulator, SimulationConfig

sim = SimpleSimulator(SimulationConfig(initial_cash=1e8))
result = sim.run(signals, market_data)
result.save("results/f_101/raw/top10pct_1d_d5/simple/", metadata={...})
```

何时用：因子第一次回测、参数扫描、与离线评测对照（看"理论"上的 alpha 上限）。**不能**作为生产决策最终依据——成本会吃掉 0.5%~3% 年化。

---

## Stage 6：详细回测（DetailedSimulator）

- **真实价**：非复权，分红事件改变股数/现金
- **完整成本**：佣金（双向）+ 印花税（卖出）+ 过户费
- **A 股规则**：涨跌停、停牌、板块交易单位
- **输出**：`nav` + `positions` + `trades` + `metrics`

```python
SimulationConfig(
    initial_cash=1e8,
    commission_rate=0.0003,     # 万 3
    min_commission=5.0,
    stamp_duty_rate=0.001,      # 卖出
    transfer_fee_rate=0.00002,
    price_type="o2o",           # T+1 开盘成交（默认）
    allow_short=False,
)
```

#### `price_type`

| price_type | 成交逻辑 | 何时用 |
|---|---|---|
| `o2o` | T+1 开盘。涨停开盘且未打开 → 不能买；跌停开盘且未打开 → 不能卖 | **默认**，贴近实盘集合竞价 |
| `c2c` | 当日收盘。close ≈ limit_up → 不能买；≈ limit_down → 不能卖 | 与离线评测 close 口径对齐 |

```python
sim = DetailedSimulator(SimulationConfig(commission_rate=0.0003, price_type="o2o"))
result = sim.run(signals, market_data, dividends)
result.save("results/f_101/raw/top10pct_1d_d5/detailed/", metadata={...})
```

何时用：简单回测筛过的候选才值得跑详细回测。看 **成本侵蚀** 和 **涨跌停滑点** 吃掉多少。详细 vs 简单的 `annual_return` 差距典型 0.5~3 pp，> 5 pp 说明因子过度依赖涨停股或停牌前后突变。

---

## Stage 7：回测评测（evaluation 模块）

把 `BacktestResult.save()` 落盘的 parquet 反推策略质量。**全项目所有指标的单一真理源**。

### 7.1 命令 / API

```bash
python -m backtest.evaluation results/f_101/raw/top10pct_1d_d5/detailed \
    --benchmark 000300.SH --rf 0.0 --rolling-window 90
```

```python
from backtest.evaluation import evaluate, render_table
report = evaluate("results/f_101/raw/top10pct_1d_d5/detailed",
                  benchmark="000300.SH", plot=True, rf=0.0)
print(render_table(report))
```

输出到 `result_dir` 下：`summary.json` + `summary.csv` + `report.png`（8 子图大图）

### 7.2 指标分组解读

#### 收益

| 指标 | 阈值参考（年化） |
|---|---|
| `annual_return` | **几何年化**。> 15% 优秀，< 5% 弃 |
| `annual_volatility` | A 股股票策略一般 18~30% |
| `worst_day` | < -5% 说明集中度过高 |

#### 风险调整

| 指标 | 看什么 |
|---|---|
| `sharpe` | **核心**：> 1.0 起步，> 1.5 优秀，> 2.0 顶级 |
| `sortino` | 一般比 Sharpe 高 30% |
| `calmar` | `ann_ret / |max_dd|`，> 1.0 稳健 |
| `information_ratio` | 超额 Sharpe；> 0.5 算稳定跑赢 |

#### 风险

| 指标 | 阈值参考 |
|---|---|
| `max_drawdown` | A 股 -20% ~ -35% 常见；< -40% 警示 |
| `recovery_days` | > 365 天说明回撤难修复 |

#### 交易（仅 Detailed）

| 指标 | 看什么 |
|---|---|
| `fees_pct_of_initial` | **> 5% 警示**，> 10% 成本结构有问题 |
| `annual_turnover` | > 20 倍说明吃成本严重 |

#### 持仓（仅 Detailed）

| 指标 | 看什么 |
|---|---|
| `avg_position_count` | 与 top_k 差距大说明涨停/停牌过滤多 |
| `avg_cash_ratio` | 持续 > 10% 说明频繁失败成交 |
| `avg_top5_weight` | > 30% 说明集中度过高 |

#### 基准对比

| 指标 | 阈值参考 |
|---|---|
| `annual_excess` | > 5% 合格，> 10% 优秀 |
| `information_ratio` | > 0.5 合格，> 1.0 优秀 |

### 7.3 八子图大图（`report.png`）

| # | 子图 | 看什么 |
|---|---|---|
| 1 | NAV (+ benchmark) | 主曲线形态、是否持续跑赢 |
| 2 | Drawdown underwater | 回撤深度 + 持续时间 |
| 3 | Monthly return heatmap | 季节性、风格切换 |
| 4 | Yearly returns bar | 年度稳定性 |
| 5 | Position count + cash ratio | 持仓数稳定性；cash 飙升 = 涨停买不进 |
| 6 | Daily turnover | 换手率波动 |
| 7 | Daily return histogram + VaR | 收益分布形态、尾部胖瘦 |
| 8 | Rolling Sharpe (90d) | Sharpe 稳定性 |

---

## Stage 8：人工决策（admit / reject）

### 8.1 核心原则

**`admit` 不再绑死评测**。当前三层评测（factor eval + simple + detailed）跑完后，由人类阅读 `results/<factor_id>/<variant>/<tag>/` 下三份 `summary.json` + `report.png`，自己做决定。

### 8.2 命令

```bash
# 看完报告觉得不错 → 入库
python -m backtest.factor.admission admit f_101 --notes "Sharpe 1.31 detailed, IR 0.92 vs 000300"

# 看完报告觉得不行 → 拒
python -m backtest.factor.admission reject f_101 --notes "RankICIR 仅 0.18"

# 看所有因子状态
python -m backtest.factor.admission status
python -m backtest.factor.admission status f_101
```

### 8.3 `admit` 干了什么

1. 从 work DB（`factors.duckdb`）读 factor 的全部数据（含 `ann_date` / `f_ann_date`）
2. UPSERT 到 library DB（`factor_library.duckdb`）
3. 从 work DB 删除该 factor 的所有行
4. 更新 `registry.json` 的 `status="admitted"` + append 到 `admission_history`

```
registry.json:
{
  "f_101": {
    "name": "my_idea",
    "category": "reversal",
    ...,
    "status": "admitted",
    "admission": {
      "action": "admitted",
      "rows_promoted": 6234521,
      "rows_cleared": 6234521,
      "timestamp": "2026-05-18T15:30:11+00:00",
      "notes": "Sharpe 1.31 detailed, IR 0.92 vs 000300"
    },
    "admission_history": [...]
  }
}
```

### 8.4 `reject` 干了什么

1. 从 work DB 删除该 factor 的所有行
2. 更新 `registry.json` 的 `status="rejected"`
3. **不动 library DB**

### 8.5 临时数据清理（不改 status）

如果想保留因子定义和评测产出、但不入库、不写 rejected，比如打算重做参数后再评：

```bash
python -m backtest.factor.cleanup f_101          # 仅清 work，status 保持 pending
python -m backtest.factor.cleanup --all          # 清空整个 work
python -m backtest.factor.cleanup --orphans      # 清掉 work 中已 admitted 的孤儿行
```

### 8.6 增量更新 library

`admit` 后该因子的"住所"是 library。日常用：

```bash
python -m backtest.factor.update      # 把所有 admitted 因子追平到最新交易日
```

只动 library 库，不动 work 库。

---

## 9. 完整示例：从灵感到入库

### Step 0：构思

"高反转 + 高换手 → 可能超买、未来跑输。负向因子。"

### Step 1：写代码（`backtest/factor/user/reversal.py` 已有 `f_rev_05`）

```python
@register("f_rev_05", name="reversal_zscore_combo", category="reversal",
          data_sources=["market_daily"],
          parameters={"ret_window": 20, "turnover_window": 20, "z_window": 60})
def reversal_zscore_combo(panel, ret_window=20, turnover_window=20, z_window=60):
    ...
    return z_score(ret_neg, z_window) * z_score(to_mean, z_window)
```

### Step 2：回填（写 work DB）

```bash
python -m backtest.factor.backfill f_rev_05
# Backfill range: 20180103 ~ 20241231 (work DB)
# f_rev_05: wrote 6,234,521 rows
```

### Step 3-7：一条命令跑完三层评测

```bash
python scripts/run_factor_pipeline.py f_rev_05 \
    --start 20210101 --end 20241231 \
    --direction asc --benchmark 000300.SH
```

控制台打印（节选）：

```
[1/4] Factor evaluation: f_rev_05
  RankICIR (h=20) = 0.31, IC+ratio = 0.55, Turnover = 0.42
  Reference thresholds: 4/4 OK
  saved: results/f_rev_05/swl2_capq5/factor_eval/f_rev_05_20d.png
  saved: results/f_rev_05/swl2_capq5/decile_backtest/f_rev_05_swl2_capq5_decile.png

[2/4] Simple backtest: f_rev_05
  Annual Return = +18.2%, Sharpe = 1.45, MaxDD = -22.1%
  saved: results/f_rev_05/swl2_capq5/top100_1w_d5/simple/report.png

[3/4] Detailed backtest: f_rev_05
  Annual Return = +16.1%, Sharpe = 1.31, MaxDD = -23.4%
  Fees % Initial = 2.8%, IR = 0.92 (vs 000300.SH)
  saved: results/f_rev_05/swl2_capq5/top100_1w_d5/detailed/report.png

Decision summary
  Factor thresholds passed : 4/4
  Decile monotonicity      : +0.612
  Decile LS ann_ret / sharpe: +8.34% / 0.72
  Simple   Sharpe / MDD    : 1.45 / -22.10%
  Detailed Sharpe / MDD    : 1.31 / -23.40%
  Cost drag (simple - det) : +2.10%

Next step:
  python -m backtest.factor.admission admit  f_rev_05 --variant swl2_capq5 --tag top100_1w_d5
  python -m backtest.factor.admission reject f_rev_05 --variant swl2_capq5 --tag top100_1w_d5
```

> pipeline 跑完会自动生成一份 `pipeline_report.md`，汇总所有阶段的关键指标（IC/RankIC、threshold checks、回测收益/风险、cost drag、决策汇总），打开即可快速决策。

输出结构：

```
results/f_rev_05/
└── swl2_capq5/                   # variant
    ├── factor_eval/              # variant-scoped: tag 无关,变 tag 不重算
    │   ├── f_rev_05_20d.png
    │   └── eval_summary.json
    ├── decile_backtest/          # variant-scoped, 仅 --decile
    │   └── f_rev_05_swl2_capq5_decile.png
    └── top100_1w_d5/              # tag = top{n|pct}_{rebalance}_d{decay}
        ├── pipeline.json            # 机器可读：所有指标 + threshold checks
        ├── pipeline_report.md       # 人类可读：汇总决策报告
        ├── simple/
        │   ├── nav.parquet
        │   ├── metadata.json
        │   ├── summary.json / summary.csv
        │   └── report.png
        └── detailed/
            ├── nav.parquet, positions.parquet, trades.parquet, metrics.parquet
            ├── metadata.json
            ├── summary.json / summary.csv
            └── report.png
```

### Step 8：人工决策

打开三份 `report.png` 和 `summary.json`（如有 `--decile` 则再加一份 `decile_backtest/*.png`），确认：
- RankICIR 0.31 达 RECOMMENDED
- detailed Sharpe 1.31，max_corr 0.78 < 0.85
- 成本侵蚀 2.1 pp 在可接受范围
- 十段分层 monotonicity +0.61，D10 > D9 > ... > D1 排列整齐

→ 入库：

```bash
python -m backtest.factor.admission admit f_rev_05 \
    --variant swl2_capq5 --tag top100_1w_d5 \
    --notes "RankICIR 0.31, detailed Sharpe 1.31, IR 0.92 vs 000300.SH"
```

```
==================================================================
Admission: f_rev_05  ->  ADMITTED
==================================================================
  rows promoted to library: 6,234,521
  rows cleared from work  : 6,234,521
  timestamp              : 2026-05-18T15:30:11+00:00
==================================================================
```

此后：
- `registry.json` 该因子状态变为 `"admitted"`
- 数据从 `factors.duckdb` 迁移到 `factor_library.duckdb`
- 下一个新因子做 evaluation 时，`f_rev_05` 进入 corr 比较名单
- 日常 `python -m backtest.factor.update` 自动维护其增量

---

## 10. 指标解读速查表

### 因子离线（factor.evaluation）

| 指标 | 顶级 | 优秀 | 合格 | 边缘 | 失败 |
|---|---|---|---|---|---|
| RankICIR (h=20) | > 0.7 | 0.4 ~ 0.7 | 0.25 ~ 0.4 | 0.15 ~ 0.25 | < 0.15 |
| RankIC_mean | > 0.10 | 0.05 ~ 0.10 | 0.03 ~ 0.05 | 0.01 ~ 0.03 | < 0.01 |
| IC+_ratio | > 0.62 | 0.56 ~ 0.62 | 0.52 ~ 0.56 | 0.50 ~ 0.52 | < 0.50 |
| Turnover | < 0.20 | 0.20 ~ 0.30 | 0.30 ~ 0.50 | 0.50 ~ 0.70 | > 0.70 |
| 与 library \|corr\| | < 0.50 | 0.50 ~ 0.70 | 0.70 ~ 0.85 | 0.85 ~ 0.95 | > 0.95 |
| **Decile monotonicity** | > 0.70 | 0.50 ~ 0.70 | 0.30 ~ 0.50 | 0.10 ~ 0.30 | < 0.10 |
| **Decile LS Sharpe** | > 1.0 | 0.7 ~ 1.0 | 0.5 ~ 0.7 | 0.3 ~ 0.5 | < 0.3 |

### 策略回测（backtest.evaluation）

| 指标 | 顶级 | 优秀 | 合格 | 警示 |
|---|---|---|---|---|
| Sharpe | > 2.0 | 1.5 ~ 2.0 | 1.0 ~ 1.5 | < 1.0 |
| Sortino | > 3.0 | 2.0 ~ 3.0 | 1.3 ~ 2.0 | < 1.3 |
| Calmar | > 2.0 | 1.0 ~ 2.0 | 0.5 ~ 1.0 | < 0.5 |
| 年化收益 | > 30% | 15% ~ 30% | 8% ~ 15% | < 8% |
| 最大回撤 | > -15% | -15%~-25% | -25%~-35% | < -35% |
| 年化超额（vs 沪深300）| > 15% | 8% ~ 15% | 3% ~ 8% | < 3% |
| Information Ratio | > 1.0 | 0.6 ~ 1.0 | 0.3 ~ 0.6 | < 0.3 |
| 日胜率 | > 0.58 | 0.54 ~ 0.58 | 0.50 ~ 0.54 | < 0.50 |
| 月胜率 | > 0.70 | 0.60 ~ 0.70 | 0.55 ~ 0.60 | < 0.55 |
| 年化换手 | < 10x | 10 ~ 20x | 20 ~ 30x | > 30x |
| fees / initial | < 2% | 2% ~ 5% | 5% ~ 8% | > 8% |

---

## 11. 常见陷阱

### 因子层

- **未来信息泄露**：因子函数里出现 `shift(-N)`、`pct_change()` 不限定 groupby、用了 `f_ann_date` 之后的数据，IC 会异常高。判别：`IC(h=1) > 0.05` 且 `IC+_ratio > 0.7` → 立刻审计代码。
- **窗口不够**：parameters 的 window 决定 lookback。多窗口时填**最大**那个。前 N 天的 NaN 会让评测样本掉到几乎为 0。
- **量纲炸裂**：因子值出现 inf / 极端值 → `np.log1p` 或 `winsorize` 或换成 `rank`/`z_score`。
- **MultiIndex 漏 set_index**：返回的 Series 索引不是 `(date, symbol)` → 入库会列名错位。

### 评测层

- **过短样本**：< 1 年的评测 ICIR 不可信。**至少 2 年，最好 4 年**。
- **基准用错**：策略选小盘多 → 对比中证 500/1000 更公平。
- **过度依赖 IC**：A 股极端值多，看 RankIC 更稳。
- **多 horizon 不衰减**：1d/5d/20d/60d IC 都差不多 → 信号可能是慢变常量，本质不预测短期收益。

### 策略层

- **没开市值中性化**：因子在小盘里好 → 实质押注小盘 beta → 牛市靓丽、熊市灾难。**生产策略默认开 `market_cap=True`**。
- **rebalance_freq 与 decay 错配**：周频 + decay=10 实质双月频，反应迟钝。常见组合：日频+decay 1~3 / 周频+decay 5 / 月频不 decay。
- **min_market_cap 漏过滤**：小盘股流动性差，简单回测能赚但详细回测打骨折。

### 回测层

- **只看 simple、不看 detailed**：成本和涨跌停可吞 1~3 pp 年化。
- **基准没拉数据**：`benchmark="000300.SH"` 但 `index_daily` 表里没数据 → 相关指标全 N/A。先：
  ```bash
  python -m backtest.data.backfill_indices --symbols 000300.SH,000905.SH,000852.SH
  ```
- **fees_pct_of_initial > 10%**：换手太高或 commission_rate 设错。

### 入库层（admission）

- **没看完三层报告就 admit**：admit 不再 gate，全靠人工。看完 `results/<fid>/<variant>/{factor_eval,<tag>/{simple,detailed}}/summary.json + report.png` 再决定。
- **work 库孤儿行**：admit 期间崩溃可能留下 work 中已 admitted 的副本。定期：
  ```bash
  python -m backtest.factor.cleanup --orphans
  ```
- **盲目降阈值**：`RECOMMENDED_THRESHOLDS` 不强制，但**不要**改去 admit 明显不达标的因子。门槛宁可调高不要调低。
- **同类因子高相关**：连续 5 个 `f_rev_*` admit 且彼此 corr > 0.7 → 策略组合实质只有 1 个因子。看 `result.corr_with_existing` 全表，主动 reject。

---

## 12. 路线图

- [x] 双 DuckDB 库（work + library）
- [x] admit / reject / cleanup 独立命令
- [x] results 分层（`results/<factor_id>/<variant>/{factor_eval,<tag>/{simple,detailed}}/`）
- [x] `scripts/run_factor_pipeline.py` 通用 driver
- [x] 十段分层回测（`--decile`）
- [ ] `sw_industry` 表落地 → 行业中性化、板块归因
- [ ] `index_members` 表落地 → 限定股票池
- [ ] 多因子组合的样本外参数选择器
- [ ] Walk-forward 滚动评测：训练窗口/评估窗口分离
- [ ] Agent 投研系统调用本 pipeline 作为底层 tool
