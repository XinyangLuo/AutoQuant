# 因子筛选 Pipeline（自动化 step1~step10）

回测系统端到端操作手册：从"灵感"经过 step1~step10 自动化门控流水线，**人工**决定 admit / reject，沉淀为可复用的库内因子。

> 各子模块的内部设计见各自的 `DESIGN.md`：
> [`factor/DESIGN.md`](factor/DESIGN.md) · [`strategy/DESIGN.md`](strategy/DESIGN.md) · [`simulation/DESIGN.md`](simulation/DESIGN.md) · [`evaluation/DESIGN.md`](evaluation/DESIGN.md) · [`data/DESIGN.md`](data/DESIGN.md) · [`pipeline/DESIGN.md`](pipeline/DESIGN.md)
> 本文档专注于"跑一整套筛选流程、阈值怎么配、指标怎么读、什么样的值代表好/坏因子、什么时候 admit"。

---

## 0. 总览

### 两个物理 DuckDB（关键设计）

```
data/duckdb/factors_pending.duckdb ← 工作区 (FactorStorage)
                                       · backfill / pipeline 写读这里
                                       · 临时数据，admit 或 cleanup 时清空

data/duckdb/factor_library.duckdb  ← 稳定库 (FactorLibrary)
                                       · 只有 admit() 写入
                                       · step8/step9 的"与现有因子相关性"只读这里
                                       · update 增量维护
                                       · append-only：禁用 delete_factor
```

**意义**：新因子的研究数据永远不会污染稳定库；与稳定库的相关性比较永远是"对照已稳定的对手"。

### 自动化流水线（step1~step10）

```
step1: 因子构建 + 覆盖率检查（截面缺失率筛）
  ↓
step2: 统一 Barra 中性化（OLS 取残差）→ 与 size/industry 相关性验证
  ↓
step3: 离线 IC/ICIR/t-stat/正IC占比 门控（日频 + 月频两套阈值）
  ↓
step4: 分 10 组单调性检验（Spearman 组号→收益）
  ↓
step5: 策略组装（默认 top_k=100, decay=5, universe=全A）
  ↓
step6: 简单回测（向量化）+ 必检阈值 → 失败回 step5，最多 3 次重试
  ↓
step7: 详细回测（含分红/成本）+ 必检阈值 → 失败回 step5，最多 3 次重试
  ↓
step8: 对全部已入库因子做逐日截面 Ridge 回归，输出 R² 分布（均值做门控）→ 不拒绝，委托 step9
  ↓
step9: 复用 step8 残差 → 计算残差 RankICIR → 决定原值入库 / 残差入库 / 拒绝
  ↓
step10: 生成报告 + 人工 admit
```

所有阈值统一定义在 `config.yaml` → `thresholds.pipeline`，通过 `StepThresholds` 读取——**代码中不 hard-code 任何阈值**。调整阈值直接修改 `config.yaml`，无需改代码。

### 端到端命令一览

```bash
conda activate AutoQuant

# === 自动化流水线（推荐） ===

# 一键跑完 step1~step10
python -m backtest.pipeline run-all f_xxx

# 或分步执行
python -m backtest.pipeline step1 f_xxx
python -m backtest.pipeline step2 f_xxx
# ... step3~step10

# === Agent 驱动（Claude Code subagent） ===

# 查询数据 schema
python -m agents.claude_cli schema --sources market_daily

# 单轮执行（完整 step1~step10）
python -m agents.claude_cli run f_auto_xxx --run-dir results/agent/runs/my_run/round_001

# === 手动流程（附录 A） ===

# 回填因子值到 work DB
python -m backtest.factor.backfill f_xxx

# 离线评测
python -m backtest.factor.evaluation f_xxx --plot --plot-horizon 20

# 策略 + 回测
python -m backtest.pipeline run-all f_xxx

# === 人工决策 ===

# 通过流水线的因子在 results/agent/candidates/<fid>/，review 后手动 admit
python -m backtest.factor.admission admit  f_xxx --notes "Sharpe 1.45"

# 拒绝
python -m backtest.factor.admission reject f_xxx --notes "RankICIR 不达标"

# 维护
python -m backtest.factor.admission status
python -m backtest.factor.cleanup f_xxx
python -m backtest.factor.update
```

### 关键约束

- **Delay = 1**：T 日已知信息 → T+1 日生效。所有阶段的评测、策略、引擎都遵守。
- **PIT 隔离**：行情靠 `get_bars(end_date=D)` 天然限制；财务靠 `get_fina_snapshot(D)` 的 `f_ann_date <= D` + QUALIFY。
- **涨跌停**：策略层不过滤、由引擎在成交日判定。离线评测默认排除涨停无法成交的样本。
- **不自动 admit**：即使全部 step 通过，也只是写入 candidates/ 目录，等待人工 review 后执行 `admit`。

---

## 1. step1：因子构建 + 覆盖率检查

### 1.1 @register 装饰器

```python
# alphas/my_factor.py
from backtest.factor.registry import register
from backtest.factor.transforms import rank, z_score

@register(
    "f_101",                         # 必填，全局唯一稳定 ID
    name="my_idea",                  # 语义别名
    category="reversal",             # 任意分类标签
    data_sources=["market_daily"],   # 决定 compute.py 注入什么 panel
    variant="barra_ind_size",        # 中性化方案：none / barra_l3 / barra_ind_size
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
- **Agent 生成**：前缀 `f_auto_`（Claude 生成）vs `f_`（人工）
- **name**：snake_case 语义名
- **parameters**：把窗口长度都暴露出来，便于参数扫描

### 1.5 回填因子值（写 work DB）

```bash
python -m backtest.factor.backfill f_101                 # 全历史 → work DB
python -m backtest.factor.backfill f_101 --test-days 60  # 调试
python -m backtest.factor.backfill --pending             # 所有 pending 因子批量
```

- **窗口因子需前置 lookback**：`parameters={"window": 20}` 决定 compute.py 多取 `int(window * 1.5)` 个日历日。多窗口时填**最大值**。
- **增量更新 library**：`python -m backtest.factor.update`（只动 library 库的 admitted 因子）。

### 1.6 覆盖率检查（门控）

**通过标准**（`config.yaml` → `thresholds.pipeline.coverage`）：

| 因子类型 | 截面缺失率上限 |
|---|---|
| 量价因子（用 `market_daily`） | < 20%（`max_missing_rate_pv`） |
| 财务因子（用 `income_q` / `balancesheet_q` / `cashflow_q`） | < 30%（`max_missing_rate_fin`） |

**测度**：每日截面 = `(NaN 股票数 / universe 内全部股票数)`，取时序平均。

**不通过 → 直接 reject**，不进入 step2。

---

## 2. step2：统一中性化 + 验证

### 2.1 中性化 pipeline

（详见 `backtest/factor/DESIGN.md`）

```
1. MAD 去极值 → 2. 申万一级行业截面中位数填充 → 3. cs_zscore
4. OLS: factor ~ industry_dummies + size_z        → 5. 取残差，cs_zscore 再标准化
```

中性化方案由 `@register(variant=...)` 声明：
- `"none"` — 不做中性化
- `"barra_l3"` — Barra L3 风格因子回归
- `"barra_ind_size"` — 行业 dummy + Size_z 回归（**推荐默认**）

### 2.2 验证标准

| 测度 | 阈值 | 用途 |
|---|---|---|
| 与 size 的 Pearson corr | `|corr| < max_corr_size` | 验中性化是否成功 |
| 与 industry dummies 的 Pearson corr（任一行业） | `|corr| < max_corr_industry` | 同上 |
| 与 library 已入库因子的 max |corr| | `|corr| < max_corr_existing` | 避免重复因子 |

> 阈值定义在 `config.yaml` → `pipeline.max_corr_size` / `max_corr_industry` / `max_corr_existing`。默认：0.05 / 0.05 / 0.50。
>
> 与 library 因子的相关性仅计算不门控（保留在 metrics 中供参考），统一推迟到 step8 Ridge R² 做风格克隆判定。

不通过 → reject。

---

## 3. step3：IC / ICIR 门控

### 3.1 年化 ICIR 公式

$$\text{ICIR}_{\text{ann}} = \frac{\mu(\text{IC})}{\sigma(\text{IC})} \cdot \sqrt{\frac{252}{h}}$$

$h$ 为预测周期（日频 = 1 或 5，月频 = 21）。

### 3.2 日频阈值（1D 和 5D 任一通过即可）

| 指标 | 阈值 | 说明 |
|---|---|---|
| \|IC\| | > `min_abs_ic`（默认 0.01） | 截面 Pearson 日均 |
| 年化 ICIR | > `min_annual_icir`（默认 1.0） | 上式 |
| t-stat | > `min_ic_tstat`（默认 2.0） | `IC_mean / (IC_std / √n)` |
| 正 IC 占比 | > `min_ic_positive_ratio`（默认 0.55） | |

### 3.3 月频阈值（h=21，单套）

| 指标 | 阈值 |
|---|---|
| \|IC\| | > 0.03 |
| 年化 ICIR | > 0.8 |
| t-stat | > 2.5 |
| 正 IC 占比 | > 0.65 |

**任一指标不达标 → reject**。

### 3.4 单指标解读速查

| 指标 | 公式 | 看什么 |
|---|---|---|
| **IC_mean** | `mean(corr(factor_t, ret_{t+h}))` | 因子与未来收益线性相关性日均值；**正值 = 正向预测** |
| **IC_std** | IC 序列波动 | 越小越稳；超过 `|IC_mean|` 4~5 倍说明噪音大 |
| **ICIR** | `IC_mean / IC_std` | **核心指标**：信号信噪比 |
| **IC_tstat** | `IC_mean / (IC_std/√n)` | IC 显著性；|t| > 2 才算统计显著 |
| **IC_pos_ratio** | 正 IC 日占比 | > 0.55 算稳健 |
| **RankIC_mean** | Spearman 秩相关 | 对极端值不敏感，**A 股更看 RankIC** |
| **RankICIR** | `RankIC_mean / RankIC_std` | 入库参考主门槛 |

### 3.5 好/中/差因子的经验区间（A 股日频，h=20）

| 档位 | RankICIR | RankIC_mean | IC+ 比例 | Turnover | 备注 |
|---|---|---|---|---|---|
| 顶级（罕见） | > 0.7 | > 0.10 | > 0.62 | < 0.30 | 老牌因子在样本外几乎不可能；做出来先怀疑数据泄露 |
| **优秀** | 0.40 ~ 0.70 | 0.05 ~ 0.10 | 0.56 ~ 0.62 | < 0.40 | 多数年份能稳定贡献 alpha |
| 合格 | 0.25 ~ 0.40 | 0.03 ~ 0.05 | 0.52 ~ 0.56 | < 0.50 | 能考虑入库但建议组合使用 |
| 边缘 | 0.15 ~ 0.25 | 0.01 ~ 0.03 | 0.50 ~ 0.52 | 0.50 ~ 0.70 | 单独用风险高；可入多因子作辅助 |
| 失败 | < 0.15 / 负 | ≈ 0 / 负 | < 0.50 | > 0.70 | 直接弃用，或反向看看 |

### 3.6 多 horizon 对比（decay 曲线）

- **理想**：单峰、缓慢衰减。峰值在 5~20d 是日/周频再平衡的甜点。
- **病态**：
  - 1d ICIR 极高、20d/60d 同样高 → 警惕未来信息泄露
  - 所有 horizon 全 < 0.1 → 因子无效
  - 1d 负、20d 突变正 → 噪音夹反转效应，需中性化或剔除短期反转

### 3.7 换手 Turnover

| Turnover | 解读 |
|---|---|
| < 0.10 | 几乎静态，多见于估值/质量类 |
| 0.10 ~ 0.30 | 中低换手，月频再平衡可保 alpha |
| 0.30 ~ 0.50 | 高换手，**周频/日频 + 成本控制**才能落地 |
| > 0.50 | 噪声多 / 日间波动 |

---

## 4. step4：单调性（十段分层）

### 4.1 测度

**Spearman corr(组号 1~10, 各组年化收益率)**。

将 universe 股票每期按因子值分 10 组（`pd.qcut`），每组等权持有，追踪 10 条净值曲线 + 多空对冲（D10 - D1）。

### 4.2 通过标准

单调性 > `min_monotonicity`（默认 0.7）。

不通过 → reject。

### 4.3 指标解读

| 指标 | 看什么 |
|---|---|
| **Monotonicity** | 年化收益与分位排名的相关系数；**> 0.5 强单调**，0.2~0.5 弱单调，< 0 反向或无序 |
| **D10 vs D1 年化收益** | 头尾差距越大，因子区分度越好 |
| **LS Sharpe** | 多空对冲的年化 Sharpe；> 0.5 说明对冲后仍有稳定 alpha |
| **LS MaxDD** | 多空对冲的回撤；> -20% 说明对冲比较干净 |

> 十段分层和 IC/RankIC、策略回测（simple/detailed）是**三个独立的验证维度**。一个有高 IC 但分层不单调（如 `z(-ret) * z(turnover)` 这类四象限因子），另一个分层单调但 IC 一般（非线性关系）。三者互补。

输出到 `results/<factor_id>/decile_backtest/<factor_id>_decile.png`：

| 面板 | 看什么 |
|---|---|
| 上：10 条 NAV 曲线（log 轴） | D1~D10 是否单调排列；RdYlGn 色谱，D1 红、D10 绿 |
| 下：Long-Short NAV | D10 - D1 的累计净值；> 1 且持续上升 = 多空有效 |

---

## 5. step5：策略组装

### 5.1 默认配置

**两层配置文件**（详见 [`pipeline/DESIGN.md`](pipeline/DESIGN.md) 配置系统章节）：

| 文件 | 位置 | 内容 |
|------|------|------|
| 全局 `config.yaml` | 项目根 | 各 step **门控阈值**（所有因子共享） |
| Per-factor `config.yaml` | `alphas/.../<factor_id>/config.yaml` | **回测参数**（本因子独立），覆盖全局 pipeline 参数 |

从 per-factor `config.yaml` → `pipeline` 读取（未指定则用默认值）：

```yaml
selection:
  method: topk
  top_k: 100          # 默认选前 100 支；与 top_pct 二选一
decay: 5              # 日频因子；月频因子可设 None
universe:
  index_members: null # 默认全 A；可选 000300/000905/000852/932000
rebalance_freq: 1D    # 日频；月频因子用 1M
delay: 1
```

月频因子例外：`rebalance_freq=1M`，`decay=None`。

per-factor `config.yaml` 非必须——不创建则全部用默认值。创建后可按因子独立调整区间、股票池、top_k、decay 等，无需改全局配置。

### 5.2 完整 YAML 范例

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
  index_members: null
  min_market_cap: 500000000.0       # 5 亿流通市值
  min_avg_amount: 10000000.0        # 1000 万 20 日均成交额

factors:
  - id: "f_101"
    direction: "desc"               # desc = 因子值大的好
    weight: 1.0

selection:
  method: "topk"
  top_pct: 0.1                      # 与 top_k 二选一

weighting:
  method: "equal"

decay: 5                            # 线性衰减加权

backtest:
  start_date: "20210101"
  end_date:   "20241231"
  benchmark:  "000300.SH"
```

### 5.3 关键字段详解

#### `rebalance_freq`

| 频率 | 何时合适 |
|---|---|
| `1D` | **默认**，适合多数 alpha 因子；高换手因子（turnover > 0.3）必须日频 |
| `2W` | 中低换手（turnover 0.1~0.3） |
| `1M` | 估值/质量等慢变因子 |
| `EOM` | 月末再平衡（避开月初波动 / 配合月报披露） |

#### `selection.method`

| 方法 | 输出特征 | 何时用 |
|---|---|---|
| `topk` | 多头 top_k 只或 top_pct%，sum = 1 | A 股主流，**生产 80% 用这个** |
| `long_short` | 多 top_k + 空 bottom_k | 研究 alpha 纯度 |
| `decile` | 10 组各 sum = 1 | 验证单调性、做分层归因 |

`top_k` 与 `top_pct` 互斥——恰好指定一个。

#### `decay`

线性加权平滑：`decay(x, n) = (n·x[T] + (n-1)·x[T-1] + ... + 1·x[T-n+1]) / (n·(n+1)/2)`

| decay | 效果 |
|---|---|
| `None` | 原始信号 |
| `3` | 弱平滑 |
| `5` | **常用默认**，明显降换手 ~30%，IC 损失轻微 |
| `10` | 强平滑；不要再叠加月频再平衡 |

#### `universe.min_market_cap` / `min_avg_amount`

**生产默认**：5 亿流通市值 + 1000 万 20 日均成交额。研究阶段可拉宽，生产/详细回测必须收紧。

---

## 6. step6：简单回测 + 门控

### 6.1 阈值

| 指标 | 类型 | 阈值（日频） | 阈值（月频） |
|---|---|---|---|
| Sharpe | 绝对 | > 0.8 | > 1.0 |
| 年化收益 | 绝对 | > 10% | > 10% |
| 最大回撤 | 绝对 | < 50% | < 50% |
| Calmar | 绝对 | > 0.5 | > 0.5 |
| 超额 Sharpe (vs HS300) | 相对 | > 0.5（若配置） | — |
| 超额年化收益 (vs HS300) | 相对 | > 5%（若配置） | — |
| 超额最大回撤 (vs HS300) | 相对 | < 30%（若配置） | — |
| 超额 Calmar (vs HS300) | 相对 | > 0.3（若配置） | — |

> 阈值定义在 `config.yaml` → `thresholds.pipeline.simple_backtest`。所有阈值均支持设为 `null` 以禁用该门控。不检查换手率（SimpleSimulator 不模拟交易）。相对 HS300/CSI500/CSI1000 的超额指标在配置了对应阈值时才会参与门控。

### 6.2 简单回测特点

- **复权价** `close * adj_factor`
- **无成本**：手续费 / 印花税 / 涨跌停 / 停牌 / 分红 **全部忽略**
- **向量化**：极快，适合大规模参数扫描
- **只产 `nav.parquet`**

```python
from backtest.simulation import SimpleSimulator, SimulationConfig

sim = SimpleSimulator(SimulationConfig(initial_cash=1e8))
result = sim.run(signals, market_data)
```

### 6.3 重试机制（最多 3 次）

全部通过 → step7；任一不通过 → 回 step5。

每次 agent 拿到失败指标后选择调整一项：
- `decay` ∈ {1, 3, 5, 10}
- `universe` ∈ {全A, 000300, 000905, 000852}
- `top_k` ∈ {20, 50, 100} 或 `top_pct` ∈ {0.05, 0.10, 0.20}

第 3 次仍不通过 → reject。

---

## 7. step7：详细回测 + 门控

### 7.1 阈值

| 指标 | 类型 | 阈值（日频） | 阈值（月频） |
|---|---|---|---|
| Sharpe | 绝对 | > 0.4 | > 0.6 |
| 年化收益 | 绝对 | > 8% | > 8% |
| 最大回撤 | 绝对 | < 50% | < 50% |
| Calmar | 绝对 | > 0.5 | > 0.5 |
| 年化双边换手率 | 绝对 | < 50 倍 | < 50 倍 |
| 超额 Sharpe (vs HS300) | 相对 | > 0.4（若配置） | — |
| 超额年化收益 (vs HS300) | 相对 | > 4%（若配置） | — |
| 超额最大回撤 (vs HS300) | 相对 | < 25%（若配置） | — |
| 超额 Calmar (vs HS300) | 相对 | > 0.3（若配置） | — |

> 阈值定义在 `config.yaml` → `thresholds.pipeline.detailed_backtest`。所有阈值均支持设为 `null` 以禁用该门控。相对 HS300/CSI500/CSI1000 的超额指标在配置了对应阈值时才会参与门控。

### 7.2 详细回测特点

- **真实价**：非复权，分红事件改变股数/现金
- **完整成本**：佣金（双向）+ 印花税（卖出）+ 过户费
- **A 股规则**：涨跌停、停牌、板块交易单位
- **输出**：`nav` + `positions` + `trades` + `metrics`

```python
from backtest.simulation import DetailedSimulator, SimulationConfig

sim = DetailedSimulator(SimulationConfig(
    initial_cash=1e8,
    commission_rate=0.0003,     # 万 3
    min_commission=5.0,
    stamp_duty_rate=0.001,      # 卖出
    transfer_fee_rate=0.00002,
    price_type="o2o",           # T+1 开盘成交（默认）
    allow_short=False,
))
result = sim.run(signals, market_data, dividends)
```

#### `price_type`

| price_type | 成交逻辑 | 何时用 |
|---|---|---|
| `o2o` | T+1 开盘。涨停开盘且未打开 → 不能买；跌停开盘且未打开 → 不能卖 | **默认**，贴近实盘集合竞价 |
| `c2c` | 当日收盘。close ≈ limit_up → 不能买；≈ limit_down → 不能卖 | 与离线评测 close 口径对齐 |

详细 vs 简单的 `annual_return` 差距典型 0.5~3 pp，> 5 pp 说明因子过度依赖涨停股或停牌前后突变。

### 7.3 重试机制

同 step6（独立计数，最多 3 次）。全部通过 → step8。

---

## 8. step8：每日截面 Ridge R² 分布

### 8.1 逻辑

候选因子对 library 中**全部已入库因子**（含 Barra L1 + 已 admit 的 alpha），逐日做**截面 Ridge 回归**，得到每日 R² 的分布。门控使用 **R² 均值**，同时输出中位数 / P90 / P95 / P99 作为参考。

| 均值 R² 范围 | 分类 | 含义 |
|---|---|---|
| R² < `pure_alpha_max`（默认 0.2） | `pure_alpha` | 与现有因子几乎正交，原值入库 |
| `pure_alpha_max` ≤ R² < `smart_beta_max`（默认 0.7） | `smart_beta` | 部分风格暴露，原值入库 |
| R² ≥ `smart_beta_max` | 标记 `needs_residual` | **不拒绝**，委托 step9 残差 ICIR 判定 |

> 阈值定义在 `config.yaml` → `thresholds.admission.ridge_r2`。与 step9 共享同一次逐日 Ridge 拟合（残差复用，不重复计算）。

### 8.2 边界情况

- Library 中尚无任何已入库因子 → 抛出 `LibraryNotBootstrappedError`

---

## 9. step9：残差 ICIR 增量信息

### 9.1 逻辑

1. 复用 step8 预计算的**逐日截面 Ridge 残差**（避免重复拟合）
2. 计算残差对 1D / 5D / 20D 远期收益的逐日 RankIC
3. 年化 ICIR = raw_icir × √(252/h)
4. **任一周期同时满足年化残差 RankICIR > 阈值（默认 0.05）且 |IC 均值| > 下限（默认 0.001）** → 通过

### 9.2 入库模式

| step8 结果 | 残差 ICIR | 入库模式 | 说明 |
|-----------|----------|---------|------|
| 非 `needs_residual` | 通过 | `raw` | 原值入库 |
| `needs_residual` | 通过 | `residual` | **残差入库**（剥离风格克隆部分，仅保留纯净 alpha） |
| 任意 | 不通过 | `reject` | 拒绝 |

> 阈值定义在 `config.yaml` → `thresholds.admission.residual_icir`。

### 9.2 边界情况

- Library 中 0 个已入库因子 → 平凡通过

---

## 10. step10：报告 + 人工入库

### 10.1 自动化行为

step1~step9 全部通过后：

1. 生成 `pipeline_report.md`（汇总所有阶段的关键指标、阈值对照、决策汇总）
2. Agent 模式下写入 `results/agent/candidates/<factor_id>/`（含 factor.py、pipeline_state.json、result.json）
3. **不自动 admit**——等待人工 review

### 10.2 人工决策

```bash
# 查看 candidates 目录，review 报告后决定

# 入库
python -m backtest.factor.admission admit f_xxx --notes "Sharpe 1.31 detailed, IR 0.92 vs 000300"

# 拒绝
python -m backtest.factor.admission reject f_xxx --notes "RankICIR 仅 0.18"

# 看所有因子状态
python -m backtest.factor.admission status
```

### 10.3 `admit` 干了什么

1. 从 work DB（`factors_pending.duckdb`）读 factor 的全部数据
2. UPSERT 到 library DB（`factor_library.duckdb`）
3. 从 work DB 删除该 factor 的所有行
4. 更新 `registry.json` 的 `status="admitted"` + append 到 `admission_history`

### 10.4 `reject` 干了什么

1. 从 work DB 删除该 factor 的所有行
2. 更新 `registry.json` 的 `status="rejected"`
3. **不动 library DB**

### 10.5 临时数据清理（不改 status）

```bash
python -m backtest.factor.cleanup f_xxx          # 仅清 work，status 保持 pending
python -m backtest.factor.cleanup --all          # 清空整个 work
python -m backtest.factor.cleanup --orphans      # 清掉 work 中已 admitted 的孤儿行
```

### 10.6 增量更新 library

```bash
python -m backtest.factor.update      # 把所有 admitted 因子追平到最新交易日
```

只动 library 库，不动 work 库。

---

## 附录 A：手动流程速查（旧版第 1~8 章）

> 以下为旧版手动操作流程的压缩参考。自动化 step1~step10 已覆盖全链路，手动命令主要用于调试和单步验证。

### A.1 构建因子 + 回填

```bash
$EDITOR alphas/my_factor.py
python -m backtest.factor.backfill f_xxx
python -m backtest.factor.backfill f_xxx --test-days 60  # 调试
```

### A.2 离线评测

```bash
python -m backtest.factor.evaluation f_xxx \
    --corr-top-k 5 --plot --plot-horizon 20
```

Python API：
```python
from backtest.factor import evaluate, print_evaluation
res = evaluate("f_101", start, end, horizons=[1, 5, 10, 20, 60], ret_type="open")
print_evaluation(res)
```

### A.3 策略 + 回测（一键）

```bash
# 策略默认通过 alphas/exp/agent/<factor_id>/config.yaml 配置
# python -m backtest.pipeline run-all f_xxx
```

### A.4 回测评测

```bash
python -m backtest.evaluation results/f_101/raw/top10pct_1d_d5/detailed \
    --benchmark 000300.SH --rf 0.0 --rolling-window 90
```

Python API：
```python
from backtest.evaluation import evaluate, render_table
report = evaluate("results/f_101/raw/top10pct_1d_d5/detailed",
                  benchmark="000300.SH", plot=True, rf=0.0)
print(render_table(report))
```

输出到 `result_dir` 下：`summary.json` + `summary.csv` + `report.png`（8 子图大图）

### A.5 八子图大图（`report.png`）

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

## 附录 B：命令行速查

```bash
conda activate AutoQuant

# === 数据 ===
python -m backtest.data.cold_start                       # 一键全量初始化
python -m backtest.data.update_daily                     # 日更

# === 自动化流水线 ===
python -m backtest.pipeline run-all f_xxx                # step1~step10 一键
python -m backtest.pipeline step1 f_xxx                  # 分步执行
# ... step2~step10

# === 因子 ===
python -m backtest.factor.backfill f_xxx                 # 回填到 work DB
python -m backtest.factor.evaluation f_xxx --plot         # 离线评测
python -m backtest.factor.admission admit f_xxx           # 人工入库
python -m backtest.factor.admission reject f_xxx          # 拒绝
python -m backtest.factor.admission status                # 状态查询
python -m backtest.factor.cleanup f_xxx                  # 清 work
python -m backtest.factor.update                          # 更新 library

# === 策略 + 回测（手动） ===
python -m backtest.pipeline run-all f_xxx
python -m backtest.evaluation <result_dir> --benchmark 000300.SH

# === Agent ===
python -m agents.claude_cli schema --sources market_daily
python -m agents.claude_cli run f_auto_xxx --run-dir <dir>
```

---

## 附录 C：指标解读速查表

### C.1 因子离线（factor.evaluation / step3）

| 指标 | 顶级 | 优秀 | 合格 | 边缘 | 失败 |
|---|---|---|---|---|---|
| RankICIR (h=20) | > 0.7 | 0.4 ~ 0.7 | 0.25 ~ 0.4 | 0.15 ~ 0.25 | < 0.15 |
| RankIC_mean | > 0.10 | 0.05 ~ 0.10 | 0.03 ~ 0.05 | 0.01 ~ 0.03 | < 0.01 |
| IC+_ratio | > 0.62 | 0.56 ~ 0.62 | 0.52 ~ 0.56 | 0.50 ~ 0.52 | < 0.50 |
| Turnover | < 0.20 | 0.20 ~ 0.30 | 0.30 ~ 0.50 | 0.50 ~ 0.70 | > 0.70 |
| 与 library \|corr\| | < 0.50 | 0.50 ~ 0.70 | 0.70 ~ 0.85 | 0.85 ~ 0.95 | > 0.95 |
| Decile monotonicity | > 0.70 | 0.50 ~ 0.70 | 0.30 ~ 0.50 | 0.10 ~ 0.30 | < 0.10 |
| Decile LS Sharpe | > 1.0 | 0.7 ~ 1.0 | 0.5 ~ 0.7 | 0.3 ~ 0.5 | < 0.3 |

### C.2 策略回测（step6/step7 / backtest.evaluation）

| 指标 | 顶级 | 优秀 | 合格 | 警示 |
|---|---|---|---|---|
| Sharpe | > 2.0 | 1.5 ~ 2.0 | 1.0 ~ 1.5 | < 1.0 |
| Sortino | > 3.0 | 2.0 ~ 3.0 | 1.3 ~ 2.0 | < 1.3 |
| Calmar | > 2.0 | 1.0 ~ 2.0 | 0.5 ~ 1.0 | < 0.5 |
| 年化收益 | > 30% | 15% ~ 30% | 8% ~ 15% | < 8% |
| 最大回撤 | > -15% | -15%~-25% | -25%~-35% | < -35% |
| 年化超额（vs 沪深300）| > 15% | 8% ~ 15% | 3% ~ 8% | < 3% |
| 超额 Sharpe（vs 沪深300）| > 1.0 | 0.6 ~ 1.0 | 0.3 ~ 0.6 | < 0.3 |
| 超额 Sharpe（vs 中证500）| > 1.0 | 0.6 ~ 1.0 | 0.3 ~ 0.6 | < 0.3 |
| 超额 Sharpe（vs 中证1000）| > 1.0 | 0.6 ~ 1.0 | 0.3 ~ 0.6 | < 0.3 |
| Information Ratio | > 1.0 | 0.6 ~ 1.0 | 0.3 ~ 0.6 | < 0.3 |
| 日胜率 | > 0.58 | 0.54 ~ 0.58 | 0.50 ~ 0.54 | < 0.50 |
| 月胜率 | > 0.70 | 0.60 ~ 0.70 | 0.55 ~ 0.60 | < 0.55 |
| 年化换手 | < 10x | 10 ~ 20x | 20 ~ 30x | > 30x |
| fees / initial | < 2% | 2% ~ 5% | 5% ~ 8% | > 8% |

---

## 附录 D：常见陷阱

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

- **没开市值中性化**：因子在小盘里好 → 实质押注小盘 beta → 牛市靓丽、熊市灾难。生产策略默认开 `variant="barra_ind_size"`。
- **rebalance_freq 与 decay 错配**：周频 + decay=10 实质双月频，反应迟钝。常见组合：日频+decay 1~3 / 周频+decay 5 / 月频不 decay。
- **min_market_cap 漏过滤**：小盘股流动性差，简单回测能赚但详细回测打骨折。

### 回测层

- **只看 simple、不看 detailed**：成本和涨跌停可吞 1~3 pp 年化。
- **基准没拉数据**：`benchmark="000300.SH"` 但 `index_daily` 表里没数据 → 相关指标全 N/A。先跑 `python -m backtest.data.cold_start`。
- **fees_pct_of_initial > 10%**：换手太高或 commission_rate 设错。

### 入库层

- **没看完报告就 admit**：admit 不再 gate，全靠人工。看完 `pipeline_report.md` + 回测 report.png 再决定。
- **work 库孤儿行**：`python -m backtest.factor.cleanup --orphans` 定期清理。
- **盲目降阈值**：门槛宁可调高不要调低。通过 `config.yaml` 统一调整。
- **同类因子高相关**：连续 5 个同类 admit 且彼此 corr > 0.7 → 策略组合实质只有 1 个因子。主动 reject。

---

## 附录 E：路线图

- [x] 双 DuckDB 库（work + library）
- [x] admit / reject / cleanup 独立命令
- [x] 自动化 step1~step10 流水线（`backtest/pipeline/`）
- [x] 十段分层回测
- [x] Barra 风险模型（7 个 L1 因子）
- [x] Ridge R² 风格分类 + 残差 ICIR 增量信息检查
- [x] Agent 投研系统接入（`agents/runner.py` 复用 pipeline step 函数）
- [x] Candidates 目录（`results/agent/candidates/`）
- [x] `sw_industry` 表落地 → 行业中性化、板块归因
- [x] `index_members` 表落地 → 限定股票池
- [ ] 多因子组合的样本外参数选择器
- [ ] Walk-forward 滚动评测：训练窗口/评估窗口分离
- [ ] OOS / IS 切分：IS 70% + OOS 30%，要求 OOS IC 衰减 < 30%
- [ ] 多 universe 稳健性：step7 同时跑全A / 沪深300 / 中证500
