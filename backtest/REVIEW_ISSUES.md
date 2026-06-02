# 回测系统审查剩余问题记录

> 生成时间：2026-06-02
> 来源：全模块深度审查（data / factor / strategy / simulation / evaluation / pipeline）

---

## 一、已修复的 P0 问题（2026-06-02）

1. **SimpleSimulator delay 对齐** — `simple.py` 中 `weight_wide.shift(1)` 与策略层 `_apply_delay` 重复，导致实际延迟为 T+2。修复：移除 weight shift，改为 `returns_wide.shift(1)` 对齐生效日收益。
2. **交易费用重复计算** — `detailed.py` 中 `Trade.commission`（总费用）与 `stamp_duty`/`transfer_fee` 被重复累加。修复：`executor.py` 新增 `calculate_cost_breakdown()`，metrics 层优先读取拆分项。
3. **送转股未调整 avg_cost** — `dividends.py` 送转股后只更新 `shares` 未稀释成本。修复：同步 `avg_cost = avg_cost / (1 + stk_div)`，送转股不再取整。
4. **财务数据 PIT delay 加固** — `get_fina_snapshot_range` 默认 `f_ann_date <= date` 在数据层无防御。修复：新增 `delay` 参数，`compute.py` 调用时传 `delay=1`。

---

## 二、待修复 / 待关注的 P1 问题

### 2.1 strategy — `multi_factor.py` IC 加权未来信息（P0 级，但尚未启用）
- **文件**：`backtest/strategy/strategies/multi_factor.py:138-165`
- **问题**：`_compute_ic_weights` 中 `end_str = date_str`（当前调仓日），IC 计算需要 forward return，导致计算 T 日信号时使用了 T 日收盘后才可知的未来 IC。
- **修复方向**：`end_str` 回退至少 `max(horizons)` 个交易日；增加数据充足性检查（可用数据 < window*0.5 时回退等权）。
- **状态**：`MultiFactorStrategy` 代码已存在，但 pipeline 中尚未启用，暂不紧急。

### 2.2 data — `adj_factor` 未来分红信息（已确认无需修改，但需记录）
- **文件**：`backtest/data/fetcher/daily_fetcher.py`
- **说明**：Tushare 的 `adj_factor` 会回溯修正历史数据（新分红发生后重写过去所有交易日的复权因子）。`SimpleSimulator` 使用 `close * adj_factor` 计算收益。
- **结论**：用户明确判定这不属于未来信息泄露，无需修改。SimpleSimulator 的定位是快速筛选工具，不代表真实收益。

### 2.3 data — 指数成分股 densify 生效时点假设（经核查：**未发现未来信息泄露，暂不修复**）
- **文件**：`backtest/data/fetcher/index_members_fetcher.py`
- **原始假设**：`densify_to_trade_dates` 假设 Tushare `pro.index_weight` 的 `trade_date` 从该日（含）开始生效。担心中证指数公司"公布日早于生效日"，若 Tushare 在公布日即入库，则公布日到生效日之间使用了未来成分股。
- **核查结果**：
  1. Tushare `pro.index_weight` **仅有 4 个字段**：`index_code`, `con_code`, `trade_date`, `weight`，**无 `ann_date`、`effective_date`、`update_time` 等时标字段**。
  2. 实证对比指数生效日与 Tushare `trade_date`：
     | 指数 | 生效日（中证规则：次半年首交易日） | Tushare 新 composition 的 `trade_date` |
     |------|-----------------------------------|----------------------------------------|
     | CSI300 | 2024-06-03（6 月首交易日） | `20240603` ✅ |
     | CSI300 | 2025-01-02（1 月首交易日） | `20250102` ✅ |
     | CSI300 | 2023-06-01（6 月首交易日） | `20230601` ✅ |
  3. 新 composition **从未在生效日之前出现**。例如 2024 年 6 月调整：5 月 31 日仍是旧列表，`20240601-02` 为周末无数据，`20240603` 才出现新列表。
  4. 常规月份（非调整月）Tushare 每月发布 2 次快照（首/末交易日），成分股不变、仅权重随市值漂移。
- **结论**：Tushare 的 `trade_date` 就是**生效日**而非公布日。`densify_to_trade_dates` 的 `merge_asof(..., direction="backward")` 逻辑与生效日对齐，不存在未来信息泄露。保持现状，无需增加 `lag_days`。
- **备注**：若日后发现 Tushare 数据更新机制改变（如在公布日即预发布新成分股），可再评估。当前 empirically 安全。

### 2.4 data — `update_flag` tiebreaker（经核查：**非问题，DESIGN.md 已覆盖**）
- **文件**：`backtest/data/storage.py`
- **原始假设**：同 `(symbol, end_date, f_ann_date)` 下若存在 `update_flag=0/1` 两条记录且数值不同，`ORDER BY update_flag DESC` 的 tiebreaker 非确定性，担心选错版本。
- **核查结果**：
  1. DESIGN.md §"业绩修正与 PIT" 明确结论： **`update_flag` 不可作为版本新旧判定**。实证案例 `920522.BJ` / `920663.BJ` 的 `update_flag` 序列均为 `(1, 0, 1)`，说明它不是严格的 "0=原始 / 1=修正"。
  2. 版本新旧的**唯一可靠判定**是 `f_ann_date DESC`。`update_flag DESC` 的用途 DESIGN.md 写得很清楚：**"仅为去重——同 `(symbol, end_date, f_ann_date)` 偶发 `update_flag=0/1` 两条同值行，任取一条不影响数值，但 outer-join 三表前必须去掉重复以免笛卡尔积。"**
  3. 同 `f_ann_date` 下 `update_flag=0/1` 两行**数值不同**的情况在实证中未被观测到；Tushare 的双行是**同值冗余**，不是竞争关系。
- **结论**：`get_fina_snapshot` 的 `ORDER BY f_ann_date DESC, update_flag DESC` 是正确的。`f_ann_date` 承担版本新旧判定的全部职责，`update_flag` 只是让去重结果稳定。无需修复，已在 DESIGN.md 文档化。

### 2.5 data — `dividends` 表主键丢失多次分红
- **文件**：`backtest/data/storage.py`
- **问题**：主键 `(symbol, end_date)` 不支持同一报告期的多次分红（特别股息）。
- **修复方向**：改为 `(symbol, end_date, ex_date)` 或 `(symbol, end_date, ann_date)`。

### 2.6 strategy — 新股过滤日历日近似（**已修复**）
- **文件**：`backtest/strategy/universe.py:58-68`
- **问题**：`exclude_new_ipo_days` 用 `(current_dt - list_dt).dt.days / 0.65` 估算交易日，长假前后系统性偏差。
- **修复**：改用 `get_trade_dates(min_list_date, date)` 获取完整交易日序列，建立 `date_to_idx` 映射，精确计算每只股票 `list_date` 到当前日的实际交易日数。对 `list_date` 不在日历中的股票（保守保留）用 `isna()` 保护。

### 2.7 pipeline — step4 重复执行 `evaluate()`（**已修复**）
- **文件**：`backtest/pipeline/steps.py:436-448`、`backtest/pipeline/state.py:67-83`、`backtest/factor/evaluation.py:295-350`
- **问题**：`PipelineState.to_dict()` 通过 `asdict()` 把 `EvaluationResult` 序列化为 dict，但 `from_dict()` 丢弃了该字段，导致 step4 加载后 `state.eval_result` 为 `None`，被迫重新执行完整因子评估。
- **修复**：
  1. `EvaluationResult` 新增 `to_dict()` / `from_dict()`，使用 `_df_` / `_s_` 标签序列化/反序列化 `pd.DataFrame` / `pd.Series` 及嵌套 dict。
  2. `PipelineState.from_dict()` 恢复 `eval_result` 字段：若 JSON 中存在且反序列化成功则重建 `EvaluationResult`，失败则回退到 `None`。
  3. step4 保留 backward-compat 的 dict→`EvaluationResult` 转换逻辑，但优先使用已恢复的对象实例。仅在反序列化失败或 `None` 时才 fallback 到重新 `evaluate()`。

### 2.8 simulation — 停牌股票目标权重未重新归一化
- **文件**：`backtest/simulation/detailed.py:430-431`
- **问题**：停牌股票直接跳过，剩余可交易股票的权重未重新归一化，导致资金闲置。
- **修复方向**：`_rebalance` 中对可交易股票的目标权重重新归一化（或按配置决定是否归一化）。
- **意见**：无需修复，停牌的资金就闲置了

### 2.9 simulation — o2o 涨停判断逻辑偏乐观
- **文件**：`backtest/simulation/executor.py:43-68`
- **问题**：开盘涨停但盘中打开时，返回以 `limit_up` 价格成交。实际上开盘涨停即无法以开盘价买入。
- **修复方向**：简化逻辑——开盘涨停（`abs(open - limit_up) <= EPS`）直接返回不可交易。
- **意见**：逻辑错误，开盘涨停但盘中打开当然可以用涨停价成交。

### 2.10 simulation — 现金不足时等比例缩减未按权重优先分配
- **文件**：`backtest/simulation/detailed.py:513-540`
- **问题**：`scale = cash / total_cost` 后逐只取整，资金利用率不足，未按目标权重优先级分配。
- **修复方向**：按目标权重排序，优先保证权重大的股票先成交。
- **意见**：留下来讨论

### 2.11 simulation — 无滑点模型
- **文件**：`backtest/simulation/executor.py`
- **问题**：成交价直接是 open/close，无价格冲击。`SimulationConfig` 中无 `slippage` 参数。
- **修复方向**：增加 `slippage_bps` 或 `slippage_model` 参数。
- **意见**：目前对开盘价冲击小，集合竞价挂单可以以开盘价成交，滑点可以加入但是设置为0

### 2.12 evaluation — 基准使用价格指数未考虑分红再投资
- **文件**：`backtest/evaluation/benchmark.py`
- **问题**：`index_daily.close` 是价格指数，长期系统性低估基准收益。
- **修复方向**：文档说明；未来切换到全收益指数（如有数据）。
- **意见**：先不修改

### 2.13 evaluation — t-statistic 未做 Newey-West 调整
- **文件**：`backtest/factor/evaluation.py:142-169`
- **问题**：假设 IC 独立同分布，标准误被低估，显著性被夸大。
- **修复方向**：实现 Newey-West 标准误 + p-value 输出。
- **意见**：先不修改

### 2.14 evaluation — `pd.qcut(duplicates="drop")` 导致分组数不一致（**已修复**）
- **文件**：`backtest/factor/evaluation.py:184-198`
- **问题**：`pd.qcut(x, n_groups, duplicates="drop")` 在单日存在重复因子值时会减少分组数，导致跨日聚合时各组定义漂移（某天 8 组、某天 10 组）。
- **修复**：`_group_returns()` 改用 `x.rank(pct=True)` 计算百分位排名，再用 `pd.cut(bins=np.linspace(0, 1, n_groups+1), include_lowest=True)` 按固定百分位边界切分。保证每天恰好 `n_groups` 组，不受重复值影响。
- **注意**： ties（重复值）会被分配到同一百分位区间，可能导致各组大小不完全相等，但组编号 0~n_groups-1 始终稳定。

---

## 三、性能优化机会（无损）

| 优先级 | 文件 | 问题 | 预期提升 | 状态 |
|--------|------|------|----------|------|
| 🔴 P0 | `factor/transforms.py` | `ts_rank` `rolling.apply` → `rolling().rank()` | ~5× | **✅ 已修复** |
| 🔴 P0 | `factor/transforms.py` | `ts_product` `rolling.apply` → `exp(sum(log))` 正数快速路径 | ~13× | **✅ 已修复** |
| 🔴 P0 | `factor/transforms.py` | `ts_corr/ts_covariance` 手动 `for` 循环 → wide-format `rolling.corr/cov` | ~5–10× | **✅ 已修复** |
| 🟡 P1 | `factor/transforms.py` | `cap_neutralize` 逐日 Python 循环 + `pd.qcut` → `rank+cut` + `groupby.transform` | ~3–5× | **✅ 已修复** |
| 🟡 P1 | `factor/evaluation.py` | `_turnover` dense pivot → long-format diff | 省 100MB+ 内存 | **✅ 已修复** |
| 🟡 P1 | `simulation/detailed.py` | `_rebalance` `sig_df.iterrows()` → 向量化 + `round_lot_for_symbol_vec` | ~2–3× | **✅ 已修复** |
| 🟢 P2 | `factor/compute.py` | `_compute_factor_chunked` 外层 1 年 chunk 与 SQL 内层 6 月 chunk 嵌套冗余 | 1.5–2× | 未修复 |
| 🟢 P2 | `pipeline/steps.py` | `step2` `_max_industry_corr` 逐日 `pd.get_dummies` | 5–10× | 未修复 |
| 🟢 P2 | `factor/transforms.py` | `z_score/ts_ir` 对 index 重复 sort | 1.2–1.3× | 未修复 |

> **备注**：`ts_argmax`/`ts_argmin`/`ts_decay_linear`/`ts_decay_exp` 未改动——`rolling.apply` 的瓶颈在于每窗口 Python 函数调用，尝试 `as_strided` 向量化后因每窗口权重/边界处理仍需 Python 循环，实测速度无提升或更慢，故保留原实现。若未来引入 `numba`/`bottleneck`，可再评估。 |

---

## 四、测试套件已知失败（与代码无关）

- `TestDetailedSimulator.test_backtest_result_summary` / `test_save_outputs` / `test_summary_extended`
- `TestEvaluateEndToEnd.test_evaluate_writes_outputs` / `test_render_table_contains_sections` / `test_compute_all_metrics_consistency`
- `TestDecileSimulator.test_ls_nav_relation`

**原因**：基准数据（HS300/CSI500/CSI1000 指数）未回填，或测试本身与当前实现不一致。需在回填 `index_daily` 后重新验证。
