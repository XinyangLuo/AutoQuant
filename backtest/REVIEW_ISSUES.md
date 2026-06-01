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

### 2.3 data — 指数成分股 densify 生效时点假设
- **文件**：`backtest/data/fetcher/index_members_fetcher.py`
- **问题**：`densify_to_trade_dates` 假设 Tushare `pro.index_weight` 的 `trade_date` 是"已知日"，从该日（含）开始生效。但中证/沪深指数公司的成分股调整通常是"月末公布、次月生效"，如果 Tushare 在公布日就更新数据，则公布日到生效日之间使用了未来成分股。
- **修复方向**：增加 `lag_days=1` 参数或 `effective_date` 字段，让 snapshot 从次交易日开始生效。

### 2.4 data — `update_flag` tiebreaker 不可靠
- **文件**：`backtest/data/storage.py`
- **问题**：同 `(symbol, end_date, f_ann_date)` 下若存在 `update_flag=0/1` 两条记录且数值不同，`ORDER BY update_flag DESC` 的 tiebreaker 非确定性。
- **修复方向**：增加 `GROUP BY` 去重或文档化"同 f_ann_date 下任取一条"的假设。

### 2.5 data — `dividends` 表主键丢失多次分红
- **文件**：`backtest/data/storage.py`
- **问题**：主键 `(symbol, end_date)` 不支持同一报告期的多次分红（特别股息）。
- **修复方向**：改为 `(symbol, end_date, ex_date)` 或 `(symbol, end_date, ann_date)`。

### 2.6 strategy — 新股过滤日历日近似
- **文件**：`backtest/strategy/universe.py:58-68`
- **问题**：`exclude_new_ipo_days` 用 `(current_dt - list_dt).dt.days / 0.65` 估算交易日，长假前后系统性偏差。
- **修复方向**：改用 `get_trade_dates()` 精确计算实际交易日数。

### 2.7 pipeline — step4 重复执行 `evaluate()`
- **文件**：`backtest/pipeline/steps.py:304-448`
- **问题**：`state.eval_result` JSON 序列化后变成 dict，step4 检查 `isinstance(eval_result, dict)` 时重新执行完整因子评估。
- **修复方向**：为 `EvaluationResult` 添加 `to_dict()` / `from_dict()`，或在 step4 直接从 `eval_summary.json` 读取。

### 2.8 simulation — 停牌股票目标权重未重新归一化
- **文件**：`backtest/simulation/detailed.py:430-431`
- **问题**：停牌股票直接跳过，剩余可交易股票的权重未重新归一化，导致资金闲置。
- **修复方向**：`_rebalance` 中对可交易股票的目标权重重新归一化（或按配置决定是否归一化）。

### 2.9 simulation — o2o 涨停判断逻辑偏乐观
- **文件**：`backtest/simulation/executor.py:43-68`
- **问题**：开盘涨停但盘中打开时，返回以 `limit_up` 价格成交。实际上开盘涨停即无法以开盘价买入。
- **修复方向**：简化逻辑——开盘涨停（`abs(open - limit_up) <= EPS`）直接返回不可交易。

### 2.10 simulation — 现金不足时等比例缩减未按权重优先分配
- **文件**：`backtest/simulation/detailed.py:513-540`
- **问题**：`scale = cash / total_cost` 后逐只取整，资金利用率不足，未按目标权重优先级分配。
- **修复方向**：按目标权重排序，优先保证权重大的股票先成交。

### 2.11 simulation — 无滑点模型
- **文件**：`backtest/simulation/executor.py`
- **问题**：成交价直接是 open/close，无价格冲击。`SimulationConfig` 中无 `slippage` 参数。
- **修复方向**：增加 `slippage_bps` 或 `slippage_model` 参数。

### 2.12 evaluation — 基准使用价格指数未考虑分红再投资
- **文件**：`backtest/evaluation/benchmark.py`
- **问题**：`index_daily.close` 是价格指数，长期系统性低估基准收益。
- **修复方向**：文档说明；未来切换到全收益指数（如有数据）。

### 2.13 evaluation — t-statistic 未做 Newey-West 调整
- **文件**：`backtest/factor/evaluation.py:142-169`
- **问题**：假设 IC 独立同分布，标准误被低估，显著性被夸大。
- **修复方向**：实现 Newey-West 标准误 + p-value 输出。

### 2.14 evaluation — `pd.qcut(duplicates="drop")` 导致分组数不一致
- **文件**：`backtest/factor/evaluation.py:184-198`
- **问题**：不同日期的分组数可能不同，跨日聚合时分组定义漂移。
- **修复方向**：改用 `rank(pct=True)` 后按百分位切分。

---

## 三、性能优化机会（无损）

| 优先级 | 文件 | 问题 | 预期提升 |
|--------|------|------|----------|
| 🔴 P0 | `factor/transforms.py` | `ts_rank/argmax/argmin/decay/product` 用 `rolling.apply(python_func)` | 10–50× |
| 🔴 P0 | `factor/transforms.py` | `ts_corr/ts_covariance` 手动 `for sym in groupby` 循环 | 5–10× |
| 🟡 P1 | `factor/transforms.py` | `cap_neutralize` 逐日 Python 循环 + `pd.qcut` | 3–5× |
| 🟡 P1 | `factor/evaluation.py` | `_turnover` dense pivot（5000×2500 矩阵） | 5–10× 内存 |
| 🟡 P1 | `simulation/detailed.py` | `_rebalance` 用 `sig_df.iterrows()` | 2–3× |
| 🟢 P2 | `factor/compute.py` | `_compute_factor_chunked` 外层 1 年 chunk 与 SQL 内层 6 月 chunk 嵌套冗余 | 1.5–2× |
| 🟢 P2 | `pipeline/steps.py` | `step2` `_max_industry_corr` 逐日 `pd.get_dummies` | 5–10× |
| 🟢 P2 | `factor/transforms.py` | `z_score/ts_ir` 对 index 重复 sort | 1.2–1.3× |

---

## 四、测试套件已知失败（与代码无关）

- `TestDetailedSimulator.test_backtest_result_summary` / `test_save_outputs` / `test_summary_extended`
- `TestEvaluateEndToEnd.test_evaluate_writes_outputs` / `test_render_table_contains_sections` / `test_compute_all_metrics_consistency`
- `TestDecileSimulator.test_ls_nav_relation`

**原因**：基准数据（HS300/CSI500/CSI1000 指数）未回填，或测试本身与当前实现不一致。需在回填 `index_daily` 后重新验证。
