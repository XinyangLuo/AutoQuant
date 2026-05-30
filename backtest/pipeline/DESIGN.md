# 因子挖掘 Pipeline — 设计文档

> **使用手册**：[`backtest/PIPELINE.md`](../PIPELINE.md)。本文是技术设计文档，描述架构、数据流和模块契约。两者冲突时以 PIPELINE.md 为准。

## 定位

串行 step1~step10 因子挖掘流水线，每步有明确的淘汰标准（pass/fail gate）。与旧版 `scripts/run_factor_pipeline.py` 的区别：

- 旧版：一次性跑完 eval + simple BT + detailed BT，无淘汰门控
- 新版：每步独立 CLI，可单独调用；失败即停；state 落盘便于 Agent 介入；**拒绝时生成完整诊断报告而非直接清理**

## 设计原则

1. **每步独立 CLI**：`python -m backtest.pipeline step3 f_001`
2. **State 落盘共享**：`results/<fid>/pipeline_state.json` 是全流程状态文件
3. **统一返回码**：0=通过, 1=淘汰, 2=基础设施错误
4. **stdout 输出 JSON**：方便 Agent 解析
5. **阈值单一来源**：所有门控阈值统一从 `config.yaml` 读取（`backtest/config_loader.py`）
6. **拒绝不清理**：因子被拒绝后保留 work DB 数据和完整诊断报告，由人工决定清理时机

## 配置系统

Pipeline 有两层配置，职责明确分离：

| 配置文件 | 位置 | 职责 |
|----------|------|------|
| **全局** `config.yaml` | 项目根 | **门控阈值** (`thresholds.pipeline.*`) + Agent 参数 (`agent.*`) |
| **Per-factor** `config.yaml` | `alphas/{admitted,exp/user,exp/agent}/<factor_id>/config.yaml` | **回测参数** (`pipeline.*` / `strategy.*` / `simulation.*`) |

### 全局 config.yaml（阈值）

```yaml
# config.yaml（项目根）
thresholds:
  pipeline:
    coverage:
      max_missing_rate_pv: 0.2
      max_missing_rate_fin: 0.30
    icir:
      min_abs_ic: 0.01
      min_annual_icir: 1.0
      min_ic_tstat: 2.0
      min_ic_positive_ratio: 0.55
    monotonicity:
      min_monotonicity: 0.7
    simple_backtest:
      min_sharpe: 0.8
      min_annual_return: 0.10
      max_max_drawdown: 0.5
      min_calmar: 0.5
      max_annual_turnover: 50.0
    detailed_backtest:
      min_sharpe: 0.4
      min_annual_return: 0.08
      min_calmar: 0.5
```

修改阈值直接编辑此文件，所有因子共用。

### Per-factor config.yaml（回测参数）

```yaml
# alphas/exp/user/f_xxx/config.yaml
pipeline:
  start_date: "20160101"
  end_date: "20251231"
  eval_horizons: [1, 5, 10, 20, 60]
  icir_check_horizons: [1, 5]
  default_top_k: 100
  default_decay: 5
  default_rebalance: 1D
  ret_type: open
  benchmark: "000300.SH"

strategy:
  universe:
    exclude_st: true
    exclude_new_ipo_days: 252
    include_cyb: true
    include_kcb: false
    include_bse: false
    min_market_cap: 500000000
    min_avg_amount: 10000000

simulation:
  initial_cash: 100000000
  commission_rate: 0.0003
  stamp_duty_rate: 0.001
  transfer_fee_rate: 0.00002
  allow_short: false
```

三个 section 均可选，未指定的字段使用 `config.py` 的硬编码默认值。Pipeline 通过 `PipelineConfig.from_factor_config(factor_id)` 自动发现并合并。

### 合并优先级

```
CLI --overrides > per-factor config.yaml > 硬编码默认值 (_DEFAULT_*)
```

阈值（`StepThresholds`）不受 per-factor 覆盖，始终从全局 `config.yaml` 读取。

## 目录结构

```
backtest/pipeline/
    __init__.py          # 公开 API
    config.py            # PipelineConfig, StepThresholds（默认值从 config.yaml 读取）
    state.py             # PipelineState（可序列化）
    steps.py             # step1~step10 函数（纯逻辑，无 CLI）
    _report.py           # markdown 报告 + 诊断图生成（拒绝时也会执行）
    _cleanup.py          # 手动清理工具（不再自动调用）
    __main__.py          # CLI dispatcher: step1~step10 + run-all
```

## CLI 接口

```bash
# 初始化 state（日期默认从 config.yaml 读取）
python -m backtest.pipeline init f_001 --frequency D

# 逐 step 执行
python -m backtest.pipeline step1 f_001   # coverage check
python -m backtest.pipeline step2 f_001   # neutralization verify
python -m backtest.pipeline step3 f_001   # ICIR gate
python -m backtest.pipeline step4 f_001   # monotonicity
python -m backtest.pipeline step5 f_001   # build strategy config
python -m backtest.pipeline step6 f_001   # simple backtest
python -m backtest.pipeline step7 f_001   # detailed backtest
python -m backtest.pipeline step8 f_001   # ridge r2
python -m backtest.pipeline step9 f_001   # residual icir
python -m backtest.pipeline step10 f_001  # report + admit

# 一键全跑（日期默认从 config.yaml 读取）
python -m backtest.pipeline run-all f_001 --frequency D

# 从某 step 重跑
python -m backtest.pipeline run-all f_001 --from-step 5

# step5 支持覆盖参数（retry 场景）
python -m backtest.pipeline step5 f_001 \
    --top-k 50 --decay 10 --universe 000300.SH
```

每个 step CLI 的执行流程：
1. 读 `pipeline_state.json`
2. 检查前置 step 是否全部通过
3. 执行 step 函数
4. 写回 `pipeline_state.json`
5. stdout 输出 JSON：`{"step": "step3", "passed": true, "metrics": {...}}`
6. exit code：0=pass, 1=reject, 2=infra error

## State 文件格式

`results/<factor_id>/pipeline_state.json`：

```json
{
  "factor_id": "f_001",
  "config": {
    "factor_id": "f_001",
    "start_date": "20160101",
    "end_date": "20251231",
    "frequency": "D",
    "thresholds": {...}
  },
  "status": "running",
  "current_step": "step3",
  "step_results": {
    "step1": {"passed": true, "metrics": {"max_missing_rate": 0.02}},
    "step2": {"passed": true, "metrics": {"size_corr": 0.001}},
    "step3": {"passed": false, "reason": "ICIR<=1.0", "metrics": {...}}
  },
  "retry_count": 0,
  "retry_params": {},
  "artifacts": {
    "eval_result": "results/f_001/factor_eval/eval_summary.json",
    "simple_bt": "results/f_001/top10pct_1d_d5/simple/",
    "detailed_bt": "results/f_001/top10pct_1d_d5/detailed/",
    "report": "results/f_001/pipeline_report.md"
  }
}
```

## Step 说明

| Step | 名称 | 功能 | 淘汰标准 |
|------|------|------|----------|
| step1 | Coverage | 截面缺失率检查（95 分位数） | 量价 > 15%，财务 > 30%（config.yaml） |
| step2 | Neutralization | 验证中性化有效性（size + industry） | size_corr ≥ 0.05 或 ind_corr ≥ 0.05。现有因子相关性仅计算不门控，推迟到 step8 |
| step3 | ICIR | 离线 ICIR 门控 | 日频：|IC|≤0.01 或 ICIR≤1.0 或 t≤2.0 或 pos_ratio≤55%（任一 horizon 通过即可）；月频阈值见 config.yaml |
| step4 | Monotonicity | 10 组单调性 | Spearman corr(group, mean_ret) ≤ 0.7 |
| step5 | Strategy Config | 构建默认策略配置 | 无淘汰（总是通过） |
| step6 | Simple Backtest | 向量化回测（无成本） | Sharpe≤0.8 或 ann_ret≤10% 或 max_dd≤-40% 或 Calmar≤0.5。不检查换手率（SimpleSimulator 不计算） |
| step7 | Detailed Backtest | 事件驱动回测（含成本） | Sharpe≤0.4 或 ann_ret≤8% 或 max_dd≤-40% 或 Calmar≤0.5 或 turnover≥50x |
| step8 | Ridge R² | 逐日截面 Ridge 回归（全部已入库因子），输出 R² 均值/中位数/P90/P95/P99 分布 | **不再淘汰**——标记 `needs_residual` 后委托 step9 判定 |
| step9 | Residual ICIR | 复用 step8 残差 → RankIC → 决定入库模式 | 残差 ICIR 不通过 → 拒绝；通过 + needs_residual → **残差入库**；通过 + 非 needs_residual → 原值入库 |
| step10 | Report | 生成诊断报告，标记 ready_for_review | 无淘汰（总是通过）。需人工 `admit` |

### 阈值来源

所有阈值定义在 `config.yaml` → `thresholds.pipeline`，通过 `backtest.config_loader.get_section()` 读取。`StepThresholds` dataclass 的 `field(default_factory=...)` 在构造时懒加载，保证单一事实来源。

### Ridge R² 分档（config.yaml → thresholds.admission.ridge_r2）

对全部已入库因子逐日截面 Ridge，取每日 R² 分布的**均值**做门控：

| 均值 R² 范围 | Tier | 行为 |
|---------|------|------|
| R² < 0.2 | `pure_alpha` | 原值入库 |
| 0.2 ≤ R² < 0.7 | `smart_beta` | 原值入库 |
| R² ≥ 0.7 | `reject`（标记 `needs_residual`） | 不拒绝，交 step9 判定。残差 ICIR 通过 → **残差入库**；不通过 → 拒绝 |

### 入库模式（step9 输出）

| step8 结果 | 残差 ICIR | `admission_mode` |
|-----------|----------|-----------------|
| 非 `needs_residual` | 通过 | `raw` — 原值入库 |
| `needs_residual` | 通过 | `residual` — **残差入库**（per-date Ridge 剥离全部已入库因子后的纯净 alpha） |
| 任意 | 不通过 | `reject` — 拒绝 |

### Retry 逻辑

仅 step6/step7 支持 retry。当 backtest 不通过时：
1. Agent 分析失败原因，建议新参数（decay/universe/top_pct）
2. 通过 `python -m backtest.pipeline step5 f_001 --top-pct 0.05 --decay 10` 覆盖
3. 重新跑 step6（最多 3 次）

当前 `run-all` 中 retry 为 stub（`state.retry_count` / `retry_params` 已定义但从未写入），未来替换为 Agent 调用。

## 拒绝处理

任一 step 失败时 `run-all` 的行为：
1. 停止后续 step 执行
2. **生成完整诊断报告**（`results/<factor_id>/pipeline_report.md` + 4 张诊断图）
3. **保留** work DB 数据和 results 目录（不自动清理）
4. 返回 exit code 1

手动清理：`python -m backtest.factor.cleanup f_xxx`

## 诊断报告

`run-all` 始终生成报告（通过或拒绝均生成）。报告包含：

- **决策横幅**：拒绝 step 和原因
- **Step 汇总表**：每步 pass/fail + 关键指标 + 拒绝原因
- **因子评估**：IC decay 图 + 分位组收益图
- **回测结果**：NAV + Drawdown 曲线（从 `nav.parquet` 加载实际数据）
- **Ridge R² 分类**：R² 值和 tier 分档

## 与现有模块的关系

| Pipeline Step | 复用的已有模块 |
|---------------|---------------|
| step1 | `FactorStorage.get_factor()`, `MarketStorage.get_bars()` |
| step2 | `FactorLibrary.get_factor(SIZE_L1_ID)`, `MarketStorage.get_industry_panel_range()`, `_corr_with_existing()` |
| step3 | `evaluate()` |
| step4 | `EvaluationResult.group_returns`（step3 缓存复用） |
| step5 | `StrategyConfig` dataclass |
| step6 | `SingleFactorStrategy`, `SimpleSimulator` |
| step7 | `DetailedSimulator`, `MarketStorage.get_dividends()` |
| step8 | `ridge_r2_check()` |
| step9 | `residual_icir_check()` |
| step10 | `admit()`, `generate_pipeline_report()` |

## Agent 交互模式

```python
# Agent 伪代码
for step in ["step1", "step2", ..., "step9"]:
    result = run_cli(f"python -m backtest.pipeline {step} {factor_id}")
    state = json.load(f"results/{factor_id}/pipeline_state.json")

    if result.returncode == 1:  # rejected
        if step in ("step6", "step7") and state["retry_count"] < 3:
            suggestion = agent_analyze_and_suggest(state)
            run_cli(f"python -m backtest.pipeline step5 {factor_id} "
                    f"--top-pct {suggestion['top_pct']} --decay {suggestion['decay']}")
            continue  # retry step6
        else:
            print("Factor rejected:", state["step_results"][step]["reason"])
            # Report already generated by run-all; review and decide.
            break
```
