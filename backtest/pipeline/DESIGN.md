# 因子挖掘 Pipeline

## 定位

串行 step1~step9 因子挖掘流水线，每步有明确的淘汰标准（pass/fail gate）。

与旧版 `scripts/run_factor_pipeline.py` 的区别：
- 旧版：一次性跑完 eval + simple BT + detailed BT，无淘汰门控
- 新版：每步独立 CLI，可单独调用；失败即停；state 落盘便于 Agent 介入

## 设计原则

1. **每步独立 CLI**：`python -m backtest.pipeline step3 f_001`
2. **State 落盘共享**：`results/<fid>/pipeline_state.json` 是全流程状态文件
3. **统一返回码**：0=通过, 1=淘汰, 2=基础设施错误
4. **stdout 输出 JSON**：方便 Agent 解析
5. **频率感知阈值**：日频(D)与月频(M)不同

## 目录结构

```
backtest/factor/pipeline/
    __init__.py          # 公开 API
    config.py            # PipelineConfig, StepThresholds
    state.py             # PipelineState（可序列化）
    steps.py             # step1~step9 函数（纯逻辑，无 CLI）
    _report.py           # markdown 报告生成
    _cleanup.py          # 淘汰时清理产物
    __main__.py          # CLI dispatcher: step1~step9 + run-all
```

## CLI 接口

```bash
# 初始化 state
python -m backtest.pipeline init f_001 \
    --start 20160101 --end 20251231 --frequency D

# 逐 step 执行
python -m backtest.pipeline step1 f_001   # coverage check
python -m backtest.pipeline step2 f_001   # neutralization verify
python -m backtest.pipeline step3 f_001   # ICIR gate
python -m backtest.pipeline step4 f_001   # monotonicity
python -m backtest.pipeline step5 f_001   # build strategy config
python -m backtest.pipeline step6 f_001   # simple backtest
python -m backtest.pipeline step7 f_001   # detailed backtest
python -m backtest.pipeline step8 f_001   # ridge r2
python -m backtest.pipeline step9 f_001   # report + admit

# 一键全跑
python -m backtest.pipeline run-all f_001 \
    --start 20160101 --end 20251231 --frequency D

# 从某 step 重跑
python -m backtest.pipeline run-all f_001 --from-step 5

# step5 支持覆盖参数（retry 场景）
python -m backtest.pipeline step5 f_001 \
    --top-pct 0.05 --decay 10 --universe 000300.SH
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
    "step3": {"passed": false, "reason": "ICIR=-0.5 <= 1.0", "metrics": {...}}
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

`step_results` 用 dict（key=step name）方便随机访问和覆盖。

## Step 说明

| Step | 名称 | 功能 | 淘汰标准 |
|------|------|------|----------|
| step1 | Coverage | 截面缺失率检查 | 量价 > 10%，财务 > 30% |
| step2 | Neutralization | 验证中性化有效性 | size/industry corr >= 0.05 或 existing corr >= 0.5 |
| step3 | ICIR | 离线 ICIR 门控 | 日频：\|IC\|<=0.01 或 ICIR<=1.0 或 t<=2.0 或 pos_ratio<=55%（1D/5D 任一通过即可）；月频：\|IC\|<=0.03 或 ICIR<=0.8 或 t<=2.5 或 pos_ratio<=65% |
| step4 | Monotonicity | 10 组单调性 | Spearman corr(group, mean_ret) <= 0.7 |
| step5 | Strategy Config | 构建默认策略配置 | 无淘汰（总是通过） |
| step6 | Simple Backtest | 向量化回测 | Sharpe<=0.8 或 ann_ret<=10% 或 max_dd<=-30% 或 Calmar<=0.5 或 turnover>=20x |
| step7 | Detailed Backtest | 事件驱动回测 | Sharpe<=0.4 或 ann_ret<=8% 或 max_dd<=-30% 或 Calmar<=0.5 或 turnover>=20x |
| step8 | Ridge R² | 风格克隆检测 | tier == reject（R² >= 0.80） |
| step9 | Admission | 报告生成 + 入库 | 入库失败则 reject |

### 频率感知阈值

```python
# 日频 (D)
StepThresholds(
    min_abs_ic=0.01,
    min_annual_icir=1.0,
    min_ic_tstat=2.0,
    min_ic_positive_ratio=0.55,
    min_sharpe_simple=0.8,
    min_sharpe_detailed=0.4,
)

# 月频 (M)
StepThresholds(
    min_abs_ic=0.03,
    min_annual_icir=0.8,
    min_ic_tstat=2.5,
    min_ic_positive_ratio=0.65,
    min_sharpe_simple=1.0,
    min_sharpe_detailed=0.6,
)
```

### Retry 逻辑

仅 step6/step7 支持 retry。当 backtest 不通过时：
1. Agent 分析失败原因，建议新参数（decay/universe/top_pct）
2. 通过 `python -m backtest.pipeline step5 f_001 --top-pct 0.05 --decay 10` 覆盖
3. 重新跑 step6（最多 3 次）

当前 `run-all` 中 retry 为 stub（确定性 fallback 规则），未来替换为 Agent 调用。

## 淘汰清理

任一 step 失败时：
1. `FactorStorage.delete_factor(factor_id)` — 清 work DB
2. `reject(factor_id)` — 标记 registry
3. `shutil.rmtree(results/<factor_id>)` — 删除回测产物

## 与现有模块的关系

| Pipeline Step | 复用的已有模块 |
|---------------|---------------|
| step1 | `FactorStorage.get_factor()`, `MarketStorage.get_bars()` |
| step2 | `FactorStorage.get_factor(SIZE_LNCAP_ID)`, `MarketStorage.get_industry_panel_range()`, `_corr_with_existing()` |
| step3 | `evaluate()` |
| step4 | `EvaluationResult.group_returns`（step3 缓存复用） |
| step5 | `StrategyConfig` dataclass |
| step6 | `SingleFactorStrategy`, `SimpleSimulator` |
| step7 | `DetailedSimulator`, `MarketStorage.get_dividends()` |
| step8 | `ridge_r2_check()` |
| step9 | `admit()`, `EvaluationReport` |

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
            continue  # 重跑 step6
        else:
            print("Factor rejected:", state["step_results"][step]["reason"])
            break
```
