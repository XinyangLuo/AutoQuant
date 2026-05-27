# /factor-iterate

交互式因子研究入口：用户给出一个自然语言因子猜测后，由 Claude Code 直接生成因子代码、运行单轮回测、分析结果、写入 trace，并在同一方向上修复或调参，直到达标或停止。

## Usage

```text
/factor-iterate 成交额放量后短期反转，尤其在小盘股里更强
/factor-iterate max_rounds=5 data_sources=market_daily,income_q 低估值盈利改善动量
```

## Operating Rules

- 默认最多迭代 `max_rounds=10`。
- 所有 Python 命令前必须使用 `conda activate AutoQuant`。
- 因子代码写入 `alphas/exp/agent/<factor_id>/factor.py`。
- 运行产物写入 `results/agent/runs/<run_id>/`。
- 每轮必须 append 一行 JSON 到 `trace.jsonl`。
- 每轮开始前必须读取 `trace.jsonl`（如果存在），避免重复错误和重复参数。
- 代码错误和 schema 错误必须同方向修复，不得直接换新因子假设。
- 只有连续 3 轮同方向没有进展时，才允许建议换方向或停止。

## Run Directory

创建：

```text
results/agent/runs/<YYYYMMDD_HHMMSS_slug>/
  hypothesis.md
  trace.jsonl
  round_001/
    factor.py
    factor_sanitized.py
    result.json
  round_002/
    ...
```

`hypothesis.md` 保存用户原始输入、解析出的 data_sources、max_rounds 和启动时间。

## Per-round Procedure

For each round:

1. Read `trace.jsonl` if it exists.
2. Query schema before writing code:

   ```bash
   conda activate AutoQuant && python -m agents.claude_cli schema --sources <data_sources>
   ```

3. Generate or repair one factor implementation.
   - Use `from __future__ import annotations`.
   - Import `register` from `backtest.factor.registry`.
   - Import only existing transforms from `backtest.factor.transforms`.
   - Use only schema columns returned by `claude_cli schema`.
   - Register with `@register("<factor_id>", ...)`.
   - Keep identifiers in English.
4. Write the candidate code to both:
   - `results/agent/runs/<run_id>/round_<NNN>/factor.py`
   - `alphas/exp/agent/<factor_id>/factor.py`
5. Run one deterministic evaluation:

   ```bash
   conda activate AutoQuant && python -m agents.claude_cli run <factor_id> \
     --run-dir results/agent/runs/<run_id>/round_<NNN> \
     --factor-file results/agent/runs/<run_id>/round_<NNN>/factor.py
   ```

6. Read `round_<NNN>/result.json`.
7. Classify result and write one JSON object to `trace.jsonl`.
8. Decide the next action.

## Trace JSONL Schema

Append exactly one object per round:

```json
{
  "round": 1,
  "factor_id": "f_auto_20260527_001",
  "status": "pass|fail|error",
  "failure_type": "code_error|schema_error|coverage_fail|neutralization_fail|icir_fail|monotonicity_fail|config_error|backtest_fail|ridge_fail|residual_fail|execution_error|metrics_fail|null",
  "error_signature": "NameError: abs_",
  "diagnosis": "The code used a missing transform; preserve the reversal-volume idea and fix the import.",
  "fix_strategy": "Replace the invalid transform and rerun the same hypothesis.",
  "code_summary": "20-day return reversal gated by abnormal amount and small-cap rank.",
  "tried_params": {"window": 20, "horizon": 20, "top_pct": 0.1},
  "metrics": {"rankicir": 0.15, "turnover": 0.42, "simple_sharpe": 0.3},
  "same_direction": true
}
```

## Failure Classification

Use `result.json.failure_type` as the first signal, then refine from traceback and metrics:

- `code_error`: SyntaxError, NameError, TypeError, ImportError, invalid transform.
- `schema_error`: KeyError, missing column, wrong data source prefix.
- `coverage_fail`: step1 — too many missing values in factor coverage check.
- `neutralization_fail`: step2 — factor too correlated with size/industry after neutralization.
- `icir_fail`: step3 — RankICIR or IC+ below pipeline threshold.
- `monotonicity_fail`: step4 — decile returns not monotonic.
- `config_error`: step5 — strategy config error (top_k/top_pct/decay).
- `backtest_fail`: step6 or step7 — simple or detailed backtest metrics below threshold.
- `ridge_fail`: step8 — Ridge R² too high, factor is a style clone.
- `residual_fail`: step9 — residual ICIR too low, no incremental predictive power.
- `execution_error`: infrastructure or unexpected runtime error.

## Repair Policy

- `code_error` / `schema_error`:
  - Set `same_direction=true`.
  - Keep the original economic hypothesis.
  - Fix only code, import, transform, or column names.
- `coverage_fail`:
  - Set `same_direction=true`.
  - Check data source availability, widen universe, or adjust missing-value handling.
- `neutralization_fail`:
  - Set `same_direction=true`.
  - Try different neutralization variant (`barra_ind_size` → `barra_l3`) or adjust factor construction.
- `icir_fail`:
  - Set `same_direction=true` while there is still a plausible adjustment.
  - Try one change at a time: window, horizon, smoothing, normalization, or sign.
  - Do not repeat parameter combinations found in trace.
- `monotonicity_fail`:
  - Set `same_direction=true`.
  - Factor may only work at extremes; add secondary filter or change construction.
- `backtest_fail`:
  - Add smoothing/decay, lengthen window, or reduce signal churn to improve Sharpe/drawdown.
- `ridge_fail`:
  - Factor is redundant with existing Barra factors; change construction approach or target a different risk dimension.
- `residual_fail`:
  - Factor has no incremental value beyond already-admitted factors; try a different hypothesis.
- Three consecutive same-direction failures with no metric improvement:
  - Stop and report why the hypothesis appears weak, or ask the user whether to continue with a new direction.

## Pass Criteria

Treat the factor as candidate if `result.json.status == "pass"`.

On pass, the CLI automatically writes the factor to `results/agent/candidates/<factor_id>/` containing:

- `factor.py` — factor source code
- `pipeline_state.json` — full step1~step10 pass/fail status and metrics
- `result.json` — complete CLI output

On pass:

1. Append final trace record with `status="pass"`.
2. Summarize factor id, path, core formula, key metrics, and candidates directory.
3. Do not automatically admit the factor. To admit, manually run: `python -m backtest.factor.admission admit <factor_id>`

## Common Column / Transform Corrections

Use aliases returned by `claude_cli schema`:

- `buy_sm` → `mf_buy_sm_amount`
- `sell_sm` → `mf_sell_sm_amount`
- `buy_lg` → `mf_buy_lg_amount`
- `net_mf` → `mf_net_mf_amount`
- `ts_zscore` → `z_score`
- `cs_rank` → `rank`

## Output Style

During iteration, keep user updates short:

- Round number and action.
- Failure type and next fix.
- Final pass/fail summary.

Do not paste full code unless the user asks.
