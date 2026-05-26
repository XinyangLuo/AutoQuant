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
  "failure_type": "code_error|schema_error|weak_signal|high_turnover|high_corr|weak_backtest|execution_error|metrics_fail|null",
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
- `weak_signal`: RankICIR or IC+ below threshold.
- `high_turnover`: turnover above threshold.
- `high_corr`: max correlation above threshold.
- `weak_backtest`: factor eval is acceptable but Sharpe/drawdown/Calmar fails.
- `execution_error`: infrastructure or unexpected runtime error.

## Repair Policy

- `code_error` / `schema_error`:
  - Set `same_direction=true`.
  - Keep the original economic hypothesis.
  - Fix only code, import, transform, or column names.
- `weak_signal`:
  - Set `same_direction=true` while there is still a plausible adjustment.
  - Try one change at a time: window, horizon, smoothing, normalization, or sign.
  - Do not repeat parameter combinations found in trace.
- `high_turnover`:
  - Add smoothing/decay, lengthen window, or reduce signal churn.
- `high_corr`:
  - Preserve the idea but change construction, conditioning, or neutralization.
- Three consecutive same-direction failures with no metric improvement:
  - Stop and report why the hypothesis appears weak, or ask the user whether to continue with a new direction.

## Pass Criteria

Treat the factor as candidate if `result.json.status == "pass"`.

On pass:

1. Append final trace record with `status="pass"`.
2. Summarize factor id, path, core formula, key metrics, and run directory.
3. Do not automatically admit the factor unless the user explicitly asks.

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
