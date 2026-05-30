# /factor-iterate

交互式因子研究入口：用户给出一个自然语言因子猜测后，由 Claude Code 直接生成因子代码、运行单轮回测、分析结果、写入 trace，并在同方向上修复或调参，直到达标或停止。

**Phase 1 增强**：失败时启动 Result Critic subagent（Agent tool）诊断 + 查 KB，替代硬编码 if/else repair policy。

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
- 失败诊断：**启动 Result Critic subagent（Agent tool）**，由其读取 result.json + trace.jsonl + KB，输出结构化诊断 JSON。父进程根据 RC 输出决定 repair / abandon / 换方向。

## Knowledge Base

KB 文件位于 `results/agent/knowledge_base/`，跨 run 积累知识：

| 文件 | 用途 |
|------|------|
| `anti_patterns.json` | 失败模式 → 修复建议，按 failure_type 分组 |
| `successful_patterns.json` | 成功模式 → SOTA 基准，按 category 分组 |
| `failed_attempts.jsonl` | 失败实验索引，仅记录失败，用于学习错误建模方案（append-only） |

**每次 `/factor-iterate` 开始时**，读取 KB 做 framing：
- `anti_patterns.json` → 此 category 的已知坑，引导初始代码避免重蹈
- `successful_patterns.json` → 同 category 成功模式，参考公式模板

**每次迭代结束时**，更新 KB（见 §Pass/Abandon 收尾）。

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

1. **Check max_rounds**：如果当前 round > max_rounds，直接跳转到 Abandon 收尾（max_rounds 耗尽）。
2. Read `trace.jsonl` if it exists.
3. Query schema before writing code:

   ```bash
   conda activate AutoQuant && python -m agents.claude_cli schema --sources <data_sources>
   ```

4. Generate or repair one factor implementation.
   - Use `from __future__ import annotations`.
   - Import `register` from `backtest.factor.registry`.
   - Import only existing transforms from `backtest.factor.transforms`.
   - Use only schema columns returned by `claude_cli schema`.
   - Register with `@register("<factor_id>", ...)`.
   - Keep identifiers in English.
   - **Repair 时**：如果上一轮 RC subagent 输出了 `repair_params`，优先采用其建议的参数。
   - **Round 1 时**：从生成的代码中提取关键参数（window、horizon 等）作为 `tried_params`，用于后续 trace 记录。如果无法提取，使用 `{}`。
   - **⚠️ 价格必须后复权**：`open`/`high`/`low`/`close` 在除权除息日会产生跳变。任何时序计算（pct_change、rolling mean/std、跨日期价格比较）**必须**乘以 `adj_factor`：
     ```python
     adj_close = panel["close"] * panel["adj_factor"]
     adj_open = panel["open"] * panel["adj_factor"]
     ```
     例外：`pct_chg` 和 `change` 已是调整后涨跌幅，无需复权；`total_mv`/`circ_mv` 和 turnover 类列也已调整。详见 `FACTOR_CODE_GUIDE.md` §5.8。
   - **⚠️ ST 股票**：ST/*ST 股票涨跌幅限制 ±5%（正常 ±10%），交易行为扭曲。必须用 `raw_signal.where(~panel["is_st"], np.nan)` 屏蔽，避免污染截面排名。详见 GUIDE §5.9。
   - **⚠️ 涨跌停**：涨停跌停日成交量接近 0、收盘价为人工价。成交量/反转类因子必须屏蔽 `(close == limit_up) | (close == limit_down)` 的数据。详见 GUIDE §5.11。
   - **⚠️ 财务数据是季度频率**：`inc_*`/`bs_*`/`cf_*` 列在每个交易日重复同一季度值，直到下季度财报发布。对财务列做时序变换（`ts_mean`/`ts_delta`/`pct_change`）**无意义**——会产生阶梯状伪影。截面比值（`inc_eps / bs_equity`）没问题；增长/斜率因子必须用 `event_driven=True` 模式。详见 GUIDE §5.12。
   - **⚠️ 成交量单位**：`volume` 单位是**股**（非手），`amount` 单位是**元**。跨股票比较成交量用 `turnover_rate`（换手率）或 `amount`（成交额），不要用原始 `volume`。详见 GUIDE §5.13。
5. Write the candidate code + per-factor config to both:
   - **因子代码**：`results/agent/runs/<run_id>/round_<NNN>/factor.py` 和 `alphas/exp/agent/<factor_id>/factor.py`
   - **回测策略配置**：`alphas/exp/agent/<factor_id>/config.yaml`（**必须生成！**）——Pipeline 通过 `PipelineConfig.from_factor_config(factor_id)` 自动发现此文件，未指定的字段使用硬编码默认值。FC 根据因子特征选择参数：

   ```yaml
   # alphas/exp/agent/<factor_id>/config.yaml
   pipeline:
     default_decay: 5          # 信号半衰期：noisy 因子 → 10~15，sharp 因子 → 3~5
     default_rebalance: "1D"   # 调仓频率：1D/5D/1W/2W/1M/EOM
     default_top_k: 100        # 持仓数：分散型 → 200~300，集中型 → 50~100
     ret_type: "open"          # 成交价：open=T+1开盘，close=T日收盘

   strategy:
     universe:
       exclude_st: true
       exclude_new_ipo_days: 252
       include_cyb: true       # 创业板
       include_kcb: false      # 科创板（默认关，波动大）
       include_bse: false      # 北交所（默认关）
       min_market_cap: 500000000     # 最小流通市值（5亿）
       min_avg_amount: 10000000      # 最小日均成交额（1000万）

   simulation:
     initial_cash: 100000000
     commission_rate: 0.0003
     stamp_duty_rate: 0.001
     allow_short: false
   ```

   - **decay 选择**：反转/换手类因子信号衰减快 → decay=3~5；趋势/质量类因子衰减慢 → decay=10~15
   - **rebalance 选择**：日频信号 → 1D；周频信号 → 5D 或 1W；月频信号 → 1M
   - **top_k 选择**：ICIR > 3 的强因子 → 50~100 集中持有；ICIR 1~3 → 100~200；ICIR < 1 → 200~300 分散
   - **Repair 时**：如果 RC 输出了 `repair_params` 中的 `decay`/`rebalance`/`top_k`，更新 config.yaml 对应值
6. Run one deterministic evaluation:

   ```bash
   conda activate AutoQuant && python -m agents.claude_cli run <factor_id> \
     --run-dir results/agent/runs/<run_id>/round_<NNN> \
     --factor-file results/agent/runs/<run_id>/round_<NNN>/factor.py
   ```

7. Read `round_<NNN>/result.json`。
   - `result.json.report_path` 指向 pipeline 诊断报告（每轮都会生成，含全部 10 步详情 + 图表）。
   - 报告同时复制到 `round_<NNN>/pipeline_report.md`，可直接 Read 查看 IC 衰减图、分层回测净值曲线等。

8. **If `result.json.status == "pass"`**：
   - Append final trace record with `status="pass"`.
   - **Update KB**（见 §Pass 收尾）。
   - End loop. Summarize factor id, path, core formula, key metrics, and candidates directory.
   - Do not automatically admit. To admit: `python -m backtest.factor.admission admit <factor_id>`

9. **If `result.json.status != "pass"`**：启动 Result Critic subagent 诊断。

   **RC Subagent 调用方式**：通过 `Agent` 工具，一次性 subagent：

   ```
   Agent tool:
     description: "诊断因子失败原因并给出修复建议"
     subagent_type: "general-purpose"
     prompt: |
       你是 Result Critic，负责诊断量化因子 pipeline 失败的原因并给出修复建议。

       ## Context
       - 原始假设: {用户输入的自然语言假设}
       - 本轮 round: {N} / {max_rounds}
       - 当前 factor_id: {factor_id}
       - 本轮参数: {tried_params}

       ## 输入文件（必须全部 Read）
       1. Read {run_dir}/round_{NNN}/result.json  — 本轮完整结果
       2. Read {run_dir}/trace.jsonl              — 本 run 完整历史
       3. Read results/agent/knowledge_base/anti_patterns.json
       4. Read results/agent/knowledge_base/successful_patterns.json

       ## result.json 关键字段说明
       result.json 结构（由 claude_cli.py 输出）：
       - `status`: "pass" | "fail" | "error"
       - `failure_type`: 失败类型字符串
       - `error`: 错误消息（如有）
       - `traceback`: 完整 traceback（如有）
       - `metrics`: FLAT dict，key 为 {annual_icir, pos_ratio, turnover, max_corr, max_existing_factor, simple_sharpe, simple_mdd, simple_annual_return, detailed_sharpe, detailed_annual_return, cost_drag, monotonicity, ridge_tier, ridge_r2, residual_annual_icir}
         - `max_corr` = step2 的 max_existing_corr（已存在的最高相关因子）
         - `residual_annual_icir` = step9 残差 ICIR
         - `ridge_tier` = step8 Ridge R² 分层（low/medium/high/extreme）
       - `experiment.step_results.{stepN}.metrics.*`: 各 step 的详细指标
         - `experiment.step_results.step2.metrics.max_existing_factor` = 相关性最高的已有因子 ID（用于判断 Barra L1 vs user alpha）
         - `experiment.step_results.step8.metrics.r2` = Ridge R²
         - `experiment.simple_bt_metrics.sharpe` = 简单回测 Sharpe
       - `experiment.category`: 因子 category（来自 @register 装饰器）
       - `experiment.data_sources`: 数据源列表（可能不存在，此时使用用户指定的 data_sources）

       ## 任务
       1. 从 result.json 提取：status, failure_type, 关键指标：
          - `annual_icir` ← `result.json.metrics.annual_icir`
          - `r2` ← `result.json.experiment.step_results.step8.metrics.r2`（注意：不在 flat metrics 中）
          - `max_existing_corr` ← `result.json.metrics.max_corr`（来自 step2）
          - `max_existing_factor` ← `result.json.experiment.step_results.step2.metrics.max_existing_factor`
          - `residual_icir` ← `result.json.metrics.residual_annual_icir`
          - 如果 step2/step8/step9 未执行（因子在更早 step 失败），对应指标为 null
       2. 从 trace.jsonl 提取：之前几轮的 failure_type 序列, 已尝试的参数组合, **连续同方向轮数和指标趋势**（特别关注：是否已连续 ≥3 轮指标无改善？）
       3. **查反模式库**：anti_patterns.json 中，匹配当前 failure_type + category 的模式？
          - 在 anti_patterns.json[failure_type] 数组中搜索
          - 匹配条件：pattern 描述是否相关、category 是否匹配
          - 如果找到 → 输出其 fix 建议
       4. **查成功模式库**：successful_patterns.json 中，同 category 的 SOTA 指标？
          - 找到同 category 的最高 annual_icir、best_sharpe
          - 用作"多好才算好"的基准
       5. 综合判断，输出诊断 JSON

       ## OutputFormat（必须严格按以下 JSON 输出，不要额外文字，不要 markdown 代码块）
       {
         "failure_type": "用 result.json 中的 failure_type",
         "diagnosis": "根因分析，一段话，说清楚为什么失败",
         "fix_strategy": "具体修复建议",
         "fix_level": "factor",
         "factor_params": {},
         "strategy_params": {},
         "same_direction": true,
         "recommend_abandon": false,
         "new_anti_pattern": null
       }

       `fix_level` 决定下一轮 FC 做什么：
       - `"factor"` → 需要改因子代码（窗口/horizon/variant/公式）。`factor_params` 包含要改的参数。
       - `"strategy_only"` → **因子代码不变**，只改 `alphas/exp/agent/<factor_id>/config.yaml`。`strategy_params` 包含新的 decay/rebalance/top_k。
       - `"both"` → 两个都要改。

       `factor_params` 示例：{"window": 10, "horizon": 5, "variant": "barra_ind_size"}。不需要时用 {}。
       `strategy_params` 示例：{"decay": 15, "rebalance": "5D", "top_k": 200}。不需要时用 {}。合法值：decay 1~30，rebalance "1D"/"5D"/"1W"/"2W"/"1M"/"EOM"，top_k 50~500，variant "none"/"barra_ind_size"/"barra_l3"。
       `new_anti_pattern` 示例：{"failure_type": "icir_fail", "pattern": "窗口过长导致反转因子失效", "category": "volume_reversal", "signature": "volume_window > 20", "fix": "缩窗到 5-10 天"}。**必须严格使用这四个字段名**：`pattern`（非 pattern_name）、`fix`（非 description）、`category`、`signature`。不需要新增反模式时用 null。

       ## Decision Rules
       - code_error / schema_error → fix_level="factor"，只修代码/列名，不换假设。factor_params 定位到具体错误行。
       - coverage_fail → fix_level="factor"。改数据源或放宽条件。
       - neutralization_fail → fix_level="factor"。换 variant 或调整构造以减少 size/industry 相关性。
       - icir_fail → fix_level="factor"。先查反模式：有匹配→采用其 fix；无匹配→改窗口/horizon（选一个，不要同时改两个）。如果同 category SOTA icir > 1.5 而我方 < 0.5 → 差距大，不急着 abandon。
       - monotonicity_fail → fix_level="factor"。加二次过滤或改构造方式（如只在极端分位取信号）。
       - config_error → fix_level="strategy_only"。修正 config.yaml 中的 top_k/top_pct/decay/rebalance。
       - **backtest_fail → 按差距分级**：
         - Sharpe ≥ 阈值的 70%（如阈值 0.8 → Sharpe ≥ 0.56）且 ICIR 达标 → fix_level="strategy_only"。因子 alpha 没问题，只是组合构建粗糙。strategy_params 调整 decay/rebalance/top_k，**不动因子代码**。
         - Sharpe < 阈值的 70% 或 ICIR 也不达标 → fix_level="factor"。策略调参救不了，需要重构因子（换窗口/horizon/variant/公式）。factor_params 给出新参数。
         - 参考：round 1 的 vol_reversal（ICIR 4.37 但 Sharpe 0.457 vs 0.8=57%）→ strategy_only 边界，可先试策略调参。
       - ridge_fail → fix_level="factor"。查 max_existing_corr + max_existing_factor：
         - >0.85 且对手是用户 alpha → recommend_abandon=true（近似重复）
         - >0.85 且对手是 Barra L1 → 换构造方式
         - 无法判断对手类型 → 给一次 retry，换构造方式
       - residual_fail → recommend_abandon=true（无增量信息）。
       - execution_error → fix_level="strategy_only"（基础设施问题）。DuckDB 锁/超时→重试；OOM→减数据量；其他→报告用户。
       - metrics_fail → 查看具体哪些指标不达标，参考最近反模式。
       - 连续 3 轮同 direction 无改善（annual_icir 或 simple_sharpe 未提升）→ recommend_abandon=true。
       - 只在确实发现新的通用性失败模式时才填充 new_anti_pattern，否则填 null。

       返回纯 JSON，不要 markdown 代码块包裹。
   ```

10. **Parse RC output**：尝试 JSON.parse RC 返回文本。
    - 如果解析失败（如 RC 输出了 markdown 代码块包裹）→ 尝试提取第一个 `{...}` 块重新解析
    - 如果仍失败 → 使用 fallback 诊断：`{"failure_type": "{from result.json}", "diagnosis": "RC output parse error", "fix_strategy": "Retry with same code", "same_direction": true, "repair_params": {}, "recommend_abandon": false, "new_anti_pattern": null}`
    - 追加一行到 `trace.jsonl`（将 RC 输出的字段合并进去，见 Trace JSONL Schema）

11. 根据 RC subagent 返回的诊断 JSON：
    - 如果 `recommend_abandon == true` 或 `same_direction == false` 或 `round >= max_rounds`：
      - **Update KB**（见 §Abandon 收尾）
      - End loop. 输出放弃报告（根因分析 + 为什么无法修复）。
    - 如果 `same_direction == true` 且 `recommend_abandon != true` 且 `round < max_rounds`：
      - 以 RC 的 `repair_params` 为指导进入下一轮。

## Trace JSONL Schema

Append exactly one object per round:

```json
{
  "round": 1,
  "factor_id": "f_auto_20260527_001",
  "category": "volume_reversal",
  "data_sources": ["market_daily"],
  "status": "pass|fail|error",
  "failure_type": "code_error|schema_error|coverage_fail|neutralization_fail|icir_fail|monotonicity_fail|config_error|backtest_fail|ridge_fail|residual_fail|execution_error|metrics_fail|null",
  "error_signature": "NameError: abs_",
  "diagnosis": "根因分析（来自 RC subagent 输出）",
  "fix_strategy": "具体修复建议（来自 RC subagent 输出）",
  "code_summary": "20-day return reversal gated by abnormal amount and small-cap rank.",
  "tried_params": {"window": 20, "horizon": 20, "top_pct": 0.1},
  "repair_params": {"window": 5},
  "recommend_abandon": false,
  "metrics": {"annual_icir": 0.15, "simple_sharpe": 0.3, "r2": null, "max_existing_corr": null, "residual_icir": null},
  "same_direction": true
}
```

### 字段来源说明

| 字段 | 来源 |
|------|------|
| `category` | 从 `result.json.experiment.category` 提取（来自 `@register` 的 category 参数）。如果不存在，从用户假设推断 |
| `data_sources` | 从 `result.json.experiment.data_sources` 提取。如果不存在，使用用户指定的 `--data_sources` 参数 |
| `error_signature` | 从 `result.json.error` 提取第一行（错误类型 + 消息），截断至 120 字符 |
| `tried_params` | Round 1：从生成的因子代码提取关键参数（window, horizon 等）；后续 round：从上一轮的 `repair_params` + 代码参数合并 |

### metrics 提取路径

`metrics` 应从 result.json 中提取关键指标：

| 指标 | 实际路径 | 说明 |
|------|---------|------|
| `annual_icir` | `result.json.metrics.annual_icir` | flat dict，直接取 |
| `simple_sharpe` | `result.json.metrics.simple_sharpe` | flat dict，优先用这个 |
| `r2` | `result.json.experiment.step_results.step8.metrics.r2` | **不在 flat metrics 中**，需通过 experiment 取 |
| `max_existing_corr` | `result.json.metrics.max_corr` | flat dict，来自 step2（非 step8） |
| `residual_icir` | `result.json.metrics.residual_annual_icir` | flat dict，key 名带 annual 前缀 |

如果某 step 未执行（因子在更早 step 失败），对应指标填 `null`。

## Pass 收尾

When loop ends with pass:

1. **Update `successful_patterns.json`**：
   - 读取当前文件
   - 在 `category` key 下追加一条记录：
   ```json
   {
     "factor_id": "{factor_id}",
     "formula_pattern": "一句话描述公式结构",
     "key_metrics": {"annual_icir": X.XX, "simple_sharpe": X.XX},
     "why_it_works": "一句话解释经济学逻辑",
     "admission_date": "{today}"
   }
   ```
   - 如果 category key 不存在则新建

2. **Append `failed_attempts.jsonl`**：
   ```json
   {"factor_id": "{factor_id}", "run_id": "{run_id}", "category": "{category}", "data_sources": [...], "status": "pass", "best_icir": X.XX, "best_sharpe": X.XX, "code_summary": "公式+构造简述", "ts": "{ISO timestamp}"}
   ```
   `code_summary` 来自 trace 最后一轮，**必须记录**——即使 pass 了，保留公式便于后续发现近似重复时快速识别。

3. 因子已由 CLI 自动写入 `results/agent/candidates/<factor_id>/`（含 `factor.py`、`pipeline_state.json`、`result.json`、`pipeline_report.md`）。

4. 总结输出：factor id、路径、核心公式、关键指标、**pipeline 报告路径**、candidates 目录。提示用户 Read 报告做最终决策。

## Abandon 收尾

When loop ends with abandon（RC 建议放弃或 max_rounds 耗尽）：

1. **Update `anti_patterns.json`**（如果最后一轮 RC 输出了 `new_anti_pattern` 非 null）：
   - 读取当前文件
   - **字段转换**：RC 输出的 new_anti_pattern 有 5 字段（`failure_type, pattern, category, signature, fix`）。追加到 KB 时需要补充 `count: 1` 和 `last_seen: "{today}"`，并**去掉 `failure_type` 字段**（它已经是顶层 key）
   - **去重判断**：在 `anti_patterns.json[failure_type]` 数组中搜索，匹配条件为 **`signature` 完全相同**（exact string match）。如果匹配到已有条目 → 该条目的 `count += 1`，更新 `last_seen`；否则 append 新条目
   - 如果 `failure_type` key 在 anti_patterns.json 中不存在 → 新建该 key 并初始化为包含此条目的数组

2. **Append `failed_attempts.jsonl`**（**仅记录失败**，用于学习错误建模方案）：
   ```json
   {"factor_id": "{factor_id}", "run_id": "{run_id}", "category": "{category}", "data_sources": [...], "status": "fail", "best_icir": X.XX, "best_sharpe": X.XX, "failure_type": "...", "code_summary": "公式+构造简述", "why_failed": "根因一句话", "ts": "{ISO timestamp}"}
   ```
   - `code_summary`：来自 trace 最后一轮，失败后因子代码被清理，这是唯一保留的公式记录
   - `why_failed`：从 RC 最后一轮 diagnosis 中提炼一句话根因（如「ICIR 优秀但波动过大导致 Sharpe 不达标」「barra_l3 切换暴露出行业聚类 0.44 超标」）
   - `failure_type`：最终失败步骤（backtest_fail / icir_fail / ridge_fail 等）
   - **不记录 `rounds`**：轮数多少不重要，重要的是为什么失败

3. 输出放弃报告：
   - 原始假设
   - 尝试的总轮数
   - 各轮 failure_type 序列
   - 最终放弃原因（来自 RC 诊断）
   - 如果 KB 中有相关成功模式，提示「同类因子 SOTA: ICIR=X.XX，建议换构造方式」

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
- RC 诊断摘要（failure_type + diagnosis 一句话 + 决策）。
- Final pass/fail summary.

Do not paste full code unless the user asks.
