# /factor-iterate

交互式因子研究入口：用户给出一个自然语言因子猜测后，由 Claude Code 直接生成因子代码、运行单轮回测、分析结果、写入 trace，并在同方向上修复或调参，直到达标或停止。

**Phase 1 增强**：失败时启动 Result Critic subagent（Agent tool）诊断 + 查 KB，替代硬编码 if/else repair policy。

## Usage

```text
/factor-iterate 成交额放量后短期反转，尤其在小盘股里更强
/factor-iterate max_rounds=5 data_sources=market_daily,income_q 低估值盈利改善动量
/factor-iterate --hypothesis agents/pdf_hypotheses/xxx/xxx_hypothesis.md
/factor-iterate --hypothesis                       # 无路径 → 弹出 hypothesis 文件列表
```

### 自然语言输入模式（无 --hypothesis）

当用户输入 `/factor-iterate` **不带 `--hypothesis`** 时，走 HG → HO → FC 完整路径：

```
用户自然语言输入
    ↓
[HG] 生成结构化 Hypothesis JSON（formula_draft + parameters + 5维自评）
    ↓
[HO] 静态评审（查重/反模式/参数建议/数据可行性/经济学逻辑）
    ↓
审阅后的 hypothesis.md → [FC] 编码 → Pipeline → [RC] 诊断
```

**父进程职责**：
1. 识别输入类型：有 `--hypothesis <path>` → 直接 Read hypothesis.md 进入 FC；无 `--hypothesis` → 启动 HG
2. 识别用户输入中是否包含**明确因子表达式**（如公式代码块、`=` 赋值、`rank(...)`/`ts_*`/列名算子链等）。若存在，设置 `formula_locked=true`，将原始表达式原样写入 hypothesis 的 `## Formula` / `formula_draft`
3. HG 输出保存到 `results/<run_id>/hypothesis.json`
4. HO 输出保存到 `results/<run_id>/hypothesis_optimized.json`
5. `formula_locked=true` 时，HO 只能做查重、风险提示、参数建议，**不得改写 Round 1 公式**；任何优化公式只能作为 Round 2+ repair 建议
6. `ho_review.recommendation == "abandon"` → 直接结束，不进入 FC
7. `ho_review.recommendation == "revise"` → 将优化后的 hypothesis 展示给用户确认
8. `ho_review.recommendation == "proceed"` → 直接进入 FC

### --hypothesis 无参数交互模式

当用户输入 `/factor-iterate --hypothesis` **不带路径**时，列出 `agents/pdf_hypotheses/` 下所有 `.md` 文件供选择：

```bash
find agents/pdf_hypotheses -name "*.md" -type f | sort
```

展示为带编号菜单，**等待用户选择**后再继续。

## Operating Rules

- 默认最多迭代 `max_rounds=10`。
- **`--hypothesis <path>` 模式**：当用户提供 hypothesis.md 路径时，先 Read 该文件提取 `## Formula`、`## Construction Logic`、`## Parameters`、`## Suggested Config`。Formula 作为 FC 编码起点，Suggested Config 作为 `config.yaml` 初始值。FC 仍需校验列名和 transforms 是否存在。此模式下不需要用户再输入自然语言假设，`hypothesis.md` 中的 `## Hypothesis` 即为假设。
- **Round 1 给定公式锁定**：如果 `/factor-iterate` 后面直接跟了明确因子表达式，或 `hypothesis.md` 中存在 `## Formula`，第一轮必须严格按给定公式复现。不得为了“更优”“更稳”“更常见”而主动改写、替换、叠加新变量、改变方向、调窗口或改组合权重。只允许做三类最小适配：① 语法落地（把伪代码翻译成可运行 pandas/AutoQuant 代码）；② 按 schema 做列名映射；③ 当原公式依赖项目缺失列/缺失 transform 时，用最接近 proxy，并在 trace 与用户摘要中明确标注。任何公式优化只能在第一轮跑完、RC 诊断后从 Round 2 开始。
- 所有 Python 命令前必须使用 `conda activate AutoQuant`。
- 因子代码写入 `alphas/exp/agent/<factor_id>/factor.py`。
- 运行产物写入 `results/<factor_id>/`（因子评估与回测）和 `results/<run_id>/`（追踪文件）。
- 每轮必须 append 一行 JSON 到 `trace.jsonl`。
- 每轮开始前必须读取 `trace.jsonl`（如果存在），避免重复错误和重复参数。
- 代码错误和 schema 错误必须同方向修复，不得直接换新因子假设。
- 失败诊断：先执行 **Pre-RC Strategy Sweep Fast Path**（强因子仅策略回测失败时直接 sweep）；不满足 fast path 时再启动 Result Critic subagent（Agent tool），由其读取 result.json + trace.jsonl + KB，输出结构化诊断 JSON。父进程根据 RC 输出决定 repair / abandon / 换方向。
- **绝对不能更改阈值（thresholds）来让因子 admit**。阈值是项目级的质量标准，降低阈值等同于自欺欺人。如果因子达不到阈值，只能改进因子或策略参数，不能改阈值。

## Knowledge Base

KB 文件位于 `agents/knowledge_base/`，跨 run 积累知识：

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
results/
  <run_id>/                                 ← 追踪文件与代码
    hypothesis.md
    trace.jsonl
    factor.py                               ← 当前因子代码（仅在 factor_change="formula" 时更新）
    config.yaml                             ← 当前策略配置
  <factor_id>/                              ← 同一因子代码共享一个目录
    factor_eval/                            ← step1-4 因子评估（同一因子共享，strategy_only 轮不重建）
    decile_backtest/                        ← 十段分层/多空测试
    <strategy>/                             ← 策略变体（step5-10）
      plots/
      simple/
      detailed/
      pipeline_report.md
      result.json
    <strategy2>/                            ← 不同参数的策略变体
      ...
      <strategy>/
  candidates/                               ← 已通过 pipeline 等待人工决策的因子
    <factor_id>/
      factor.py
      pipeline_report.md
      pipeline_state.json
      result.json
```

**目录规则**：
- `factor.py` 和 `factor_eval/` / `decile_backtest/` 仅在 `factor_change="formula"` 的轮次更新。`factor_change="params"` 的轮次**也必须重建** factor_eval / decile_backtest（因为窗口/horizon/variant 变化导致因子值完全不同）。只有 `strategy_only` 的轮次**不重建**这些目录，只新增策略变体目录。
- 策略变体目录按 `{top_k}_{rebalance}_d{decay}` 命名（如 `top100_1d_d5`），由 pipeline 自动生成。
- `result.json` 与 `pipeline_report.md` 放在同一策略变体目录下。
- `sweep` 组合目录必须是 `results/<factor_id>/sweep_runs/<strategy>/`，不得再嵌套 `<factor_id>/<strategy>`。

## Per-round Procedure

For each round:

1. **Check max_rounds**：如果当前 round > max_rounds，直接跳转到 Abandon 收尾（max_rounds 耗尽）。
2. Read `trace.jsonl` if it exists.
3. Query schema before writing code:

   ```bash
   conda activate AutoQuant && python -m agents.claude_cli schema --sources <data_sources>
   ```

4. Generate or repair one factor implementation.
   - **Round 1 + 给定公式**：如果 `formula_locked=true` 或 hypothesis 中有 `## Formula`，FC 的首要任务是“复现公式”，不是“改进公式”。代码结构、列名、helper 函数可以为运行而适配，但信号的经济含义、方向、窗口、权重、分组逻辑必须与给定公式一致；不确定处先用最保守的字面解释，并在 trace 中记录假设。
   - Use `from __future__ import annotations`.
   - Import `register` from `backtest.factor.registry`.
   - Import only existing transforms from `backtest.factor.transforms`.
   - Use only schema columns returned by `claude_cli schema`.
   - Register with `@register("<factor_id>", ...)`.
   - Keep identifiers in English.
   - **Repair 时**：如果上一轮 RC subagent 输出了 `factor_params`，优先采用其建议的参数。
   - **Round 1 时**：从生成的代码中提取关键参数（window、horizon 等）作为 `tried_params`，用于后续 trace 记录。如果无法提取，使用 `{}`。
   - **⚠️ 价格必须后复权**：`open`/`high`/`low`/`close` 在除权除息日会产生跳变。任何时序计算（pct_change、rolling mean/std、跨日期价格比较）**必须**乘以 `adj_factor`：
     ```python
     adj_close = panel["close"] * panel["adj_factor"]
     adj_open = panel["open"] * panel["adj_factor"]
     ```
     例外：`pct_chg` 和 `change` 已是调整后涨跌幅，无需复权；`total_mv`/`circ_mv` 和 turnover 类列也已调整。详见 `agents/FACTOR_CODE_GUIDE.md` §5.8。
   - **⚠️ ST 股票**：ST/*ST 股票的剔除由 `strategy.universe.exclude_st: true`（config.yaml）处理，**因子代码中不需要手动屏蔽**。
   - **⚠️ 涨跌停**：涨跌停过滤由 simulation 层在交易执行时处理，**因子代码中不需要手动屏蔽**。
   - **⚠️ 因子输出公式信号值**：去极值、中性化和最终标准化由 pipeline 统一处理，因子函数不要重复加 `cs_mad_winsorize`、行业/市值中性化、`~is_st`、`limit_up/down` 过滤。`rank()`/截面排名**允许**用于不同量纲变量的组合或公式构造（例如 `rank(price_signal) * rank(volume_signal)`），但不要仅为了重复 pipeline 的最终归一化而嵌套。因子方向可在代码中用 `*(-1)` 明确表达。
   - **⚠️ 财务数据是季度频率**：`inc_*`/`bs_*`/`cf_*` 列在每个交易日重复同一季度值，直到下季度财报发布。对财务列做时序变换（`ts_mean`/`ts_delta`/`pct_change`）**无意义**——会产生阶梯状伪影。截面比值（`inc_eps / bs_equity`）没问题；增长/斜率因子必须用 `event_driven=True` 模式。详见 `agents/FACTOR_CODE_GUIDE.md` §5.12。
   - **⚠️ 成交量单位**：`volume` 单位是**股**（非手），`amount` 单位是**元**。跨股票比较成交量用 `turnover_rate`（换手率）或 `amount`（成交额），不要用原始 `volume`。详见 `agents/FACTOR_CODE_GUIDE.md` §5.13。
5. Write the factor code + config:

   **Round 1 或 factor_change="formula"**（因子代码变化）：
   - 因子代码写入 `results/<run_id>/factor.py` **和** `alphas/exp/agent/<factor_id>/factor.py`
   - 策略配置写入 `alphas/exp/agent/<factor_id>/config.yaml`，同时 copy 到 `results/<run_id>/config.yaml`

   **factor_change="params" 或 strategy_only**（因子代码不变）：
   - **只更新** `alphas/exp/agent/<factor_id>/config.yaml`（改 decay/rebalance/top_k）
   - 因子代码**不重写**，factor_eval/ 结果可复用

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
   - **Repair 时**：如果 RC 的 `fix_level` 是 `strategy_only` 或 `both`，用 `strategy_params` 中的 `decay`/`rebalance`/`top_k` 更新 config.yaml；`fix_level=factor` 时用 `factor_params` 修改因子代码
6. Run pipeline（区分两类修复）：

   **Round 1 或 factor_change="formula"**（全量 pipeline）：
   ```bash
   conda activate AutoQuant && python -m agents.claude_cli run <factor_id> \
     --factor-file results/<run_id>/factor.py
   ```
   Pipeline 自动将产物写入 `results/<factor_id>/factor_eval/`、`results/<factor_id>/decile_backtest/` 以及默认策略变体目录下。

   **strategy_only**（仅策略变化）：
   ```bash
   conda activate AutoQuant && python -m agents.claude_cli run <factor_id> \
     --factor-file results/<run_id>/factor.py \
     --from-step 5 \
     --top-k 200 --decay 10 --rebalance 5D
   ```
   > factor_eval/ / decile_backtest/ 结果不变，不重复生成。`--from-step 5` 会复用已回填因子值与 step1~4 结果，只重建策略配置与后续回测。新的策略参数会产生一个新的策略变体目录（如 `top200_5d_d10`）。

   **factor_change="params"**（窗口/horizon/variant 等因子值变化）：仍需从 step1 重新运行，因为因子值已改变。

   **Quick 模式（默认用于 strategy-only 参数扫描）**：
   当因子公式已锁定、只需要比较策略参数时，使用 `--quick`（等价于 `--to-step 6`）跳过 detailed backtest / Ridge / residual，只跑到 simple backtest。这会快很多，且仍然生成包含 step1-6 的 `result.json` 和 `pipeline_report.md`。
   ```bash
   conda activate AutoQuant && python -m agents.claude_cli run <factor_id> \
     --factor-file results/<run_id>/factor.py \
     --top-k 100 --decay 10 --rebalance 5D \
     --quick
   ```

   **两级 sweep（universe → strategy 参数网格）**：
   当强因子只卡在策略参数提取，或 RC 明确建议尝试多个参数组合时，不要逐个 `run`，直接用 `sweep`。系统先跑 base 因子 step1~4（统一 IC/单调性），然后**按 universe 串行、策略 combo 并行**扫参数网格。使用真实 factor_id，不创建 clone 目录。
   ```bash
   conda activate AutoQuant && python -m agents.claude_cli sweep <factor_id> \
     --factor-file results/<run_id>/factor.py \
     --workers 4
   ```
   - **Universe 维度**：默认覆盖 hs300 / csi500 / csi1000 / csi2000 四大宽基指数。串行执行避免 DB 争用。
   - **策略维度**：固定 `top_pct=10%`，按因子类型自动选 decay × rebalance 网格：
     - 量价/技术因子：`decay=5,10,15 × rebalance=1D,5D`（6 combo）
     - 基本面/财务因子：`decay=5 × rebalance=1M,3M`（2 combo）
   - **默认跑 step5~7**（含 detailed backtest），相当于旧版 `--full`。全过则直接进 candidates/ 对应 universe 目录。
   - `--validate-top-n N`（默认 1）：每个 universe 保留 top N combo 的 full result。
   - `--to-step 6`：退化为 quick 模式（只到 simple backtest）。
   - `--universes hs300=000300.SH,csi500=000905.SH`：自定义 universe 集合。
   - 结果写到 `results/<factor_id>/<universe>/<strategy_tag>/`，汇总到 `results/<factor_id>/cross_universe.json`。

   **Sweep 后的标准 workflow**：
   1. 公式确定后先跑一次 `--quick` 确认 simple metrics 有信号；
   2. 直接 `sweep --workers 4` 扫全量 universe × strategy 网格；
   3. Sweep 已内置 `--validate-top-n 1`，每条 universe 的最优 combo 自动跑完 step7 detailed；
   4. 如需手动测试单个参数组合，用 `run --from-step 5 --top-k ... --decay ... --rebalance ...`，不要从 step1 重跑；
   5. 只有完整 run `pass` 才进 `candidates/`。

7. Read `result.json`（位于 `results/<factor_id>/<strategy>/result.json`）。
   - 如果用了 `sweep`，每个组合的 `result.json` 路径以 `results/<factor_id>/sweep_summary.json` 中的 `result_path` / `full_result_path` 为准；sweep 不会创建 clone factor_id。
   - `result.json.report_path` 指向 pipeline 诊断报告。
   
8. **If `result.json.status == "pass"`**：
   - Append final trace record with `status="pass"`.
   - **Update KB**（见 §Pass 收尾）。
   - End loop. Summarize factor id, path, core formula, key metrics, and candidates directory.
   - Do not automatically admit. To admit: `python -m backtest.factor.admission admit <factor_id>`

9. **Pre-RC Strategy Sweep Fast Path**：在启动 RC 前先判断是否属于“强因子但策略参数提取失败”。

   若同时满足：
   - `result.json.failure_type == "backtest_fail"`；
   - failed step 是 step6（simple backtest），且 step1~step5 passed；
   - `annual_icir`、`monotonicity` 已通过阈值，且 simple Sharpe 为正 / 接近阈值（例如 ≥ 阈值 70%）或 ICIR 极强（例如 annual_icir ≥ 3）；
   - 没有 code/schema/execution 错误；
   - 因子公式未计划修改；

   则**不要启动 RC**，直接运行两级 universe × strategy sweep：
   ```bash
   conda activate AutoQuant && python -m agents.claude_cli sweep <factor_id> \
     --factor-file results/<run_id>/factor.py \
     --workers 4
   ```
   - 默认覆盖 hs300/csi500/csi1000/csi2000 四大宽基 + 自动按因子类型选 decay/rebalance 网格（量价：5,10,15 × 1D,5D；基本面：5 × 1M,3M），不再手动传 `--top-k/--decay/--rebalance`。
   - 如果任一 universe 有 full `pass`，进入 Pass 收尾。
   - 如果全部 universe 无 pass，或 full 验证仍失败，再进入 RC，并把 `cross_universe.json` 注入 prompt。

10. **If `result.json.status != "pass"` 且不满足 Pre-RC fast path**：启动 Result Critic subagent 诊断。

   **条件注入：父进程组装 RC Prompt**

   RC 不再自己 Read `trace.jsonl` 和 KB 文件。父进程按以下步骤准备：

   a. **提取 `failure_type`**：从 `result.json.failure_type` 读取
   b. **查询 KB**（调用 `kb_query.py`）：
      ```bash
      conda activate AutoQuant && python -m agents.kb_query \
        --category <category> \
        --failure-type <failure_type> \
        --limit 3
      ```
   c. **组装 trace 摘要**：按 `failure_type` 决定注入最近 N 轮
      - `code_error` / `schema_error` / `execution_error` / `residual_fail` → 最近 1 轮
      - `coverage_fail` / `monotonicity_fail` → 最近 2 轮
      - `icir_fail` / `backtest_fail` → 最近 3 轮 + trend 分析
      - `ridge_fail` → max_existing_factor 信息 + 相关成功模式
   d. **组装完整 prompt**：
      - Role (from `.claude/prompts/shared/role.md` RC persona)
      - Scenario Description
      - Context (round, factor_id, hypothesis_text, tried_params)
      - This Round's Result (`result.json` 关键字段直接注入，不让 RC 自己读文件)
      - Trace Summary (条件注入的最近 N 轮)
      - SOTA Reference (同 category)
      - KB Query Results (`kb_query.py` 输出)
      - Task + Output Format (from `.claude/prompts/result_critic.md`)

   **RC Subagent 调用方式**：

   ```
   Agent tool:
     description: "诊断因子失败原因并给出修复建议"
     subagent_type: "general-purpose"
     prompt: |
       [父进程组装好的完整 prompt，包含上述所有 sections]
   ```

   **RC 输出扩展**：
   - 新增 `new_hypothesis` 字段：当 `same_direction=false` 时，RC 可输出具体的新方向假设文本
   - 新增 `factor_change` 字段：明确区分 "params"（只调参数）和 "formula"（改公式结构）
   - 保留 `new_anti_pattern` 字段

   **完整 Diagnosis JSON Schema** 见 `.claude/prompts/shared/output_formats.md`。

11. **Parse RC output**：尝试 JSON.parse RC 返回文本。
    - 如果解析失败（如 RC 输出了 markdown 代码块包裹）→ 尝试提取第一个 `{...}` 块重新解析
    - 如果仍失败 → 使用 fallback 诊断：`{"failure_type": "{from result.json}", "diagnosis": "RC output parse error", "fix_level": "factor", "factor_params": {}, "strategy_params": {}, "same_direction": true, "recommend_abandon": false, "new_anti_pattern": null}`
    - 追加一行到 `trace.jsonl`（将 RC 输出的字段合并进去，见 Trace JSONL Schema）

12. 根据 RC subagent 返回的诊断 JSON：

    **RC 职责边界**：RC 的核心任务是诊断**因子构造**问题（公式、窗口、算子组合、方向、数据列选择）。当失败原因是策略参数空间问题时，RC 只需指出"这是策略参数问题，建议 sweep"，**不要**让 RC 输出具体的 top_k/decay/rebalance 数值组合——那是 sweep 的工作，RC 逐个猜参数效率低且浪费 token。

    具体规则：
    - **fix_level="strategy_only"**：不要按 RC 的 `strategy_params` 逐个单点跑，直接启动 `sweep` 扫 grid。RC 的诊断只用于确认"公式方向正确，瓶颈在参数提取"；参数组合选择交给 sweep。
    - **fix_level="factor"**：按 RC 建议修改因子代码（窗口、horizon、variant、公式结构），这才是 RC 的主战场。
    - **fix_level="both"**：先改因子代码，然后启动 sweep 验证策略参数，不要手调单个参数。

    执行决策：
    - 如果 `recommend_abandon == true` 或 `same_direction == false` 或 `round >= max_rounds`：
      - **Update KB**（见 §Abandon 收尾）
      - End loop. 输出放弃报告（根因分析 + 为什么无法修复）。
    - 如果 `same_direction == true` 且 `recommend_abandon != true` 且 `round < max_rounds`：
      - **fix_level="factor" + factor_change="params"**：以 RC 的 `factor_params` 为指导修改因子代码（窗口/horizon/variant），进入下一轮。
      - **fix_level="factor" + factor_change="formula"**：按 RC 的 `fix_strategy` 中的**公式改进方向**重构因子代码（变换算子/归一化/组合增强），进入下一轮。
      - **fix_level="strategy_only"**：因子代码不变，不做单点手调；直接启动 `sweep`（两级 universe × strategy 自动网格），系统自动按因子类型选 combo 并在每条 universe 内 validate-top-n。
      - **fix_level="both"**：两个都改。factor_change 遵循与 fix_level="factor" 相同的规则（params 或 formula），FC 修改 factor.py 后启动 sweep 验证策略参数。
      - **fix_level="retry"**：什么都不改，原样重试。

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
  "fix_level": "factor",
  "factor_change": "params",
  "factor_params": {"window": 5},
  "strategy_params": {},
  "code_summary": "20-day return reversal gated by abnormal amount and small-cap rank.",
  "tried_params": {"window": 20, "horizon": 20, "top_pct": 0.1},
  "recommend_abandon": false,
  "metrics": {"annual_icir": 0.15, "simple_sharpe": 0.3, "r2": null, "max_existing_corr": null, "residual_icir": null},
  "same_direction": true,
  "new_hypothesis": null,
  "parent_round_id": null,
  "branch_id": "main",
  "fork_reason": null,
  "ts": "2026-06-02T12:00:00Z"
}
```

### 字段来源说明

| 字段 | 来源 |
|------|------|
| `category` | 从 `result.json.experiment.category` 提取（来自 `@register` 的 category 参数）。如果不存在，从用户假设推断 |
| `data_sources` | 从 `result.json.experiment.data_sources` 提取。如果不存在，使用用户指定的 `--data_sources` 参数 |
| `error_signature` | 从 `result.json.error` 提取第一行（错误类型 + 消息），截断至 120 字符 |
| `tried_params` | Round 1：从生成的因子代码提取关键参数（window, horizon 等）；后续 round：从上一轮的 `factor_params` + 代码参数合并 |

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

1. **Auto-update KB**（替代手动文件操作）：
   ```bash
   conda activate AutoQuant && python -m agents.claude_cli kb-update \
     --result results/<factor_id>/<strategy>/result.json \
     --status pass
   ```
   该命令自动更新 `hypothesis_index.jsonl`（upsert）和 `successful_patterns.json`（按 category 追加，factor_id 去重）。

2. **Auto-append trace**（若 run 时未使用 `--auto-kb-update`）：
   ```bash
   conda activate AutoQuant && python -m agents.claude_cli trace-append \
     --run-dir results/<run_id>/ \
     --result results/<factor_id>/<strategy>/result.json \
     --round <N> --category <category> --code-summary "<summary>"
   ```

3. 因子已由 CLI 自动写入 `results/candidates/<factor_id>/`（含 `factor.py`、`pipeline_state.json`、`result.json`、`pipeline_report.md`）。

4. 总结输出：factor id、路径、核心公式、关键指标、**pipeline 报告路径**、candidates 目录。提示用户 Read 报告做最终决策。

## Abandon 收尾

When loop ends with abandon（RC 建议放弃或 max_rounds 耗尽）：

1. **Auto-update KB**（含 conditional anti-pattern 更新）：
   ```bash
   conda activate AutoQuant && python -m agents.claude_cli kb-update \
     --result results/<factor_id>/<strategy>/result.json \
     --status fail \
     --rc-output results/<run_id>/rc_diagnosis.json
   ```
   该命令自动：
   - 更新 `hypothesis_index.jsonl`（status=fail，upsert best_icir）
   - 追加 `failed_attempts.jsonl`
   - 若 RC 输出了 `new_anti_pattern`，按 `signature` exact match 去重更新 `anti_patterns.json`（count += 1）

2. **Auto-append trace**：
   ```bash
   conda activate AutoQuant && python -m agents.claude_cli trace-append \
     --run-dir results/<run_id>/ \
     --result results/<factor_id>/<strategy>/result.json \
     --rc-output results/<run_id>/rc_diagnosis.json \
     --round <N> --category <category> --code-summary "<summary>"
   ```

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
