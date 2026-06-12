# Factor Iterate Workflow

本 workflow 用于在 AutoQuant 中执行单个因子方向的交互式研究循环。它是独立 skill：负责 hypothesis 或自然语言想法到 factor code、pipeline、trace、RC 修复和 KB 更新，不负责 PDF 提取。

## 1. 输入解析

支持三类入口：

- 自然语言想法：直接生成结构化 hypothesis，再进入编码与回测。
- 明确公式：第一轮必须锁定公式，只做语法落地、列名映射和必要 proxy。
- PDF hypothesis 选择：当用户说“刚才第 1 个”“最新 PDF 结果”“从 PDF hypothesis 继续”等，先发现候选并展示菜单。

用户主动提供 `agents/pdf_hypotheses/...` 路径时可以直接读取；否则不要要求用户 hardcode 路径。

## 2. PDF Hypothesis 候选发现

当需要从 PDF 结果继续时：

1. 遍历 `agents/pdf_hypotheses/` 下的批次目录，按目录修改时间倒序排序。
2. 如果只有一个批次，直接展示该批次内的 hypothesis 菜单。
3. 如果有多个批次，先展示批次菜单，让用户选择。
4. 优先读取批次内 `manifest.json` 的 `generated_hypotheses`，用于展示因子名、优先级、Sharpe 和 HO 建议。
5. 如果 `manifest.json` 缺失或不完整，fallback 到 `find <batch> -name "*_hypothesis.md" -type f | sort`。
6. 用户选择编号后，由 skill 读取对应文件继续。

批次菜单：

```text
可继续的 PDF hypothesis 批次：
1. 20260612_103000_volume_report | 3 个 hypothesis | 最新
2. 20260610_221500_valuation_report | 5 个 hypothesis
```

文件菜单：

```text
请选择要迭代的 hypothesis：
1. 01_volume_reversal_hypothesis.md | 高优 | Sharpe 0.95 | HO proceed
2. 02_liquidity_discount_hypothesis.md | 中优 | Sharpe 0.62 | HO revise
```

## 3. 运行前检查

- 读取 `AGENTS.md`、`agents/AGENTS.md`、`agents/FACTOR_CODE_GUIDE.md` 和相关 `DESIGN.md`。
- 所有 Python 命令前使用 `conda activate AutoQuant`。
- 查询 schema：

```bash
conda activate AutoQuant && python -m agents.codex_cli schema --sources <data_sources>
```

- 读取 KB framing：

```bash
conda activate AutoQuant && python -m agents.kb_query --category <category> --limit 3
```

## 4. 因子代码规则

- 代码只写入 `alphas/exp/agent/<factor_id>/factor.py`。
- 配置写入 `alphas/exp/agent/<factor_id>/config.yaml`。
- 使用 `from __future__ import annotations`。
- 使用 `@register("<factor_id>", ...)`。
- 只使用 schema 返回的列和 `backtest.factor.transforms` 中真实存在的 transform。
- 价格时序计算必须使用后复权价格，例如 `close * adj_factor`。
- ST、新股、涨跌停、去极值、中性化和最终标准化交给 pipeline/strategy/simulation，不在因子代码里重复处理。
- 财务字段是季度频率；不要对财务列做普通日频 rolling 伪时序。
- 成交量单位是股；跨股票比较优先使用 `amount`、`turnover_rate` 或 `turnover_rate_free`。

## 5. 每轮执行

默认最多 10 轮，除非用户指定。

### 自主迭代约束

因子迭代是闭环执行任务，不是单轮报告任务。除非用户明确说“只分析”“先别改/别跑”
或“暂停”，否则每次失败后都必须自己进入下一轮，直到满足终止条件。

允许向用户提问的情况只有：

- PDF/hypothesis 选择存在多个候选且无法从上下文唯一确定。
- 下一步需要破坏性清理、admit、提交代码、长时间外部数据任务或付费/权限变更。
- 已连续尝试到最大轮数，且没有新的、有根据的修复方向。
- 运行环境阻塞，例如缺数据、缺权限、命令连续失败且无法本地修复。

不允许停下提问的情况：

- step1-step4 失败但 metrics 已指出缺口；应直接改公式或数据 proxy。
- step5-step7 失败但因子统计强；应直接做 strategy 参数验证或 sweep。
- sweep 没跑完或 top candidates 没验证；应继续等待/验证，而不是询问用户。
- 只差一个门槛指标；应先尝试有逻辑依据的公式/参数邻域。
- 已经想到 1 个以上合理 next experiment；应选最保守、信息量最大的一个执行。

每轮结束要写明“下一步为什么这样选”，然后直接执行；只有达到终止条件才总结并交还用户。

每轮：

1. 读取已有 `results/<run_id>/trace.jsonl`，避免重复尝试。
2. 根据当前 hypothesis 或 RC 修复建议生成/更新 factor code 和 config。
3. 先跑因子评估 step1-step4：

```bash
conda activate AutoQuant && python -m agents.codex_cli run <factor_id> \
  --factor-file alphas/exp/agent/<factor_id>/factor.py \
  --to-step 4 --keep-work-db
```

默认不要为这条命令传 `--run-dir`；让执行层使用 canonical
`results/<factor_id>/` 布局。`--run-dir` 只用于独立 round/trace 目录，
例如 `results/runs/<factor_id>_round01/`，不要指向 `results/<factor_id>`，
否则旧版实现会产生 `results/<factor_id>/<factor_id>/` 嵌套目录。

4. 如果 step1-step4 通过，再根据因子类型运行策略参数测试或 sweep：

```bash
conda activate AutoQuant && python -m agents.codex_cli sweep <factor_id> \
  --factor-file alphas/exp/agent/<factor_id>/factor.py \
  --workers 4
```

5. 读取 `result.json` 或 `cross_universe.json`。
6. 追加 trace；如果失败，生成 RC 诊断并决定 repair、strategy_only、params change 或 abandon。
7. 只要未达到终止条件，立即执行所选下一步，不要以“是否继续”结束当前 turn。

强因子但策略参数失败时，优先使用 sweep；不要手动逐个组合重复跑 step1-step4。

### Pre-RC Strategy Sweep Fast Path

当 step1-step4 已经通过、但 step5-step7 的策略参数或组合表现不足时，优先运行 sweep 并校验最优组合：

```bash
conda activate AutoQuant && python -m agents.codex_cli sweep <factor_id> \
  --factor-file alphas/exp/agent/<factor_id>/factor.py \
  --validate-top-n 3
```

如果 sweep 仍在进行或尚未验证 top candidates，不要启动 RC。Sweep 只复用原始因子代码和 pipeline state，不会创建 `alphas/exp/agent/<factor_id>_sw_*` 这类 clone 因子目录。

如果 sweep 被中断或只产生部分 universe/strategy 目录，本轮必须：

1. 检查是否仍有后台进程。
2. 清理半截 sweep artifacts，保留中断前已经存在的 canonical result。
3. 记录未完成状态。
4. 下一次继续时重跑完整 sweep，而不是依据半截 universe 下结论。

## 6. Trace 与 KB

Trace 写入独立 round 目录的 `trace.jsonl`，推荐路径为
`results/runs/<factor_id>_roundXX/trace.jsonl`。不要把 trace 目录与
canonical factor result dir `results/<factor_id>/` 混用；`results/<factor_id>/`
只放 pipeline/sweep artifacts，例如 `result.json`、`pipeline_state.json`、
`cross_universe.json` 和各 universe/strategy 子目录。每条 trace record
包含 round、status、failure_type、关键 metrics、code_summary、tried_params、
RC 诊断和 fix strategy。

Pass 时：

```bash
conda activate AutoQuant && python -m agents.codex_cli kb-update \
  --result <result.json> --status pass
```

Fail/abandon 时：

```bash
conda activate AutoQuant && python -m agents.codex_cli kb-update \
  --result <result.json> --status fail \
  --rc-output results/<run_id>/rc_diagnosis.json
```

## 7. Pass 收尾

- 总结 factor id、代码路径、核心公式、关键指标、报告路径和 candidates 目录。
- 不自动 admit。
- 告诉用户可人工审阅后执行 admission。

## 8. Abandon 收尾

- 总结原始 hypothesis、尝试轮数、failure_type 序列和最终放弃原因。
- 确认 KB 已记录失败或指出缺口。
- 如用户要清理该因子，引导其使用独立的 `reject-factor` skill；不要在本 skill 中直接做 reject 清理。
