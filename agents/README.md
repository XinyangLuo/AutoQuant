# AutoQuant Agent 投研系统使用手册

> 版本: 2026-06-12
> 定位: Codex skill 驱动的 A 股因子研究系统。Codex 负责交互、决策、代码生成和结果分析；Python 侧保留 schema、pipeline、trace、KB 等执行工具。

## 一分钟快速开始

### 自然语言因子想法

对 Codex 说：

```text
迭代这个因子想法：成交额放量后短期反转，尤其在小盘股里更强。
```

`factor-iterate` skill 会生成结构化假设、写因子代码、运行 pipeline、分析失败并同方向修复。

### 研报 PDF 提取

对 Codex 说：

```text
分析 research_papers/华泰多因子系列之四.pdf，提取可迭代的单因子。
```

如果你没有给 PDF 路径，`pdf-hypothesis` skill 会遍历 `research_papers/` 并展示编号菜单。提取完成后，它会展示生成的 hypothesis 菜单；你之后只需要说“迭代第 1 个”或“用刚才第 2 个继续”。

### 清理失败因子

对 Codex 说：

```text
拒绝一个失败因子。
```

`reject-factor` skill 会从 admission status、`alphas/exp/agent/` 和 `results/` 发现候选，展示编号菜单，并在任何删除、移动或 DB reject 前请求确认。

## 三个独立 Skill

### `pdf-hypothesis`

职责：

- 从研报 PDF 提取文本。
- 穷举报告中的单因子。
- 做 schema 可用性、KB 查重、反模式和静态风险评审。
- 生成 PDF hypothesis 批次目录。

边界：

- 不运行回测。
- 不启动因子迭代。
- 不要求用户复制 `agents/pdf_hypotheses/...` 路径。

### `factor-iterate`

职责：

- 从自然语言想法、明确公式或已生成 hypothesis 开始。
- 生成 `alphas/exp/agent/<factor_id>/factor.py` 和 `config.yaml`。
- 调用 `agents.codex_cli run` / `sweep` 执行 pipeline。
- 维护 trace，并在 pass/fail 时更新 KB。

边界：

- 可以读取 `pdf-hypothesis` 产物，但不做 PDF 提取。
- 成功后只进入 `results/candidates/`，不自动 admit。
- 不降低 pipeline 阈值。

### `reject-factor`

职责：

- 正式 reject 失败因子。
- 清理 work DB / registry 状态。
- 删除或保留生成代码。
- 将 results 归档到 `results/rejected/`。
- 检查 KB 是否记录失败经验。

边界：

- 不创建 hypothesis。
- 不做因子迭代。
- destructive 操作前必须确认。

## Skill 间协作

三个 skill 通过菜单选择衔接，而不是要求用户 hardcode 路径：

```text
pdf-hypothesis
  -> 生成 agents/pdf_hypotheses/<batch>/
  -> 展示 hypothesis 编号菜单

factor-iterate
  -> 用户说“用刚才第 1 个继续”
  -> 遍历最新 batch / manifest.json
  -> 读取选中的 hypothesis

reject-factor
  -> 用户说“拒绝失败因子”
  -> 遍历 admission、代码目录、结果目录
  -> 展示 factor 编号菜单
```

如果有多个 PDF hypothesis 批次，`factor-iterate` 先让用户选批次，再选具体 hypothesis。`manifest.json` 存在时优先用它展示因子名、优先级、Sharpe 和 HO 评审；缺失时 fallback 到 `*_hypothesis.md` 文件列表。

## PDF Hypotheses 目录

`agents/pdf_hypotheses/` 是 gitignored 运行时目录。新生成方式固定为每次 PDF 分析一个批次目录：

```text
agents/pdf_hypotheses/
└── <YYYYMMDD_HHMMSS_slug>/
    ├── manifest.json
    ├── extracted.md
    ├── 01_<factor_slug>_hypothesis.md
    ├── 02_<factor_slug>_hypothesis.md
    └── ...
```

规则：

- 顶层不再生成 `.json` 或 `.md`。
- `manifest.json` 必须生成。
- `extracted.md` 仅在需要持久化 PDF 提取文本时生成。
- hypothesis 文件使用编号前缀，方便用户直接选择。

## Python 执行层命令

这些命令通常由 Codex skill 自动调用；手动调试时可直接使用。

```bash
conda activate AutoQuant

# 查询数据 schema
python -m agents.codex_cli schema --sources market_daily
python -m agents.codex_cli schema --sources market_daily,income_q

# 单轮 pipeline
python -m agents.codex_cli run f_auto_001 \
    --factor-file alphas/exp/agent/f_auto_001/factor.py

# 多 universe 策略扫描
python -m agents.codex_cli sweep f_auto_001 \
    --factor-file alphas/exp/agent/f_auto_001/factor.py

# KB 查询
python -m agents.kb_query --category volume_reversal --limit 3

# 人工 admit
python -m backtest.factor.admission admit f_auto_001
```

`sweep` 默认 quick scan 跑到 step6：每个 universe 内的策略组合共享同一份 factor/market panel，并用批量 simple backtest 生成各自的 `signals.parquet`、`simple/nav.parquet`、`result.json`。`--validate-top-n` 会对最优组合从 step7 继续跑详细回测；只有显式传 `--to-step 7` 或更高时才回到逐 combo worker 路径。

## Knowledge Base

KB 位于 `agents/knowledge_base/`，同样是 gitignored 本地运行时状态：

```text
agents/knowledge_base/
├── anti_patterns.json
├── successful_patterns.json
├── failed_attempts.jsonl
└── hypothesis_index.jsonl
```

使用方式：

- `pdf-hypothesis` 用 KB 做查重、反模式和 SOTA 参考。
- `factor-iterate` 在 pass/fail 收尾时更新 KB。
- `reject-factor` 只检查 KB 是否已有失败记录，不负责补写新的研究结论。

## 目录结构

```text
agents/
├── codex_cli.py
├── runner.py
├── evaluator.py
├── experiment.py
├── schema.py
├── trace.py
├── kb_query.py
├── kb_update.py
├── sweep.py
├── FACTOR_CODE_GUIDE.md
├── knowledge_base/
└── pdf_hypotheses/

.codex/
├── skills/
│   ├── pdf-hypothesis/
│   ├── factor-iterate/
│   └── reject-factor/
└── prompts/
```

## 常见问题

### 我需要复制 hypothesis 路径吗？

不需要。让 Codex “用刚才第 1 个继续”即可。skill 会遍历 `agents/pdf_hypotheses/` 并读取对应文件。

### PDF hypothesis 可以手动改吗？

可以。它是人机协作契约文件。你可以改公式、参数、数据源或 suggested config，再让 `factor-iterate` 从该 hypothesis 继续。

### 因子通过后会自动入库吗？

不会。通过 pipeline 的因子会进入 `results/candidates/<factor_id>/`，人工审阅后再执行：

```bash
conda activate AutoQuant && python -m backtest.factor.admission admit <factor_id>
```

### 失败因子为什么要走 `reject-factor`？

因为 reject 需要同时处理 work DB、registry、代码目录、results 归档和 KB 检查。独立 skill 可以把 destructive 操作集中到一个有确认门禁的流程里。
