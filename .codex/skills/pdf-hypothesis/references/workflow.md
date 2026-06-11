# PDF Hypothesis Workflow

本 workflow 用于把 A 股研报 PDF 转成可供 `factor-iterate` 继续使用的结构化 hypothesis 文件。它是独立 skill：只做 PDF 提取、因子穷举、静态评审和 hypothesis 生成，不运行回测。

## 1. 输入识别与文件选择

- 如果用户给出 PDF 文件路径，先确认文件存在，然后处理该文件。
- 如果用户说“最新研报”“刚下载的研报”，列出 `research_papers/*.pdf`，按修改时间倒序展示编号菜单，让用户选择。
- 如果用户没有给路径，列出 `research_papers/*.pdf`，展示编号菜单，并等待用户选择。
- 如果用户要求批量分析目录，遍历目录下 PDF，按文件名排序后逐个处理；每次运行仍只生成一个批次目录。

菜单格式：

```text
可分析的研报：
1. <file-name> | <size> | <mtime>
2. <file-name> | <size> | <mtime>
a. 全部分析
```

不要要求用户复制 `research_papers/...` 路径；用户只需选择编号。

## 2. PDF 提取

- 优先使用当前 Codex/GPT 会话的原生多模态 PDF 阅读能力（例如 GPT-5.5 支持直接读取 PDF 输入时），覆盖正文、表格、图、脚注和扫描件页面。记录页码、章节或图表位置，便于复核。
- 如果当前会话没有可用原生多模态 PDF 输入，再使用项目 `.mcp.json` 配置的 `mcp-pdf` 能力提取 markdown。
- 如果原生多模态与 PDF MCP 都不可用，可使用本地 Python PDF 文本提取库作为 fallback。
- 必须在 `manifest.json.source.extraction_method` 中记录实际方法：`native_multimodal` / `mcp-pdf` / `local_text_extractor`。
- 原生多模态读取时，如果无法导出完整逐页文本，写入 `evidence.md` 或 `extracted.md` 保存可复核的页码、图表/表格依据和关键摘录；不得只依赖未落盘的模型记忆。
- 超长 PDF 先读取元信息或目录，再按章节处理；最终分析仍需覆盖全篇。
- 只有当完整提取文本对复核有价值或使用文本 fallback 时，才把全文写入批次目录的 `extracted.md`。

## 3. 因子穷举与筛选

先穷举，再筛选。不要因为因子普通、常见、研报没有完整回测指标而跳过。

每个候选因子至少记录：

- `name`
- `category`
- `hypothesis_text`
- `formula`
- `construction_logic`
- `data_sources`
- `required_columns`
- `report_metrics`
- `source_location`
- `source_quote`

筛选顺序：

1. 数据可用性：运行 `conda activate AutoQuant && python -m agents.codex_cli schema --sources <sources>`，确认列名和可用 proxy。
2. KB 检查：用 `python -m agents.kb_query` 查询同 category 的 successful patterns、anti-patterns 和 duplicate risk。
3. 静态评审：评估经济逻辑、参数合理性、前视风险、重复风险和 pipeline 兼容性。
4. 排序：优先使用研报 Sharpe/ICIR；缺失时基于同类因子和 KB 做保守估计，并标注估算来源。

## 4. 批次目录契约

所有输出写入单个批次目录：

```text
agents/pdf_hypotheses/
└── <YYYYMMDD_HHMMSS_slug>/
    ├── manifest.json
    ├── extracted.md
    ├── evidence.md
    ├── 01_<factor_slug>_hypothesis.md
    ├── 02_<factor_slug>_hypothesis.md
    └── ...
```

规则：

- 顶层 `agents/pdf_hypotheses/` 不生成 `.json` 或 `.md`。
- `manifest.json` 必须生成。
- `extracted.md` 可选，用于保存完整或接近完整的文本提取结果。
- `evidence.md` 可选，用于保存原生多模态读取时的页码、图表/表格依据和关键摘录。
- 只为用户选择或默认高优候选生成 `NN_<factor_slug>_hypothesis.md`。
- 文件名使用 ASCII slug；hypothesis 正文可以包含中文。

`manifest.json` 最小结构：

```json
{
  "version": 1,
  "source": {
    "pdf_paths": ["research_papers/example.pdf"],
    "pdf_title": "研报标题",
    "extraction_ts": "2026-06-12T10:00:00+08:00",
    "extraction_method": "native_multimodal"
  },
  "summary": {
    "total_factors_found": 0,
    "feasible_count": 0,
    "high_priority_count": 0
  },
  "ranked_factors": [],
  "rejected_factors": [],
  "generated_hypotheses": [
    {
      "menu_id": 1,
      "rank": 1,
      "name": "因子名称",
      "category": "volume_reversal",
      "file": "01_factor_slug_hypothesis.md",
      "priority": "high",
      "estimated_sharpe": 0.8,
      "ho_recommendation": "proceed"
    }
  ]
}
```

## 5. hypothesis.md 模板

每个 hypothesis 文件必须包含这些标题，供 `factor-iterate` 稳定解析：

````markdown
# Factor Hypothesis: <factor name>

## Source
- Report: <report title>
- PDF Path: <path>
- Extraction Batch: <batch directory>
- Rank: <rank>

## Hypothesis
<one paragraph>

## Formula
```python
<AutoQuant-style pseudo formula>
```

## Construction Logic
<calculation steps and economic intuition>

## Data Sources
- market_daily

## Required Columns
- close
- adj_factor

## Parameters
<window, horizon, rebalance, decay, top_k>

## Suggested Config
```yaml
pipeline:
  default_decay: 5
  default_rebalance: "1D"
  default_top_k: 100
  ret_type: "open"
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
  allow_short: false
```

## Static Review
<HO and KB review summary>

## Source Evidence
<short source quote or page/section pointer>
````

## 6. 用户交互输出

完成后先展示包含预期收益和迭代优先级的排名表，再展示可继续迭代的编号菜单。优先级使用绿/黄/红三级：

- 绿：数据可落地，报告收益强，HO/KB 无明显阻碍，建议优先迭代。
- 黄：可落地但依赖 proxy、收益中等、或有轻微风险，建议审阅后迭代。
- 红：关键数据缺失、强反模式、或不可作为单因子落地，暂不建议迭代。

排名表格式：

```text
| 排名 | 因子 | 预期收益 | 报告 ICIR | 数据可用性 | 迭代优先级 | 建议 |
|------|------|----------|-----------|------------|------------|------|
| 1 | 隔夜价量信号强度 | Sharpe 3.26 | 0.620 | proxy 可落地 | 绿 | 优先迭代 |
| 2 | 日内量价相关性 | Sharpe 1.83 | 0.481 | proxy 可落地 | 黄 | 审阅后迭代 |
| 3 | 隔夜-日内成交量相关性 | Sharpe 3.25 | 0.638 | 缺集合竞价量 | 红 | 暂不迭代 |
```

随后展示 hypothesis 文件菜单：

```text
已生成 hypothesis：
1. 01_volume_reversal_hypothesis.md | 高优 | Sharpe 0.95 | HO proceed
2. 02_liquidity_discount_hypothesis.md | 中优 | Sharpe 0.62 | HO revise
```

告诉用户可以直接说：

- “迭代第 1 个”
- “用刚才第 2 个继续”
- “从最新 PDF 结果里选”

不要要求用户复制 `agents/pdf_hypotheses/...` 路径。
