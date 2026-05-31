# /pdf-hypothesis

读取券商/学术研报 PDF，通过 mcp-pdf 提取文本后，**穷举研报中列出的所有单因子**，按预期 Sharpe/ICIR 排序，筛选出数据可得、非重复、高夏普的可行因子短名单。审阅后可直接转入 `/factor-iterate` 执行。

## Usage

```text
/pdf-hypothesis                                          # 无参数 → 弹出 research_papers/ 文件列表供选择
/pdf-hypothesis research_papers/xxx.pdf                  # 指定单个 PDF
/pdf-hypothesis --top 5 research_papers/xxx.pdf          # 只取 Sharpe 最高的 top 5
/pdf-hypothesis --dir research_papers/                    # 批量扫描目录下所有 PDF
/pdf-hypothesis --latest                                  # 自动选目录下最新修改的 PDF
```

### 无参数交互模式（推荐）

当用户直接输入 `/pdf-hypothesis` 不带任何参数时，**必须先列出 `research_papers/` 目录下所有 PDF 文件供用户选择**，而不是报错：

1. 列出目录内容：

   ```bash
   ls -lt research_papers/*.pdf
   ```

2. 展示为带编号的菜单：

   ```
   ## 选择要分析的研报：

   | # | 文件名 | 大小 | 修改时间 |
   |---|--------|------|---------|
   | 1 | 20161220-华泰证券-多因子系列之四：单因子测试之动量类因子.pdf | 3.0 MB | 2026-05-31 |
   | 2 | 华泰证券-多因子系列之五：单因子测试之估值类因子.pdf | 2.5 MB | 2026-05-30 |
   | 3 | 申万宏源-A股因子有效性报告2025Q4.pdf | 1.8 MB | 2026-05-29 |
   | a | 扫描全部 PDF |
   ```

3. 等待用户选择（输入编号或 `a`），确认后再进入 Step 1。

## Operating Rules

- 所有 Python 命令前必须使用 `conda activate AutoQuant`。
- PDF 文本提取通过 **mcp-pdf** MCP server 完成（`pdf_to_markdown` 工具），不依赖模型原生多模态。
- **无参数时必须先列出文件菜单**，不得直接报错或自动选第一个。
- **穷举优先**：遍历研报中出现的每一个单因子，不遗漏。即使研报只给一句话描述也要列入。
- **Sharpe/ICIR 驱动排序**：最终输出按预期 Sharpe 降序排列。研报给出了回测数据的直接用；没给的根据同类因子 + KB successful_patterns 估算。
- 输出保存到 `agents/pdf_hypotheses/<YYYYMMDD_HHMMSS_slug>.json`。
- 必须验证每个因子的数据可用性（调用 `claude_cli schema` 确认所需列名存在）。
- 必须查 KB 反模式库，标记与已知失败模式高度相似的因子。
- **不筛选 novelty**：即使和已有因子方向类似，只要 Sharpe 高且构造方式有差异，也列入候选。

## Prerequisites

1. **mcp-pdf server 已配置**：项目根 `.mcp.json` 已配置（`conda run -n AutoQuant mcp-pdf`）。安装依赖：`pip install mcp-pdf`（已加入 `environment.yml`）。

2. **Knowledge Base 可用**（非强制）：如果 `agents/knowledge_base/` 存在，则交叉验证；否则跳过 KB 检查。

## Procedure

### Step 0: 文件选择（无参数时）

如果用户**未提供 PDF 路径**（仅输入 `/pdf-hypothesis`），列出并展示菜单（同 §无参数交互模式）。

将输出格式化为可选择的表格，然后**等待用户选择**——不要自动继续，不要假设用户想跑哪个文件。

如果用户提供了 `--latest`，自动选择修改时间最新的 PDF。

如果用户提供了 `--dir <path>`，扫描目录下所有 PDF，逐个处理。

### Step 1: 提取 PDF 文本

使用 mcp-pdf 的 `pdf_to_markdown` 工具提取 PDF 全文为 markdown：

- 自动包含表格（转 markdown table）
- 保留章节结构
- 如果 PDF 超过 80 页，先调 `get_metadata` 获取目录，然后分章节提取

### Step 2: 穷举所有单因子

逐章节扫描研报文本，**穷举所有被提及的单因子**，不做主观筛选。

**什么是「单因子」**：
- 研报中明确给出构造公式 + 参数 + 回测结果的（如「5 日动量因子，ICIR=1.2」）
- 研报中描述了构造逻辑但未给出回测结果的（如「用 ROE 增速选股」）
- 研报回测表格中列出的因子，即使正文未详细描述
- 对研报图表中隐含的因子（如分层回测的排序变量），也要推断提取

**每条因子必须提取 6 项信息**：

1. **计算公式**：从研报原文推断的类 Python 伪代码（最重要！）
2. **构造逻辑**：原文描述 + 推断的 transforms 调用链
3. **研报回报数据**：年化收益 / Sharpe / ICIR / 最大回撤 / 胜率
4. **所需数据列**：映射到 data_sources 的实际列名
5. **参数**：窗口 / horizon / 调仓频率 / 持仓数
6. **原文定位**：章节 / 页码 / 原文引用

**公式提取要求**：
- 每条因子**必须**有一个 `formula` 字段，用类 Python/pandas 伪代码表达
- 公式用 AutoQuant transforms 命名风格：`rank()`, `ts_mean()`, `cs_zscore()`, `cs_rank()`, `ts_delta()`, `cs_mad_winsorize()`, `ts_regression_residual()` 等
- 研报只给了文字描述 → 根据描述推断公式
- 研报给了原始数学公式 → 直接转录为伪代码
- 样式：`cs_rank` / 方向（`*(-1)` 等）**不要**写在因子公式里——因子只输出原始信号值，排名和方向由 pipeline strategy 层处理
- 示例：「1 个月反转」→ `ts_delta(close * adj_factor, 21)`（原始累计收益，负值即反转）
- 示例：「换手率加权的 6 个月」→ `ts_decay_exp(ret_1d * turnover_rate, 126, halflife=63)`（加权均值）
- 示例：「5 日放量反转，小盘加权」→ `(ts_mean(amount, 5) / ts_mean(amount, 20)) * ts_delta(close * adj_factor, 5) * (1 / circ_mv)`

**不做的事**：
- ❌ 不因为「太简单 / 太常见」跳过
- ❌ 不因为研报没给回测数据就丢弃
- ❌ 不自行发明研报没提的因子

### Step 3: 可行性筛选 + Sharpe 排序

对穷举出的因子做**三轮筛选**：

**第一轮：数据可用性**

```bash
conda activate AutoQuant && python -m agents.claude_cli schema --sources <data_sources>
```

- 所需列全部可用 → 通过
- 关键列缺失且无法 proxy → 标记 `feasible=false`，扔到 rejected
- 部分列可用 proxy 替代 → 通过，在 construction_logic 中注明替代方案

**第二轮：KB 反模式检查**

查 `anti_patterns.json`：
- 构造逻辑高度匹配已知反模式的 `signature` → 标记 `kb_warning`，但不自动丢弃（让用户决定）
- 匹配到反模式且该模式 count ≥ 3 → 标记高风险，建议谨慎

查 `successful_patterns.json`：
- 同 category 的 SOTA ICIR/Sharpe → 用于校准预期回报

**第三轮：预期 Sharpe 估算 + 排序**

`estimated_sharpe` 按以下优先级确定：

1. 研报直接给出了 Sharpe → 直接采用（标注 `source: "report"`）
2. 研报给了 ICIR + 回撤 → 用 `ICIR × 行业经验系数` 估算 Sharpe（标注 `source: "estimated_from_icir"`）
3. 研报只给了收益/t 值 → 按 category 经验估算（标注 `source: "estimated_from_category"`）
4. 研报只描述了逻辑，无任何量化数据 → 以同 category successful_patterns SOTA 的 50% 估算（标注 `source: "rough_guess"`）

**最终排序**：按 `estimated_sharpe` 降序 → top N 输出。

### Step 4: 输出排序短名单

按 Step 3 排序结果，输出结构化短名单。每条因子附加：

- 可行性风险提示（数据 proxy / KB 警告）
- 推荐优先级（见 Decision Rules）
- 如果研报中有详细构造描述 → 附 `construction_logic`

### Step 5: 生成 hypothesis.md + 提示下一步

1. 展示**排名表**（名次 / 因子名 / 公式 / 预期 Sharpe / 数据可用 / 推荐）
2. 询问用户确认要执行的因子（可多选）
3. 对每个选中的因子，生成 `hypothesis.md` 到：

   ```
   agents/pdf_hypotheses/<YYYYMMDD_HHMMSS_slug>/<factor_name>_hypothesis.md
   ```

4. 告知用户文件路径，提示可手动审阅修改 `hypothesis.md`（公式/参数/config），确认后执行：

   ```text
   /factor-iterate --hypothesis agents/pdf_hypotheses/<slug>/<factor_name>_hypothesis.md
   ```

   **绝对不要**在本命令中直接调用 `/factor-iterate`。`hypothesis.md` 是两条命令之间的唯一契约。

## Decision Rules

**优先级分档**：

| 档位 | 条件 | 行动 |
|------|------|------|
| 🟢 高优 | `estimated_sharpe ≥ 0.8` 且 `feasible=true` 且 `kb_warning=null` | 直接进入 `/factor-iterate` |
| 🟡 中优 | `estimated_sharpe ≥ 0.5` 或 `kb_warning` 存在但 count < 3 | 审阅后决定 |
| 🔴 低优 | `estimated_sharpe < 0.5` 或数据依赖 proxy | 最后考虑，或直接 skip |
| ⚫ 不可行 | 关键数据缺失且无 proxy | rejected |

**去重判断**：
- 两个因子构造逻辑和参数高度重合（>80%）→ 只保留 Sharpe 更高的那个
- 与 KB successful_patterns 中已有因子 MC > 0.85 → 标记「近似重复」，仍保留但降低优先级

## Output Format

保存到 `agents/pdf_hypotheses/<slug>.json`：

```json
{
  "source": {
    "pdf_path": "research_papers/xxx.pdf",
    "pdf_title": "研报标题",
    "extraction_date": "2026-05-31T10:00:00"
  },
  "summary": {
    "total_factors_found": 15,
    "feasible_count": 8,
    "high_priority_count": 3,
    "top_sharpe": 1.25
  },
  "ranked_factors": [
    {
      "rank": 1,
      "priority": "🟢 高优",
      "name": "研报中的因子名称",
      "category": "volume_reversal",
      "hypothesis_text": "一句话描述",
      "formula": "ts_mean(amount, 5) / ts_mean(amount, 20) * ts_delta(close * adj_factor, 5) * (-1)",
      "construction_logic": "分步骤 transforms 调用链",
      "data_sources": ["market_daily"],
      "report_metrics": {
        "sharpe": 0.95,
        "annual_return": 0.12,
        "icir": 1.55,
        "max_drawdown": -0.068,
        "win_rate": 0.63
      },
      "estimated_sharpe": 0.95,
      "sharpe_source": "report",
      "feasible": true,
      "data_availability": "全部列可用",
      "kb_check": {
        "similar_successful": "volume_reversal SOTA: ICIR=1.55, Sharpe=0.95",
        "kb_warning": null,
        "duplicate_of": null
      },
      "source_quote": "研报原文引用段落",
      "suggested_config": {
        "decay": 5,
        "rebalance": "1D",
        "top_k": 100
      }
    }
  ],
  "rejected": [
    {
      "name": "因子名称",
      "rejection_reason": "数据不可用: 需要 northbound_flow，schema 中不存在",
      "reported_sharpe": 1.1
    }
  ]
}
```

## Chinese Research Report Factor Patterns

券商/学术研报中常见单因子列示形式：

| 研报常见表述 | 因子方向 | 所需数据 | 典型参数 |
|-------------|---------|---------|---------|
| 「动量因子（MOM）」 | 价格动量 | `close`, `adj_factor` | window=20, skip=1 |
| 「反转因子（REV）」 | 短期反转 | `close`, `adj_factor` | window=5 |
| 「换手率因子（TO）」 | 换手率 | `turnover_rate` / `turnover_rate_free` | window=20 |
| 「波动率因子（VOL）」 | 低波 | `close`, `adj_factor` | window=20, std |
| 「市值因子（SIZE）」 | 小市值 | `circ_mv` | ln(circ_mv) |
| 「估值因子（EP/BP）」 | 价值 | `pe_ttm`, `pb`, `ps_ttm` | 截面 rank |
| 「盈利因子（ROE）」 | 质量 | `inc_return_on_equity` | 截面 rank |
| 「成长因子（GROWTH）」 | 成长 | `inc_revenue_ps`, `inc_net_profit` | yoy / qoq |
| 「资金流因子（MF）」 | 资金流向 | `mf_net_mf_amount`, `mf_buy_lg_amount` | window=5/20 |
| 「流动性因子（LIQ）」 | 非流动性 | `amount`, `turnover_rate` | Amihud / Pastor |
| 「Beta 因子」 | 低 Beta | `close`, `adj_factor` + 指数收益 | window=252 |
| 「波动率偏度」 | 偏度效应 | `close`, `adj_factor` | window=60 |
| 「最大日收益（MAX）」 | MAX 效应 | `high`, `low`, `adj_factor` | window=20 |
| 「异质波动率（IVOL）」 | 异质波动 | `close`, `adj_factor` + FF3 残差 | window=60 |
| 「盈余公告后漂移」 | PEAD | `inc_*` + `f_ann_date` | SUE / CAR |
| 「应计异象」 | 应计 | `bs_*`, `cf_*` | 应计项目计算 |

**术语映射（研报 → 实际列名）**：

| 研报用语 | 实际列名 |
|---------|---------|
| 「成交额」「成交金额」 | `amount`（单位：元） |
| 「成交量」 | `volume`（单位：股，不推荐直接用，用 `turnover_rate` 替代） |
| 「流通市值」「市值」 | `circ_mv` |
| 「总市值」 | `total_mv` |
| 「换手率」 | `turnover_rate` / `turnover_rate_free` |
| 「涨跌幅」 | `pct_chg`（已调整，无需复权） |
| 「收盘价」 | `close`（需乘 `adj_factor` 复权） |
| 「主力资金」「大单」 | `mf_buy_lg_amount` / `mf_sell_lg_amount` / `mf_net_lg_amount` |
| 「散户资金」「小单」 | `mf_buy_sm_amount` / `mf_sell_sm_amount` |
| 「市盈率」 | `pe_ttm` |
| 「市净率」 | `pb` |
| 「ROE」「净资产收益率」 | `inc_return_on_equity`（**季度频率**，不做 ts_mean） |
| 「营收增速」 | `inc_revenue_ps` yoy（**季度频率**） |
| 「净利润增速」 | `inc_net_profit` yoy（**季度频率**） |
| 「股息率」 | `dv_ttm` / `dv_ratio` |
| 「ST」 | `is_st`（bool，strategy.universe.exclude_st 处理，因子不屏蔽） |
| 「涨跌停」 | `limit_up / limit_down`（simulation 层处理，因子不屏蔽） |

**风险提示**：
- 研报通常有「看好」倾向（券商卖方报告），报告 Sharpe 建议打 7~8 折
- 研报回测可能存在前视偏差、幸存者偏差 → 实际跑可能低 20~30%
- 研报参数可能过拟合 → 实际执行建议做参数敏感性测试
- 表格中的数字可能不精确 → 以构造逻辑为准，不以数字为准

## Output Style

**每次运行必须首先输出排名总览表**：

```
## 研报因子排名：{pdf_title}

| 排名 | 因子 | 公式 | 预期 Sharpe | 研报来源 | 数据可用 | 推荐 |
|------|------|------|-----------|---------|---------|------|
| 1 | 放量反转 | `ts_mean(amount,5)/ts_mean(amount,20)*ts_delta(close*adj,5)*(-1)` | 0.95 | 研报回测 | ✅ | 🟢 |
| 2 | ROE 质量 | `inc_return_on_equity` | 0.72 | ICIR 估算 | ✅ | 🟡 |
| 3 | 流动性折价 | `1 / ts_mean(turnover_rate, 20) * (-1)` | 0.55 | 逻辑推断 | ⚠️ | 🔴 |
| ... | ... | ... | ... | ... | ... | ... |

数据不可用 (rejected): 北向资金因子、机构调研因子
```

然后展示高优（🟢）因子的**公式 + 构造概要**。告知 hypothesis.md 已生成及下一步命令。不贴完整 PDF 文本，除非用户要求查看原文段落。

### hypothesis.md 输出格式

每个选中因子生成一个独立文件，格式如下：

```markdown
# Factor Hypothesis: <因子名称>

## Source
- **Report**: <研报标题>
- **Report Date**: <日期>
- **Extraction Date**: <ISO timestamp>
- **PDF Path**: research_papers/xxx.pdf

## Hypothesis
<一句话假设描述>

## Formula
```
<类 Python 伪代码，AutoQuant transforms 风格>
```

## Construction Logic
（仅描述因子原始值计算步骤。去极值/中性化/排名/方向由 pipeline 统一处理，不在此列出。）
1. Step 1
2. Step 2
...

## Parameters
- **window**: 63
- **halflife**: 31
- **variant**: barra_ind_size
- **category**: momentum_reversal
- **data_sources**: market_daily

## Report Metrics
- **多空 Sharpe**: 2.57
- **IR Ratio**: 0.96
- **IC Mean**: -7.74%
- **TOP 组合夏普**: 1.15

## Data Columns Required
- `close`, `adj_factor`, `turnover_rate`
- (ST/涨跌停过滤由 strategy/simulation 层处理，因子不需使用 `is_st`, `limit_up`, `limit_down`)

## Suggested Config
\`\`\`yaml
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
\`\`\`
```

### 与 /factor-iterate 的联用

```text
# 第一步：提取因子
/pdf-hypothesis research_papers/华泰多因子系列之四.pdf
# → 排名表 → 选择因子 → 生成 hypothesis.md

# 第二步：审阅 hypothesis.md（手动编辑公式/参数/config）

# 第三步：执行
/factor-iterate --hypothesis agents/pdf_hypotheses/<slug>/<name>_hypothesis.md
```
