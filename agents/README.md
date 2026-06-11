# AutoQuant Agent 投研系统 — 使用手册

> **版本**: 2026-06-02  
> **定位**: Codex subagent 驱动的 A 股因子迭代研究系统。不写独立 agent 循环，由 Codex 本身承担决策、代码生成和结果分析。Python 侧只保留最小执行层。

---

## 目录

1. [一分钟快速开始](#一分钟快速开始)
2. [三种研究入口](#三种研究入口)
   - 自然语言输入（最常用）
   - 研报 PDF 提取
   - 已有 hypothesis 文件
3. [内部工作流](#内部工作流)
4. [命令速查](#命令速查)
5. [Knowledge Base（知识库）](#knowledge-base知识库)
6. [目录结构](#目录结构)
7. [常见问题](#常见问题)

---

## 一分钟快速开始

### 方式一：自然语言描述一个因子想法

```text
/factor-iterate 成交额放量后短期反转，尤其在小盘股里更强
```

系统会自动：
1. **HG** 将你的想法扩展为结构化假设（含公式草稿、参数、数据列）
2. **HO** 静态评审该假设（查重/反模式/参数建议/数据可行性）
3. **FC** 将假设写成 Python 因子代码
4. 运行 **Pipeline**（step1~step10 门控回测）
5. **RC** 分析结果；如果不达标 → 自动修复/调参，最多 10 轮

### 方式二：从研报 PDF 中提取因子

```text
/pdf-hypothesis research_papers/华泰多因子系列之四.pdf
```

系统会：
1. 提取 PDF 中所有单因子
2. 按预期 Sharpe 排序
3. 附加 **HO 静态评审**（重复风险/反模式警告）
4. 生成 `hypothesis.md` 文件
5. 你确认后直接 `/factor-iterate --hypothesis <文件路径>` 执行

### 方式三：已有 hypothesis 文件

```text
/factor-iterate --hypothesis agents/pdf_hypotheses/20260501_120000_momentum/5日反转_hypothesis.md
```

跳过 HG/HO，直接进入 FC 编码 + Pipeline。

---

## 三种研究入口

### 1. 自然语言输入（`factor-iterate` 无 `--hypothesis`）

**适合场景**：你有一个模糊的因子直觉，想快速验证。

**用法**：
```text
/factor-iterate "低波动股票在月末效应中表现更好"
/factor-iterate max_rounds=5 data_sources=market_daily,income_q "ROE改善且估值偏低的股票"
```

**完整流程**：

```
用户自然语言输入
    ↓
[HG] Hypothesis Generator
    - 扩展为结构化假设（formula_draft + parameters + category）
    - 查 schema 确认列名可用
    - 5 维自评（alignment/impact/novelty/feasibility/risk-reward）
    ↓
[HO] Hypothesis Optimizer（静态评审，不回测！）
    - 查重：与已有因子对比，标记近似重复
    - 反模式：construction_logic 是否触发已知失败模式
    - 参数建议：基于同类 SOTA 调整 window/decay
    - 数据可行性：所需列是否全部可用
    - 经济学逻辑：是否有反直觉构造
    ↓
审阅后的 hypothesis.md → [FC] Factor Coder 编码
    ↓
Pipeline（step1~step10 门控回测）
    ↓
[RC] Result Critic 诊断
    - 分析失败原因
    - 查 KB 反模式/成功模式
    - 决策：repair（同方向修复）/ abandon（放弃）/ new_hypothesis（换方向）
    ↓
repair → 下一轮循环（最多 max_rounds 轮）
pass → 写入 candidates/，等待人工 admit
abandon → 更新 KB，输出放弃报告
```

**关键决策点**：
- `ho_review.recommendation == "abandon"` → 系统在 FC 前拦截，直接结束
- `ho_review.recommendation == "revise"` → 展示优化后的假设给用户确认
- `ho_review.recommendation == "proceed"` → 直接进入 FC

### 2. 研报 PDF 提取（`/pdf-hypothesis`）

**适合场景**：拿到一篇券商/学术研报，想批量提取其中的单因子并筛选。

**用法**：
```text
/pdf-hypothesis                              # 无参数 → 列出 research_papers/ 目录供选择
/pdf-hypothesis research_papers/xxx.pdf      # 指定 PDF
/pdf-hypothesis --latest                     # 自动选最新修改的 PDF
/pdf-hypothesis --dir research_papers/       # 批量扫描目录下所有 PDF
```

**流程**：

```
PDF 文本提取（mcp-pdf）
    ↓
Step 1-2: 穷举所有单因子（不遗漏任何一个）
    ↓
Step 3: 可行性筛选 + Sharpe 排序
    - 数据可用性（schema 校验）
    - KB 反模式检查
    - 预期 Sharpe 估算
    ↓
Step 4-5: 输出排名短名单 + 生成 hypothesis.md
    ↓
Step 6 [新增]: HO 批量评审（对每个候选因子）
    - duplicate_risk（查重）
    - anti_pattern_warnings（反模式）
    - param_suggestions（参数建议）
    ↓
Step 7: 展示最终排名表（含 HO 评审结果）
```

**排名表示例**：

```
| 排名 | 因子 | 公式 | 预期 Sharpe | HO 评审 | 推荐 |
|------|------|------|-----------|---------|------|
| 1 | 放量反转 | `...` | 0.95 | ✅ 无警告 | 🟢 |
| 2 | ROE 质量 | `...` | 0.72 | ⚠️ 近似 f_barra_quality | 🟡 |
| 3 | 流动性折价 | `...` | 0.55 | ❌ 触发反模式 | 🔴 |
```

**推荐规则**：
- 🟢 **高优**：Sharpe ≥ 0.8 + HO 无警告 → 推荐立即执行
- 🟡 **中优**：Sharpe ≥ 0.5 或 HO 有轻微警告 → 建议审阅
- 🔴 **低优**：Sharpe < 0.5 或 HO 发现严重问题 → 降级或 skip

**与 `/factor-iterate` 的联用**：

```text
# 第一步：提取
/pdf-hypothesis research_papers/华泰多因子系列之四.pdf

# 第二步：审阅生成的 hypothesis.md（可手动修改）

# 第三步：执行
/factor-iterate --hypothesis agents/pdf_hypotheses/20260501_120000_momentum/5日反转_hypothesis.md
```

### 3. 已有 hypothesis 文件（`factor-iterate --hypothesis`）

**适合场景**：
- PDF 提取后已生成 hypothesis.md，想直接执行
- 自己手写了一个 hypothesis.md
- 上一轮 RC 输出了 `new_hypothesis`，想在新方向上启动

**用法**：
```text
/factor-iterate --hypothesis agents/pdf_hypotheses/xxx/xxx_hypothesis.md
/factor-iterate --hypothesis              # 无路径 → 列出 pdf_hypotheses/ 目录供选择
```

此模式下跳过 HG/HO，直接 Read hypothesis.md → FC 编码 → Pipeline → RC 诊断。

---

## 内部工作流

### Subagent 分工（4 个角色）

| Subagent | 全称 | 职责 | 触发时机 |
|----------|------|------|----------|
| **HG** | Hypothesis Generator | 将自然语言扩展为结构化假设 | 用户无 `--hypothesis` 时；RC 输出 `new_hypothesis` 时 |
| **HO** | Hypothesis Optimizer | **不回测**的前提下做静态专家评审 | HG 输出后；PDF 提取后 |
| **FC** | Factor Coder | 写因子代码 + 执行 pipeline | HO 审阅通过后；RC 诊断 repair 时 |
| **RC** | Result Critic | 诊断失败 + 决定 repair/abandon/new_hypothesis | Pipeline `status != "pass"` 时 |

### 条件注入 Prompt（核心优化）

RC 不是每次都读取完整的 `trace.jsonl` 和整个 KB。父进程根据 `failure_type` 动态组装 prompt，只注入必要信息：

| failure_type | 注入的历史深度 | 额外注入 |
|---|---|---|
| `code_error` / `schema_error` | 最近 1 轮 | code diff |
| `coverage_fail` | 最近 2 轮 | data_sources 列表 |
| `icir_fail` | 最近 3 轮 + trend 分析 | SOTA 参考 + 反模式 |
| `backtest_fail` | 策略参数历史 + metrics trend | SOTA 参考 |
| `ridge_fail` | max_existing_factor 信息 | 相关成功模式 |
| `residual_fail` | 最近 1 轮 + 残差指标 | — |

**效果**：省 token、聚焦问题、避免历史信息淹没当前诊断。

### Trace DAG（为分支预留）

每轮运行 append 一行 JSON 到 `trace.jsonl`，记录：
- `round`: 轮次编号
- `parent_round_id`: 父轮次（线性时为 `round-1`，fork 时为源轮次）
- `branch_id`: 分支标识（目前固定为 `"main"`，Phase 3 启用多分支）
- `failure_type`, `diagnosis`, `fix_strategy`, `metrics` 等

```json
{
  "round": 3,
  "parent_round_id": 2,
  "branch_id": "main",
  "factor_id": "f_auto_20260527_001",
  "status": "fail",
  "failure_type": "icir_fail",
  "diagnosis": "窗口过长导致反转信号衰减",
  "fix_strategy": "缩窗到 5-10 天",
  "fix_level": "factor",
  "factor_change": "params",
  "same_direction": true,
  "metrics": {"annual_icir": 0.15, "simple_sharpe": 0.3}
}
```

### 一轮运行的目录结构

```text
results/
  <run_id>/                         # 追踪文件
    hypothesis.md
    hypothesis_optimized.json       # HO 输出
    trace.jsonl                     # 每轮记录
    factor.py                       # 当前因子代码
    config.yaml                     # 当前策略配置
  <factor_id>/                      # 因子评估结果
    factor_eval/                    # step1-4（同一因子共享）
    decile_backtest/                # 十段分层测试
    top100_1d_d5/                   # 策略变体 1
      result.json
      pipeline_report.md
      plots/
    top200_5d_d10/                  # 策略变体 2
      ...
  candidates/                       # 通过 pipeline 等待人工 admit
    <factor_id>/
      factor.py
      pipeline_report.md
      result.json
```

---

## 命令速查

### 交互式命令（Codex slash commands）

```text
# 自然语言输入 → 完整 HG→HO→FC→Pipeline→RC 流程
/factor-iterate "你的因子想法"
/factor-iterate max_rounds=5 "限制最多 5 轮"

# 已有 hypothesis → 跳过 HG/HO，直接 FC
/factor-iterate --hypothesis agents/pdf_hypotheses/xxx/xxx_hypothesis.md
/factor-iterate --hypothesis              # 无路径 → 列出 hypothesis 文件供选择

# PDF 提取 → 穷举因子 → 排序 → 生成 hypothesis.md
/pdf-hypothesis                           # 无参数 → 列出 PDF 文件供选择
/pdf-hypothesis research_papers/xxx.pdf   # 指定 PDF
/pdf-hypothesis --latest                  # 自动选最新 PDF
/pdf-hypothesis --dir research_papers/    # 批量扫描
/pdf-hypothesis --top 5 research_papers/xxx.pdf   # 只取 top 5
```

### CLI 工具（Python 执行层）

```bash
conda activate AutoQuant

# 查询数据 schema（FC 编码前必查，确认列名存在）
python -m agents.codex_cli schema --sources market_daily
python -m agents.codex_cli schema --sources market_daily,income_q

# 单轮执行（通常由 Codex 自动调用，无需手动）
python -m agents.codex_cli run f_auto_001 \
    --factor-file results/<run_id>/factor.py

# KB 查询（通常由父进程自动调用）
python -m agents.kb_query \
    --category volume_reversal \
    --failure-type icir_fail \
    --limit 3

python -m agents.kb_query \
    --check-duplicate \
    --formula-fingerprint "ts_mean(amount,5)/ts_mean(amount,20)*ts_delta(close*adj,5)"

# 人工 admit 通过 pipeline 的因子
python -m backtest.factor.admission admit f_auto_001
```

---

## Knowledge Base（知识库）

KB 是跨 run 持久化的知识积累，分 **三级** 设计防止 context 爆炸：

### 三级分层

| 层级 | 数据 | 加载方式 | 大小控制 |
|------|------|----------|----------|
| **L1 热数据** | 当前 trace、当前 category 的 SOTA（1 条）、匹配当前 failure_type 的 anti_patterns（≤3 条） | **每次必带** | 自然控制 |
| **L2 温数据** | 同 category 成功模式（≤5 条）、同 failure_type 反模式（≤5 条） | **父进程过滤后注入 prompt** | 按 category + count 排序截断 |
| **L3 冷数据** | 完整 failed_attempts.jsonl、已归档 hypothesis | **不直接进 prompt**，Python 层 keyword 查询 | 定期归档 |

### 文件位置

```text
agents/knowledge_base/
  ├── anti_patterns.json          # 失败模式 → 修复建议
  ├── successful_patterns.json    # 成功模式 → SOTA 基准
  ├── failed_attempts.jsonl       # 失败实验索引（append-only）
  └── hypothesis_index.jsonl      # 轻量查重索引（factor_id + formula_fingerprint + status）
```

### 自动更新时机

- **Pass 时**：写入 `successful_patterns.json`（同 category）+ `failed_attempts.jsonl`（status=pass）
- **Abandon 时**：写入 `anti_patterns.json`（如果 RC 发现新反模式）+ `failed_attempts.jsonl`（status=fail）
- **每次 HO 查重时**：读取 `hypothesis_index.jsonl` 做 formula fingerprint 匹配

---

## 目录结构

```text
agents/
├── README.md                   # 本文（使用手册）
├── AGENTS.md                   # 系统总览与定位
├── DESIGN.md                   # 架构设计文档（v2.0）
├── TODO.md                     # P0~P4 工单池
├── codex_cli.py               # CLI 入口：schema + run
├── runner.py                   # 流水线执行器（对接 backtest）
├── evaluator.py                # 评估器：result → feedback
├── experiment.py               # 实验数据类
├── config.py                   # AgentConfig
├── schema.py                   # 数据 schema 查询
├── helpers.py                  # 工具函数（代码校验、@register 注入）
├── kb_query.py                 # KB 分层查询脚本（新增）
├── FACTOR_CODE_GUIDE.md        # LLM 因子代码参考手册
├── knowledge_base/             # 跨 run 本地知识库（gitignore）
│   ├── anti_patterns.json
│   ├── successful_patterns.json
│   ├── failed_attempts.jsonl
│   └── hypothesis_index.jsonl  # 轻量查重索引（新增）
└── pdf_hypotheses/             # PDF→hypothesis 中间产物（gitignore）
    └── <slug>/
        └── *_hypothesis.md

.codex/
├── commands/
│   ├── factor-iterate.md       # /factor-iterate 命令定义
│   └── pdf-hypothesis.md       # /pdf-hypothesis 命令定义
└── prompts/
    ├── shared/
    │   ├── role.md             # 4 个 subagent 角色 persona
    │   ├── output_formats.md   # JSON schema 模板
    │   └── context_sections.md # 条件注入规则 + Section 模板
    ├── factor_coder.md         # FC system prompt
    ├── result_critic.md        # RC system prompt
    ├── hypothesis_gen.md       # HG system prompt
    ├── hypothesis_optimizer.md # HO system prompt（新增）
    └── pdf_hypothesis.md       # PDF 分析 prompt
```

---

## 常见问题

### Q: 自然语言输入和 `--hypothesis` 模式有什么区别？

| | 自然语言输入 | `--hypothesis` 模式 |
|---|---|---|
| 入口 | `/factor-iterate "..."` | `/factor-iterate --hypothesis <path>` |
| HG | ✅ 自动生成结构化假设 | ❌ 跳过 |
| HO | ✅ 静态评审 | ❌ 跳过 |
| 适用 | 模糊的直觉/想法 | 已有明确假设（PDF 提取或手写） |

### Q: HO 静态评审能发现哪些问题？

- **近似重复**：你的公式和已有因子（KB 中）高度相似
- **反模式**：构造逻辑触发已知失败模式（如财务数据做 ts_mean）
- **参数不合理**：window 远超同类 SOTA 的范围
- **数据不可用**：所需列在 schema 中不存在
- **经济学逻辑问题**：如未复权价格做时序比较、ST/涨跌停在因子中过滤

### Q: 一轮最多迭代多少轮？

默认 `max_rounds=10`。可以在输入时覆盖：
```text
/factor-iterate max_rounds=5 "成交额放量后短期反转"
```

### Q: 因子通过后如何入库？

通过 pipeline 的因子自动写入 `results/candidates/<factor_id>/`，**不会自动 admit**。需要人工审阅后执行：

```bash
conda activate AutoQuant && python -m backtest.factor.admission admit <factor_id>
```

### Q: 为什么放弃时还要更新 KB？

失败的因子同样有价值：
- `anti_patterns.json`：记录"这种构造方式在这种 category 下会失败"
- `failed_attempts.jsonl`：记录"这个公式尝试了 X 轮，最终因为 Y 失败"

后续 HO 查重时会参考这些记录，避免重复踩坑。

### Q: 可以手动修改 hypothesis.md 吗？

**可以，且推荐**。`hypothesis.md` 是人机协作的契约文件：
- PDF 提取后 → 你可以修改公式/参数/config
- HG 生成后 → 你可以修改 HO 的优化建议
- 确认无误后再 `--hypothesis` 执行

### Q: RC 输出 `new_hypothesis` 是什么意思？

当 RC 判断当前方向已耗尽（如 3+ 轮无改善），但发现一个有希望的替代方向时，会输出 `same_direction=false` + `new_hypothesis="..."`。

父进程会：
1. 结束当前 branch
2. 启动 **新 run**，以 `new_hypothesis` 作为 HG 输入
3. 新 run 的 `parent_round_id` 指向当前 run 的最后一轮（Trace DAG 分叉）

### Q: 系统需要 API Key 吗？

- **Codex 本身**：需要 `OPENAI_API_KEY`（在 `.env` 中配置）
- **Python 执行层**：需要 `TUSHARE_TOKEN`（数据下载）
- **mcp-pdf**：PDF 提取走本地 MCP server，不消耗 API token

### Q: 如何查看某个 category 的 SOTA？

```bash
conda activate AutoQuant && python -m agents.kb_query \
    --category volume_reversal \
    --limit 5
```

或查看 `agents/knowledge_base/successful_patterns.json`。
