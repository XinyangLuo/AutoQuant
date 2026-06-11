# Agent 投研系统 — 设计文档

> **版本**: 2026-06-07 v2.1
> **定位**: Codex subagent 模式驱动的因子迭代研究系统。Codex 直接承担决策、代码生成、结果分析；Python 侧只保留执行层与轻量文件工具。

---

## 1. 架构总览

### 1.1 核心思想

**从 RD-Agent 学到的关键洞察**：
1. **Hypothesis 是一等公民** —— 假设不是代码的附属品，而是独立的设计文档，需经结构化生成 → 静态评审 → 编码 → 回测的完整 DAG
2. **选择性上下文注入** —— Prompt 模板固定，但按场景动态选择注入 `sota` / `last` / `full` 历史片段，避免堆叠全部 trace
3. **Trace 即 DAG** —— 迭代历史不是线性数组，而是支持分支和 SOTA 回溯的有向无环图

**AutoQuant 的差异化选择**（不照搬 RD-Agent 重型架构）：
- ❌ 不用 pickle KB、Docker workspace、embedding 图
- ✅ 保留轻量文件 IPC（JSON/JSONL/Markdown）
- ✅ 保留 Codex subagent 模式（不建独立 Python agent 循环）
- ✅ 父进程（Codex 对话层）负责 prompt 组装和决策

### 1.2 系统架构图

```
Codex 父进程
  ├── factor-iterate skill    单方向迭代（Phase 1~2）
  ├── pdf-hypothesis skill    研报提取 + HO 评审（Phase 2）
  ├── factor-explore skill    多方向并行探索（Phase 3+）
  └── library-audit skill     库健康审计（Phase 4+）
        ↓
Subagent 类型（4 个）：
  ├── HG — Hypothesis Generator   结构化假设生成
  ├── HO — Hypothesis Optimizer   静态评审（不回测）
  ├── FC — Factor Coder           编码 + 执行 pipeline
  └── RC — Result Critic          诊断 + 修复决策
        ↓
Python 执行层（agents/）
  ├── codex_cli.py  — schema / run / trace-append / kb-update
  ├── runner.py      — 因子注册 + pipeline 调用
  ├── evaluator.py   — result → layered feedback
  ├── trace.py       — trace.jsonl 读写（§3.1）
  ├── kb_query.py    — KB 分层查询
  ├── kb_update.py   — KB 自动积累（§3.3）
  └── ...
        ↓
Backtest 流水线
  ├── factor/compute   — 因子计算
  ├── pipeline/steps   — step1~step10
  └── evaluation/      — 回测评估
```

### 1.3 Subagent 类型（4 个）

| Subagent | 职责 | 触发时机 | 输入 | 输出 |
|----------|------|----------|------|------|
| **HG** | 将自然语言/PDF/RC 输出转化为结构化假设 | 用户输入无 `--hypothesis` 时；RC 输出 `new_hypothesis` 时 | 自然语言文本 / `new_hypothesis` 字符串 / schema 列名 | 结构化 Hypothesis JSON |
| **HO** | **不回测**的前提下做静态专家评审 | HG 输出后；PDF-Hypothesis 生成后 | Hypothesis JSON + KB 查询结果 | 优化后的 Hypothesis JSON / 风险提示 |
| **FC** | 写因子代码 + 执行 pipeline | HO 审阅通过后；RC 诊断 repair 时 | Hypothesis / RC 修复指令 / schema | `factor.py` + `config.yaml` + `result.json` |
| **RC** | 诊断失败 + 决定 repair/abandon | Pipeline `status != pass` 时 | result.json + trace summary + KB + SOTA | Diagnosis JSON |

---

## 2. 模块接口契约

| 边界 | 提供方 | 消费方 | 形式 |
|------|--------|--------|------|
| 数据 schema | `schema.py` | HG / FC | CLI `schema` 子命令 → JSON |
| 流水线执行 | `runner.py` | `codex_cli.py` | `AutoQuantFactorRunner.run(experiment)` |
| 反馈评估 | `evaluator.py` | `codex_cli.py` | `AutoQuantFactorEvaluator.evaluate(experiment)` → `QuantFeedback` |
| Trace 读写 | `trace.py` | `codex_cli.py` | `TraceManager.append(record)` / `read_all()` |
| KB 查询 | `kb_query.py` | 父进程 / RC prompt | CLI → JSON |
| KB 更新 | `kb_update.py` | `codex_cli.py` | `KbUpdater.update_on_pass/fail(experiment)` |

---

## 3. 核心子系统设计

### 3.1 Trace 系统（`agents/trace.py`）

#### 3.1.1 设计目标

- **自动化**：`codex_cli.py run` 在 `--run-dir` 模式下自动追加 trace，替代完全手动的文件操作
- **DAG 支持**：`parent_round_id` + `branch_id` 支持线性迭代和分支探索
- **纯函数**：`TraceRecord.from_result_json()` 只接收 dict，不读取文件，便于单元测试

#### 3.1.2 TraceRecord 字段

对齐 `.codex/prompts/shared/output_formats.md` 中的 Trace JSONL Schema。关键字段：

| 字段 | 来源 | 说明 |
|------|------|------|
| `round` | 调用方传入 / `TraceManager.get_next_round()` | 轮次编号 |
| `parent_round_id` | 线性：`max_round`；分支：显式传入 | DAG 父节点 |
| `branch_id` | 默认 `"main"`，分支时 `"explore_xxx"` | 分支标识 |
| `status` | `result.json["status"]` | pass / fail / error |
| `failure_type` | `result.json["failure_type"]` | 见 failure_type 枚举 |
| `metrics` | `result.json["metrics"]` + 深层路径 | 含 `annual_icir`, `simple_sharpe`, `r2` (step8), `max_existing_corr`, `residual_icir` |
| `diagnosis` / `fix_strategy` / `fix_level` | `rc_output` | RC 诊断输出 |
| `code_summary` | 调用方传入 | 因子公式 20 词描述 |
| `tried_params` | 调用方传入 | 已尝试参数 |

#### 3.1.3 原子写

`TraceManager.append()` 使用 **tmp → replace** 模式：
1. 写入 `.trace.jsonl.tmp`
2. `os.replace()` 原子重命名
确保并发读取者不会看到半行 JSON。

#### 3.1.4 向后兼容

- 不指定 `--run-dir` 的 auto-layout 模式**不写 trace**（auto-layout 按 factor_id 组织，无 run 概念）
- Trace 只在明确指定 `--run-dir` 的迭代模式下写入

---

### 3.2 QuantFeedback 分层（`agents/evaluator.py`）

#### 3.2.1 问题

旧版 `QuantFeedback` 是 ~25 个字段的 monolithic dataclass。RC prompt 按 `failure_type` 本应只看相关字段，但当前注入完整 `result.json`，浪费 token 且降低聚焦度。

#### 3.2.2 三层拆分

按 failure_type 归属分为三层：

| 层 | 字段 | 对应 failure_type |
|----|------|------------------|
| **ExecutionFeedback** | error, traceback, code_valid, imports_valid, coverage_ratio, failed_step, failure_reason | `code_error`, `schema_error`, `execution_error`, `coverage_fail`, `config_error` |
| **EvaluationFeedback** | annual_icir, pos_ratio, turnover, max_corr, monotonicity, simple/detailed sharpe/mdd/calmar, cost_drag, ridge_tier, ridge_r2, residual_annual_icir | `neutralization_fail`, `icir_fail`, `monotonicity_fail`, `backtest_fail`, `ridge_fail`, `residual_fail`, `metrics_fail` |
| **HypothesisFeedback** | category, data_sources, construct_valid, direction_consistent | 所有（元信息层） |

#### 3.2.3 核心方法

```python
QuantFeedback.get_relevant_layer(failure_type: str | None) -> dict
```

按 `_LAYER_MAP` 返回对应层 + 顶层决策字段（decision, observation, suggestion, passed_steps, failed_step, failure_reason）。

RC prompt 注入时优先使用 `result.json["feedback"]["relevant"]`，而非完整 result.json。

#### 3.2.4 向后兼容

- `to_flat_dict()` 保持与旧版完全相同的输出格式
- `result.json` 中同时输出 `feedback.flat`（兼容旧消费者）和 `feedback.layered`（新消费者）

---

### 3.3 KB 自动积累（`agents/kb_update.py`）

#### 3.3.1 设计目标

- **自动化**：Pass/Abandon 时自动更新 `hypothesis_index.jsonl`（核心），连带更新其他 KB 文件
- **Upsert 而非 Append**：`hypothesis_index.jsonl` 和 `successful_patterns.json` 按 `factor_id` 去重更新
- **Signature 去重**：`anti_patterns.json` 按 `signature` 去重，`count` 累加

#### 3.3.2 文件更新逻辑

| 文件 | Pass 时 | Fail 时 | 去重键 |
|------|---------|---------|--------|
| `hypothesis_index.jsonl` | Upsert（更新 best_icir/best_sharpe/status） | Upsert（更新 status=fail） | `factor_id` |
| `successful_patterns.json` | 按 category 追加 | 不操作 | `factor_id` |
| `anti_patterns.json` | 不操作 | 条件更新（RC 输出 `new_anti_pattern` 非 null） | `signature` |
| `failed_attempts.jsonl` | Append（status=pass） | Append（status=fail） | N/A（append-only） |

#### 3.3.3 hypothesis_index.jsonl Upsert 逻辑

```
读取现有记录 → 查找 factor_id
  ├─ 存在 → 更新 status, best_icir=max(old, current), best_sharpe=max(old, current), ts
  └─ 不存在 → 追加新记录
```

`formula_fingerprint` 提取：优先从 `experiment.factor_code` 正则提取（`@register` 后的首行描述或函数体首行），失败 fallback 到 `factor_id`。

#### 3.3.4 anti_patterns.json 去重逻辑

```
从 rc_output 提取 new_anti_pattern（pattern, category, signature, fix）
  ├─ null → 跳过
  └─ 非 null → 在 anti_patterns[failure_type] 数组中 exact match signature
        ├─ 匹配 → count += 1, last_seen = today
        └─ 未匹配 → append 新条目（count=1, last_seen=today）
```

#### 3.3.5 触发方式

- **CLI 子命令**：`python -m agents.codex_cli kb-update --result <path> --status pass|fail [--rc-output <path>]`
- **自动触发**：`cmd_run` 新增 `--auto-kb-update` 标志，run 结束后自动调用
- 默认不启用（向后兼容），`factor-iterate` skill 迭代模式建议启用

---

## 4. Prompt 工程

### 4.1 选择性上下文注入

Prompt 模板固定（存放在 `.codex/prompts/`），但父进程按场景动态选择注入哪些 section。

**RC Prompt 条件注入规则**（按 `failure_type`）：

| failure_type | Surface Layer | Deep Layer | Structural Layer |
|--------------|--------------|------------|------------------|
| `code_error` / `schema_error` / `execution_error` | `feedback.relevant` (execution) | Last 1 round trace | — |
| `coverage_fail` / `monotonicity_fail` | `feedback.relevant` (evaluation) | Last 2 rounds trace | — |
| `icir_fail` / `backtest_fail` | `feedback.relevant` (evaluation) | Last 3 rounds + trend | Param exhaustion if 3+ rounds |
| `ridge_fail` | `feedback.relevant` (evaluation) | max_existing_factor + SOTA | — |
| `residual_fail` | `feedback.relevant` (evaluation) | Last 1 round | — |

### 4.2 输出格式

所有 subagent 输出结构化 JSON，schema 定义见 `.codex/prompts/shared/output_formats.md`：
- Hypothesis JSON Schema（HG / HO）
- HO Review JSON Schema
- Diagnosis JSON Schema（RC）
- Trace JSONL Schema
- Hypothesis Index Entry Schema

---

## 5. 数据流

### 5.1 `factor-iterate` 单轮数据流

```
[父进程] 组装 prompt（Role + Scenario + Context + Result + Trace + KB + Task）
    ↓
[FC] 生成 factor.py + config.yaml
    ↓
python -m agents.codex_cli run <factor_id> --run-dir <dir>
    ├─ runner.py → backtest pipeline step1~step10
    ├─ evaluator.py → QuantFeedback (layered)
    ├─ (if --run-dir) trace.py → auto-append trace.jsonl
    └─ result.json 输出
    ↓
[父进程] 读取 result.json
    ├─ status == pass → kb-update --status pass → 结束
    └─ status != pass → 组装 RC prompt → [RC] 诊断
         ↓
    [父进程] 解析 RC 输出 → trace-append --rc-output <path>
         ↓
    决策：repair（next round）/ abandon（kb-update --status fail）
```

### 5.2 `pdf-hypothesis` 数据流

```
PDF → MCP pdfplumber 提取 → 因子穷举 → [HO] 评审排序
    ↓
批次目录 manifest.json + hypothesis 菜单
    ↓
用户选择编号 → factor-iterate skill 读取选中 hypothesis
```

---

## 6. 演进路线

| Phase | 范围 | 触发条件 | 状态 |
|-------|------|----------|------|
| **Phase 1** | 增强 `factor-iterate` skill + KB 驱动修复 | 当前 | ✅ 已完成 |
| **Phase 1.5** | Trace DAG 落地 + Feedback 分层 + KB 自动积累 | 本设计文档 | 🚧 实施中 |
| **Phase 2** | PDF-hypothesis 完整链路 | 有研报 PDF 时 | ✅ 已完成 |
| **Phase 3** | 多方向并行探索（background exec session + 分支） | 需要同时探索 2+ 方向 | 📋 规划中 |
| **Phase 4** | 库健康审计 + Bandit 方向选择 | admitted factor > 50 | 📋 远期 |

---

## 7. 编码约定

- Python 3.11；`from __future__ import annotations` + 类型注解
- 所有文件写操作必须原子（tmp → replace）
- 单元测试覆盖每个 public method 的主路径和边界条件
- 中文注释允许，标识符必须英文
