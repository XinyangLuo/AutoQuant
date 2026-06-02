# Agent 投研系统 — 设计文档

> **版本**: 2026-06-01 v2.0
> **定位**: Claude Code subagent 模式驱动的因子迭代研究系统。Claude Code 直接承担决策、代码生成、结果分析；Python 侧只保留最小执行层。

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
- ✅ 保留 Claude Code subagent 模式（不建独立 Python agent 循环）
- ✅ 父进程（Claude Code 对话层）负责 prompt 组装和决策

### 1.2 系统架构图

```
Claude Code 父进程
  ├── /factor-iterate    单方向迭代（Phase 1~2）
  ├── /pdf-hypothesis    研报提取 + HO 评审（Phase 2）
  ├── /factor-explore    多方向并行探索（Phase 3+）
  └── /library-audit     库健康审计（Phase 4+）
        ↓
Subagent 类型（4 个）：
  ├── HG — Hypothesis Generator   结构化假设生成
  ├── HO — Hypothesis Optimizer   静态评审（不回测）
  ├── FC — Factor Coder           编码 + 执行 pipeline
  └── RC — Result Critic          诊断 + 修复决策
        ↓
Python 执行层（agents/）
  ├── claude_cli.py  — schema / run
  ├── runner.py      — 因子注册 + pipeline 调用
  ├── evaluator.py   — result → feedback
  ├── kb_query.py    — KB 分层查询（新增）
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
| **RC** | 诊断失败 + 决定 repair/abandon | Pipeline `status != 