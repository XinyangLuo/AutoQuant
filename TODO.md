# TODO

> 临时工单池。每项完成后从本文件删除；全部完成后删除整份 TODO.md。
> 分级：P0=立刻/阻塞，P1=依赖 P0，P2=有价值但不紧急，P3=维护/锦上添花，P4=远期/想法池
> 上次整理: 2026-06-01

---

## P0

> 阻塞性，必须立刻修。


---

## P1

### Agent 系统核心架构

- [ ] **P1.A.0 Prompt 模板系统**：将 `factor-iterate.md` 内联 prompt 拆到 `.claude/prompts/`（`shared/` 公共块 + `factor_coder.md` + `result_critic.md` + `hypothesis_gen.md` + `hypothesis_optimizer.md`）。引入 Role Anchoring + Section Labeling + 条件注入模式。
- [ ] **P1.A.1 Trace DAG 字段**：`trace.jsonl` 增加 `parent_round_id` + `branch_id`，Phase 1.5 固定为 `"main"` / `round-1`，保持向后兼容，为 Phase 3 分支预留。
- [ ] **P1.A.2 条件注入机制**：父进程按 `failure_type` 动态组装 RC prompt（code_error 只需 last round，icir_fail 需要 trend + SOTA）。定义在 `shared/context_sections.md`。
- [ ] **P1.A.3 Hypothesis Generator (HG)**：新增 HG subagent，将自然语言/PDF/RC `new_hypothesis` 转化为结构化 Hypothesis JSON（含 5 维自评）。
- [ ] **P1.A.4 Hypothesis Optimizer (HO)**：新增 HO subagent，在**不回测**前提下做静态专家评审（查重/反模式/参数建议/数据可行性/经济学逻辑）。
- [ ] **P1.A.5 KB 分层查询层**：新增 `agents/kb_query.py` + `agents/knowledge_base/hypothesis_index.jsonl`。L1（热，必带）/ L2（温，过滤后注入）/ L3（冷，keyword 查询不进 prompt）。
- [ ] **P1.A.6 `/factor-iterate` 无参模式**：当用户输入自然语言（无 `--hypothesis`）时，自动走 HG → HO → FC 路径。

### 测试覆盖

- [ ] Barra L1 smoke test（`barra_ind_size` pipeline 端到端）
- [ ] Data 模块 multi-type fetch + snapshot
- [ ] Transforms 单测（`single_quarter` / `ttm` / `yoy`）
- [ ] 策略模块测试：`SingleFactorStrategy` + `MultiFactorStrategy` 基础路径
- [ ] Pipeline 集成测试：step1~step9 顺序调用 + state JSON 累积验证

### 基础设施

- [ ] `backtest/data/backfill_indices.py` standalone CLI：benchmark 报错信息仍引用旧路径
- [ ] 分钟级数据：fetcher → backfill → update → 读取 API（pyarrow.dataset 按日期分区）

### 仿真引擎

- [x] `SimulationConfig.benchmark` 字段实现：benchmark 对比逻辑已下沉到 evaluation 层。
- [ ] `DetailedSimulator` 输入校验：检查 market_data 包含必要列
- [ ] Daily metrics fee 一致性：`detailed.py` 中 transfer_fee/stamp_duty 从 `t.amount * rate` 重算，与 `Trade.commission` 可能不一致
- [ ] **2.10** 现金不足时权重优先分配：`detailed.py:513-540` 等比例缩减改为按目标权重排序优先分配（待讨论）
- [ ] **2.11** 滑点模型：`SimulationConfig` 增加 `slippage_bps` 参数，executor 成交价按滑点调整，默认值为 0（待讨论）

---

## P2

### Agent 系统增强

- [ ] **P2.A.1 PDF-Hypothesis HO 联动**：`/pdf-hypothesis` Step 5 后增加 HO 批量评审，排名表附加 `ho_review` 字段（重复风险/反模式警告/参数建议）。
- [ ] **P2.A.2 Diff 注入**：RC/FC prompt 中对比上一轮代码变化（`diff` 而非完整文件），省 token 并聚焦修复。
- [ ] **P2.A.3 QuantFeedback 多层拆分**：将当前单层 `QuantFeedback` 拆为 `execution` / `evaluation` / `hypothesis` 三层。
- [ ] **P2.A.4 RC 输出 `new_hypothesis`**：RC 诊断可输出新方向假设，触发 HG → HO → 新 branch 流程。
- [ ] **P2.A.5 KB 自动积累**：Pass/Abandon 时自动更新 `hypothesis_index.jsonl`，HO 查重精度提升。

### 性能优化（P0/P1 见 REVIEW_ISSUES.md §三）

- [ ] `FactorStorage.get_factors_wide(factor_ids, start, end)`：单次 SQL 出多列宽表
- [ ] `_pooled_r2` 用 numpy 切 aligned arrays 替代 `merge + dropna`
- [ ] `momentum.py:_ewm_log_return_sum` 的 `rolling.apply` 向量化
- [ ] backfill 多因子并行：`ProcessPoolExecutor`
- [ ] `cs_mad_winsorize` / `cs_zscore` 等从 `groupby.apply` 改为 `groupby.transform` + numpy
- [ ] `get_factors_long` 把 melt 推到 SQL（`UNION ALL` per column）
- [ ] `MultiFactorStrategy._compute_ic_weights` 加缓存
- [ ] **P2** `factor/compute.py` 外层 1 年 chunk 与 SQL 内层 6 月 chunk 嵌套冗余 → 1.5–2×
- [ ] **P2** `pipeline/steps.py` `step2` `_max_industry_corr` 逐日 `pd.get_dummies` → 5–10×
- [ ] **P2** `factor/transforms.py` `z_score/ts_ir` 对 index 重复 sort → 1.2–1.3×

### 文档

- [ ] `backtest/strategy/DESIGN.md` 更新：补充 `selection.py`、`decay`、`RiskConfig`/`BacktestConfig`
- [ ] `backtest/simulation/DESIGN.md` 更新：补充 `decile.py` 文档

### 其他

- [ ] 所有因子报告整合成 web 浏览页面

---

## P3

### Agent 系统

- [ ] **P3.A.1 DuckDB vss 向量检索**：在 `factors_pending.duckdb` 或独立表中引入 embedding + HNSW 索引（DuckDB `vss` 扩展），用于 L3 冷数据的语义查重。当前 L3 用 keyword 过滤足够，vss 是备选加速方案。
- [ ] **P3.A.2 Workspace checkpoint/rollback**：引入 `FactorWorkspace` dataclass（`dict[str, str]` 文件字典 + zip checkpoint），支持 round 间回滚。
- [ ] **P3.A.3 并行探索（Phase 3 初版）**：父进程手动指定 2 个方向，每个方向独立跑 `/factor-iterate`（不同 run dir + `run_in_background`）。验证 DuckDB 并发安全 + token 消耗可控。`branch_id` 开始真正使用。

### 交易模块

- [ ] 推送渠道选型（企微 / 飞书 / Server酱 / 邮件）
- [ ] 信号渲染：策略信号 → 可读推送消息
- [ ] 仓位 CLI：手动录入/编辑本地持仓 YAML

### Evaluation 增强

- [ ] 个股贡献 top/bottom 10
- [ ] 行业归因（依赖 sw_industry）
- [ ] 多策略对比、滚动 IS/OOS
- [ ] Brinson 归因

### 代码清理

- [ ] `backtest/strategy/neutralize.py`：已标记 deprecated，确认无调用后删除
- [ ] `backtest/evaluation/metrics.py` docstring 修正：`CLAUDE.md` → `DESIGN.md`

### 工程

- [ ] `pyproject.toml` 落地：`pip install -e .`
- [ ] `environment.yml` 补全缺失依赖

---

## P4

### Agent 系统远期

- [ ] **P4.A.1 库审计（Phase 4）**：触发条件 admitted factor > 10；冗余/缺口/衰减检测；audit → 发现缺口 → 自动生成 HG 输入启动迭代。
- [ ] **P4.A.2 Bandit 方向选择**：当方向数 > 5 且手动选择不可持续时，引入 bandit 自动选择探索方向。
- [ ] **P4.A.3 全自动触发**：定时扫描覆盖缺口 → 自动启动 HG → HO → FC 循环，无需人工输入。

### 分钟级数据

- [ ] 分钟级数据 → 天级因子合成

### Pipeline 第二阶段

- [ ] OOS / IS 切分：IS 70% + OOS 30%，OOS IC 衰减 < 30%
- [ ] 多 universe 稳健性：全A / 沪深300 / 中证500，至少 2 个通过

### 交易模块第二阶段

- [ ] 对接 QMT / Ptrade / easytrader（实盘）
