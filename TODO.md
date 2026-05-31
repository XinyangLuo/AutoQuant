# TODO

> 临时工单池。每项完成后从本文件删除；全部完成后删除整份 TODO.md。
> 分级：P0=立刻/阻塞，P1=依赖 P0，P2=有价值但不紧急，P3=维护/锦上添花，P4=远期/想法池
> 上次整理: 2026-05-31

---

## P0

> 阻塞性，必须立刻修。


---

## P1

### Agent 系统

- [ ] **P1.A.0 Prompt 模板系统**：将 `factor-iterate.md` 内联 prompt 拆到 `.claude/prompts/`（`shared/` 公共块 + `factor_coder.md` + `result_critic.md` + `hypothesis_gen.md`）。引入 Role Anchoring + Section Labeling + SOTA 锚定 + 结构化 JSON schema 模式。参考 RD-Agent 的 Jinja2 YAML 共享块思路，AutoQuant 用 Markdown 文件级复用（够轻量）。
- [ ] **P1.A.0b Hypothesis Generation 步骤**：在 `/factor-iterate` Round loop 前增加显式假设生成。FC 先输出 hypothesis JSON（`hypothesis_text` / `expected_icir` / `category` / `data_sources` / `construction_logic`），父进程审核后再进入编码。避免"直接写代码"的盲目性。
- [ ] **P1.A.1 KB 积累 + 自动引导（Phase 2）**：**依赖 P1.A.0**（prompt 模板到位后才能稳定地查 KB 写进 prompt）。触发条件：≥20 次迭代，≥10 条反模式，≥3 条成功模式；父进程在 framing 阶段自动查 KB 引导初始代码。
- [ ] **P1.A.2 并行探索（Phase 3）**：**依赖 P2.A.1**（DAG Trace 支持分支后才能真正并行）。触发条件：Phase 2 稳定 + 单方向成功率 >20%；2 方向手动并行（不同 run dir + background）；验证 DuckDB 并发安全 + token 消耗可控。
- [ ] **P1.A.3 库审计（Phase 4）**：触发条件：admitted factor > 10；冗余/缺口/衰减检测；在 `claude_cli.py` 新增 `admit-correlations` 子命令。

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

- [ ] `SimulationConfig.benchmark` 字段实现：已定义但无功能逻辑
- [ ] `DetailedSimulator` 输入校验：检查 market_data 包含必要列
- [ ] Daily metrics fee 一致性：`detailed.py` 中 transfer_fee/stamp_duty 从 `t.amount * rate` 重算，与 `Trade.commission` 可能不一致

---

## P2

### Agent 系统

- [ ] **P2.A.1 Trace DAG 结构**：`trace.jsonl` 增加 `parent_round_id` + `branch_id` 字段，支持从任意历史节点 fork 新分支。为 Phase 3 并行探索预留数据结构，Phase 1 即可写入（branch_id 固定为 `"main"`）。
- [ ] **P2.A.2 QuantFeedback 多层拆分**：将当前单层 `QuantFeedback` 拆为 `execution` / `evaluation` / `hypothesis` 三层，对应 RD-Agent 的 Execution → Evaluation → Hypothesis Feedback 层级。`AutoQuantFactorEvaluator` 负责汇总，RC subagent 消费结构化诊断。
- [ ] **P2.A.3 Diff 注入**：RC/FC prompt 中对比上一轮代码变化（`diff` 而非完整文件），省 token 并聚焦修复。需维护 `sota_files` 对比当前 `files` 的 diff 生成逻辑。
- [x] **P2.A.4 研报 PDF → Hypothesis**：通过 `mcp-pdf` server（pdfplumber/pymupdf 文本提取）让 Agent 读取研报 PDF，穷举所有单因子并按 Sharpe/IR 排序，生成 `hypothesis.md` 中间体，经用户审阅后通过 `/factor-iterate --hypothesis` 执行。`/pdf-hypothesis` 命令 + `.mcp.json` 配置已落地。**架构约束**：文本提取，不依赖模型原生多模态。

### 性能优化

- [ ] `FactorStorage.get_factors_wide(factor_ids, start, end)`：单次 SQL 出多列宽表
- [ ] `_pooled_r2` 用 numpy 切 aligned arrays 替代 `merge + dropna`
- [ ] `momentum.py:_ewm_log_return_sum` 的 `rolling.apply` 向量化
- [ ] backfill 多因子并行：`ProcessPoolExecutor`
- [ ] `cs_mad_winsorize` / `cs_zscore` 等从 `groupby.apply` 改为 `groupby.transform` + numpy
- [ ] `get_factors_long` 把 melt 推到 SQL（`UNION ALL` per column）
- [ ] `MultiFactorStrategy._compute_ic_weights` 加缓存

### 文档

- [ ] `backtest/strategy/DESIGN.md` 更新：补充 `selection.py`、`decay`、`RiskConfig`/`BacktestConfig`
- [ ] `backtest/simulation/DESIGN.md` 更新：补充 `decile.py` 文档

### 其他

- [ ] 所有因子报告整合成 web 浏览页面

---

## P3

### Agent 系统

- [ ] **P3.A.1 DuckDB vss 向量检索**：在 `factors_pending.duckdb` 或独立表中引入 embedding + HNSW 索引（DuckDB `vss` 扩展），替代 JSON 全文匹配反模式查询。参考 RD-Agent 的 `PDVectorBase` + `UndirectedGraph` 混合检索思路，但先用轻量向量相似度（cosine）+ constraint label 过滤。
- [ ] **P3.A.2 Workspace checkpoint/rollback**：引入 `FactorWorkspace` dataclass（`dict[str, str]` 文件字典 + zip checkpoint），支持 round 间回滚。参考 RD-Agent `FBWorkspace` 的 `create_ws_ckp` / `recover_ws_ckp` 模式，但不做 Docker 隔离（本地 conda 执行是 feature）。

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

### 分钟级数据

- [ ] 分钟级数据 → 天级因子合成

### Pipeline 第二阶段

- [ ] OOS / IS 切分：IS 70% + OOS 30%，OOS IC 衰减 < 30%
- [ ] 多 universe 稳健性：全A / 沪深300 / 中证500，至少 2 个通过

### 交易模块第二阶段

- [ ] 对接 QMT / Ptrade / easytrader（实盘）
