# TODO

> 临时工单池。每项完成后从本文件删除；全部完成后删除整份 TODO.md。
> 分级：P0=立刻/阻塞，P1=依赖 P0，P2=有价值但不紧急，P3=维护/锦上添花，P4=远期/想法池
> 上次整理: 2026-05-24

---

## P0

### Barra L1 收尾

- [ ] **P0.1** smoke test 一个 alpha 走 `barra_ind_size` pipeline，验证 7/7 admitted 的 Barra L1 能被读出来做中性化

### 测试覆盖

- [ ] **P0.2** data 模块测试：multi-type fetch + 5-列 PK + snapshot 行为
- [ ] **P0.3** transforms 测试：`single_quarter` / `ttm` / `yoy` 三个助手函数（函数已实现，缺测试覆盖）
- [ ] **P0.4** strategy 模块测试：`SingleFactorStrategy` + `MultiFactorStrategy` 基础路径

### 因子挖掘 pipeline 剩余项

- [ ] **P0.5** 集成测试：CLI step1~step9 顺序调用 + state JSON 累积验证
- [ ] **P0.6** 端到端验证：用一个因子跑通全链路
- [ ] **P0.7** `run-all` 中 retry 逻辑落地（step6/7 失败后自动调参重试，state.retry_count/retry_params 已定义但从未写入）

### 缺失基础设施

- [ ] **P0.8** `backtest/data/__init__.py`：模块缺少公共 API 入口
- [ ] **P0.9** `backtest/data/index_fetcher.py` + `backfill_indices.py`：evaluation 模块的 benchmark 功能依赖这两个文件（`benchmark.py` 报错信息已引用不存在的脚本）
- [ ] **P0.10** strategy CLI 入口：`python -m backtest.strategy.run --config strategy_config.yaml`（`run.py` / `__main__.py` 不存在）

---

## P1

### Agent 因子投研系统（`agents/rdagent/`）

> **代码已落地。** 7 个 Phase 全部完成实现。
> 准入策略：半自动 —— Agent 生成候选列表和审核报告，人工最终确认后 `admit()`。
> 前置依赖：P0.5~P0.7（pipeline 集成测试 + retry 逻辑）。

- [x] **P1.1** Phase 1: 复制 rdagent/core 抽象基类（Scenario / Proposal / Experiment / Evaluator / Trace / KnowledgeBase）
- [x] **P1.2** Phase 2: 实现 `AShareQuantScenario` + Prompt 模板（scenario_desc.md）
- [x] **P1.3** Phase 3: 实现 `AutoQuantFactorExperiment` + `AutoQuantFactorRunner`（对接 compute / evaluate / strategy / simulation / evaluation）
- [x] **P1.4** Phase 4: 实现 `AutoQuantFactorEvaluator`（指标 → `QuantFeedback`）
- [x] **P1.5** Phase 5: 实现 `AutoQuantFactorHypothesisGen` + `Hypothesis2Experiment` + 4 个 Prompt 模板
- [x] **P1.6** Phase 6: 实现 `AShareKnowledgeBase`（经验积累 + 相似案例检索）
- [x] **P1.7** Phase 7: 实现主循环 `run.py` + CLI 入口 + 审核报告生成
- [ ] **P1.8** 集成测试：用已知简单因子验证 Runner + Evaluator 全流程
- [ ] **P1.9** 端到端测试：跑一次 3-round Agent 循环，验证假设 → 代码 → 回测 → 反馈链路
- [ ] **P1.10** PDF 研报作为种子输入：Claude 多模态 API 读取研报 → 提取因子假设 → 作为 Round-1 seed（`--seed-pdf` / `--seed-pages` CLI）
- [ ] **P1.11** 文集/批量研报输入：支持目录批量读取多篇研报，提取多因子想法队列逐个跑 Agent 循环（`--seed-dir` CLI）

### 基础设施

- [ ] `pyproject.toml` 落地：便于 `pip install -e .`
- [x] `environment.yml` 已补 `anthropic` SDK（Agent 投研依赖）
- [ ] `environment.yml` 其他缺失依赖：ruff/black/matplotlib/httpx/lxml/feedparser
- [ ] `allow_short` 默认值改 `False`：A 股不支持做空

### 仿真模块补全

- [ ] **P1.12** benchmark 字段实现：`SimulationConfig.benchmark` 已定义但无功能逻辑
- [x] **P1.13** T+1 结算：日频调仓天然满足（T 日收盘算因子 → T+1 开盘调仓），无需额外逻辑
- [ ] **P1.14** DetailedSimulator 输入校验：检查 market_data 包含 `open/close/low/high/limit_up/limit_down` 列
- [ ] **P1.15** Daily metrics fee 一致性：`detailed.py` 中 transfer_fee/stamp_duty 从 `t.amount * rate` 重算，与 `Trade.commission` 可能不一致

---

## P2

### 性能优化

- [ ] `FactorStorage.get_factors_wide(factor_ids, start, end)`：单次 SQL 出 7 列对齐宽表，消除 Ridge check 6× DuckDB 往返 + ~5× 1.4GB 峰值
- [ ] `_pooled_r2` 用 numpy 切 aligned arrays 替代 `merge + dropna` 双拷贝（依赖 `get_factors_wide`）
- [ ] `compute.py` 财务因子 panel 拼接走流式（依赖 `get_fina_snapshot_range`，已实现）
- [ ] `momentum.py:_ewm_log_return_sum` 的 `rolling.apply` 用 `sliding_window_view` 向量化
- [ ] backfill 多因子并行：`ProcessPoolExecutor` 并发跑独立因子
- [ ] `cs_mad_winsorize` / `cs_zscore` 等从 `groupby.apply` 改为 `groupby.transform` + numpy 直算
- [ ] Storage 共用底座：`_quote_ident` / `_upsert` / `_registered` 抽到共享模块，`FactorStorage` 与 `MarketStorage` 共用 DuckDB 底座
- [ ] `get_factors_long` 把 melt 推到 SQL（`UNION ALL` per column），避免宽表全量 melt 到内存
- [ ] `MultiFactorStrategy._compute_ic_weights` 加缓存：当前每因子 × 每 rebalance date 调用 `evaluate()`，O(N×D) 无缓存

### 因子库可视化

- [ ] 所有因子报告整合成 web 浏览页面

### 策略模块文档同步

- [ ] `backtest/strategy/DESIGN.md` 更新：P0-3 checklist 状态已过期（代码已改但文档未勾），补充 `selection.py`、`decay`、`RiskConfig`/`BacktestConfig` 的文档
- [ ] `backtest/simulation/DESIGN.md` 更新：补充 `decile.py` 文档

---

## P3

### 交易模块（第一阶段：信号推送 + 仓位跟踪）

- [ ] 推送渠道选型（企微 / 飞书 / Server酱 / 邮件）
- [ ] 信号渲染：策略信号 → 可读推送消息
- [ ] 仓位 CLI：手动录入/编辑本地持仓 YAML

### Evaluation 模块增强

- [ ] 个股贡献 top/bottom 10
- [ ] 行业归因（依赖 sw_industry）
- [ ] 多策略对比
- [ ] 滚动 IS/OOS
- [ ] Brinson 归因（依赖 sw_industry + index_members）

### 代码清理

- [ ] `backtest/strategy/neutralize.py`：已标记 deprecated 但仍包含完整实现，确认无调用后删除或恢复
- [ ] `backtest/evaluation/metrics.py` docstring 修正：`CLAUDE.md` → `DESIGN.md`

---

## P4

### 数据模块远期

- [ ] 分钟级数据：parquet 格式设计与接入
- [ ] 分钟级数据 → 天级因子合成（依赖上一项）

### 因子挖掘 pipeline 第二阶段

- [ ] OOS / IS 切分：IS 70% + OOS 30%，OOS IC 衰减 < 30%
- [ ] 多 universe 稳健性：全A / 沪深300 / 中证500，至少 2 个通过

### Agent 投研系统第二阶段

- [ ] 文档解析方案（unstructured / PyMuPDF / Claude 多模态）
- [ ] 网页抓取方案（feedparser / Playwright / httpx+bs4）
- [ ] 向量检索（看因子库规模）
- [ ] 多因子组合策略迭代
