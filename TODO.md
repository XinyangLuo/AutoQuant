# TODO

> 临时工单池。每项完成后从本文件删除；全部完成后删除整份 TODO.md。
> 分级：P0=立刻/阻塞，P1=依赖 P0，P2=有价值但不紧急，P3=维护/锦上添花，P4=远期/想法池
> 创建时间: 2026-05-18
> 上次整理: 2026-05-22

---

## P0

### 因子挖掘流程优化 — 剩余项

- [ ] 集成测试：CLI step1~step9 顺序调用 + state JSON 累积验证
- [ ] 端到端验证：用已有 Barra L1 因子跑通全链路
- [ ] retry 逻辑在 `run-all` 中落地（step6/7 失败后自动调参重试）
- [ ] Agent stub (`_agent_stub.py`) 从确定性 fallback 替换为实际 Agent 调用接口

---

## P1

### 文档更新，目录整理
- 整理下目录结构，使其更合理
- 更新所有文档使其与代码匹配，删掉已经废弃的功能

### 基础设施

- `pyproject.toml` 落地：便于 `pip install -e .`
- `environment.yml` 完善：补充 ruff/black/matplotlib/httpx/lxml/feedparser 等缺失依赖
- CLI 入口：`python -m backtest.strategy.run --config strategy_config.yaml`
- `allow_short` 默认值改 `False`：A 股不支持做空

### 数据模块扩展

- 指数成分股表：`index_members`(symbol, index_code, trade_date, weight)
- `get_fina_snapshot_range(start, end)`：区间批量 join，替换当前每个 trade_date 单查再 concat 的模式

---

## P2

### 性能优化

- `FactorStorage.get_factors_wide(factor_ids, start, end)`：单次 SQL 出 7 列对齐宽表，消除 Ridge check 6× DuckDB 往返 + ~5× 1.4GB 峰值
- `_pooled_r2` 用 numpy 切 aligned arrays 替代 `merge + dropna` 双拷贝（依赖 `get_factors_wide`）
- `compute.py` 财务因子 panel 拼接走流式（或 `get_fina_snapshot_range`）
- `momentum.py:_ewm_log_return_sum` 的 `rolling.apply` 用 `sliding_window_view` 向量化
- backfill 多因子并行：`ProcessPoolExecutor` 并发跑独立因子
- `cs_mad_winsorize` / `cs_zscore` 等从 `groupby.apply` 改为 `groupby.transform` + numpy 直算
- Storage 共用底座：`_quote_ident` / `_upsert` / `_registered` 抽到共享模块，`FactorStorage` 与 `MarketStorage` 共用 DuckDB 底座
- `get_factors_long` 把 melt 推到 SQL（`UNION ALL` per column），避免宽表全量 melt 到内存

### 因子库可视化

- 所有因子报告整合成 web 浏览页面。详见 PLAN.md §5。

---

## P3

### Agent 投研系统（`agents/rdagent/`）

- Phase 1: 复制 `rdagent/core/` 抽象基类到 `agents/rdagent/core/`
- Phase 2: 实现 `AShareQuantScenario` + Prompt 模板
- Phase 3: 实现 `AutoQuantFactorExperiment` + `AutoQuantFactorRunner`
- Phase 4: 实现 `AutoQuantFactorEvaluator`（指标转换）
- Phase 5: 实现 `HypothesisGen` + `Hypothesis2Experiment`
- Phase 6: 实现 `AShareKnowledgeBase`
- Phase 7: 实现主循环 `run.py` + 集成测试

### 交易模块（第一阶段：信号推送 + 仓位跟踪）

- 推送渠道选型（企微 / 飞书 / Server酱 / 邮件）
- 信号渲染：策略信号 → 可读推送消息
- 仓位 CLI：手动录入/编辑本地持仓 YAML

### Evaluation 模块增强

- 个股贡献 top/bottom 10
- 行业归因（依赖 sw_industry）
- 多策略对比
- 滚动 IS/OOS
- Brinson 归因（依赖 sw_industry + index_members）

### 数据模块远期

- 分钟级数据：parquet 格式设计与接入
- 分钟级数据 → 天级因子合成（PLAN.md §6，依赖上一项）

### 因子挖掘 pipeline 第二阶段

- OOS / IS 切分：IS 70% + OOS 30%，OOS IC 衰减 < 30%
- 多 universe 稳健性：全A / 沪深300 / 中证500，至少 2 个通过

### Agent 投研系统第二阶段

- 文档解析方案（unstructured / PyMuPDF / Claude 多模态）
- 网页抓取方案（feedparser / Playwright / httpx+bs4）
- 向量检索（看因子库规模）
- 多因子组合策略迭代
