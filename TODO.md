# TODO

> 临时工单池。每项完成后从本文件删除；全部完成后删除整份 TODO.md。
> 分级：P0=立刻/阻塞，P1=依赖 P0，P2=有价值但不紧急，P3=维护/锦上添花，P4=远期/想法池
> 上次整理: 2026-05-30

---

## P0

> 严格「立刻 / 阻塞其他工作」。

- [ ] **P0.1** `run-all` 中 Agent 驱动 retry 落地：step6/7 失败后，Agent 分析 feedback 决定调参策略（如放宽 top_pct、缩短 horizon），通过 step5 override 重新生成信号再跑，最多 3 次。`state.retry_count` / `retry_params` 已定义但从未写入
- [x] **P0.2** **残差入库因子的 DAG 回补**：step8 R² 超标 + step9 残差 ICIR 通过 → 因子以残差值（per-date Ridge 剥离全部已入库因子）入库。这类因子在冷启动/日更时必须按依赖拓扑排序——先算 Barra L1（无依赖），再算直接依赖它们的残差因子，再算二层依赖（残差因子可能被后续残差因子依赖，形成 chain）。已实现：
  - registry 记录每个因子的 `depends_on: [factor_id, ...]` 依赖列表 + `admission_mode`
  - 回补/更新模块支持拓扑执行（`graphlib.TopologicalSorter` / `backtest.factor.dag`）
  - 残差值写入时标记 `admission_mode=residual` 以便下游区分

---

## P1

### Multi-Agent 自动因子挖掘系统

> 实施计划：[`agents/PLAN.md`](agents/PLAN.md)（渐进式，每个 Phase 有明确触发条件）
> 旧版设计（已废弃）：[`.claude/plans/desktop-rd-agent-main-rd-agent-claude-s-typed-crab.md`](../.claude/plans/desktop-rd-agent-main-rd-agent-claude-s-typed-crab.md)

- [x] **P1.A.1 KB 地基（Phase 1）**：创建 `results/agent/knowledge_base/` + 3 个空 schema 文件（`anti_patterns.json`、`successful_patterns.json`、`run_index.jsonl`）；无历史 trace 可 bootstrap，从空开始积累
- [x] **P1.A.2 增强单 agent（Phase 1）**：修改 `.claude/commands/factor-iterate.md`，在 fail 后启动 RC subagent（Agent tool）诊断 + 查 KB；端到端验证待下次实际运行时完成
- [ ] **P1.A.3 KB 积累 + 自动引导（Phase 2）**：触发条件：≥20 次迭代，≥10 条反模式，≥3 条成功模式；父进程在 framing 阶段自动查 KB 引导初始代码；RC prompt 抽到 `.claude/prompts/result_critic.md`
- [ ] **P1.A.4 并行探索（Phase 3）**：触发条件：Phase 2 稳定 + 单方向成功率 >20%；2 方向手动并行（不同 run dir + background）；验证 DuckDB 并发安全 + token 消耗可控
- [ ] **P1.A.5 库审计（Phase 4）**：触发条件：admitted factor > 10；冗余/缺口/衰减检测；在 `claude_cli.py` 新增 `admit-correlations` 子命令

### 测试覆盖

- [ ] 单模块测试：Barra L1 smoke test（`barra_ind_size` pipeline 端到端）、data 模块 multi-type fetch + snapshot、transforms（`single_quarter` / `ttm` / `yoy`）
- [ ] 策略模块测试：`SingleFactorStrategy` + `MultiFactorStrategy` 基础路径
- [ ] Pipeline 集成测试：step1~step9 顺序调用 + state JSON 累积验证，用一个因子跑通全链路

### 基础设施补全

- [ ] `backtest/data/__init__.py`：模块缺少公共 API 入口
- [ ] `backtest/data/backfill_indices.py` standalone CLI：benchmark 报错信息仍引用旧路径，需更新
- [ ] Strategy CLI 入口：`python -m backtest.strategy.run --config strategy_config.yaml`
- [ ] `pyproject.toml` 落地：`pip install -e .`
- [ ] `environment.yml` 补全缺失依赖：ruff/black/matplotlib/httpx/lxml/feedparser
- [ ] `allow_short` 默认值改 `False`：A 股不支持做空（已下沉到 `config.yaml`）
- [ ] **分钟级数据 fetcher**：`backtest/data/fetcher/minute_fetcher.py`（Tushare `pro_bar` 1min/5min 封装，单股长区间获取以应对 1 次/分钟速率限制）
- [ ] **分钟级数据 backfill**：`backtest/data/backfill/minute.py`（全市场历史回填，按日期分区 parquet，断点续传）
- [ ] **分钟级数据 update**：`backtest/data/update_minute.py`（日更增量，扫描已有日期自动补新）
- [ ] **分钟级数据读取 API**：`get_minute_bars(symbols, start, end, freq)`（pyarrow.dataset 按日期分区过滤）

### 仿真引擎补全

- [ ] `SimulationConfig.benchmark` 字段实现：已定义但无功能逻辑
- [ ] `DetailedSimulator` 输入校验：检查 market_data 包含 `open/close/low/high/limit_up/limit_down` 列
- [ ] Daily metrics fee 一致性：`detailed.py` 中 transfer_fee/stamp_duty 从 `t.amount * rate` 重算，与 `Trade.commission` 可能不一致

---

## P2

### 性能优化

- [ ] `FactorStorage.get_factors_wide(factor_ids, start, end)`：单次 SQL 出多列宽表，消除 Ridge check 多次 DuckDB 往返
- [ ] `_pooled_r2` 用 numpy 切 aligned arrays 替代 `merge + dropna` 双拷贝
- [ ] `compute.py` 财务因子 panel 拼接走流式（依赖 `get_fina_snapshot_range`，已实现）
- [ ] `momentum.py:_ewm_log_return_sum` 的 `rolling.apply` 向量化
- [ ] backfill 多因子并行：`ProcessPoolExecutor` 并发跑独立因子
- [ ] `cs_mad_winsorize` / `cs_zscore` 等从 `groupby.apply` 改为 `groupby.transform` + numpy 直算
- [ ] Storage 共用底座：`_quote_ident` / `_upsert` / `_registered` 抽到共享模块
- [ ] `get_factors_long` 把 melt 推到 SQL（`UNION ALL` per column）
- [ ] `MultiFactorStrategy._compute_ic_weights` 加缓存：当前每因子 × 每 rebalance date 调用 `evaluate()`，O(N×D) 无缓存

### 因子库可视化

- [ ] 所有因子报告整合成 web 浏览页面

### 文档同步

- [ ] `backtest/strategy/DESIGN.md` 更新：补充 `selection.py`、`decay`、`RiskConfig`/`BacktestConfig`
- [ ] `backtest/simulation/DESIGN.md` 更新：补充 `decile.py` 文档

---

## P3

### 交易模块（信号推送 + 仓位跟踪）

- [ ] 推送渠道选型（企微 / 飞书 / Server酱 / 邮件）
- [ ] 信号渲染：策略信号 → 可读推送消息
- [ ] 仓位 CLI：手动录入/编辑本地持仓 YAML

### Evaluation 增强

- [ ] 个股贡献 top/bottom 10
- [ ] 行业归因（依赖 sw_industry）
- [ ] 多策略对比
- [ ] 滚动 IS/OOS
- [ ] Brinson 归因（依赖 sw_industry + index_members）

### 代码清理

- [ ] `backtest/strategy/neutralize.py`：已标记 deprecated 但仍包含完整实现，确认无调用后删除
- [ ] `backtest/evaluation/metrics.py` docstring 修正：`CLAUDE.md` → `DESIGN.md`

---

## P4

### 数据模块远期

- [x] 分钟级数据：parquet 格式设计与接入（方案已定，fetcher/backfill/update 落地中，见 P1 基础设施补全）
- [ ] 分钟级数据 → 天级因子合成

### 因子挖掘 pipeline 第二阶段

- [ ] OOS / IS 切分：IS 70% + OOS 30%，OOS IC 衰减 < 30%
- [ ] 多 universe 稳健性：全A / 沪深300 / 中证500，至少 2 个通过

### 交易模块第二阶段
