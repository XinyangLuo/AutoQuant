# TODO

> 临时工单池。每项完成后从本文件删除；全部完成后删除整份 TODO.md。
> 分级：P0=立刻/阻塞，P1=依赖 P0，P2=有价值但不紧急，P3=维护/锦上添花，P4=远期/想法池
> 上次整理: 2026-05-31

---

## P0

> 阻塞性，必须立刻修。

- [x] **P0.1** Pipeline 自动 retry：step6/7 失败后自动以渐进宽松策略参数重试 step5→6→7，最多 3 次（double decay → widen top_k → 5D rebalance）。`state.retry_count` / `retry_params` 已写入，报告展示重试次数和最终参数。

---

## P1

### Agent 系统

- [ ] **P1.A.1 KB 积累 + 自动引导（Phase 2）**：触发条件：≥20 次迭代，≥10 条反模式，≥3 条成功模式；父进程在 framing 阶段自动查 KB 引导初始代码；RC prompt 抽到 `.claude/prompts/result_critic.md`
- [ ] **P1.A.2 并行探索（Phase 3）**：触发条件：Phase 2 稳定 + 单方向成功率 >20%；2 方向手动并行（不同 run dir + background）；验证 DuckDB 并发安全 + token 消耗可控
- [ ] **P1.A.3 库审计（Phase 4）**：触发条件：admitted factor > 10；冗余/缺口/衰减检测；在 `claude_cli.py` 新增 `admit-correlations` 子命令

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
