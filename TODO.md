# TODO

> 临时工单池。每项完成后从本文件删除；全部完成后删除整份 TODO.md。
> 分级：P0=立刻/阻塞，P1=依赖 P0，P2=有价值但不紧急，P3=维护/锦上添花，P4=远期/想法池
> 上次整理: 2026-05-25

---

## P0

> 严格「立刻 / 阻塞其他工作」。

- [ ] **P0.1** `run-all` 中 Agent 驱动 retry 落地：step6/7 失败后，Agent 分析 feedback 决定调参策略（如放宽 top_pct、缩短 horizon），通过 step5 override 重新生成信号再跑，最多 3 次。`state.retry_count` / `retry_params` 已定义但从未写入
- [ ] **P0.2** Agent 同方向迭代机制：区分「代码错误」vs「指标不达标」。代码错误时要求 LLM 修复同一思路的代码；指标不达标时在同一思路上调参/改构造，而非每次换全新方向（当前 Round 1 代码 `abs_` 未导入后，Round 2~7 全部偏离 seed）
- [ ] **P0.3** Agent Feedback 深度化：代码错误时提供 **行号 + 修复建议**（而非仅 `name 'abs_' is not defined`）；指标不达标时分析 **方向是否正确、哪个子项拖后腿**（而非仅 `RankICIR is low`）
- [ ] **P0.4** Agent Prompt 列名精准注入：根据 hypothesis 的 `data_sources` 只注入相关列名，而非列出全部 50+ 列，减少 LLM hallucinate 列名（如 `buy_sm` → `mf_buy_sm_amount`、`ts_zscore` → `z_score`）
- [ ] **P0.5** Agent Trace 增强记忆：history 中增加 **代码摘要 + 关键错误信息 + 已尝试过的参数组合**，防止重复踩坑，让 LLM 能从过往失败中学习

---

## P1

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

### Agent 测试与扩展

- [ ] Runner + Evaluator 集成测试：用已知简单因子验证全流程
- [ ] 端到端 Agent 循环测试：跑一次 3-round 验证假设 → 代码 → 回测 → 反馈链路
- [ ] PDF 研报种子输入：`--seed-pdf` / `--seed-pages` CLI 读取研报 → 提取因子假设
- [ ] 批量研报输入：`--seed-dir` CLI 支持目录批量读取多篇研报

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

### Agent 投研系统第二阶段

- [ ] 文档解析方案（unstructured / PyMuPDF / Claude 多模态）
- [ ] 网页抓取方案（feedparser / Playwright / httpx+bs4）
- [ ] 向量检索（看因子库规模）
- [ ] 多因子组合策略迭代
