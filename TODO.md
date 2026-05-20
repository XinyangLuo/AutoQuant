# TODO

> 临时工单池。每项完成后从本文件删除；全部完成后删除整份 TODO.md。
> 分级：P0=立刻/阻塞，P1=依赖 P0，P2=有价值但不紧急，P3=维护/锦上添花，P4=远期/想法池
> 创建时间: 2026-05-18

---

## P0

### 因子算子库（`backtest/factor`）

- [ ] 参考 https://platform.worldquantbrain.com/learn/operators 实现一线常用算子，避免每次因子代码重复构造样板。详见 PLAN.md §1。

### Barra 风险模型（`backtest/factor`）

- [ ] 删除旧因子表，新建以 Barra 风格因子为底的正式因子表
- [ ] 三级因子计算：Size(LNCAP) / Beta(BETA) / Momentum(RSTR) / Value(BTOP+ETOP+DTOP) / Quality(ROA+GP+AGRO) / Liquidity(STOM) / Growth(EGRO)，公式见 PLAN.md §2.1
- [ ] 合成流程：MAD 去极值（median ± 3×1.4826×MAD clip）→ 申万一级行业中位数填充 → 截面 z-score → 等权合成二级/一级
- [ ] 财务因子防穿越，缺失值处理
- [ ] 因子中性化：因子 ~ 行业哑变量 + Size_z 截面 OLS，取残差再标准化，替代原分层中性化（PLAN.md §2.2）
- [ ] 入库检查：与剩余 6 个一级因子 Ridge Regression，按 R² 判 pure alpha / smart beta（PLAN.md §2.3）

### 交易日历（`backtest/data` + `backtest/strategy` + `backtest/simulation`）

- [ ] 表记下周/月第一个交易日，支撑周频/月频因子策略调仓。详见 PLAN.md §3。

## P1

### 因子挖掘流程优化（依赖 P0，配合 rd agent）

- [ ] 串行 step1~step9 pipeline + 每步淘汰标准：截面缺失率 < 30% → 中性化后与 size/industry corr < 0.05 且与已有因子 max corr < 0.5 → 离线 ICIR（日频 1D/5D，月频 1M 阈值见 PLAN）→ 分 10 组单调性 > 0.7 → 默认策略 top10% decay=5 → 向量化简单回测（日频 Sharpe > 0.8 / 月频 > 1.0）→ 详细回测（日频 Sharpe > 0.4 / 月频 > 0.6）→ Ridge R² 入库分流 → markdown 报告生成。详见 PLAN.md §4。

## P2

### 因子库可视化

- [ ] 所有因子报告整合成 web 浏览页面。详见 PLAN.md §5。

## P3

### 基础设施

- [ ] `pyproject.toml` 落地：便于 `pip install -e .`
- [ ] `environment.yml` 完善：补充 ruff/black/matplotlib/httpx/lxml/feedparser 等缺失依赖
- [ ] CLI 入口：`python -m backtest.strategy.run --config strategy_config.yaml`
- [ ] `allow_short` 默认值改 `False`：A 股不支持做空

### Agent 投研系统（`agents/rdagent/DESIGN.md`）

- [ ] Phase 1: 复制 `rdagent/core/` 抽象基类到 `agents/rdagent/core/`
- [ ] Phase 2: 实现 `AShareQuantScenario` + Prompt 模板
- [ ] Phase 3: 实现 `AutoQuantFactorExperiment` + `AutoQuantFactorRunner`
- [ ] Phase 4: 实现 `AutoQuantFactorEvaluator`（指标转换）
- [ ] Phase 5: 实现 `HypothesisGen` + `Hypothesis2Experiment`
- [ ] Phase 6: 实现 `AShareKnowledgeBase`
- [ ] Phase 7: 实现主循环 `run.py` + 集成测试

### 交易模块（第一阶段：信号推送 + 仓位跟踪）

- [ ] 推送渠道选型（企微 / 飞书 / Server酱 / 邮件）
- [ ] 信号渲染：策略信号 → 可读推送消息
- [ ] 仓位 CLI：手动录入/编辑本地持仓 YAML

### 数据模块扩展

- [ ] 指数成分股表：`index_members`(symbol, index_code, trade_date, weight)
- [ ] 分钟级数据：parquet 格式设计与接入
- [ ] 分钟级数据 → 天级因子合成（PLAN.md §6，依赖上一项）

### Evaluation 模块增强（`backtest/evaluation/DESIGN.md`）

- [ ] 个股贡献 top/bottom 10
- [ ] 行业归因（依赖 sw_industry）
- [ ] 多策略对比
- [ ] 滚动 IS/OOS
- [ ] Brinson 归因（依赖 sw_industry + index_members）

### Agent 投研系统第二阶段（远期）

- [ ] 文档解析方案（unstructured / PyMuPDF / Claude 多模态）
- [ ] 网页抓取方案（feedparser / Playwright / httpx+bs4）
- [ ] 向量检索（看因子库规模）
- [ ] 多因子组合策略迭代

### 因子挖掘 pipeline 第二阶段

- [ ] OOS / IS 切分：IS 70% + OOS 30%，要求 OOS IC 衰减 < 30% 才能入 step8
- [ ] 多 universe 稳健性：step7 同时跑全A / 沪深300 / 中证500，至少 2 个通过阈值才入 step8
