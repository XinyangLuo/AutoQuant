# TODO

> 临时工单池。每项完成后从本文件删除；全部完成后删除整份 TODO.md。
> 分级：P0=立刻/阻塞，P1=依赖 P0，P2=有价值但不紧急，P3=维护/锦上添花，P4=远期/想法池
> 创建时间: 2026-05-18

---

## P0

### Barra 风险模型分 4 commit 串行（`backtest/factor`）

进度：

- [x] **Commit 1: wide-format schema + variant 元数据化**（`44da10c`，2026-05-21）
  - `factors_daily` 从 long 改 wide：PK=(date, symbol)，每个因子一列
  - 删因子由 6 分钟 DELETE 改为毫秒级 `ALTER TABLE DROP COLUMN`
  - `@register(variant=..., frequency="D"|"W"|"M")` 新接口；variant 退化为 registry 元数据
  - 旧 13 个 `f_rev_*` 因子清空（4.4G → 268K）；alphas/ 仅保留 `__init__.py`
  - 下游 admission / strategy / evaluation / pipeline script / tests 全部迁移
  - `_existing_columns` 加缓存消除 N+1；DRY admission 三态查询；76/76 因子测试通过

- [x] **Commit 2: Barra 三级因子注册（11 个）+ 一级合成（7 个）**（2026-05-22）
  - `backtest/factor/builtin/barra/` 落地：`size / beta / momentum / value / quality / liquidity / growth / composite + _common`
  - 11 个 L3 因子按 PLAN.md §2.1 公式实现，全部 `variant="barra_l3"`（MAD → 行业中位数填充 → cs_zscore 自动应用）
  - 严谨版：Beta WLS 向量化（sliding_window_view + 闭式 2 参解，窗口 252、半衰期 63）；Momentum EWMA（半衰期 126）+ lag 11 + 11 日平滑；AGRO/EGRO 按 trade_date 分组的 PIT-safe 20 季回归
  - 7 个一级合成因子（`variant="none"`）：Size=LNCAP、Beta=BETA、Momentum=RSTR、Value=(BTOP+ETOP+DTOP)/3、Quality=(ROA+GP+AGRO)/3、Liquidity=STOM、Growth=EGRO
  - DTOP 修复：dividend fetch 加 400 天回看 buffer，早期 trade date 的 TTM 不再被截断
  - composite 改 `groupby.mean()` 避免 pivot 内存峰值；共享 helper 抽到 `_common.py`

- [x] **Commit 3: 中性化 pipeline 替换为 PLAN.md §2.2 OLS 版**（2026-05-22）
  - `compute.apply_variant_pipeline` `barra_ind_size` 分支替换为完整 PLAN.md §2.2 pipeline：MAD 去极值 → SW-L1 行业中位数填充 → cs_zscore → 截面 OLS（intercept + 行业 dummies drop_first + Size_z）→ 取残差 → re-cs_zscore
  - Size_z 直接读 `f_barra_size_lncap`（Commit 2 已落地，barra_l3 pipeline 后已是 z-score）
  - 新增 `transforms.cs_ols_residualize(values, design_panel, dummy_col, numeric_cols)`：通用 OLS 残差算子，dummy 用 `Categorical` 在循环外一次性编码，循环内按 codes 切 identity-style block，避免 `pd.get_dummies` N+1 开销；`np.linalg.lstsq` 闭式解
  - `apply_variant_pipeline` 增加 `factor_storage` 参数，`backfill.py` 透传
  - 验证：synthetic alpha = industry_effect + 1.2·size_z + noise → 残差对 size_z 和所有行业 dummies 的截面 Pearson corr < 1e-6（OLS 保证精确正交）
  - 测试：4 个 `cs_ols_residualize` 单元 + 2 个 pipeline 集成（barra_l3 / barra_ind_size），共 6 个新测试全部通过；全 factor 测试 206/206 通过

- [ ] **Commit 4: Ridge 入库检查 + R² 分层**
  - 新增 `backtest/factor/admission_check.py::ridge_r2_check(factor_id) -> {r2, tier, residual_icir}`
  - 候选因子 vs 6 个一级 Barra（除 Size 和 Industry）做 Ridge regression
  - R² 分层（PLAN.md §4 step8）：< 0.10 pure_alpha / 0.10-0.50 smart_beta / 0.50-0.80 edge_smart_beta（需残差 ICIR 日频 > 1.0 月频 > 0.8）/ ≥ 0.80 reject
  - `admit()` 在写入 library 时把 `tier` 和 `r2` 写入 `registry.json` meta

### 因子算子库（`backtest/factor`）

- [x] 已合并到 main（`c6700c2`）：23 个 ts/cs/math/conditional 算子 + 完整 test suite

### 交易日历（`backtest/data` + `strategy` + `simulation`）

- [x] 已合并到 main（`6d3dd9c`）：`trade_calendar` 表 + `is_week_first/is_month_first` 标志

## P1

### 因子挖掘流程优化（依赖 P0，配合 rd agent）

- [ ] 串行 step1~step9 pipeline + 每步淘汰标准：截面缺失率 < 30% → 中性化后与 size/industry corr < 0.05 且与已有因子 max corr < 0.5 → 离线 ICIR（日频 1D/5D，月频 1M 阈值见 PLAN）→ 分 10 组单调性 > 0.7 → 默认策略 top10% decay=5 → 向量化简单回测（日频 Sharpe > 0.8 / 月频 > 1.0）→ 详细回测（日频 Sharpe > 0.4 / 月频 > 0.6）→ Ridge R² 入库分流 → markdown 报告生成。详见 PLAN.md §4。

## P2

### 因子库可视化

- [ ] 所有因子报告整合成 web 浏览页面。详见 PLAN.md §5。

### Storage 共用底座（从 Commit 1 的 /simplify review 延期）

- [ ] `_quote_ident` / `_upsert` / `_registered` 抽到共享模块，让 `FactorStorage` 和 `backtest/data/storage.py:MarketStorage` 共用 DuckDB 底座，避免两套并行实现长期漂移
- [ ] `get_factors_long` 把 melt 推到 SQL（`UNION ALL` per column with `WHERE col IS NOT NULL`），避免 25M 行 × N 列宽表全部 melt 到内存

### Barra 因子计算性能 follow-up（Commit 2 延期）

- [ ] `compute.py` 财务因子 panel 拼接走流式：现在每个 trade_date 跑一次 `get_fina_snapshot` 再 `pd.concat` N 个 snapshot，最坏 ~14B 单元；改成按 trade_date 分块计算再拼小结果，或者在 storage 层加 `get_fina_snapshot_range(start, end)` 用区间 join 一次出长表
- [ ] `momentum.py:_ewm_log_return_sum` 的 `rolling.apply` 用 `sliding_window_view` 向量化（与 beta 已做的对应），节省全市场 5000 股 × 1000 天 backfill 时间
- [ ] backfill 多因子并行：现在 `compute_all` 串行循环 registry，可用 `ProcessPoolExecutor` 并发跑独立因子
- [ ] `cs_mad_winsorize` / `cs_zscore` 等 `cs_*` 算子从 `groupby.apply(_one)` 改为 `groupby.transform('mean'/'std')` + numpy 直算；`barra_ind_size` pipeline 跑 5 次 `groupby(date)`，向量化后预计 3-5× 加速

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
