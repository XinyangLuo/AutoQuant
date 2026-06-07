# TODO

> 临时工单池。每项完成后从本文件删除；全部完成后删除整份 TODO.md。
> 分级逻辑（2026-06-03 重分类）：
> - **阻塞项**：直接影响单因子研究结论正确性（仿真偏差、静默错误）
> - **效果/效率**：不阻塞，但解决后显著提升迭代速度或因子质量
> - **多因子 & 风控**：多因子合成、组合优化、实盘交易
> - **远期**：依赖规模或前置条件，当前不紧急
> 上次整理: 2026-06-04

---

## 一、阻塞项

> 这些不修，回测结果可能误导因子准入决策。

（当前无阻塞项）

---

## 二、效果/效率

> 不阻塞，但显著影响迭代速度和因子质量。

### 2.1 验证体系

- [ ] **OOS/IS 时间切分**：`PipelineConfig` 增加 `oos_ratio` / `oos_start` 参数，step3~step7 在 IS 上训练/筛选，OOS 上独立验证。要求 OOS IC 衰减 < 30%。
- [ ] **多 universe 稳健性**：因子至少在 2 个 universe（全A / HS300 / CSI500）上通过 step3 ICIR gate。

### 2.3 性能优化

> 按预估加速比排序。

- [ ] **`pipeline/steps.py` `step2` `_max_industry_corr`**：逐日 `pd.get_dummies` → 5–10×。预计算行业哑变量复用。
- [ ] **`factor/compute.py` chunk 嵌套冗余**：外层 1 年 chunk + SQL 内层 6 月 chunk 重复切分 → 1.5–2×。合并为一层。
- [ ] **backfill 多因子并行**：`ProcessPoolExecutor` 并行回填多个 factor_id。
- [ ] **`FactorStorage.get_factors_wide`**：单次 SQL `UNION ALL` 出多列宽表，替代多次查询 + 客户端 merge。
- [ ] **Transforms 向量化**：`z_score`/`ts_ir` 去重复 sort，`cs_mad_winsorize`/`cs_zscore` 从 `groupby.apply` → `groupby.transform` + numpy。

### 2.4 测试覆盖

- [ ] **Pipeline 集成测试**：step1~step10 顺序调用 + state JSON 累积验证，确保门控逻辑不退化。
- [ ] **Barra L1 smoke test**：`barra_ind_size` pipeline 端到端，验证中性化不破坏因子结构。
- [ ] Data 模块：multi-type fetch + PIT snapshot 正确性
- [ ] Transforms 单测：`single_quarter` / `ttm` / `yoy` / `ts_ir`
- [ ] 策略模块：`SingleFactorStrategy` + `MultiFactorStrategy` 基础路径

### 2.5 文档 & 工程

- [ ] `backtest/strategy/DESIGN.md`：补充 `selection.py`、`decay`、`RiskConfig`/`BacktestConfig`
- [ ] `backtest/simulation/DESIGN.md`：补充 `decile.py`
- [ ] `backtest/evaluation/metrics.py` docstring：`CLAUDE.md` → `DESIGN.md`
- [ ] `backtest/strategy/neutralize.py`：已标记 deprecated，确认无调用后删除
- [ ] `backtest/data/backfill_indices.py`：benchmark 报错信息仍引用旧路径
- [ ] `pyproject.toml` 落地 + `environment.yml` 补全缺失依赖

---

## 三、多因子合成 & 策略风控

> 单因子只是第一步；这些是组合管理和实盘的基础设施。

### 3.1 仿真增强

- [ ] **滑点模型**：`SimulationConfig` 增加 `slippage_bps` 参数，executor 成交价按滑点调整，默认 0。
- [ ] **现金不足权重优先分配**：等比例缩减 → 按目标权重排序优先分配，更贴近实盘执行。

### 3.2 归因与对比

- [ ] **个股贡献 top/bottom 10**：`positions × daily_return` 矩阵，定位收益/亏损最大来源。
- [ ] **行业归因**：依赖 `sw_industry`（已落地），按 L1 行业聚合暴露 + 收益。
- [ ] **Brinson 归因**：配置效应 + 选择效应 + 交互效应，需 `index_members` 基准权重。
- [ ] **多策略对比**：多因子/多参数组合的并排指标表 + 滚动 IS/OOS 窗口对比。

### 3.3 交易模块

- [ ] **Phase 1 — 信号推送**：推送渠道选型（企微/飞书/Server酱）→ 信号渲染（策略信号 → 可读消息）→ 仓位 CLI（本地 YAML 录入/编辑）。
- [ ] **Phase 2 — 实盘对接**：QMT / Ptrade / easytrader。

---

## 四、远期/探索

> 依赖规模（因子库 > 50）或前置条件未满足，当前不紧急。

### 4.1 Agent 规模化

- [ ] **并行探索**：父进程手动指定 2 个方向，独立 `/factor-iterate`（不同 run dir + `run_in_background`），验证 DuckDB 并发安全 + token 消耗。
- [ ] **Workspace checkpoint/rollback**：`FactorWorkspace` dataclass + zip checkpoint，支持 round 间回滚。
- [ ] **DuckDB vss 向量检索**：L3 冷数据语义查重，当前 keyword 过滤够用，vss 是备选加速。
- [ ] **库审计**：触发条件 admitted factor > 50；冗余/缺口/衰减检测 → 自动生成 HG 输入。
- [ ] **Bandit 方向选择**：方向数 > 5 且手动选择不可持续时，bandit 自动分配探索预算。
- [ ] **全自动触发**：定时扫描覆盖缺口 → HG → HO → FC 全自动循环。

### 4.2 数据

- [ ] **分钟级数据**：fetcher → backfill → update → 读取 API（pyarrow.dataset 按日期分区）→ 天级因子合成。
