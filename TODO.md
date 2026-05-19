# TODO

> 临时工单池。每项完成后从本文件删除;全部完成后删除整份 TODO.md。
> 创建时间: 2026-05-18

---

## 4. 散落 TODO 合并(本表已统一登记)

下列待办原本散在各 `DESIGN.md` 和根 `CLAUDE.md`,统一在本文件追踪。各 DESIGN.md 内的对应条目可保留作为模块自身 roadmap,但**所有 cross-cutting 优先级以本文件为准**。

### Cross-cutting 阻塞依赖

- [ ] **T+1 交割制度建模(simulation)**
- [ ] **CLI 入口**: `python -m backtest.strategy.run --config strategy_config.yaml`
- [ ] **`environment.yml` 落地**: 当前未创建,首次开工需先建 conda env
- [ ] **`pyproject.toml` 落地**: 便于 `pip install -e .`

### Agent / 推送 / 文档解析(尚未启动)

- [ ] 推送渠道选型(企微 / 飞书 / Server酱 / 邮件)
- [ ] 文档解析方案(unstructured / PyMuPDF / Claude 多模态)
- [ ] 网页抓取方案(feedparser / Playwright / httpx + bs4)
- [ ] 是否需要向量检索(看因子库规模)

### Evaluation 模块 roadmap(来自 `backtest/evaluation/DESIGN.md`)

- [ ] 个股贡献 top/bottom 10
- [ ] 行业归因(依赖 sw_industry)
- [ ] 多策略对比
- [ ] 滚动 IS/OOS
- [ ] Brinson 归因(依赖 sw_industry + index_members)
