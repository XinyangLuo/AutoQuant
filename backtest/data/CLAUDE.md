# 数据模块

## 表结构

### `market_daily`（日频主表，回测主用）

- **主键**：`(date, symbol)`，按 `date` 分区
- **列**：`open / high / low / close / volume / amount / pre_close / change / pct_chg / adj_factor / is_st / list_date / limit_up / limit_down / turnover_rate / turnover_rate_f / volume_ratio / pe / pe_ttm / pb / ps / ps_ttm / dv_ratio / dv_ttm / total_share / float_share / free_share / total_mv / circ_mv`
- **日更**：只 append 最新交易日的行，历史行永不动
- **扩列**：偶发，走 `ALTER TABLE ... ADD COLUMN` + 历史回填脚本
- **查询路径**：`get_panel(date)` 横截面 / `get_bars(symbols, start, end)` 时序

### `factors_daily`（因子长表，研究主用）

- **Schema**：`(date, symbol, factor_name, value)`
- 加新因子**零 schema 变化**
- 多因子组合时 pivot 成宽表给策略/引擎

### `fina_indicator_quarterly`（财务指标季度表）

- **数据源**：Tushare `pro.fina_indicator`（108 列衍生指标），不走原始三表
- **Schema**：`(symbol VARCHAR, end_date VARCHAR, ann_date VARCHAR, eps DOUBLE, roe DOUBLE, ...)`
- **主键**：`(symbol, end_date)`
- `end_date` / `ann_date` 为 `YYYYMMDD` 字符串，避免日期解析

## Fetch/Merge 模式

日频数据（`market_daily`）的流水线：

```
pro.daily → DataFrame
pro.adj_factor → DataFrame → pandas merge (LEFT JOIN on date+symbol)
pro.stock_st → DataFrame → merge
pro.stk_limit → DataFrame → merge
pro.daily_basic → DataFrame → merge
→ 统一宽 DataFrame → UPSERT INTO market_daily
```

- 每个数据源**单独 fetch**，然后**pandas left-merge**
- 空响应自动填充 None/False，不中断流水线

## 增量更新

| 表 | 增量方式 | 对齐依据 |
|---|---|---|
| `market_daily` | 按交易日历，从 `MAX(date)+1` 开始 | 交易日历 |
| `fina_indicator_quarterly` | 按 `start_ann_date`，只拉新公告的报表 | 公告日期 |

## 回填约定

- **不走 pandas 全量重跑**。只 fetch 目标数据源，写入 DuckDB 临时表，通过 SQL `INSERT ... ON CONFLICT DO UPDATE SET target_col = EXCLUDED.target_col` 只更新目标列
- `insert_daily()` 为**动态列模式**：DataFrame 有什么列就 INSERT/UPDATE 什么列，其他列不动
- 适用于：新增列的历史回填、部分列的修复重跑

## 去重与修正

### `fina_indicator` 重复行

- Tushare 对同一 `(ts_code, end_date)` 可能返回 **2 行**
- 大部分是 100% 相同重复，少数是部分 NaN 的不完整记录
- **入库前必须去重**：
  ```python
  df = df.drop_duplicates(subset=["ts_code", "end_date"])
  df["_nan_count"] = df.isna().sum(axis=1)
  df = df.sort_values("_nan_count").drop_duplicates(subset=["ts_code", "end_date"], keep="first")
  df = df.drop(columns=["_nan_count"])
  ```

### 修正问题

- Tushare 对同一报告期**只保留最终修正版**，不存在旧版记录
- `income`/`balancesheet` 虽有 `f_ann_date`，但实测 `ann_date == f_ann_date`
- 回测引擎在日期 D 只能查 `ann_date <= D` 的报表

## 未来信息隔离

- 离线因子计算时只使用 `ann_date <= 当前日期` 的财务数据
- 计算结果连同 `ann_date` 一起写入**独立的财务因子表**
- 回测引擎按 `ann_date <= 回测日期` 过滤使用

## 对外接口

```python
get_panel(date, columns=[...])           # 主表某日横截面
get_bars(symbols, start, end, columns=[...])  # 主表时序
get_fina(symbol, end_date)               # 财务指标单条
get_factor(factor_name, start, end)      # 因子表单因子时序
get_factor_panel(factor_names, date)     # 因子表 pivot 宽表
```
