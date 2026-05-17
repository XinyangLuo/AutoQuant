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

### `income_q` / `balancesheet_q` / `cashflow_q`（三大财报独立表）

- **数据源**：Tushare `pro.income` / `pro.balancesheet` / `pro.cashflow`，各自独立入库。**不**使用 `pro.fina_indicator`（该表丢失 `update_flag` / `f_ann_date`，无法做合法 PIT 隔离）
- **入库过滤**：只保留 `report_type=1`（合并报表）。其它口径（母公司、调整版）当前不入库
- **主键**：`(symbol, end_date, f_ann_date, update_flag)`——Tushare 偶尔对同一 `(symbol, end_date, f_ann_date)` 返回 `update_flag=0` 和 `update_flag=1` 两行，必须同时保留
- **Schema（物理表保留 Tushare 原始列名，不加前缀）**：
  ```
  symbol, end_date, ann_date, f_ann_date, report_type, comp_type, end_type, update_flag,
  <各表原始 numeric 列>
  ```
  | 表 | 约多少 numeric 列 | 关键列示例 |
  |---|---|---|
  | income_q | ~77 | `basic_eps`, `total_revenue`, `n_income`, `n_income_attr_p`, `operate_profit` ... |
  | balancesheet_q | ~144 | `total_assets`, `total_liab`, `total_hldr_eqy_inc_min_int`, `total_cur_assets` ... |
  | cashflow_q | ~89 | `n_cashflow_act`, `n_cashflow_inv_act`, `n_cash_flows_fnc_act`, `free_cashflow` ... |
- **版本语义**：
  - `update_flag='0'`：原始公告，`ann_date == f_ann_date`
  - `update_flag='1'`：修正版，`ann_date` 仍为原始公告日，`f_ann_date` 为修正日（可能晚 1~5 年）
  - 同一 `(symbol, end_date)` 可能有 1 行（无修正）或 2+ 行（每次修正多一行）
- **物理表保留所有版本，不在存储层去重**——可溯源、可回放历史。"D 日只看一条"的语义由 `get_fina_snapshot()` 在查询时实现

### `dividends`（分红送股事件表）

- **数据源**：Tushare `pro.dividend`（14 列）
- **Schema**：`(symbol VARCHAR, end_date VARCHAR, ann_date VARCHAR, ex_date VARCHAR, record_date VARCHAR, pay_date VARCHAR, cash_div DOUBLE, cash_div_tax DOUBLE, stk_div DOUBLE, stk_bo_rate DOUBLE, div_proc VARCHAR)`
- **主键**：`(symbol, end_date)`
- **入库过滤**：只保留 `div_proc = '实施'`
- `ex_date`（除权除息日）是回测最关键日期：价格跳空、送转股生效
- 预估总量 < 20 万行，事件型查询 `WHERE ex_date = ?`

## Fetch/Merge 模式

### 日频数据 (`market_daily`)

```
pro.daily        → DataFrame
pro.adj_factor   → DataFrame → pandas merge (LEFT JOIN on date+symbol)
pro.stock_st     → DataFrame → merge
pro.stk_limit    → DataFrame → merge（列名 up_limit/down_limit → rename 为 limit_up/limit_down）
pro.daily_basic  → DataFrame → merge
→ 统一宽 DataFrame → UPSERT INTO market_daily
```

- 每个数据源**单独 fetch**，然后**pandas left-merge**
- 空响应自动填充 None/False，不中断流水线
- **列名映射**：`pro.stk_limit` 返回 `up_limit`/`down_limit`，需 rename 为 `limit_up`/`limit_down` 以匹配 `DAILY_COLUMNS`

### 财务数据（`income_q` / `balancesheet_q` / `cashflow_q`）

```
pro.income(report_type=1)       → DataFrame  → rename ts_code→symbol
pro.balancesheet(report_type=1) → DataFrame  → rename ts_code→symbol
pro.cashflow(report_type=1)     → DataFrame  → rename ts_code→symbol
→ 各自独立 UPSERT 到对应物理表
   ON CONFLICT (symbol, end_date, f_ann_date, update_flag) DO UPDATE
```

- **三表独立入库**：`income_q`、`balancesheet_q`、`cashflow_q` 各有一张物理表，不预先合并
- **各自保留完整版本**：每张表的原始/修正版由自身 PK 自然区分
- **三表 ann_date / f_ann_date 大部分一致**，但存在约 1% 的独立修正（如 income 在 2022-06-28 修正，balancesheet 在 2022-04-26 修正）——独立表天然处理这种情况，无需在入库时做 awkward merge
- **读取时合并**：`get_fina_snapshot(D)` 内部对三张表分别做 PIT 快照，再按 `(symbol, end_date, ann_date, f_ann_date, update_flag, comp_type, end_type, report_type)` outer-join，非 key 列自动加 `inc_/bs_/cf_` 前缀

## 增量更新

| 表 | 增量方式 | 对齐依据 |
|---|---|---|
| `market_daily` | 按交易日历，从 `MAX(date)+1` 开始 | 交易日历 |
| `income_q` / `balancesheet_q` / `cashflow_q` | 按 `f_ann_date` 游标，从 `MAX(f_ann_date)` 起扫到今天 | 实际见报日（含修正） |
| `dividends` | 每日查 `ex_date=today` 和 `ann_date=today` | 除权日/公告日 |

**注意**：财务表必须以 `f_ann_date` 而非 `ann_date` 为增量游标——只有 `f_ann_date` 能捕获多年后回头发布的修正版（修正版的 `ann_date` 是当年的旧日期，按 `ann_date` 排序会被认为"早已拉过"而漏掉）。

## 回填约定

- **不走 pandas 全量重跑**。只 fetch 目标数据源，写入 DuckDB 临时表，通过 SQL `INSERT ... ON CONFLICT DO UPDATE SET target_col = EXCLUDED.target_col` 只更新目标列
- `insert_daily()` / `insert_fundamentals()` 为**动态列模式**：DataFrame 有什么列就 INSERT/UPDATE 什么列，其他列不动
- 适用于：新增列的历史回填、部分列的修复重跑

## 业绩修正与 PIT（point-in-time）

A 股上市公司可在原始年报发布后多年回头修正报表。典型案例：**300237.SZ 在 2022-06-28 修正了 2018 年报，净利润从 3.79 亿下修到 4116 万**，跨度 3 年多。Tushare 在 `income`/`balancesheet`/`cashflow` 三个原始表里通过 `update_flag` 和 `f_ann_date` 暴露了这一事实，但在派生表 `fina_indicator` 里把这两个字段都丢了——这是我们放弃 `fina_indicator` 的根因。

**字段语义：**

- `ann_date` = 原始公告日（任何版本都不变）
- `f_ann_date` = 该版本的实际见报日：原始版 `== ann_date`，修正版 `>= ann_date`
- `update_flag` = `'0'` 原始 / `'1'` 修正

**正确的因子取数 / 回测取数要满足两条**：

1. 只看 `f_ann_date <= D` 的行（D 日真实可见的版本）
2. 同一 `(symbol, end_date)` 多版本时，取 `f_ann_date` 最大的那条（D 日"最新已知"的事实）

**实现为 `get_fina_snapshot(as_of_date)`**：内部对三张表分别跑 QUALIFY，再 outer-join：

```sql
-- 以 income_q 为例（balancesheet_q / cashflow_q 同理）
SELECT *
FROM income_q
WHERE f_ann_date <= ?            -- ① 隔离未来信息
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY symbol, end_date
    ORDER BY f_ann_date DESC, update_flag DESC  -- ② 取最新可见版本
) = 1
```

- **查询时跑，不修改物理表**。存储层永远保留全部版本，可溯源、可回放任意历史 D
- DuckDB 原生支持 `QUALIFY`，免 CTE / 子查询
- 三张表各自 QUALIFY 后，按 8 个共享 key 列 outer-join，非 key 列加 `inc_/bs_/cf_` 前缀
- 1% 的"三表独立修正"case 自然处理：每张表取 D 日各自最新版本，wide 行可能 `inc_*` 来自 f_ann_date=A、`bs_*` 来自 f_ann_date=B——这正是 D 日真实可见的状态
- 全市场 ~5500 股 × ~70 季度，三张表各自约 30~40 万行，单次查询 < 200ms 量级

**对派生因子的影响**：离线计算财务因子时，先调用 `get_fina_snapshot(D)` 拿到 wide DataFrame，再算因子。结果连同 `ann_date` / `f_ann_date` 一并写入因子表（便于二次审计与回放）。

## 对外接口

```python
get_panel(date, columns=[...])                                # market_daily 横截面
get_bars(symbols, start, end, columns=[...])                  # market_daily 时序
get_fina_snapshot(as_of_date, symbols=None, columns=None)     # D 日财报 wide 快照（PIT 安全），
                                                              # 三张表各自 QUALIFY 后 outer-join，
                                                              # 非 key 列自动加 inc_/bs_/cf_ 前缀
get_factor(factor_name, start, end)                           # 因子表单因子时序
get_factor_panel(factor_names, date)                          # 因子表 pivot 宽表
get_dividend(symbol, start, end)                              # 分红事件
```
