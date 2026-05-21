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

### `dividends`(分红送股事件表)

- **数据源**：Tushare `pro.dividend`（14 列）
- **Schema**：`(symbol VARCHAR, end_date VARCHAR, ann_date VARCHAR, ex_date VARCHAR, record_date VARCHAR, pay_date VARCHAR, cash_div DOUBLE, cash_div_tax DOUBLE, stk_div DOUBLE, stk_bo_rate DOUBLE, div_proc VARCHAR)`
- **主键**：`(symbol, end_date)`
- **入库过滤**：只保留 `div_proc = '实施'`
- `ex_date`（除权除息日）是回测最关键日期：价格跳空、送转股生效
- 预估总量 < 20 万行，事件型查询 `WHERE ex_date = ?`

### `index_daily`(宽基指数日行情)

- **数据源**：Tushare `pro.index_daily`
- **Schema**：`(date DATE, symbol VARCHAR, open / high / low / close / pre_close / change / pct_chg / volume / amount DOUBLE)`
- **主键**：`(date, symbol)`
- **默认 4 大宽基**（`DEFAULT_INDICES` in `backfill/indices.py`,从各自基日起回填):
  - `000300.SH`(沪深 300, 基日 2004-12-31)
  - `000905.SH`(中证 500, 基日 2004-12-31)
  - `000852.SH`(中证 1000, 基日 2004-12-31)
  - `932000.CSI`(中证 2000, 基日 2013-12-31)
  - 此外含 `000001.SH`(上证综指) `399006.SZ`(创业板指)
- **后缀注意**：中证 2000 走 `.CSI`(中证指数公司),Tushare `pro.index_daily` 原生支持
- **查询路径**:`get_index_bars(symbols, start, end)`(签名同 `get_bars`,读 `index_daily` 表)

### `sw_industry`(申万行业归属历史, SW2021 体系)

- **数据源**：Tushare `pro.index_classify`(行业代码 → 名称映射) + `pro.index_member`(成分股历史,**注意不是 `pro.index_member_all`** —— 后者只返回当前归属,丢失历史)
- **Schema**：`(symbol VARCHAR, level VARCHAR, industry_code VARCHAR, industry_name VARCHAR, in_date DATE, out_date DATE)`
- **主键**：`(symbol, level, industry_code, in_date)`
  - 同一 `(symbol, level)` 在不同时段可属于不同行业,也可多次进出同一行业
- **level 取值**：`'L1'`(申万一级,31 个行业) / `'L2'`(申万二级,约 125 个行业,SW2021 体系下有 9 个 L2 行业暂无成分股)
- **out_date 语义**：`NULL` = 截至最新数据日仍在该行业;否则为剔除日
- **典型样本**:`000034.SZ` 历史上 2007-07-03 ~ 2009-06-01 在农林牧渔(L1),2015-07-01 ~ 2017-06-30 又回到农林牧渔(L1) —— 表里保留两行
- **入库脚本**:`python -m backtest.data.backfill.sw_industry`(默认 L1+L2,全量重拉。行业变更不频繁,可周/月频跑)
- **查询路径**：
  - `get_industry_panel(date, level='L1')` —— D 日横截面 `[symbol, industry_code, industry_name]`(按 `in_date <= D AND (out_date IS NULL OR out_date > D)` 过滤)
  - `get_industry_history(symbol, level=None)` —— 某股票的全历史归属

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
| `index_daily` | 按 `(symbol, MAX(date)+1)` 起增量(per symbol) | 各指数自身最大日 |
| `sw_industry` | 全量重拉(UPSERT),周/月频跑 | 不依赖时间游标(行业变更稀疏,直接全表覆盖最便宜) |

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
get_index_bars(symbols, start, end, columns=[...])            # 指数日行情时序
get_industry_panel(date, level='L1')                          # 申万行业归属 D 日横截面
get_industry_history(symbol, level=None)                      # 申万行业归属全历史(各分段)
```

---

# P0 实施计划

## P0-3: 交易日历表（数据模块部分）

### 目标
把现在每次调用 `trade_calendar.py:fetch_trade_calendar` 都打 Tushare API 的逻辑，改成 DuckDB 物理表 + 预计算的 `is_week_first` / `is_month_first` / `is_week_last` / `is_month_last` 布尔列，支撑策略层周/月频调仓。

### 现状
- `backtest/data/trade_calendar.py` 27 行，纯 API 直连：每次都 `pro.trade_cal(is_open="1")` → DataFrame
- 没有持久化表。`storage.py` 里没有 `trade_calendar` 任何痕迹
- 消费方:`update_daily.py / cold_start.py / _pipeline.py / strategy/base.py / factor/{update,compute,backfill}.py / evaluation/metrics.py`

### 新表 schema

在 `backtest/data/storage.py` 的 schema 区域新增：

```sql
CREATE TABLE IF NOT EXISTS trade_calendar (
    cal_date DATE PRIMARY KEY,
    is_open BOOLEAN NOT NULL,           -- 是否交易日
    is_week_first BOOLEAN NOT NULL,     -- 本 ISO 周第一个交易日
    is_week_last BOOLEAN NOT NULL,      -- 本 ISO 周最后一个交易日
    is_month_first BOOLEAN NOT NULL,    -- 本月第一个交易日
    is_month_last BOOLEAN NOT NULL      -- 本月最后一个交易日
);
```

注：
- 包含**所有日期**（含非交易日），用 `is_open` 区分。这样查 `is_open=true AND cal_date >= ?` 即可获得交易日序列；不需要额外做日历推算
- 4 个布尔标志只对 `is_open=true` 行可能为 true；周/月边界基于交易日序列计算（不是自然日历），与 strategy/base.py 现有逻辑一致

### 新建文件 `backtest/data/backfill_trade_calendar.py`

```python
def backfill_trade_calendar(start_date: str = "20000101",
                            end_date: str | None = None) -> int:
    """
    1. pro.trade_cal(start_date=start, end_date=end) → 全量日期 + is_open
    2. 按 is_open=1 过滤出交易日序列
    3. 按 ISO week / month groupby 计算 first/last 标志
    4. UPSERT 到 trade_calendar 表
    Returns: 写入行数
    """
```

CLI 入口：`python -m backtest.data.backfill_trade_calendar`

### 改造 `backtest/data/trade_calendar.py`

新签名（保留旧 `get_trade_dates(start, end)` 名字，向后兼容）：

```python
def get_trade_dates(start: str, end: str) -> list[str]:
    """DuckDB-first: 读 trade_calendar WHERE is_open=true。空时 fallback 到 pro.trade_cal 并写表。"""

def get_rebalance_dates(start: str, end: str, freq: str) -> list[str]:
    """
    freq: '1D' / '5D' / '1W' / '2W' / '1M' / 'EOM'
      '1D' → 所有交易日
      '5D' → 每 5 个交易日取一个（从 start 开始计数）
      '1W' → is_week_first = true
      '2W' → is_week_first = true 且 ISO 周号为偶数（或从 start 起隔一周）
      '1M' → is_month_first = true
      'EOM' → is_month_last = true
    """
```

`get_rebalance_dates` 是给 strategy 层用的新接口，把 `strategy/base.py:_get_rebalance_dates` 里运行时算边界的 100 行逻辑下沉到 SQL 查询。

### `cold_start.py` 集成
在主流水线**最开始**调用 `backfill_trade_calendar(start, end)`（必须先于 `market_daily` 回填，因为后者依赖交易日序列）。

### 增量更新
`update_daily.py` 在每日开始时调用 `backfill_trade_calendar(start=last_cal_date+1, end=today)`，确保新交易日入表。

### 完成标准
- [ ] `trade_calendar` 表落地 + UPSERT 工作
- [ ] `backfill_trade_calendar.py` CLI 可用
- [ ] `cold_start.py` / `update_daily.py` 已接入
- [ ] `get_trade_dates` DuckDB-first，Tushare 调用频率从每次 → 仅缺失时
- [ ] `get_rebalance_dates(start, end, freq)` 实现并通过样例测试：2024 年 1 月 `1M` 应返回 `['20240102']`，`EOM` 应返回 `['20240131']`
- [ ] 与 strategy 模块协调：`strategy/base.py:_get_rebalance_dates` 改为调 `get_rebalance_dates`（详见 strategy/DESIGN.md P0-3 节）
