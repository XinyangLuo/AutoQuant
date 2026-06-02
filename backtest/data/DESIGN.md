# 数据模块

## 表结构

### `market_daily`（日频主表，回测主用）

- **主键**：`(date, symbol)`，按 `date` 分区
- **列**：`open / high / low / close / volume / amount / pre_close / change / pct_chg / adj_factor / is_st / list_date / limit_up / limit_down / turnover_rate / turnover_rate_f / volume_ratio / pe / pe_ttm / pb / ps / ps_ttm / dv_ratio / dv_ttm / total_share / float_share / free_share / total_mv / circ_mv`
  - **融资融券**（`pro.margin_detail`）：`margin_rzye / margin_rqye / margin_rzmre / margin_rqyl / margin_rzche / margin_rqchl / margin_rqmcl / margin_rzrqye`
  - **资金流向**（`pro.moneyflow`）：`mf_buy_sm_vol / mf_buy_sm_amount / mf_sell_sm_vol / mf_sell_sm_amount / mf_buy_md_vol / mf_buy_md_amount / mf_sell_md_vol / mf_sell_md_amount / mf_buy_lg_vol / mf_buy_lg_amount / mf_sell_lg_vol / mf_sell_lg_amount / mf_buy_elg_vol / mf_buy_elg_amount / mf_sell_elg_vol / mf_sell_elg_amount / mf_net_mf_vol / mf_net_mf_amount`
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
- **主键**：`(symbol, end_date, ann_date, ex_date)` — 同一报告期可能多次分红（如中期 + 末期、常规 + 特别股息），仅靠 `(symbol, end_date)` 会丢失数据
- **入库过滤**：只保留 `div_proc = '实施'`
- `ex_date`（除权除息日）是回测最关键日期：价格跳空、送转股生效
- **NULL 兜底**：Tushare 对 0.08% 的老记录缺失 `ex_date` / `pay_date`，入库时用 `pay_date → ann_date → end_date` 链式回填，避免 PK 冲突同时保留事件
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

### `index_members`(宽基指数成分股, 已密集化到每个交易日)

- **数据源**：Tushare `pro.index_weight`(月度再平衡日的权重快照,每月一次)
- **Schema**：`(index_code VARCHAR, symbol VARCHAR, trade_date DATE, weight DOUBLE)`
- **主键**：`(index_code, symbol, trade_date)`
- **入库语义**：月度快照展开到下次发布日前的每个交易日,日期等值查询直接可用,不需要 as-of 逻辑
- **默认指数**：`000300.SH` / `000905.SH` / `000852.SH` / `932000.CSI`(配置在 `backfill/index_members.py:DEFAULT_INDICES`)。创业板指 `399006.SZ` 因 Tushare 账户权限限制暂未含
- **入库脚本**：`python -m backtest.data.backfill.index_members`(增量从 `get_max_index_member_date(idx)+1` 开始,日更走 `update_daily.py` Phase 5)
- **查询路径**：`get_index_members(date, index_code) -> set[symbol]`,用于策略层 universe 过滤(`backtest/strategy/universe.py` line 77)

### 分钟级行情（Parquet 存储）

> **状态**：接入中（见 `backtest/data/fetcher/minute_fetcher.py` / `backfill/minute.py` / `update_minute.py`）。
> **数据量**：全市场 5000+ 股 × 240 条/天 ≈ 120 万行/天，parquet 压缩后约 20–40 MB/天。

**存储格式**：按 `freq/symbol/year.parquet` 分区的双层目录结构：

```
data/minute/
  1min/
    000001.SZ/
      2024.parquet
      2025.parquet
    000002.SZ/
      2024.parquet
      2025.parquet
  5min/
    000001.SZ/
      2025.parquet
```

- 每只股票每年一个 parquet 文件
- 列：`date DATE, time TIMESTAMP, symbol VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT, amount DOUBLE, pre_close DOUBLE, change DOUBLE, pct_chg DOUBLE`
- 使用 `pyarrow` + `pandas` 读写；`pyarrow.dataset` 支持按 `symbol` / `date` 分区过滤
- 不复权入库，复权在读取/计算层处理（与日频 `market_daily` 一致）

**数据源**：Tushare `ts.pro_bar(ts_code=..., freq='1min', start_date=..., end_date=...)`，底层调用 `pro.stk_mins`

**关键约束**：
- 单股模式：`ts_code` 必填，不支持多值批量输入
- 速率限制：基础账户约 1 次/分钟；充值后频次提升
- **单次返回上限**：8000 行（约 33 个交易日/次 @1min，或 166 个交易日/次 @5min）
- **Fetcher 内部自动分块**：当请求区间超过 8000 行时，自动拆分为多个子区间循环拉取并合并
- 全市场 backfill：5000 股 × N 年，按 symbol 逐个获取，支持断点续传（检查年文件存在则跳过）

**增量更新**：扫描各 symbol 目录最新年文件中的最大日期，从 `max_date + 1` 循环到今天，每只 symbol 拉取缺失区间并追加到对应年 parquet（跨年时新建下一年文件）。

**查询路径**（规划中）：
```python
get_minute_bars(symbols, start, end, freq='1min', columns=None) -> pd.DataFrame
```
使用 `pyarrow.dataset` 按 `symbol` / `date` 分区过滤。

### `cyq_chips`（筹码分布，DuckDB LIST 存储）

> **状态**：已接入。见 `backtest/data/cyq_storage.py`、`backtest/data/fetcher/cyq_fetcher.py`、`backtest/data/backfill/cyq_chips.py`。
> **数据量**：全市场 5000 股 × 250 交易日 × 平均 100 档 ≈ 1.25 亿个 price/percent 对。

**为什么用 LIST 数组而不是宽表/长表**：

| 方案 | 问题 |
|---|---|
| 宽表（每档一列）| 档位数 59~175 不固定，schema 会膨胀到 175+ 列 |
| 长表（`date, symbol, price, percent`）| ~1.25 亿行/年，按 `(date, symbol)` 读需扫多行再 reassemble |
| **LIST（本方案）** | ~125 万行/年，`(date, symbol)` 点查**一行**返回完整分布 |

**Schema**：

```sql
CREATE TABLE cyq_chips (
    date      DATE,
    symbol    VARCHAR,
    n_bins    INTEGER,        -- len(prices) == len(percents)
    prices    DOUBLE[],       -- 价格档位数组（升序）
    percents  DOUBLE[],       -- 筹码占比数组
    PRIMARY KEY (date, symbol)
);
```

- 每只股票每天**只有一行**
- `prices` 和 `percents` 是等长变长数组，长度即该股票当天档位数
- DuckDB 列式压缩后约 **0.5–1 GB/年**，10 年全量 ~5–10 GB

**数据源**：Tushare `pro.cyq_chips(ts_code=..., trade_date=...)`

**关键约束**：
- `ts_code` 是**必填参数**，不支持全量一次拉取
- 支持 `start_date` / `end_date` 按 symbol 批量拉取一段区间
- **单次返回上限**：约 6000 行（约 35 个交易日/次，对高 bin 股票）
- **Backfill 按 symbol + chunk 循环**：每个 symbol 拆成 30 交易日 chunk，避免触发上限
- 基础账户速率限制较严，fetcher 内留 `sleep_sec` 参数

**增量更新**：从 `MAX(date)+1` 开始，按 symbol 逐个 chunk 拉取并 UPSERT（`ON CONFLICT` 覆盖）。

**查询路径**：
```python
from backtest.data.cyq_storage import CyqStorage

with CyqStorage() as store:
    # 单票单日 → tidy DataFrame [price, percent]
    store.get_cyq(date="20250526", symbol="600519.SH")

    # 某日横截面 → [date, symbol, n_bins, prices, percents]
    store.get_cyq_panel(date="20250526", symbols=[...])

    # 单票历史 → [date, n_bins, prices, percents]
    store.get_cyq_history(symbol="600519.SH", start="20240101", end="20250526")

    # SQL 层聚合示例：筹码重心
    store.get_weighted_prices(date="20250526")

    # SQL 层聚合示例：峰值档位价格
    store.get_peak_prices(date="20250526")
```

## Fetch/Merge 模式

### 日频数据 (`market_daily`)

```
pro.daily        → DataFrame
pro.adj_factor   → DataFrame → pandas merge (LEFT JOIN on date+symbol)
pro.stock_st     → DataFrame → merge
pro.stk_limit    → DataFrame → merge（列名 up_limit/down_limit → rename 为 limit_up/limit_down）
pro.daily_basic  → DataFrame → merge
pro.margin_detail → DataFrame → merge（8 列，rename 加 margin_ 前缀）
pro.moneyflow    → DataFrame → merge（18 列，rename 加 mf_ 前缀；单位转换 手→股 / 万元→元）
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

## 财报数据使用指南 (Shi Chuan 4-case + 单季度 + TTM)

> 本节是对上节 "业绩修正与 PIT" 的扩展：在 `update_flag` 之外，Tushare 还通过 `report_type` 暴露报表版本族（初始 / 基准 / 修正前 / 修正后）。石川《因子投资》第 3 章给出 4 种典型披露场景，下文给出对照与实现契约。
>
> **当前状态**：data 层方案已定稿，待代码落地（见 §6 路线图 Round 1）。单季度推导 / TTM / YoY 不在 data 层完成，由因子层 `backtest/factor/transforms.py` 助手函数承担（Round 2）。

### 1. Tushare `report_type` 取值

来源：Tushare `pro.income` / `pro.balancesheet` / `pro.cashflow` 文档（<https://tushare.pro/document/2?doc_id=33>）。

| code | 含义 | 备注 |
|---|---|---|
| 1 | 合并报表（默认） | 初始合并报表 |
| 2 | 单季合并 | 已是单季度，但保留作为冗余 |
| 3 | 调整（合并） | "前次披露" 合并版本 |
| 4 | 调整（合并报表）/ 基准 | 次年年报附带披露的同年基准 |
| 5 | 调整前（合并） | 更正发生时的 "原值" |
| 6 | 母公司报表 | 单体口径，不入库 |
| 11 / 12 | 母公司调整 / 调整前 | 单体口径，不入库 |

**入库范围**：`report_type ∈ {1, 2, 3, 4, 5}`（合并口径全集）；`6 / 11 / 12`（母公司口径）不入库。

**实证结论 — `update_flag` 不可作为版本新旧判定**：

Tushare 数据中 `update_flag` 并非严格的 "0=原始 / 1=修正"。实测案例：

- `920522.BJ` 2023H1（end_date=20230630）三条记录按 `f_ann_date` 升序的 `update_flag` 序列为 `(1, 0, 1)`
- `920663.BJ` 2022 年报（end_date=20221231）同样为 `(1, 0, 1)`
- `300237.SZ` 2018 年报（end_date=20181231）只两条：`f_ann_date=20190426 update_flag=0`（原始 3.79 亿）+ `f_ann_date=20220628 update_flag=1`（修正后 4116 万）—— **修正只新增一行，原行不改写**

因此：

- **`f_ann_date` 是唯一可靠的版本时间戳**。同一 `(symbol, end_date)` 在不同 `report_type`（如初始 1、基准 4、修正前 5）下各自可能有多个 `f_ann_date` 版本；存储层保留所有物理行（PK 含 `report_type` 不互覆盖），snapshot 层按 `WHERE f_ann_date <= D` + 取最新即可。
- 同 `f_ann_date` 偶发 `update_flag=0/1` 两条**同值**冗余行（如 `688981.SH 20250829`），仅需任取一条避免 outer-join 三表重复。

**石川语义 vs. Tushare 取值**（仅作语义参考，snapshot 实现不依赖该映射）：

| 石川语义角色 | 实际含义 | 候选 Tushare `report_type` | 说明 |
|---|---|---|---|
| 初始（类型 1）| 当期首次披露的合并报表 | 1 | 已确认 |
| 基准（类型 2）| 次年年报随发的同年合并 | 4 | 仅语义参考 |
| 修正前（类型 3）| 更正公告前的初始报表（历史快照）| 5（或 3）| 仅语义参考 |
| 修正后（类型 4）| 更正后的值 | 1（含 `update_flag='1'`）| 仅语义参考 |

由于 4-case 框架本质上等价于「同 `(symbol, end_date)` 按 `f_ann_date` 取最新」（详见 §2），snapshot 不需要按 `report_type` 分支决策。

### 2. 石川 4 种披露场景与取数契约

verbatim 引用（出自《因子投资》第 3 章图 3.7）：

> (a) x 年年报在 t1 日披露后没有发生调整和更正，此时只有 1 条数据即类型 1。t1 日后如果要使用该财报数据，则提取类型 1 即可。
>
> (b) x 年年报在 t1 日首次披露，在 t2 日 x+1 年年报披露时顺便再一次披露了 x 年年报（基准报表）。因此，若在 t1 日到 t2 日之间使用该财报数据，则应提取类型 1；若在 t2 日后使用该数据，则提取类型 2。
>
> (c) 除了常规的初始报表和基准报表，还发生了数据更正，且更正发生在基准报表之前。在 t1 日披露 x 年年报，此时记录为类型 1；在 t2 日对年报进行了更正，此时更正后的数据记为类型 1，更正前的数据记为类型 3；在 t3 日披露调整数据，此时记录为类型 2。因此，如果在 t1 日到 t2 日使用 x 年年报，应提取类型 3；若在 t2 日到 t3 日之间使用该数据，则提取类型 1；若在 t3 日之后，提取类型 2 即可。
>
> (d) 更正发生在基准报表之后。在 t1 日披露 x 年年报，此时记为类型 1；在 t2 日披露 x 年年报基准报表，记为类型 2；t3 日发出 x 年年报更正公告，此时修改类型 1 和类型 2 为更正后的最新值，原来的数据分别记为类型 3 和类型 4。因此，如果在 t1 日到 t2 日期间使用该年报，则应提取类型 3；如果在 t2 日到 t3 日之间使用该数据，则提取类型 4；在 t3 日之后提取类型 2 即可。

D 日取数对照表（**语义参考**——实际实现无需按场景分支，原理见表后说明）：

| 场景 | D 位置 | 应读类型 | Tushare 实现（语义参考） |
|---|---|---|---|
| (a) | D ≥ t1 | 类型 1 | `report_type='1'` |
| (b) | t1 ≤ D < t2 | 类型 1 | `report_type='1'` |
| (b) | D ≥ t2 | 类型 2 | `report_type='4'` |
| (c) | t1 ≤ D < t2 | 类型 3 | t1 时点 `report_type='1'` |
| (c) | t2 ≤ D < t3 | 类型 1 | t2 时点 `report_type='1'` |
| (c) | D ≥ t3 | 类型 2 | `report_type='4'` |
| (d) | t1 ≤ D < t2 | 类型 3 | t1 时点 `report_type='1'` |
| (d) | t2 ≤ D < t3 | 类型 4 | t2 时点 `report_type='4'` |
| (d) | D ≥ t3 | 类型 2 | 最新 `report_type='4'` |

**关键洞察**：上表的每一行都等价于 "D 日按 `f_ann_date` 取同 `(symbol, end_date)` 下最新可见版本"。只要：

1. fetch 把 `report_type ∈ {1,2,3,4,5}` 全部入库；
2. 存储层 PK 加入 `report_type`，让不同 type 的物理行不互覆盖；

则按 `WHERE f_ann_date <= D + QUALIFY ROW_NUMBER OVER (PARTITION BY symbol, end_date ORDER BY f_ann_date DESC, update_flag DESC) = 1` 即可**自然实现** 4-case 框架，**无需** CASE WHEN report_type 优先级判定。

**`get_fina_snapshot(D)` 实际实现 SQL**（Round 1 落地）：

```sql
SELECT *
FROM income_q  -- balancesheet_q / cashflow_q 同理
WHERE f_ann_date <= ?
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY symbol, end_date
    ORDER BY f_ann_date DESC,
             update_flag DESC  -- 仅作同 f_ann_date 下的 stable tiebreaker
) = 1;
```

说明：

- `f_ann_date DESC` 取 D 日最新已见版本——这是 4-case 框架的核心：场景 (a) 取唯一的 `f_ann_date=t1`；场景 (b) 在 `D≥t2` 时自然取到 t2 时点的基准版本（因为它的 `f_ann_date=t2 > t1`）；场景 (c)/(d) 同理。
- `update_flag DESC` 仅为去重——同 `(symbol, end_date, f_ann_date)` 偶发 `update_flag=0/1` 两条同值行（如 `688981.SH 20250829`），任取一条不影响数值，但 outer-join 三表前必须去掉重复以免笛卡尔积。
- 同 `(symbol, end_date)` 下不同 `report_type` 的行（如初始 1、基准 4、修正前 5）通过 `f_ann_date DESC` 自然分先后；多 type 共存仅在物理层保留可溯源，对 snapshot 输出无影响。

### 3. 单季度数据推导

资产负债表是时点数据，所以单季度数据 = 报告期数据。利润表 / 现金流量表是累计值，单季度数据 = 当期累计 − 上一报告期累计。

| 报告期 | `end_date` 末月 | 利润表 / 现金流量表 公式 | 资产负债表 公式 |
|---|---|---|---|
| Q1 | 03 | `Q1 = report 原值` | `= report`（时点）|
| Q2 | 06 | `Q2 = H1 − Q1` | `= report` |
| Q3 | 09 | `Q3 = 9M − H1` | `= report` |
| Q4 | 12 | `Q4 = FY − 9M` | `= report` |

NaN 传播规则：若任一参与运算的环比报告期缺失（前一报告期 row 在 PIT 快照中不存在），对应单季度结果为 NaN。

**实现位置**：**不在 data 层完成**。data 层只负责 PIT 快照取数，单季度推导是因子构建时的 transform —— 不同因子需要的科目不同，统一在 data 层算等于浪费。助手函数 `single_quarter(panel, value_col)` 计划放在 `backtest/factor/transforms.py`，由 Round 2 (P0.5) 提供。本节定义公式契约。

### 4. TTM (Trailing Twelve Months)

**利润表 / 现金流量表（流量科目）**：

```
若 latest.end_date 末月 == 12（年报）：
    TTM = latest.value
否则若 LY_FY 存在 且 LY_same_period 存在：
    TTM = latest.value + LY_FY.value − LY_same_period.value
否则（兜底年化）：
    TTM = annualize(latest.value, end_date)
    其中系数：Q1=4, H1=2, 9M=4/3, FY=1
```

**资产负债表（时点科目）**：

**默认 `TTM_BS = latest reported value`**（即 D 日 PIT 快照中最新可见的报告期值）。

该默认与现有 `backtest/factor/builtin/barra/_common.py:latest_quarter_per_day` 行为一致；BS-only 因子（BTOP、AGRO）迁移到 `ttm()` 助手后**数值不变**。

简述但**不启用**的替代选项（仅留 future opt-in，需要时再加 flag）：

- 最近 4 个报告期平均
- 当前 + 同比 平均

**实现位置**：**不在 data 层完成**（同 §3）。助手函数 `ttm(panel, value_col, kind='flow'|'stock')` 计划放在 `backtest/factor/transforms.py`，由 Round 2 (P0.5) 提供。本节定义公式契约。

### 5. 当前实现差距 (gap analysis)

| 功能 | 现状 | 正确做法 | 状态 |
|---|---|---|---|
| Fetch 多 `report_type` | `fetcher/fundamentals_fetcher.py:18-25` 只保留 `'1'` | 入库 1/2/3/4/5 | Round 1 待落地 |
| Storage PK | `storage.py` PK=4 列 | 加入 `report_type` → 5 列 PK | Round 1 待落地 |
| Snapshot 取数 | 已按 `(f_ann_date DESC, update_flag DESC)` 取最新 | 保持现状（与 fetch 多 type 配合即实现 4-case）| Round 1 与 fetch+PK 一起验证 |
| 单季度推导 | 无 | `transforms.py:single_quarter` | Round 2 (P0.5) |
| TTM | `_common.py:annualize_ytd` 近似 | `transforms.py:ttm` | Round 2 (P0.5) |
| YoY / 同比 | 无 | `transforms.py:yoy` | Round 2 (P0.5) |
| 测试 | `tests/test_fundamentals.py` 仅随机抽样一致性 | + multi-type 一致性 + 单季度 + TTM + YoY 单测 | Round 1 / Round 2 分摊 |

**Factor 消费者影响清单**（六个 Barra 因子）：

| 因子 | 文件 | 当前依赖 | 未来 (Round 2 P0.6) 改读 |
|---|---|---|---|
| `f_barra_growth_egro` | `barra/growth.py` | `pit_quarterly_slope(inc_basic_eps)` | 同（slope 不受 TTM 影响）|
| `f_barra_quality_roa` | `barra/quality.py` | `annualize_ytd(inc_n_income_attr_p) / bs_total_assets` | `ttm(inc_n_income_attr_p) / bs_ttm` |
| `f_barra_quality_gp` | `barra/quality.py` | `annualize_ytd(rev) − annualize_ytd(cost)` | `ttm(rev) − ttm(cost)` |
| `f_barra_quality_agro` | `barra/quality.py` | `pit_quarterly_slope(bs_total_assets)` | 同 |
| `f_barra_value_btop` | `barra/value.py` | `bs_total_hldr_eqy_inc_min_int / circ_mv` | 同（BS 默认 = latest）|
| `f_barra_value_etop` | `barra/value.py` | `annualize_ytd(inc_n_income_attr_p) / circ_mv` | `ttm(inc_n_income_attr_p) / circ_mv` |

**数值漂移风险**：

- **Round 1（data 层）落地后**：现有 Barra 因子的输入会变化—— `get_fina_snapshot` 同 `(symbol, end_date)` 的取值从 "仅 type=1 中最新" 变成 "type 1/2/3/4/5 全集中按 f_ann_date 最新"。多数情况一致，但**修正后版本族（type=5 或 update_flag=1）可能改写 type=1 之前的"原始已知值"**——这是石川 4-case 框架的正确行为，但意味着历史回测结果会有微小数值差异。落地后需跑现有 Barra 因子的 IC sanity，对比 PR 前后。
- **Round 2（factor 层）迁移后**：BTOP / EGRO / AGRO 不变（BS 默认 = latest = 现行语义；slope 因子不触及年化）；ROA / GP / ETOP 会变（`annualize_ytd` → 真 TTM）。迁移时再做一次前后 IC sanity。

### 6. 下一步路线图

落到 [`TODO.md`](../../TODO.md) P0 §"基本面因子修正"，分两轮：

**Round 1（data 层，代码即将落地）**

- (P0.1) `fundamentals_fetcher.py`：`_keep_consolidated` 放宽到 `report_type ∈ {1,2,3,4,5}`（合并口径全集，剔除母公司 6/11/12）
- (P0.2) `storage.py`：三张表 PK 改为 `(symbol, end_date, f_ann_date, update_flag, report_type)`。DuckDB 不支持 `ALTER PRIMARY KEY` → init 时若检测到旧 schema 则 drop 三表
- (P0.3) `get_fina_snapshot`：保持现有 SQL（`WHERE f_ann_date <= ? + QUALIFY ORDER BY f_ann_date DESC, update_flag DESC`），不引入 CASE-rank（实证 §1 表明 update_flag 不可靠 + f_ann_date 已足够实现 4-case 语义）
- (P0.4) backfill：代码改完，用户手动跑 `python -m backtest.data.backfill.fundamentals`（或重新 `cold_start`）

**Round 2（factor 层，下一轮）**

- (P0.5) `backtest/factor/transforms.py`：新增 `single_quarter(panel, value_col)` / `ttm(panel, value_col, kind='flow'|'stock')` / `yoy(panel, value_col)` 助手函数（基于 PIT 多期快照）
- (P0.6) Barra 因子迁移：ROA / GP / ETOP 改用 `ttm`；前后 IC sanity 对比，记录数值漂移
- (P0.7) 测试：multi-type fetch 一致性 + 5-列 PK + snapshot 行为（Round 1）+ 单季度公式 + TTM 公式 + YoY 单测（Round 2）

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

# cyq_chips (筹码分布) —— 见 backtest/data/cyq_storage.py
CyqStorage.get_cyq(date, symbol)                              # 单票单日 → [price, percent]
CyqStorage.get_cyq_panel(date, symbols=None)                  # 某日横截面 → [date, symbol, n_bins, prices, percents]
CyqStorage.get_cyq_history(symbol, start, end)                # 单票历史 → [date, n_bins, prices, percents]
CyqStorage.get_weighted_prices(date, symbols=None)            # SQL 层：筹码重心
CyqStorage.get_peak_prices(date, symbols=None)                # SQL 层：峰值档位价格
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
