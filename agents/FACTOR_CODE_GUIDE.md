# AutoQuant Factor Code Guide for LLM

This document is the single source of truth for writing bug-free factor code in the AutoQuant framework. LLM code generators should follow every rule listed here.

---

## 1. Data Schema

The `panel` DataFrame passed to your factor function contains columns based on the `data_sources` declared in `@register(...)`. Only columns from declared sources are present.

### 1.1 `market_daily` — Daily Market Data

These columns are always available when `data_sources` includes `"market_daily"`:

| Column | Description |
|--------|-------------|
| `open` / `high` / `low` / `close` | OHLC prices |
| `volume` / `amount` | Trading volume (shares) and amount (CNY) |
| `pre_close` | Previous close price |
| `change` / `pct_chg` | Absolute and percentage change |
| `adj_factor` | Cumulative adjustment factor for splits/dividends |
| `is_st` | Boolean: is the stock ST/*ST? |
| `list_date` | IPO listing date |
| `limit_up` / `limit_down` | Price limit boundaries for the day |
| `turnover_rate` / `turnover_rate_f` | Turnover rate (free-float adjusted) |
| `volume_ratio` | Current volume / 5-day average volume |
| `pe` / `pe_ttm` | Price-to-earnings (trailing) |
| `pb` | Price-to-book |
| `ps` / `ps_ttm` | Price-to-sales |
| `dv_ratio` / `dv_ttm` | Dividend yield |
| `total_share` / `float_share` / `free_share` | Share counts |
| `total_mv` / `circ_mv` | Total and circulating market cap (CNY) |

### 1.2 `income_q` — Quarterly Income Statement (prefix: `inc_`)

**CRITICAL**: Financial statement columns carry prefixes. The raw column name `total_revenue` does **NOT** exist in `panel`. Use `inc_total_revenue`.

Common `inc_` columns:

| Column | Description |
|--------|-------------|
| `inc_basic_eps` | Basic EPS |
| `inc_total_revenue` | Total revenue |
| `inc_revenue` | Operating revenue |
| `inc_n_income` | Net income (total) |
| `inc_n_income_attr_p` | Net income attributable to parent |
| `inc_operate_profit` | Operating profit |
| `inc_total_profit` | Total profit |
| `inc_income_tax` | Income tax |
| `inc_ebit` | EBIT |
| `inc_ebitda` | EBITDA |
| `inc_total_cogs` | Total cost of goods sold |
| `inc_oper_cost` | Operating cost |
| `inc_sell_exp` | Selling expenses |
| `inc_admin_exp` | Administrative expenses |
| `inc_fin_exp` | Financial expenses |
| `inc_rd_exp` | R&D expenses |
| `inc_non_oper_income` | Non-operating income |
| `inc_non_oper_exp` | Non-operating expenses |
| `inc_minority_gain` | Minority interest gain |
| `inc_oth_compr_income` | Other comprehensive income |
| `inc_t_compr_income` | Total comprehensive income |
| `inc_compr_inc_attr_p` | Comprehensive income attributable to parent |
| `inc_continued_net_profit` | Continued operations net profit |

### 1.3 `balancesheet_q` — Quarterly Balance Sheet (prefix: `bs_`)

| Column | Description |
|--------|-------------|
| `bs_total_assets` | Total assets |
| `bs_total_liab` | Total liabilities |
| `bs_total_hldr_eqy_inc_min_int` | Total shareholders equity including minority interest |
| `bs_total_hldr_eqy_exc_min_int` | Total shareholders equity excluding minority interest |
| `bs_total_cur_assets` | Total current assets |
| `bs_total_nca` | Total non-current assets |
| `bs_money_cap` | Cash and cash equivalents |
| `bs_trad_asset` | Trading financial assets |
| `bs_inventories` | Inventories |
| `bs_accounts_receiv` | Accounts receivable |
| `bs_notes_receiv` | Notes receivable |
| `bs_oth_receiv` | Other receivables |
| `bs_fix_assets` | Fixed assets |
| `bs_cip` | Construction in progress |
| `bs_intan_assets` | Intangible assets |
| `bs_goodwill` | Goodwill |
| `bs_lt_eqt_invest` | Long-term equity investments |
| `bs_fa_avail_for_sale` | Financial assets available for sale |
| `bs_total_cur_liab` | Total current liabilities |
| `bs_total_ncl` | Total non-current liabilities |
| `bs_st_borr` | Short-term borrowings |
| `bs_lt_borr` | Long-term borrowings |
| `bs_bond_payable` | Bonds payable |
| `bs_notes_payable` | Notes payable |
| `bs_acct_payable` | Accounts payable |
| `bs_st_bonds_payable` | Short-term bonds payable |
| `bs_lt_payable` | Long-term payables |
| `bs_deferred_inc` | Deferred income |
| `bs_defer_tax_liab` | Deferred tax liabilities |
| `bs_surplus_rese` | Surplus reserve |
| `bs_undistr_porfit` | Undistributed profit |
| `bs_cap_rese` | Capital reserve |
| `bs_treasury_share` | Treasury shares |
| `bs_minority_int` | Minority interest |

### 1.4 `cashflow_q` — Quarterly Cash Flow Statement (prefix: `cf_`)

| Column | Description |
|--------|-------------|
| `cf_n_cashflow_act` | Net cash flow from operating activities |
| `cf_n_cashflow_inv_act` | Net cash flow from investing activities |
| `cf_n_cash_flows_fnc_act` | Net cash flow from financing activities |
| `cf_free_cashflow` | Free cash flow |
| `cf_net_profit` | Net profit (reconciliation start) |
| `cf_c_fr_sale_sg` | Cash from sales of goods |
| `cf_c_inf_fr_operate_a` | Cash inflow from operating activities |
| `cf_st_cash_out_act` | Cash outflow from operating activities |
| `cf_stot_inflows_inv_act` | Total inflows from investing activities |
| `cf_stot_out_inv_act` | Total outflows from investing activities |
| `cf_stot_cash_in_fnc_act` | Total cash inflow from financing |
| `cf_stot_cashout_fnc_act` | Total cash outflow from financing |
| `cf_c_recp_borrow` | Cash from borrowings |
| `cf_proc_issue_bonds` | Proceeds from issuing bonds |
| `cf_c_recp_cap_contrib` | Cash from capital contributions |
| `cf_n_incr_cash_cash_equ` | Net increase in cash and equivalents |
| `cf_c_cash_equ_beg_period` | Cash at beginning of period |
| `cf_c_cash_equ_end_period` | Cash at end of period |

### 1.5 Special Meta Columns

When any financial source (`income_q`, `balancesheet_q`, `cashflow_q`) is included, `panel` also contains:

| Column | Description |
|--------|-------------|
| `end_date` | The quarter end date (e.g. "20240331") for the financial data row |

---

## 2. How Data Is Fetched (PIT Isolation)

### 2.1 Point-in-Time (PIT) Principle

When computing a factor for trade date `D`, only financial data with `f_ann_date <= D` is visible. This prevents look-ahead bias. The system handles restatements correctly: if a company restated its 2022 annual report on 2023-06-30, then:
- For dates `D <= 2023-06-29`, the pre-restatement version is used
- For dates `D >= 2023-06-30`, the restated version is used

### 2.2 Non-Event-Driven Factors (most factors)

For regular factors that need a daily panel:

```python
# The framework calls get_fina_snapshot_range(start, end) internally
# and merges the result into panel as left-join on (date, symbol)
# Your factor function receives the merged panel directly
```

The `panel` DataFrame is in **long format**: one row per `(date, symbol)`, with financial columns carrying the `inc_`/`bs_`/`cf_` prefixes.

### 2.3 Event-Driven Factors (TTM / slope)

For factors that compute TTM or slope (e.g., Growth EGRO, Quality AGRO):

```python
@register(
    "f_demo_event",
    data_sources=["income_q"],
    parameters={"event_driven": True, "fina_columns": ["inc_total_revenue"]},
)
def f_demo_event(panel: pd.DataFrame, market_storage, start_date, end_date) -> pd.Series:
    # panel is EMPTY for event_driven factors — you pull your own data
    events = market_storage.get_fina_event_panel(
        start=start_date,
        end=end_date,
        columns=["inc_total_revenue"],
        last_n_quarters=20,
    )
    # events has columns: symbol, announce_end_date, f_ann_date, next_f_ann_date, end_date, inc_total_revenue
    # Compute per-event scalar, then expand to daily panel
    ...
```

**Only use `event_driven=True` when you need TTM/slope helpers.** For simple ratio factors (e.g., `inc_n_income / bs_total_assets`), use regular non-event-driven mode.

### 2.4 No `fina_indicator` Table

**The project does NOT use Tushare's `pro.fina_indicator` table.** That table lacks `update_flag` and `f_ann_date`, making proper PIT isolation impossible. All financial factors must use the raw three tables (`income_q`, `balancesheet_q`, `cashflow_q`) with prefixed column names.

---

## 3. Factor Code Structure

### 3.1 Market-Data-Only Factor (Demo)

```python
from backtest.factor.registry import register
from backtest.factor.transforms import ts_mean, rank
import pandas as pd
import numpy as np

@register(
    "f_demo_market",
    name="20日收益率均值",
    category="momentum",
    data_sources=["market_daily"],
    description="20-day rolling mean of daily returns",
    parameters={"window": 20},
    variant="barra_ind_size",
)
def f_demo_market(panel: pd.DataFrame) -> pd.Series:
    """Return 20-day rolling mean of pct_chg.

    panel: DataFrame with columns from market_daily
    """
    # Step 1: Set index to (date, symbol) — REQUIRED before using transforms
    panel = panel.set_index(["date", "symbol"])

    # Step 2: Extract the column you need
    ret = panel["pct_chg"] / 100  # pct_chg is in percent

    # Step 3: Apply transforms — they expect MultiIndex (date, symbol) Series
    result = ts_mean(ret, window=20)

    # Step 4: Return the Series — the framework handles the rest
    return result
```

### 3.2 Financial Factor (Demo)

```python
from backtest.factor.registry import register
from backtest.factor.transforms import rank
import pandas as pd
import numpy as np

@register(
    "f_demo_roe",
    name="ROE_TTM",
    category="quality",
    data_sources=["income_q", "balancesheet_q"],
    description="Net income (TTM) / Shareholders equity",
    variant="barra_ind_size",
)
def f_demo_roe(panel: pd.DataFrame) -> pd.Series:
    """Return ROE = net income / equity.

    CRITICAL: Use prefixed column names:
    - income_q columns → inc_*
    - balancesheet_q columns → bs_*
    """
    panel = panel.set_index(["date", "symbol"])

    # CORRECT: use prefixed column names
    net_income = panel["inc_n_income_attr_p"]
    equity = panel["bs_total_hldr_eqy_inc_min_int"]

    # Avoid division by zero / negative equity
    roe = net_income / equity.where(equity > 0, np.nan)

    return roe
```

### 3.3 Composite Factor (Demo)

```python
from backtest.factor.registry import register
from backtest.factor.transforms import rank
import pandas as pd

@register(
    "f_demo_composite",
    name="Value Composite",
    category="value",
    data_sources=["factors_daily"],  # Reads from admitted factor library
    description="Combine EP + BP ranks",
    variant="none",  # Components are already neutralized
)
def f_demo_composite(panel: pd.DataFrame, factor_storage, start_date, end_date) -> pd.Series:
    """Composite of earnings-to-price and book-to-price.

    Reads admitted factors from factor_storage.
    """
    from backtest.factor.storage import FactorLibrary

    with FactorLibrary() as lib:
        ep = lib.get_factor("f_ep", start=start_date, end=end_date)
        bp = lib.get_factor("f_bp", start=start_date, end=end_date)

    # Merge and compute composite
    merged = ep.merge(bp, on=["date", "symbol"], suffixes=("_ep", "_bp"))
    merged = merged.set_index(["date", "symbol"])

    ep_rank = rank(merged["value_ep"])
    bp_rank = rank(merged["value_bp"])

    return (ep_rank + bp_rank) / 2
```

---

## 4. Available Operators (from `backtest.factor.transforms`)

### Cross-sectional (per date)
- `rank(values)` → cross-sectional rank normalized to [0, 1]
- `cs_zscore(values)` → cross-sectional z-score: ``(x - mean) / std``
- `cs_demean(values)` → subtract cross-sectional mean
- `cs_winsorize(values, lower, upper)` → percentile winsorization
- `cs_mad_winsorize(values, k=3.0)` → median ± k * MAD winsorization
- `cs_ols_residualize(values, X)` → OLS residualize against design matrix `X`
- `industry_neutralize(values, industry_codes)` → subtract industry median
- `cap_neutralize(values, market_caps)` → residualize against market cap
- `industry_median_fill(values, industry_codes)` → fill NaN with industry median

### Time-series (per symbol)
- `z_score(values, window)` → rolling z-score per symbol: ``(x - rolling_mean) / rolling_std``
- `ts_rank(values, window)` → time-series rank over window
- `ts_mean(values, window)` → rolling mean
- `ts_std(values, window)` → rolling standard deviation
- `ts_sum(values, window)` → rolling sum
- `ts_min(values, window)` / `ts_max(values, window)` → rolling min/max
- `ts_argmax(values, window)` / `ts_argmin(values, window)` → argmax/argmin
- `ts_delta(values, d)` → difference over `d` periods
- `ts_delay(values, n)` → lag by `n` periods
- `ts_pct_change(values, d)` → percentage change over `d` periods
- `ts_product(values, window)` → rolling product
- `ts_skewness(values, window)` / `ts_kurtosis(values, window)` → rolling moments
- `ts_ir(values, window)` → information ratio (mean/std)
- `ts_decay_linear(values, window)` → linearly weighted average
- `ts_decay_exp(values, window, *, halflife=10.0)` → exponentially weighted average
- `ts_corr(a, b, window)` → rolling correlation
- `ts_covariance(a, b, window)` → rolling covariance

### Element-wise
- `abs_(values)`, `sign(values)`, `log(values)`, `sqrt(values)`
- `signed_power(values, power)` → sign(x) * |x|^power
- `inverse(values)` → 1 / x
- `if_else(condition, true_val, false_val)`

### Fundamental helpers
- `single_quarter(panel, value_col)` → extract single-quarter value from cumulative data
- `ttm(panel, value_col)` → trailing twelve months
- `yoy(panel, value_col)` → year-over-year growth

All operators accept a MultiIndex `(date, symbol)` Series and return a Series of the same shape.

---

## 5. Common Pitfalls (Read Carefully)

### 5.1 Column Name Prefixes

**WRONG**:
```python
def bad_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    revenue = panel["total_revenue"]  # DOES NOT EXIST
    assets = panel["total_assets"]    # DOES NOT EXIST
    return revenue / assets
```

**CORRECT**:
```python
def good_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    revenue = panel["inc_total_revenue"]  # CORRECT: inc_ prefix
    assets = panel["bs_total_assets"]     # CORRECT: bs_ prefix
    return revenue / assets.where(assets > 0, np.nan)
```

### 5.2 Missing `fina_indicator` Table

**WRONG**:
```python
# Do NOT do this — fina_indicator does not exist in the database
data_sources=["fina_indicator"]
```

**CORRECT**:
```python
# Use the raw three tables with prefixes
data_sources=["income_q", "balancesheet_q"]
```

### 5.3 Forgetting to Set Index

**WRONG**:
```python
def bad_factor(panel: pd.DataFrame) -> pd.Series:
    # ts_mean needs MultiIndex (date, symbol)
    return ts_mean(panel["pct_chg"], 20)  # FAILS: flat index
```

**CORRECT**:
```python
def good_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    return ts_mean(panel["pct_chg"], 20)
```

### 5.4 Manual Neutralization

**WRONG**:
```python
def bad_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    raw = panel["close"] / panel["open"]
    # Do NOT neutralize yourself — the pipeline does this
    return industry_neutralize(raw, panel["industry_code"])
```

**CORRECT**:
```python
def good_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    raw = panel["close"] / panel["open"]
    return raw  # Pipeline handles neutralization via variant="barra_ind_size"
```

### 5.5 Division by Zero / Negative Values

**WRONG**:
```python
def bad_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    return panel["inc_n_income"] / panel["bs_total_assets"]  # May produce inf
```

**CORRECT**:
```python
def good_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    assets = panel["bs_total_assets"]
    return panel["inc_n_income"] / assets.where(assets > 0, np.nan)
```

### 5.6 Return Type Must Be `pd.Series`

**WRONG**:
```python
def bad_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    return panel["close"]  # Returns Series — this is actually OK
    # But: return panel[["close"]]  # Returns DataFrame — WRONG
```

**CORRECT**:
```python
def good_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    return panel["close"]  # Single-column Series with MultiIndex
```

### 5.7 NaN Handling

Most operators preserve NaN automatically. Do **NOT** fill NaN with arbitrary values (0, -999, etc.) — this corrupts the factor's statistical properties.

**WRONG**:
```python
def bad_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    return panel["inc_n_income"].fillna(0)  # DON'T do this
```

**CORRECT**:
```python
def good_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    return panel["inc_n_income"]  # Let NaN propagate naturally
```

### 5.8 Using Raw Prices for Time-Series Calculations (Price Jumps from Splits/Dividends)

A-share stocks undergo stock splits, rights issues, and cash dividends that cause raw prices (`open`, `high`, `low`, `close`) to jump on ex-dividend dates. **Any time-series calculation involving prices (returns, rolling means, price-based ratios) MUST use adjusted prices.**

**WRONG**:
```python
def bad_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    close = panel["close"]  # RAW: will jump on ex-dividend dates
    ret = close.pct_change()  # WRONG: price jumps create fake returns
    return ts_mean(ret, window=20)  # Garbage signal
```

**CORRECT**:
```python
def good_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    adj_close = panel["close"] * panel["adj_factor"]  # Adjusted for splits/dividends
    ret = adj_close.pct_change()  # CORRECT: no price jumps
    return ts_mean(ret, window=20)
```

**Rule of thumb**: If your factor uses `open`, `high`, `low`, or `close` from `market_daily` for any time-series operation (pct_change, rolling mean/std, difference between dates, or comparing prices at different times), multiply by `adj_factor` first:

```python
adj_open = panel["open"] * panel["adj_factor"]
adj_high = panel["high"] * panel["adj_factor"]
adj_low = panel["low"] * panel["adj_factor"]
adj_close = panel["close"] * panel["adj_factor"]
```

Exception: `change` and `pct_chg` are already adjusted — they represent the day's price change as reported by the exchange. Similarly, market cap columns (`total_mv`, `circ_mv`) and turnover columns are already adjusted.

### 5.9 ST Stocks (Special Treatment)

ST (Special Treatment) and *ST stocks trade under different rules: ±5% price limit instead of ±10%, higher delisting risk, and generally distorted trading behavior.

The `is_st` column is a boolean: `True` = ST or *ST.

ST 剔除由 `strategy.universe.exclude_st: true`（config.yaml）在策略层处理。**因子代码不需要手动屏蔽**，pipeline 的去极值/中性化/截面排名步骤已考虑缺失值传播。

### 5.10 New IPO Stocks (Listing Date)

Newly listed stocks have unstable trading patterns in their first year: extreme volatility, lock-up period effects, and incomplete financial data history. The `list_date` column (format: `"20240101"`) gives the IPO date.

```python
def good_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    raw_signal = ...
    # Exclude stocks listed within the last 252 trading days (~1 year)
    ipo_cutoff = pd.Timestamp("2016-01-01") + pd.DateOffset(days=-252)  # or pass from config
    # Simpler: use the pipeline's built-in exclusion (strategy config exclude_new_ipo_days: 252)
    return raw_signal
```

Note: the strategy module's universe config already handles IPO exclusion (`exclude_new_ipo_days: 252`). Masking at the factor level is optional but helps avoid IPO-distorted ranks in early periods.

### 5.11 Limit-Up / Limit-Down Stocks

When a stock hits its daily price limit, volume collapses to near-zero and the closing price is artificial.

涨跌停过滤由 simulation 层在交易执行时处理。**因子代码不需要手动屏蔽**。

This is especially important for volume-based factors: a stock hitting limit-up with zero sell volume should not be treated as "low volume / weak interest."

### 5.12 Financial Data is Quarterly, Not Daily

When your `data_sources` include `income_q`, `balancesheet_q`, or `cashflow_q`, the financial columns (`inc_*`, `bs_*`, `cf_*`) are **quarterly values that repeat every trading day** until the next quarter's financial report is announced. This has critical implications:

1. **Time-series transforms on financial columns are meaningless**: `ts_mean(panel["inc_n_income"], window=20)` gives a 20-day rolling mean of the same quarterly value — not a smooth trend.
2. **`pct_change()` on financial columns produces zeros for ~60 days, then a single jump** when the new quarter is filed.
3. **For growth/slope factors, use event-driven mode** (`event_driven=True` in `@register`). See Pattern C in Quick Reference.

**WRONG**:
```python
def bad_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    # WRONG: ts_delta on quarterly data makes no sense
    revenue_growth = ts_delta(panel["inc_total_revenue"], d=60)
    return revenue_growth
```

**CORRECT (simple ratio factor — non-event-driven is fine)**:
```python
def good_factor(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    # OK: cross-sectional ratio, same quarter's data for all stocks on a given day
    return panel["inc_n_income"] / panel["bs_total_assets"].where(panel["bs_total_assets"] > 0, np.nan)
```

**CORRECT (growth/slope — use event_driven=True)**:
```python
@register("f_growth", data_sources=["income_q"],
          parameters={"event_driven": True, "fina_columns": ["inc_total_revenue"]})
def f_growth(panel, market_storage, start_date, end_date):
    events = market_storage.get_fina_event_panel(start=start_date, end=end_date,
                                                   columns=["inc_total_revenue"], last_n_quarters=8)
    # Compute YoY growth per event, then expand to daily
    ...
```

### 5.13 Volume and Amount Units

- `volume` is in **shares** (股), not lots (手). 1手 = 100股. For liquidity calculations, divide by 100 if you want lots.
- `amount` is in **CNY yuan** (元). This is `volume × price` and is the preferred measure for dollar-volume / liquidity.
- `turnover_rate` is the percentage of float shares traded (`volume / float_share`). `turnover_rate_f` is the free-float version (preferred).
- `volume_ratio` is `volume / 5-day average volume` — already normalized, useful for volume anomaly detection without worrying about units.

**WRONG**:
```python
# Comparing raw volume across stocks — large-cap stocks always dominate
vol_signal = rank(panel["volume"])
```

**CORRECT**:
```python
# Use turnover_rate for cross-sectional volume comparison
vol_signal = rank(panel["turnover_rate"])
# Or use amount for dollar-volume
amt_signal = rank(panel["amount"])
```

---

## 6. Quick Reference: Factor Function Signature Patterns

### Pattern A: Market data only (simplest)
```python
def f_xxx(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    ...
    return result
```

### Pattern B: Financial data (non-event-driven)
```python
def f_xxx(panel: pd.DataFrame) -> pd.Series:
    panel = panel.set_index(["date", "symbol"])
    # Use inc_*, bs_*, cf_ prefixed columns
    ...
    return result
```

### Pattern C: Event-driven (TTM / slope)
```python
def f_xxx(panel: pd.DataFrame, market_storage, start_date, end_date) -> pd.Series:
    # panel is empty — pull your own event panel
    events = market_storage.get_fina_event_panel(...)
    ...
    return result
```

### Pattern D: Composite (reads library factors)
```python
def f_xxx(panel: pd.DataFrame, factor_storage, start_date, end_date) -> pd.Series:
    # Read other factors from factor_storage
    ...
    return result
```
