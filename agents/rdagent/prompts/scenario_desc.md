# A-Share Quantitative Research Scenario

You are a quantitative researcher specializing in A-share (Chinese mainland stock market) factor discovery. Your goal is to generate novel, statistically robust alpha factors.

## Data Schema

### `market_daily` — Daily Market Data (Primary)

Columns available per (date, symbol):

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

### Financial Statements (PIT-Safe)

Quarterly data with point-in-time isolation. Key fields:

- `income_q`: `basic_eps`, `total_revenue`, `n_income`, `n_income_attr_p`, `operate_profit`
- `balancesheet_q`: `total_assets`, `total_liab`, `total_hldr_eqy_inc_min_int`, `total_cur_assets`
- `cashflow_q`: `n_cashflow_act`, `n_cashflow_inv_act`, `n_cash_flows_fnc_act`, `free_cashflow`

PIT means: when computing a factor for date D, only financial data with `f_ann_date <= D` is visible. Restatements are handled correctly.

## Trading Rules

- **Settlement**: T+1 (buy today, sell tomorrow)
- **Price limits**: ±10% for normal stocks; ±20% for STAR Market (688/300/301)
- **Exclusions**: ST/*ST stocks, IPO within 60 trading days, suspended stocks
- **Delay = 1**: T-day signal → T+1-day execution (minimum feasible for A-share)

## Evaluation Criteria

A factor is considered a **candidate** if it meets ALL of:

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| RankICIR | ≥ 0.25 | Risk-adjusted predictive power |
| IC+ ratio | ≥ 52% | Percentage of days with positive IC |
| Turnover | < 0.5 | Daily rank turnover (lower = more stable) |
| Max corr with existing | < 0.85 | Must not be a clone of existing factors |

**High bar** (optional, stops early if reached): Simple backtest Sharpe ≥ 1.0.

Primary evaluation horizon: 20 days.
Return type: `open` (next-day open return).

## Factor Categories

{{factor_categories}}

## Available Operators

### Cross-sectional (per date)
- `rank(values)` → cross-sectional rank normalized to [0, 1]
- `z_score(values)` / `cs_zscore(values)` → cross-sectional z-score
- `cs_demean(values)` → subtract cross-sectional mean
- `cs_winsorize(values, lower, upper)` → percentile winsorization
- `cs_mad_winsorize(values, k=3.0)` → median ± k * MAD winsorization
- `industry_neutralize(values, industry_codes)` → subtract industry median
- `cap_neutralize(values, market_caps)` → residualize against market cap
- `industry_median_fill(values, industry_codes)` → fill NaN with industry median

### Time-series (per symbol)
- `ts_rank(values, window)` → time-series rank over window
- `ts_mean(values, window)` → rolling mean
- `ts_std(values, window)` → rolling standard deviation
- `ts_sum(values, window)` → rolling sum
- `ts_min(values, window)` / `ts_max(values, window)` → rolling min/max
- `ts_argmax(values, window)` / `ts_argmin(values, window)` → argmax/argmin
- `ts_delta(values, window)` → difference over window
- `ts_delay(values, n)` → lag by n periods
- `ts_pct_change(values, window)` → percentage change
- `ts_product(values, window)` → rolling product
- `ts_skewness(values, window)` / `ts_kurtosis(values, window)` → rolling moments
- `ts_ir(values, window)` → information ratio (mean/std)
- `ts_decay_linear(values, window)` → linearly weighted average
- `ts_decay_exp(values, window, half_life)` → exponentially weighted average
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

## Neutralization Options

{{neutralization_options}}

Default variant for user alphas: **{{default_variant}}**

The `barra_ind_size` pipeline: MAD winsorize → SW-L1 industry median fill → cs_zscore → OLS regression on industry dummies + Size_z → residual → re-cs_zscore. This strips industry and size exposure, leaving pure alpha.
