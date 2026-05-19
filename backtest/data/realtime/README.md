# `data/realtime/` — Intraday quote adapters

This subpackage adds **realtime** market data on top of the EOD Tushare
channel in `backtest/data/`. Output frames are intentionally column-compatible
with `market_daily` so downstream code (signals / push / position tracking)
does not need to branch on data source.

## Why a separate channel

Tushare Pro publishes most A-share daily fields on a **T+0 evening** schedule
(usually after 17:00 CST). For §6 signal generation that means the earliest
trade decision is "next morning, based on yesterday's close". `xtquant` reads
直接 from the local QMT / 迅投 client (国金, 招商, etc.) and returns L1 ticks
plus minute bars **during the session**, which moves signal generation from
overnight to intraday.

Realtime data is **never persisted to `market_daily.duckdb`** — it is volatile
by definition and would corrupt the EOD audit trail. Consumers treat it as a
hot snapshot.

## What's in scope

- `xtquant_quote.fetch_full_tick(symbols)` — L1 snapshot, `market_daily`-shaped DataFrame
- `xtquant_quote.fetch_bars(symbols, period, count)` — 1m / 5m / ... / 1d bars
- `xtquant_quote.fetch_instrument_details(symbols)` — name, limit_up, limit_down, pre_close

## What's **out** of scope

- Order placement / position queries / fund queries — these require account
  authorization and an unlocked trader session. They belong in a future
  `trading/broker/` module (CLAUDE.md §6.5, Phase 2).
- Any persistence of realtime data.

## Setup (macOS / Linux)

`xtquant` ships only as a Windows DLL bundle. The supported route on
non-Windows hosts:

1. Install **CrossOver** (or plain Wine) + a QMT mini client from any
   supporting broker (国金 / 招商 / 国信 / ...).
2. Inside the Wine prefix, install **Windows Python 3.9** and `pip install
   xtquant` from the broker's wheel.
3. Run the realtime module under that Wine Python — *not* native macOS Python.
   The rest of AutoQuant continues to run under native conda `AutoQuant`.

### macOS gotcha: down_queue accumulation

QMT mini writes per-connection cache dirs (`down_queue_win_<ts>`,
`down_queue_xtmodel-N`) of ~72 MB each under
`~/Downloads/gjzqqmt_binary/userdata_mini/` and never garbage-collects them.
Left unchecked these will fill a small SSD within weeks. Add a `launchd`
cleanup job:

```xml
<!-- ~/Library/LaunchAgents/com.autoquant.qmt-cleanup.plist -->
<plist version="1.0">
<dict>
  <key>Label</key><string>com.autoquant.qmt-cleanup</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string>
    <string>-c</string>
    <string>find ~/Downloads/gjzqqmt_binary/userdata_mini -maxdepth 1 -type d \
      -name 'down_queue_*' -mtime +1 -print -exec rm -rf {} +</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>4</integer><key>Minute</key><integer>30</integer></dict>
</dict>
</plist>
```

Then `launchctl load ~/Library/LaunchAgents/com.autoquant.qmt-cleanup.plist`.

## Environment variables

See `.env.example`. The quote adapter itself does not consume credentials —
read-only market data requires only that the QMT client be running and logged
in interactively (or via the broker's own SSO). Variables exist for the future
broker module:

| Var                | Purpose                                          |
|--------------------|--------------------------------------------------|
| `QMT_PYTHON_PATH`  | Path to Wine Python 3.9 (CLI launcher convenience) |
| `QMT_USERDATA_DIR` | Override default `userdata_mini` location         |

**Never** commit `.env` with real account numbers. The bundled
`qmt_market_test.py` script (see PR description) shows the full integration
end-to-end with anonymous symbols only.
