"""Realtime intraday market data sources.

These complement the EOD Tushare channel with in-session quotes and minute bars.
Realtime data is volatile and is NOT persisted to ``market_daily``; consumers
read it as a hot snapshot only.
"""
