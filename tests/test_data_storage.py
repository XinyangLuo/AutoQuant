"""Offline unit tests for data storage and fetch transforms."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.data.fetcher.daily_fetcher import (
    merge_adj_factor,
    merge_daily_basic,
    merge_limit_prices,
    merge_margin_detail,
    merge_moneyflow,
    merge_st_status,
    transform_daily,
)
from backtest.data.fetcher.index_members_fetcher import densify_to_trade_dates
from backtest.data.fetcher.auction_fetcher import transform_stock_auction
from backtest.data.backfill.stock_auction import backfill_stock_auction
from backtest.data.tushare_client import _find_project_root
from backtest.data.storage import MarketStorage


@pytest.fixture
def tmp_market_storage(tmp_path, monkeypatch):
    import backtest.data.storage as storage_module

    db_dir = tmp_path / "duckdb"
    monkeypatch.setattr(storage_module, "DATA_DIR", db_dir)
    monkeypatch.setattr(storage_module, "DB_PATH", db_dir / "market.duckdb")

    with MarketStorage() as storage:
        yield storage


def _meta(
    symbol: str,
    end_date: str,
    f_ann_date: str,
    *,
    update_flag: str = "0",
    report_type: str = "1",
) -> dict:
    return {
        "symbol": symbol,
        "end_date": end_date,
        "ann_date": f_ann_date,
        "f_ann_date": f_ann_date,
        "report_type": report_type,
        "comp_type": "1",
        "end_type": "1",
        "update_flag": update_flag,
    }


def test_tushare_client_finds_worktree_project_root():
    root = _find_project_root()

    assert (root / "AGENTS.md").exists()
    assert (root / "environment.yml").exists()


class TestIndexMembers:
    def test_densify_uses_only_visible_snapshot_not_future_members(self):
        monthly = pd.DataFrame(
            [
                {
                    "index_code": "000300.SH",
                    "symbol": "OLD.SZ",
                    "trade_date": pd.Timestamp("2024-01-02").date(),
                    "weight": 1.0,
                },
                {
                    "index_code": "000300.SH",
                    "symbol": "NEW.SZ",
                    "trade_date": pd.Timestamp("2024-01-05").date(),
                    "weight": 2.0,
                },
            ]
        )
        trade_dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        ).date.tolist()

        dense = densify_to_trade_dates(monthly, trade_dates)

        by_date = {
            pd.Timestamp(date).strftime("%Y%m%d"): set(grp["symbol"])
            for date, grp in dense.groupby("trade_date")
        }
        assert by_date["20240103"] == {"OLD.SZ"}
        assert by_date["20240104"] == {"OLD.SZ"}
        assert by_date["20240105"] == {"NEW.SZ"}

    def test_get_index_members_requires_exact_trade_date_no_latest_fallback(
        self, tmp_market_storage
    ):
        storage = tmp_market_storage
        storage.insert_index_members(
            pd.DataFrame(
                [
                    {
                        "index_code": "000300.SH",
                        "symbol": "OLD.SZ",
                        "trade_date": pd.Timestamp("2024-01-02").date(),
                        "weight": 1.0,
                    },
                    {
                        "index_code": "000300.SH",
                        "symbol": "NEW.SZ",
                        "trade_date": pd.Timestamp("2024-01-05").date(),
                        "weight": 2.0,
                    },
                ]
            )
        )

        assert storage.get_index_members("000300.SH", "20240102") == {"OLD.SZ"}
        assert storage.get_index_members("000300.SH", "20240104") == set()
        assert storage.get_index_members("000300.SH", "20240105") == {"NEW.SZ"}


class TestMarketStorageFundamentals:
    def test_get_fina_snapshot_uses_latest_visible_version_per_table(
        self, tmp_market_storage
    ):
        storage = tmp_market_storage
        storage.insert_income(
            pd.DataFrame(
                [
                    {**_meta("000001.SZ", "20231231", "20240401"), "n_income": 100.0},
                    {**_meta("000001.SZ", "20231231", "20240430"), "n_income": 120.0},
                    {**_meta("000001.SZ", "20231231", "20240515"), "n_income": 999.0},
                ]
            )
        )
        storage.insert_balancesheet(
            pd.DataFrame(
                [
                    {
                        **_meta("000001.SZ", "20231231", "20240402"),
                        "total_assets": 1_000.0,
                    },
                    {
                        **_meta("000001.SZ", "20231231", "20240510"),
                        "total_assets": 1_200.0,
                    },
                ]
            )
        )
        storage.insert_cashflow(
            pd.DataFrame(
                [
                    {
                        **_meta("000001.SZ", "20231231", "20240403"),
                        "n_cashflow_act": 50.0,
                    }
                ]
            )
        )

        before_revision = storage.get_fina_snapshot(
            "20240420",
            symbols=["000001.SZ"],
            columns=["inc_n_income", "bs_total_assets", "cf_n_cashflow_act"],
        )
        after_income_revision = storage.get_fina_snapshot(
            "20240505",
            symbols=["000001.SZ"],
            columns=["inc_n_income", "bs_total_assets", "cf_n_cashflow_act"],
        )
        after_bs_revision = storage.get_fina_snapshot(
            "20240512",
            symbols=["000001.SZ"],
            columns=["inc_n_income", "bs_total_assets", "cf_n_cashflow_act"],
        )

        assert before_revision.iloc[0]["inc_n_income"] == pytest.approx(100.0)
        assert before_revision.iloc[0]["bs_total_assets"] == pytest.approx(1_000.0)
        assert before_revision.iloc[0]["cf_n_cashflow_act"] == pytest.approx(50.0)
        assert after_income_revision.iloc[0]["inc_n_income"] == pytest.approx(120.0)
        assert after_income_revision.iloc[0]["bs_total_assets"] == pytest.approx(1_000.0)
        assert after_bs_revision.iloc[0]["inc_n_income"] == pytest.approx(120.0)
        assert after_bs_revision.iloc[0]["bs_total_assets"] == pytest.approx(1_200.0)

    def test_get_fina_snapshot_range_respects_trade_calendar_and_visibility_delay(
        self, tmp_market_storage
    ):
        storage = tmp_market_storage
        storage.insert_trade_calendar(
            pd.DataFrame(
                {
                    "cal_date": pd.to_datetime(["2024-04-01", "2024-04-02"]),
                    "is_open": [True, True],
                    "is_week_first": [True, False],
                    "is_week_last": [False, False],
                    "is_month_first": [True, False],
                    "is_month_last": [False, False],
                }
            )
        )
        storage.insert_income(
            pd.DataFrame(
                [
                    {**_meta("000001.SZ", "20231231", "20240401"), "n_income": 100.0}
                ]
            )
        )

        same_day_visible = storage.get_fina_snapshot_range(
            "20240401",
            "20240402",
            symbols=["000001.SZ"],
            columns=["inc_n_income"],
            delay=0,
        )
        next_day_visible = storage.get_fina_snapshot_range(
            "20240401",
            "20240402",
            symbols=["000001.SZ"],
            columns=["inc_n_income"],
            delay=1,
        )
        same_day_visible = same_day_visible.sort_values("date").reset_index(drop=True)
        next_day_visible = next_day_visible.sort_values("date").reset_index(drop=True)

        assert same_day_visible["date"].dt.strftime("%Y%m%d").tolist() == [
            "20240401",
            "20240402",
        ]
        assert next_day_visible["date"].dt.strftime("%Y%m%d").tolist() == ["20240402"]
        assert next_day_visible.iloc[0]["inc_n_income"] == pytest.approx(100.0)


class TestDailyFetcherMerge:
    def test_process_like_multi_type_merge_keeps_daily_rows_and_unit_conversions(self):
        daily = transform_daily(
            pd.DataFrame(
                {
                    "trade_date": ["20240102", "20240102"],
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "open": [10.0, 20.0],
                    "high": [11.0, 21.0],
                    "low": [9.0, 19.0],
                    "close": [10.5, 20.5],
                    "pre_close": [10.0, 20.0],
                    "change": [0.5, 0.5],
                    "pct_chg": [5.0, 2.5],
                    "vol": [12.0, 34.0],
                    "amount": [56.0, 78.0],
                }
            )
        )
        merged = merge_adj_factor(
            daily,
            pd.DataFrame(
                {
                    "trade_date": ["20240102"],
                    "ts_code": ["000001.SZ"],
                    "adj_factor": [1.1],
                }
            ),
        )
        merged = merge_st_status(
            merged,
            pd.DataFrame({"ts_code": ["000002.SZ"]}),
        )
        merged = merge_limit_prices(
            merged,
            pd.DataFrame(
                {
                    "trade_date": ["20240102"],
                    "ts_code": ["000001.SZ"],
                    "up_limit": [11.55],
                    "down_limit": [9.45],
                }
            ),
        )
        merged = merge_daily_basic(
            merged,
            pd.DataFrame(
                {
                    "trade_date": ["20240102"],
                    "ts_code": ["000001.SZ"],
                    "turnover_rate": [1.2],
                    "circ_mv": [500_000.0],
                }
            ),
        )
        merged = merge_margin_detail(
            merged,
            pd.DataFrame(
                {
                    "trade_date": ["20240102"],
                    "ts_code": ["000001.SZ"],
                    "rzye": [10.0],
                    "rqye": [2.0],
                }
            ),
        )
        merged = merge_moneyflow(
            merged,
            pd.DataFrame(
                {
                    "trade_date": ["20240102"],
                    "ts_code": ["000001.SZ"],
                    "buy_sm_vol": [3.0],
                    "buy_sm_amount": [4.0],
                    "net_mf_vol": [5.0],
                    "net_mf_amount": [6.0],
                }
            ),
        )

        row1 = merged.set_index("symbol").loc["000001.SZ"]
        row2 = merged.set_index("symbol").loc["000002.SZ"]

        assert len(merged) == 2
        assert row1["volume"] == 1_200
        assert row1["amount"] == pytest.approx(56_000.0)
        assert row1["adj_factor"] == pytest.approx(1.1)
        assert row1["is_st"] == pytest.approx(0)
        assert row2["is_st"] == pytest.approx(1)
        assert row1["limit_up"] == pytest.approx(11.55)
        assert row1["circ_mv"] == pytest.approx(500_000.0)
        assert row1["margin_rzye"] == pytest.approx(10.0)
        assert row1["mf_buy_sm_vol"] == 300
        assert row1["mf_buy_sm_amount"] == pytest.approx(40_000.0)
        assert row1["mf_net_mf_vol"] == 500
        assert row1["mf_net_mf_amount"] == pytest.approx(60_000.0)
        assert pd.isna(row2["limit_up"])


class TestAuctionData:
    def test_transform_stock_auction_normalizes_fields_and_units(self):
        raw = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102"],
                "ts_code": ["000001.SZ", "000002.SZ"],
                "open": [10.0, 20.0],
                "high": [10.2, 20.2],
                "low": [9.9, 19.9],
                "close": [10.1, 20.1],
                "vol": [12.0, 34.5],
                "amount": [56.0, 78.25],
                "vwap": [10.05, 20.05],
            }
        )

        df = transform_stock_auction(raw)

        assert df.columns.tolist() == [
            "date", "symbol", "open", "high", "low", "close",
            "volume", "amount", "vwap",
        ]
        assert df.iloc[0]["date"].strftime("%Y%m%d") == "20240102"
        assert df.iloc[0]["symbol"] == "000001.SZ"
        assert df.iloc[0]["volume"] == 12
        assert df.iloc[1]["volume"] == pytest.approx(34.5)
        assert df.iloc[0]["amount"] == pytest.approx(56.0)
        assert df.iloc[1]["amount"] == pytest.approx(78.25)

    def test_storage_upserts_and_queries_open_and_close_auction(self, tmp_market_storage):
        storage = tmp_market_storage
        open_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02"]).date,
                "symbol": ["000001.SZ", "000002.SZ"],
                "open": [10.0, 20.0],
                "high": [10.2, 20.2],
                "low": [9.9, 19.9],
                "close": [10.1, 20.1],
                "volume": [1_200, 3_400],
                "amount": [56_000.0, 78_000.0],
                "vwap": [10.05, 20.05],
            }
        )
        close_df = open_df.assign(close=[10.3, 20.3], vwap=[10.15, 20.15])

        storage.insert_stock_auction_open(open_df)
        storage.insert_stock_auction_close(close_df)
        storage.insert_stock_auction_open(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02"]).date,
                    "symbol": ["000001.SZ"],
                    "close": [10.11],
                    "volume": [1_300],
                }
            )
        )

        open_rows = storage.get_stock_auction_open(
            start="20240102", end="20240102", symbols=["000001.SZ"]
        )
        close_rows = storage.get_stock_auction_close(date="20240102")

        assert len(open_rows) == 1
        assert open_rows.iloc[0]["close"] == pytest.approx(10.11)
        assert open_rows.iloc[0]["volume"] == 1_300
        assert open_rows.iloc[0]["amount"] == pytest.approx(56_000.0)
        assert len(close_rows) == 2
        assert storage.get_max_stock_auction_open_date() == "20240102"
        assert storage.get_max_stock_auction_close_date() == "20240102"

    def test_backfill_stock_auction_writes_requested_sessions(
        self, tmp_market_storage, monkeypatch
    ):
        import backtest.data.backfill.stock_auction as module

        calls = []

        def fake_trade_dates(start: str, end: str) -> list[str]:
            assert start == "20240102"
            assert end == "20240103"
            return ["20240102", "20240103"]

        def fake_fetch_open(trade_date: str) -> pd.DataFrame:
            calls.append(("open", trade_date))
            return pd.DataFrame(
                {
                    "date": pd.to_datetime([trade_date], format="%Y%m%d").date,
                    "symbol": ["000001.SZ"],
                    "volume": [100],
                    "amount": [1_000.0],
                }
            )

        def fake_fetch_close(trade_date: str) -> pd.DataFrame:
            calls.append(("close", trade_date))
            return pd.DataFrame(
                {
                    "date": pd.to_datetime([trade_date], format="%Y%m%d").date,
                    "symbol": ["000001.SZ"],
                    "volume": [200],
                    "amount": [2_000.0],
                }
            )

        monkeypatch.setattr(module, "get_trade_dates", fake_trade_dates)
        monkeypatch.setattr(module, "fetch_stock_auction_open", fake_fetch_open)
        monkeypatch.setattr(module, "fetch_stock_auction_close", fake_fetch_close)

        counts = backfill_stock_auction(
            storage=tmp_market_storage,
            start="20240102",
            end="20240103",
        )

        assert calls == [
            ("open", "20240102"),
            ("close", "20240102"),
            ("open", "20240103"),
            ("close", "20240103"),
        ]
        assert counts == {"open": 2, "close": 2}
        assert len(tmp_market_storage.get_stock_auction_open()) == 2
        assert len(tmp_market_storage.get_stock_auction_close()) == 2

    def test_update_stock_auction_uses_independent_session_watermarks(
        self, monkeypatch
    ):
        from datetime import datetime
        import backtest.data.update_daily as module

        calls = []

        class FakeDatetime(datetime):
            @classmethod
            def now(cls):
                return cls(2024, 1, 7)

        class FakeStorage:
            def get_max_stock_auction_open_date(self):
                return "20240105"

            def get_max_stock_auction_close_date(self):
                return None

            def get_min_date(self):
                return "20240101"

        def fake_backfill_stock_auction(*, storage, start, end, sessions):
            calls.append((start, end, tuple(sessions)))
            return {session: 1 for session in sessions}

        monkeypatch.setattr(module, "datetime", FakeDatetime)
        monkeypatch.setattr(module, "backfill_stock_auction", fake_backfill_stock_auction)

        module.update_stock_auction(FakeStorage())

        assert calls == [
            ("20240106", "20240107", ("open",)),
            ("20240101", "20240107", ("close",)),
        ]

    def test_cold_start_recent_days_skips_stock_auction(
        self, tmp_market_storage, monkeypatch
    ):
        import sys
        import backtest.data.cold_start as module

        calls = []
        stock_list = pd.DataFrame(
            {"ts_code": ["000001.SZ"], "list_date": ["20200101"]}
        )

        monkeypatch.setattr(sys, "argv", ["cold_start", "--recent-days", "1"])
        monkeypatch.setattr(module, "fetch_stock_list", lambda: stock_list)
        monkeypatch.setattr(module, "MarketStorage", lambda: tmp_market_storage)
        monkeypatch.setattr(module, "backfill_trade_calendar", lambda **kwargs: 1)
        monkeypatch.setattr(
            module,
            "cold_start_market_daily",
            lambda storage, *, stock_list, recent_days=None: calls.append(
                ("market_daily", recent_days)
            ),
        )
        monkeypatch.setattr(
            module,
            "backfill_stock_auction",
            lambda **kwargs: calls.append(("stock_auction", kwargs)),
        )

        module.main()

        assert calls == [("market_daily", 1)]
