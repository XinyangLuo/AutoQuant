"""AutoQuant factor runner — executes the full backtest pipeline for a generated factor."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.evaluation import evaluate as bt_evaluate
from backtest.factor.compute import apply_variant_pipeline, compute_factor
from backtest.factor.evaluation import evaluate as factor_evaluate
from backtest.factor.registry import get_factor_meta, sync_registry
from backtest.factor.storage import FactorStorage
from backtest.simulation.config import SimulationConfig
from backtest.simulation.detailed import DetailedSimulator
from backtest.simulation.simple import SimpleSimulator
from backtest.strategy.config import (
    BacktestConfig,
    FactorConfig,
    SelectionConfig,
    StrategyConfig,
    UniverseConfig,
    WeightingConfig,
)
from backtest.strategy.strategies.single_factor import SingleFactorStrategy

from .experiment import AutoQuantFactorExperiment


class AutoQuantFactorRunner:
    """Run the complete AutoQuant pipeline for a single factor experiment.

    Steps
    -----
    1. Write code to disk → import to trigger ``@register``
    2. Backfill: ``compute_factor()`` + neutralization → work DB
    3. Factor evaluation: ``evaluate()`` → IC / RankIC / turnover / corr
    4. Simple backtest: ``SingleFactorStrategy`` + ``SimpleSimulator``
    5. Detailed backtest (conditional): ``DetailedSimulator``
    6. Collect all metrics into the experiment
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        *,
        results_root: Path | str = "results/agent",
        market_storage: MarketStorage | None = None,
        factor_storage: FactorStorage | None = None,
        benchmark: str = "000300.SH",
        agent_config: Any = None,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.results_root = Path(results_root)
        self.market_storage = market_storage
        self.factor_storage = factor_storage
        self.benchmark = benchmark
        self.agent_config = agent_config

        # Lazily create storages only if not provided
        self._market_storage_owned = market_storage is None
        self._factor_storage_owned = factor_storage is None
        if self.market_storage is None:
            self.market_storage = MarketStorage()
        if self.factor_storage is None:
            self.factor_storage = FactorStorage()

        # Default strategy config for single-factor backtests
        self._default_strategy_config = StrategyConfig(
            strategy_type="single_factor_topk",
            rebalance_freq="1D",
            delay=1,
            universe=UniverseConfig(
                exclude_st=True,
                exclude_new_ipo_days=252,
                include_cyb=True,
                include_kcb=False,
            ),
            selection=SelectionConfig(
                method="topk",
                top_pct=0.1,
            ),
            weighting=WeightingConfig(method="equal"),
            backtest=BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                benchmark=benchmark,
            ),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_storages()
        return False

    def _close_storages(self) -> None:
        """Explicit cleanup — idempotent."""
        if self._market_storage_owned and self.market_storage is not None:
            self.market_storage.close()
            self.market_storage = None
        if self._factor_storage_owned and self.factor_storage is not None:
            self.factor_storage.close()
            self.factor_storage = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        experiment: AutoQuantFactorExperiment,
        strategy_config: StrategyConfig | None = None,
    ) -> AutoQuantFactorExperiment:
        """Execute the full pipeline for an experiment.

        Parameters
        ----------
        experiment : AutoQuantFactorExperiment
            Must have ``factor_id`` and ``factor_code`` set.
        strategy_config : StrategyConfig | None
            Override the default strategy configuration.

        Returns
        -------
        AutoQuantFactorExperiment
            The same object, mutated with results.
        """
        experiment.status = "running"
        strategy_config = strategy_config or self._default_strategy_config

        try:
            # Step 1: Write code and register
            self._register_factor(experiment)

            # Step 2: Backfill → work DB
            self._backfill_factor(experiment)

            # Step 3: Factor evaluation
            self._evaluate_factor(experiment)

            # Step 4: Simple backtest
            self._run_simple_backtest(experiment, strategy_config)

            # Step 5: Detailed backtest (conditional)
            self._maybe_run_detailed_backtest(experiment, strategy_config)

            experiment.status = "passed"

        except Exception as e:
            experiment.status = "rejected"
            experiment.error = f"{type(e).__name__}: {e}"
            raise

        return experiment

    # ------------------------------------------------------------------
    # Step 1: Code registration
    # ------------------------------------------------------------------

    def _register_factor(self, experiment: AutoQuantFactorExperiment) -> None:
        """Write factor code to disk and import to trigger @register."""
        if not experiment.factor_code:
            raise ValueError("experiment.factor_code is empty")
        if not experiment.factor_id:
            raise ValueError("experiment.factor_id is empty")

        # Write to generated/ directory
        gen_dir = Path(__file__).parent / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        # Ensure generated/ is a valid Python package for dynamic imports
        init_file = gen_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Auto-generated factor modules\n", encoding="utf-8")
        file_path = gen_dir / f"{experiment.factor_id}.py"

        # Clean up stale module from sys.modules if reusing a factor_id
        mod_name = f"agents.rdagent.generated.{experiment.factor_id}"
        sys.modules.pop(mod_name, None)

        file_path.write_text(experiment.factor_code, encoding="utf-8")
        experiment.factor_file_path = file_path

        # Import the module to trigger @register
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise

        # Persist registry
        sync_registry()

    # ------------------------------------------------------------------
    # Step 2: Backfill
    # ------------------------------------------------------------------

    def _backfill_factor(self, experiment: AutoQuantFactorExperiment) -> None:
        """Compute factor values and write to work DB."""
        meta = get_factor_meta(experiment.factor_id)

        # Compute raw values
        raw_df = compute_factor(
            experiment.factor_id,
            self.start_date,
            self.end_date,
            market_storage=self.market_storage,
        )

        # Apply neutralization pipeline if variant != "none"
        variant = meta.get("variant", "barra_ind_size")
        if variant != "none":
            try:
                df = apply_variant_pipeline(
                    raw_df,
                    experiment.factor_id,
                    market_storage=self.market_storage,
                    factor_storage=self.factor_storage,
                )
            except RuntimeError as e:
                if "admitted" in str(e).lower() or "requires" in str(e).lower():
                    # Required library factor (e.g. f_barra_size) not available;
                    # fall back to raw factor without neutralization.
                    df = raw_df
                else:
                    raise
        else:
            df = raw_df

        # Insert into work DB
        self.factor_storage.insert_factors(df)

    # ------------------------------------------------------------------
    # Step 3: Factor evaluation
    # ------------------------------------------------------------------

    def _evaluate_factor(self, experiment: AutoQuantFactorExperiment) -> None:
        """Run static factor evaluation (IC / RankIC / turnover / corr)."""
        eval_result = factor_evaluate(
            experiment.factor_id,
            self.start_date,
            self.end_date,
            ret_type="open",
            corr_top_k=5,
            exclude_limit_up=True,
        )

        # Flatten to dict
        thresholds = eval_result.threshold_metrics(primary_horizon=20)
        experiment.eval_result = {
            "factor_id": eval_result.factor_id,
            "variant": eval_result.variant,
            "horizons": eval_result.horizons,
            "rankicir": thresholds.get("rankicir"),
            "ic_positive_ratio": thresholds.get("ic_positive_ratio"),
            "turnover": thresholds.get("turnover"),
            "max_corr": thresholds.get("max_corr"),
            "summary": eval_result.summary().to_dict("records"),
        }

    # ------------------------------------------------------------------
    # Step 4: Simple backtest
    # ------------------------------------------------------------------

    def _run_simple_backtest(
        self,
        experiment: AutoQuantFactorExperiment,
        strategy_config: StrategyConfig,
    ) -> None:
        """Run vectorized simple backtest."""
        # Build strategy config for this factor
        config = self._build_strategy_config(strategy_config, experiment.factor_id)

        # Load factor panel
        factor_panel = self.factor_storage.get_factors_long(
            factor_ids=[experiment.factor_id],
            start=self.start_date,
            end=self.end_date,
        )
        factor_panel = factor_panel.pivot_table(
            index=["date", "symbol"],
            columns="factor_id",
            values="value",
        ).reset_index()

        # Load market data
        market_panel = self.market_storage.get_bars(
            symbols=None,
            start=self.start_date,
            end=self.end_date,
        )

        # Generate signals
        strategy = SingleFactorStrategy(config)
        rebalance_dates = self._get_rebalance_dates(config)
        signals = strategy.generate_signals(factor_panel, market_panel, rebalance_dates)

        # Run simulation
        sim = SimpleSimulator(SimulationConfig())
        result = sim.run(signals, market_panel)

        # Save results
        result_dir = self.results_root / experiment.factor_id / "simple"
        result.save(str(result_dir))
        experiment.simple_bt_dir = result_dir

        # Evaluate post-simulation
        report = bt_evaluate(result_dir, benchmark=self.benchmark, plot=False)
        experiment.simple_bt_metrics = report.metrics if hasattr(report, "metrics") else {}

    # ------------------------------------------------------------------
    # Step 5: Detailed backtest (conditional)
    # ------------------------------------------------------------------

    def _maybe_run_detailed_backtest(
        self,
        experiment: AutoQuantFactorExperiment,
        strategy_config: StrategyConfig,
    ) -> None:
        """Run detailed backtest only if factor passes initial thresholds."""
        cfg = self.agent_config
        min_rankicir = getattr(cfg, "min_rankicir", 0.25) if cfg else 0.25
        min_sharpe_simple = getattr(cfg, "min_sharpe_simple", 0.5) if cfg else 0.5

        rankicir = experiment.eval_result.get("rankicir", float("-inf"))
        simple_sharpe = (experiment.simple_bt_metrics or {}).get("sharpe", 0.0)

        if rankicir < min_rankicir or simple_sharpe < min_sharpe_simple:
            return  # Skip detailed backtest

        config = self._build_strategy_config(strategy_config, experiment.factor_id)

        factor_panel = self.factor_storage.get_factors_long(
            factor_ids=[experiment.factor_id],
            start=self.start_date,
            end=self.end_date,
        )
        factor_panel = factor_panel.pivot_table(
            index=["date", "symbol"],
            columns="factor_id",
            values="value",
        ).reset_index()

        market_panel = self.market_storage.get_bars(
            symbols=None,
            start=self.start_date,
            end=self.end_date,
        )

        strategy = SingleFactorStrategy(config)
        rebalance_dates = self._get_rebalance_dates(config)
        signals = strategy.generate_signals(factor_panel, market_panel, rebalance_dates)

        # Load dividends for detailed simulation
        dividends = self.market_storage.get_dividends(
            start=self.start_date,
            end=self.end_date,
        )

        sim = DetailedSimulator(SimulationConfig())
        result = sim.run(signals, market_panel, dividends_data=dividends)

        result_dir = self.results_root / experiment.factor_id / "detailed"
        result.save(str(result_dir))
        experiment.detailed_bt_dir = result_dir

        report = bt_evaluate(result_dir, benchmark=self.benchmark, plot=False)
        experiment.detailed_bt_metrics = report.metrics if hasattr(report, "metrics") else {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_strategy_config(
        self,
        base: StrategyConfig,
        factor_id: str,
    ) -> StrategyConfig:
        """Clone base config and inject the target factor."""
        return StrategyConfig(
            strategy_type=base.strategy_type,
            rebalance_freq=base.rebalance_freq,
            delay=base.delay,
            universe=base.universe,
            factors=[FactorConfig(id=factor_id, direction="desc", weight=1.0)],
            combine_method=base.combine_method,
            selection=base.selection,
            weighting=base.weighting,
            neutralize=base.neutralize,
            risk=base.risk,
            backtest=base.backtest,
            decay=base.decay,
        )

    def _get_rebalance_dates(self, config: StrategyConfig) -> list[str]:
        """Generate rebalancing dates from trade calendar."""
        from backtest.data.trade_calendar import get_trade_dates

        dates = get_trade_dates(self.start_date, self.end_date)
        freq = config.rebalance_freq

        if freq == "1D":
            return dates
        elif freq == "5D":
            return dates[::5]
        elif freq == "1W":
            # Weekly: every 5 trading days (~1 week)
            return dates[::5]
        elif freq == "2W":
            return dates[::10]
        elif freq == "1M" or freq == "EOM":
            # Monthly: pick last trading day of each month
            df = pd.DataFrame({"date": pd.to_datetime(dates)})
            df["ym"] = df["date"].dt.to_period("M")
            return df.groupby("ym")["date"].last().dt.strftime("%Y%m%d").tolist()
        else:
            return dates[::5]  # Default to weekly
