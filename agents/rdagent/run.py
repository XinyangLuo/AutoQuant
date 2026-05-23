"""Agent factor research main loop and CLI entry point.

Usage::

    # Run the agent loop
    python -m agents.rdagent.run \
        --max-rounds 10 --start 20160101 --end 20231231 \
        --output-dir results/agent/run_001

    # List candidates from a run
    python -m agents.rdagent.run --list-candidates results/agent/run_001

    # Admit a candidate (manual review)
    python -m agents.rdagent.run --admit f_auto_001 --run-dir results/agent/run_001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AgentConfig
from .core.evolving_framework import Trace
from .core.proposal import Hypothesis
from .core.utils import save_json
from .evaluator import AutoQuantFactorEvaluator, QuantFeedback
from .experiment import AutoQuantFactorExperiment
from .hypothesis import (
    AutoQuantFactorHypothesis2Experiment,
    AutoQuantFactorHypothesisGen,
)
from .knowledge import AShareKnowledgeBase
from .runner import AutoQuantFactorRunner
from .scenario import AShareQuantScenario

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_agent_loop(
    max_rounds: int = 10,
    start_date: str = "20160101",
    end_date: str = "20231231",
    output_dir: Path = Path("results/agent"),
    *,
    agent_config: AgentConfig | None = None,
    seed_hypothesis: Hypothesis | None = None,
    resume: bool = False,
) -> list[AutoQuantFactorExperiment]:
    """Run the agent factor research loop using DeepSeek.

    Parameters
    ----------
    max_rounds : int
        Maximum number of hypothesis→experiment iterations.
    start_date, end_date : str
        Backtest date range (YYYYMMDD).
    output_dir : Path
        Directory to save checkpoints and reports.
    agent_config : AgentConfig | None
        Thresholds and knobs.  Defaults are drawn from the pipeline / admission
        configs (see ``agents.rdagent.config.AgentConfig``).
    seed_hypothesis : Hypothesis | None
        If provided, use this as the Round-1 hypothesis instead of asking the
        LLM to generate one.  The LLM will still generate code from your idea.
        Subsequent rounds iterate based on feedback as usual.
    resume : bool
        If True, resume from the latest checkpoint in ``output_dir``.

    Returns
    -------
    list[AutoQuantFactorExperiment]
        Candidates that passed all thresholds.
    """
    if OpenAI is None:
        raise RuntimeError(
            "openai SDK is not installed. Install: pip install openai"
        )

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable not set")

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    scenario = AShareQuantScenario()
    llm_client = OpenAI(api_key=api_key, base_url=base_url)
    llm_client._default_model = "deepseek-chat"  # type: ignore[attr-defined]

    kb = AShareKnowledgeBase(db_path=output_dir / "kb.json")
    hypothesis_gen = AutoQuantFactorHypothesisGen(scenario, llm_client, kb)
    h2e = AutoQuantFactorHypothesis2Experiment(scenario, llm_client)
    cfg = agent_config or AgentConfig()
    evaluator = AutoQuantFactorEvaluator(
        min_rankicir=cfg.min_rankicir,
        min_ic_positive_ratio=cfg.min_ic_positive_ratio,
        max_turnover=cfg.max_turnover,
        max_corr=cfg.max_corr,
        min_simple_sharpe=cfg.min_sharpe_simple,
    )

    with AutoQuantFactorRunner(
        start_date=start_date,
        end_date=end_date,
        results_root=output_dir,
        agent_config=cfg,
    ) as runner:
        # Try resume from checkpoint
        trace = Trace()
        candidates: list[AutoQuantFactorExperiment] = []
        completed_rounds = 0
        round_num = completed_rounds  # safe lower bound; overridden by loop

        if resume:
            checkpoint = _load_checkpoint(output_dir)
            if checkpoint:
                trace, candidates, completed_rounds = checkpoint
                round_num = completed_rounds
                print(
                    f"\n[RESUME] Loaded checkpoint from round {completed_rounds} "
                    f"with {len(candidates)} candidates"
                )
            else:
                print("\n[RESUME] No checkpoint found, starting from scratch")

        for round_num in range(completed_rounds + 1, max_rounds + 1):
            print(f"\n{'='*60}")
            print(f"  Round {round_num} / {max_rounds}")
            print(f"{'='*60}")

            # 1. Generate hypothesis (use seed on Round 1 if provided)
            try:
                if round_num == 1 and seed_hypothesis is not None:
                    hypothesis = seed_hypothesis
                    print(f"  [SEED] Using provided hypothesis: {hypothesis.hypothesis_text[:80]}...")
                    # Clear seed so subsequent rounds use LLM
                    seed_hypothesis = None
                else:
                    hypothesis = hypothesis_gen.gen(trace=trace)
                    print(f"  Hypothesis: {hypothesis.hypothesis_text[:80]}...")
                print(f"  Category: {hypothesis.category}")
            except (RuntimeError, ValueError, SyntaxError) as e:
                print(f"  Hypothesis generation failed: {e}")
                continue

            # 2. Convert to experiment (code generation)
            try:
                experiment = h2e.convert(hypothesis, trace)
                # Propagate hypothesis metadata for knowledge base
                experiment.category = hypothesis.category
                experiment.keywords = hypothesis.keywords
                print(f"  Generated code for: {experiment.factor_id}")
            except (RuntimeError, ValueError, SyntaxError) as e:
                print(f"  Code generation failed: {e}")
                continue

            # 3. Execute pipeline
            try:
                experiment = runner.run(experiment)
            except (RuntimeError, ValueError, ImportError, SyntaxError) as e:
                print(f"  Pipeline execution failed: {e}")
                runner.cleanup_work_db(experiment.factor_id)
                feedback = QuantFeedback(
                    decision=False,
                    observation=f"Execution error: {type(e).__name__}: {e}",
                    suggestion="Check code syntax and data source compatibility. "
                              "Ensure all referenced columns exist in the data schema.",
                )
                trace.add(experiment, feedback)
                kb.add_experience(experiment, feedback)
                _save_checkpoint(trace, candidates, round_num, output_dir)
                continue

            # 4. Evaluate
            feedback = evaluator.evaluate(experiment)

            # 5. Record
            trace.add(experiment, feedback)
            kb.add_experience(experiment, feedback)

            # 6. Save checkpoint
            _save_checkpoint(trace, candidates, round_num, output_dir)

            # 7. Print results
            print(f"  RankICIR: {feedback.rankicir:.3f}")
            print(f"  IC+: {feedback.ic_positive_ratio:.1%}")
            print(f"  Turnover: {feedback.turnover:.3f}")
            print(f"  Simple Sharpe: {feedback.simple_sharpe}")
            print(f"  Decision: {'CANDIDATE' if feedback.decision else 'REJECTED'}")

            # 8. Check candidate
            if feedback.decision:
                experiment.status = "candidate"
                candidates.append(experiment)
                print(f"  --> CANDIDATE: {experiment.factor_id}")

                # Early stop on high bar
                if feedback.simple_sharpe is not None and feedback.simple_sharpe > cfg.high_bar_sharpe:
                    print(f"  --> High bar reached (Sharpe {feedback.simple_sharpe:.3f} > {cfg.high_bar_sharpe}), stopping early.")
                    break
            else:
                # Rejected — clean up work DB to prevent indefinite growth
                runner.cleanup_work_db(experiment.factor_id)

    # Generate final review report
    _generate_review_report(candidates, output_dir)
    _save_run_metadata(output_dir, max_rounds, start_date, end_date, len(candidates))

    total_rounds = max(completed_rounds, round_num)
    print(f"\n{'='*60}")
    print(f"  Run complete: {len(candidates)} candidates from {total_rounds} rounds")
    print(f"  Report: {output_dir / 'candidates.md'}")
    print(f"{'='*60}")

    return candidates


# ---------------------------------------------------------------------------
# Checkpoint & report
# ---------------------------------------------------------------------------


def _save_checkpoint(
    trace: Trace,
    candidates: list[AutoQuantFactorExperiment],
    round_num: int,
    output_dir: Path,
) -> None:
    """Save a checkpoint of the current state (atomic write)."""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Atomic write for trace
    trace_path = checkpoint_dir / f"trace_round_{round_num:03d}.json"
    trace_tmp = trace_path.with_suffix(".tmp")
    with trace_tmp.open("w", encoding="utf-8") as f:
        json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2, default=str)
    os.replace(str(trace_tmp), str(trace_path))

    # Atomic write for candidates
    cand_path = checkpoint_dir / f"candidates_round_{round_num:03d}.json"
    cand_tmp = cand_path.with_suffix(".tmp")
    with cand_tmp.open("w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in candidates], f, ensure_ascii=False, indent=2, default=str)
    os.replace(str(cand_tmp), str(cand_path))


def _load_checkpoint(
    output_dir: Path,
) -> tuple[Trace, list[AutoQuantFactorExperiment], int] | None:
    """Load latest checkpoint if exists.

    Returns
    -------
    (trace, candidates, completed_rounds) or None
    """
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None

    trace_files = sorted(checkpoint_dir.glob("trace_round_*.json"))
    if not trace_files:
        return None

    latest_trace_file = trace_files[-1]
    # Extract round number from filename: trace_round_NNN.json
    import re
    m = re.search(r"_(\d{3,})$", latest_trace_file.stem)
    if not m:
        return None
    completed_rounds = int(m.group(1))

    # Load trace with correct factories for subclass round-trip
    try:
        trace_data = json.loads(latest_trace_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    trace = Trace.from_dict(
        trace_data,
        experiment_factory=AutoQuantFactorExperiment.from_dict,
        feedback_factory=QuantFeedback.from_dict,
    )

    # Load candidates from the same round
    candidates: list[AutoQuantFactorExperiment] = []
    candidate_file = checkpoint_dir / f"candidates_round_{completed_rounds:03d}.json"
    if candidate_file.exists():
        try:
            candidate_data = json.loads(candidate_file.read_text(encoding="utf-8"))
            candidates = [AutoQuantFactorExperiment.from_dict(c) for c in candidate_data]
        except json.JSONDecodeError:
            candidates = []

    return trace, candidates, completed_rounds


def _generate_review_report(
    candidates: list[AutoQuantFactorExperiment],
    output_dir: Path,
) -> None:
    """Generate a Markdown review report of all candidates."""
    lines: list[str] = [
        "# Agent Factor Research — Candidate Review Report",
        "",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
        f"**Candidates**: {len(candidates)}",
        "",
        "## Candidate Factors",
        "",
    ]

    for i, exp in enumerate(candidates, 1):
        er = exp.eval_result or {}
        sm = exp.simple_bt_metrics or {}
        dm = exp.detailed_bt_metrics or {}

        lines.append(f"### {i}. {exp.factor_id}")
        lines.append("")
        if exp.factor_code:
            # Show first few lines of code
            code_preview = "\n".join(exp.factor_code.split("\n")[:8])
            lines.append(f"```python\n{code_preview}\n```")
            lines.append("")
        lines.append(f"- **RankICIR**: {er.get('rankicir', 'N/A')}")
        lines.append(f"- **IC+ ratio**: {er.get('ic_positive_ratio', 'N/A')}")
        lines.append(f"- **Turnover**: {er.get('turnover', 'N/A')}")
        lines.append(f"- **Max corr with existing**: {er.get('max_corr', 'N/A')}")
        lines.append(f"- **Simple Sharpe**: {sm.get('sharpe', 'N/A')}")
        lines.append(f"- **Simple Annual Return**: {sm.get('annual_return', 'N/A')}")
        lines.append(f"- **Simple MDD**: {sm.get('max_drawdown', 'N/A')}")
        if dm:
            lines.append(f"- **Detailed Sharpe**: {dm.get('sharpe', 'N/A')}")
            lines.append(f"- **Detailed Annual Return**: {dm.get('annual_return', 'N/A')}")
        lines.append("")
        lines.append("#### Suggested Actions")
        lines.append(f"```bash")
        lines.append(f"# To admit this factor to the library:")
        lines.append(f"python -m backtest.factor.admission admit {exp.factor_id}")
        lines.append("")
        lines.append(f"# To reject:")
        lines.append(f"python -m backtest.factor.admission reject {exp.factor_id}")
        lines.append(f"```")
        lines.append("")

    if not candidates:
        lines.append("_No candidates met the threshold in this run._")
        lines.append("")

    report_path = output_dir / "candidates.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _save_run_metadata(
    output_dir: Path,
    max_rounds: int,
    start_date: str,
    end_date: str,
    num_candidates: int,
) -> None:
    """Save run metadata for later inspection."""
    meta = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "max_rounds": max_rounds,
        "start_date": start_date,
        "end_date": end_date,
        "num_candidates": num_candidates,
        "output_dir": str(output_dir),
    }
    save_json(meta, output_dir / "run_metadata.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AutoQuant Agent Factor Research",
    )
    sub = parser.add_subparsers(dest="command")

    # run loop
    p_run = sub.add_parser("run", help="Run the agent research loop")
    p_run.add_argument("--max-rounds", type=int, default=10)
    p_run.add_argument("--start", default="20160101")
    p_run.add_argument("--end", default="20231231")
    p_run.add_argument("--output-dir", default="results/agent/run_001")
    p_run.add_argument("--min-rankicir", type=float, default=None)
    p_run.add_argument("--min-ic-pos", type=float, default=None)
    p_run.add_argument("--max-turnover", type=float, default=None)
    p_run.add_argument("--min-sharpe", type=float, default=None)
    p_run.add_argument("--high-bar-sharpe", type=float, default=None)
    p_run.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in --output-dir.",
    )
    p_run.add_argument(
        "--seed",
        default=None,
        help="Seed hypothesis text for Round 1 (e.g. '20-day momentum x volume spike deviation'). "
             "If omitted, the LLM generates the first hypothesis.",
    )

    # list candidates
    p_list = sub.add_parser("list-candidates", help="List candidates from a run")
    p_list.add_argument("run_dir")

    # admit
    p_admit = sub.add_parser("admit", help="Admit a candidate factor")
    p_admit.add_argument("factor_id")
    p_admit.add_argument("--run-dir")

    # reject
    p_reject = sub.add_parser("reject", help="Reject a candidate factor")
    p_reject.add_argument("factor_id")
    p_reject.add_argument("--reason", default="")
    p_reject.add_argument("--run-dir")

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    cfg = AgentConfig()
    # Override from CLI if explicitly provided
    if args.min_rankicir is not None:
        cfg.min_rankicir = args.min_rankicir
    if args.min_ic_pos is not None:
        cfg.min_ic_positive_ratio = args.min_ic_pos
    if args.max_turnover is not None:
        cfg.max_turnover = args.max_turnover
    if args.min_sharpe is not None:
        cfg.min_sharpe_simple = args.min_sharpe
    if args.high_bar_sharpe is not None:
        cfg.high_bar_sharpe = args.high_bar_sharpe

    # Build seed hypothesis if --seed is provided
    seed: Hypothesis | None = None
    if args.seed:
        seed = Hypothesis(
            hypothesis_text=args.seed,
            category="user_seed",
            data_sources=["market_daily"],
            rationale="User-provided seed hypothesis.",
            expected_behavior="",
            keywords=["seed"],
        )

    try:
        candidates = run_agent_loop(
            max_rounds=args.max_rounds,
            start_date=args.start,
            end_date=args.end,
            output_dir=Path(args.output_dir),
            agent_config=cfg,
            seed_hypothesis=seed,
            resume=args.resume,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_list_candidates(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    report_path = run_dir / "candidates.md"
    if report_path.exists():
        print(report_path.read_text(encoding="utf-8"))
        return 0

    # Fallback: try loading from checkpoint
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"No candidates found in {run_dir}", file=sys.stderr)
        return 1

    # Load latest candidates
    candidate_files = sorted(checkpoint_dir.glob("candidates_round_*.json"))
    if not candidate_files:
        print(f"No candidates found in {run_dir}", file=sys.stderr)
        return 1

    data = json.loads(candidate_files[-1].read_text(encoding="utf-8"))
    print(f"Candidates from {args.run_dir}:")
    for c in data:
        print(f"  - {c.get('factor_id')}: status={c.get('status')}")
    return 0


def cmd_admit(args: argparse.Namespace) -> int:
    from backtest.factor.admission import admit

    try:
        result = admit(args.factor_id)
        print(f"Admitted: {result}")
    except Exception as e:
        print(f"Admit failed: {e}", file=sys.stderr)
        return 1
    return 0


def _find_factor_code(factor_id: str, run_dir: Path | None = None) -> str | None:
    """Locate generated factor code on disk.

    Searches the standard generated/ directory and optionally a run-specific dir.
    """
    from pathlib import Path

    search_paths: list[Path] = []
    if run_dir is not None:
        search_paths.append(run_dir)
    # Default generated directory
    search_paths.append(Path(__file__).parent / "generated")

    for base in search_paths:
        candidate = base / f"{factor_id}.py"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return None


def cmd_reject(args: argparse.Namespace) -> int:
    from backtest.factor.admission import reject

    try:
        result = reject(args.factor_id, notes=args.reason or None)
        print(f"Rejected: {result}")
    except Exception as e:
        print(f"Reject failed: {e}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "list-candidates":
        return cmd_list_candidates(args)
    elif args.command == "admit":
        return cmd_admit(args)
    elif args.command == "reject":
        return cmd_reject(args)
    else:
        parser.print_help()
        return 2


if __name__ == "__main__":
    sys.exit(main())
