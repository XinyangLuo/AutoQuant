"""生成中文诊断报告 + 全部图表。"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# CJK font support — try common macOS / Linux Chinese fonts in order.
for _font in ("PingFang HK", "Heiti TC", "STHeiti", "Arial Unicode MS",
              "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"):
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

from backtest.evaluation.report import _fmt

from .state import PipelineState

_FIGSIZE_WIDE = (14, 5)
_DPI = 120

# ===========================================================================
# Tag
# ===========================================================================


def _build_tag(state: PipelineState) -> str:
    cfg = state.strategy_config
    if cfg is None:
        step5 = state.step_results.get("step5")
        if step5 and step5.metrics:
            top_k = step5.metrics.get("top_k")
            top_pct = step5.metrics.get("top_pct")
            decay = step5.metrics.get("decay", 5)
            rebalance = step5.metrics.get("rebalance", "1D")
            if top_k is not None:
                tag = f"top{top_k}"
            elif top_pct is not None:
                tag = f"top{int(round(top_pct * 100))}pct"
            else:
                tag = "top10pct"
            return f"{tag}_{rebalance.lower()}_d{decay}"
        return "default"

    if isinstance(cfg, dict):
        selection = cfg.get("selection", {})
        top_k = selection.get("top_k")
        top_pct = selection.get("top_pct")
        decay = cfg.get("decay", 5)
        rebalance = cfg.get("rebalance_freq", "1D")
        if top_k is not None:
            tag = f"top{top_k}"
        elif top_pct is not None:
            tag = f"top{int(round(top_pct * 100))}pct"
        else:
            tag = "top10pct"
        return f"{tag}_{rebalance.lower()}_d{decay}"

    sel = cfg.selection
    if sel.top_pct is not None:
        tag = f"top{int(round(sel.top_pct * 100))}pct"
    else:
        tag = f"top{sel.top_k}"
    decay = cfg.decay or 0
    return f"{tag}_{cfg.rebalance_freq.lower()}_d{decay}"


_STEP_NAMES = {
    "step1": "覆盖率",
    "step2": "中性化验证",
    "step3": "ICIR 门控",
    "step4": "单调性",
    "step5": "策略配置",
    "step6": "简单回测",
    "step7": "详细回测",
    "step8": "Ridge R² 分档",
    "step9": "残差 ICIR 增量信息",
    "step10": "报告生成",
}


def _step_label(step_key: str) -> str:
    return _STEP_NAMES.get(step_key, step_key)


# ===========================================================================
# 主入口
# ===========================================================================


def generate_pipeline_report(state: PipelineState) -> Path:
    config = state.config
    tag = _build_tag(state)
    tag_dir = config.results_dir() / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = tag_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    report_path = tag_dir / "pipeline_report.md"

    lines: list[str] = [
        f"# 因子 Pipeline 报告：`{config.factor_id}`",
        "",
        f"- **策略标签**：{tag}",
        f"- **回测区间**：{config.start_date} ~ {config.end_date}",
        f"- **成交类型**：{config.ret_type}（{'开盘' if config.ret_type == 'open' else '收盘'}价成交）",
        f"- **频率**：{config.frequency}",
        f"- **状态**：{state.status}",
        "",
    ]

    # 决策横幅 + 汇总表
    lines.extend(_decision_banner(state))
    lines.extend(_summary_table(state))

    # 逐步明细
    lines.extend(_step_section(state, plots_dir, "step1", _render_step1))
    lines.extend(_step_section(state, plots_dir, "step2", _render_step2))
    lines.extend(_step_section(state, plots_dir, "step3", _render_step3))
    lines.extend(_step_section(state, plots_dir, "step4", _render_step4))
    lines.extend(_step_section(state, plots_dir, "step5", _render_step5))
    lines.extend(_step_section(state, plots_dir, "step6", _render_step6))
    lines.extend(_step_section(state, plots_dir, "step7", _render_step7))
    lines.extend(_step_section(state, plots_dir, "step8", _render_step8))
    lines.extend(_step_section(state, plots_dir, "step9", _render_step9_residual_icir))
    lines.extend(_step_section(state, plots_dir, "step10", _render_step10))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _step_section(state: PipelineState, plots_dir: Path,
                  step_key: str, render_fn) -> list[str]:
    """Render a single step section with heading, content, and pass/fail."""
    result = state.step_results.get(step_key)
    name = _STEP_NAMES[step_key]
    status = "通过" if (result and result.passed) else ("**拒绝**" if result else "未执行")
    lines = [f"## {name}", ""]
    lines.append(f"**结果**：{status}")
    if result and result.reason:
        lines.append(f"<br>**原因**：{result.reason}")
    lines.append("")
    lines.extend(render_fn(state, plots_dir))
    return lines


# ===========================================================================
# 决策横幅 + 汇总表
# ===========================================================================


def _decision_banner(state: PipelineState) -> list[str]:
    lines: list[str] = []
    if state.status == "admitted":
        lines.append("> **结果：已入库**")
    elif state.status == "ready_for_review":
        lines.append("> **结果：待人工审核**")
        lines.append(">")
        lines.append(f"> 所有步骤通过，请查看报告后手动入库：")
        lines.append(f"> `python -m backtest.factor.admission admit {state.config.factor_id}`")
    elif state.status == "rejected":
        reason = _find_rejection_reason(state)
        step_name = _find_rejection_step(state)
        label = _step_label(step_name) if step_name else "未知"
        lines.append(f"> **结果：已拒绝**（{label}）")
        if reason:
            lines.append(f">")
            lines.append(f"> **原因**：{reason}")
    else:
        lines.append(f"> **结果**：{state.status}")
    lines.append("")
    return lines


def _summary_table(state: PipelineState) -> list[str]:
    lines = ["## 汇总", ""]
    lines.append("| 步骤 | 名称 | 结果 | 关键指标 | 原因 |")
    lines.append("|------|------|------|----------|------|")
    for step_key in _STEP_NAMES:
        result = state.step_results.get(step_key)
        if result is None:
            lines.append(f"| {step_key} | {_STEP_NAMES[step_key]} | — | — | — |")
            continue
        status = "通过" if result.passed else "**拒绝**"
        name = _STEP_NAMES[step_key]
        metrics_str = _summarise_metrics(result.metrics, step_key)
        reason = result.reason or "-"
        lines.append(f"| {step_key} | {name} | {status} | {metrics_str} | {reason} |")
    lines.append("")
    return lines


def _summarise_metrics(metrics: dict, step_key: str) -> str:
    if not metrics:
        return "-"
    step_priorities = {
        "step1": ["pct95_missing_rate", "max_missing_rate", "n_dates"],
        "step2": ["max_existing_corr", "size_corr", "max_industry_corr"],
        "step3": ["annual_icir", "abs_ic", "best_horizon"],
        "step4": ["spearman", "n_groups"],
        "step5": ["top_pct", "decay", "rebalance"],
        "step6": ["sharpe", "annual_return", "max_drawdown"],
        "step7": ["sharpe", "annual_return", "annual_turnover"],
        "step8": ["r2", "tier"],
        "step9": ["annual_icirs", "n_regressors"],
        "step10": [],
    }
    keys = step_priorities.get(step_key, list(metrics.keys())[:2])
    # Prioritise the first available excess_sharpe_* key (e.g. hs300, csi500, csi1000)
    # so the summary shows the relative metric when present.
    if step_key in ("step6", "step7"):
        excess_keys = [k for k in metrics if k.startswith("excess_sharpe_")]
        if excess_keys:
            keys = [keys[0], excess_keys[0]] + keys[1:]
    parts = []
    for key in keys:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                parts.append(f"{key}={val:.3f}")
            else:
                parts.append(f"{key}={val}")
    return ", ".join(parts[:2]) if parts else "-"


# ===========================================================================
# 各 step 渲染函数
# ===========================================================================


def _metrics_sub_table(metrics: dict, exclude: set | None = None) -> list[str]:
    """Render a subset of metrics as a key-value table."""
    exclude = exclude or set()
    lines = ["| 指标 | 数值 |", "|------|------|"]
    for k, v in metrics.items():
        if k in exclude:
            continue
        if isinstance(v, float):
            lines.append(f"| {k} | {_fmt(v, 'f4')} |")
        elif v is None:
            lines.append(f"| {k} | — |")
        elif isinstance(v, dict):
            pass  # skip nested dicts
        else:
            lines.append(f"| {k} | {v} |")
    return lines


def _render_step1(state: PipelineState, plots_dir: Path) -> list[str]:
    result = state.step_results.get("step1")
    if not result or not result.metrics:
        return ["*无数据。*", ""]
    return _metrics_sub_table(result.metrics, exclude={"all_ic_metrics"})


def _render_step2(state: PipelineState, plots_dir: Path) -> list[str]:
    result = state.step_results.get("step2")
    if not result or not result.metrics:
        return ["*无数据。*", ""]
    return _metrics_sub_table(result.metrics)


def _render_step3(state: PipelineState, plots_dir: Path) -> list[str]:
    result = state.step_results.get("step3")
    if not result:
        return ["*无数据。*", ""]

    lines = []

    # Factor formula
    formula = _get_factor_formula(state.config.factor_id)
    if formula:
        lines.extend(formula)

    # IC metrics table
    all_ic = result.metrics.get("all_ic_metrics", {})
    if all_ic:
        lines.append("### IC 指标（Pearson）")
        lines.append("")
        lines.append(f"*成交类型：{state.config.ret_type}（{'开盘' if state.config.ret_type == 'open' else '收盘'}价），已排除涨停无法成交样本*")
        lines.append("")
        lines.append("| 周期 | IC 均值 | IC 标准差 | ICIR | IC t 值 | IC 正向占比 |")
        lines.append("|------|---------|-----------|------|---------|-------------|")
        for h_str, ic in sorted(all_ic.items(), key=lambda x: int(x[0])):
            lines.append(
                f"| {h_str}天 | {_fmt(ic.get('ic_mean'), 'f4')} | "
                f"{_fmt(ic.get('ic_std'), 'f4')} | {_fmt(ic.get('icir'), 'f4')} | "
                f"{_fmt(ic.get('ic_tstat'), 'f4')} | {_fmt(ic.get('ic_positive_ratio'), 'pct')} |"
            )
        lines.append("")

        # RankIC (Spearman) — less sensitive to outliers, preferred for A-shares
        has_rankic = any("rank_ic_mean" in ic for ic in all_ic.values())
        if has_rankic:
            lines.append("### RankIC 指标（Spearman）")
            lines.append("")
            lines.append("| 周期 | RankIC 均值 | RankIC 标准差 | RankICIR | RankIC t 值 | RankIC 正向占比 |")
            lines.append("|------|-------------|---------------|----------|--------------|----------------|")
            for h_str, ic in sorted(all_ic.items(), key=lambda x: int(x[0])):
                lines.append(
                    f"| {h_str}天 | {_fmt(ic.get('rank_ic_mean'), 'f4')} | "
                    f"{_fmt(ic.get('rank_ic_std'), 'f4')} | {_fmt(ic.get('rank_icir'), 'f4')} | "
                    f"{_fmt(ic.get('rank_ic_tstat'), 'f4')} | {_fmt(ic.get('rank_ic_positive_ratio'), 'pct')} |"
                )
            lines.append("")

    # IC decay — generated during step3, copy into report
    _copy_eval_plot(state, "eval_ic_decay.png", plots_dir)
    if (plots_dir / "eval_ic_decay.png").exists():
        lines.append("![IC 衰减图](plots/eval_ic_decay.png)")
        lines.append("")

    # IC time series — generated during step3 in factor_eval/plots/.
    # Copy into the report plots dir so markdown references stay local.
    eval_plots = state.artifacts.get("eval_plots_dir")
    if eval_plots:
        src_dir = Path(eval_plots)
        for h in [1, 5, 20]:
            src = src_dir / f"ic_ts_h{h}.png"
            if src.exists():
                dst = plots_dir / src.name
                shutil.copy2(src, dst)
                lines.append(f"![IC 时序图 (h={h})](plots/{src.name})")
                lines.append("")
    return lines


def _render_step4(state: PipelineState, plots_dir: Path) -> list[str]:
    result = state.step_results.get("step4")
    if not result or not result.metrics:
        return ["*无数据。*", ""]

    # 分组收益（静态截面）
    lines = ["### 分组收益", ""]
    lines.extend(_metrics_sub_table(result.metrics))
    lines.append("")
    group_rets = result.metrics.get("group_mean_returns", {})
    if group_rets:
        _copy_eval_plot(state, "eval_group_returns.png", plots_dir)
        if (plots_dir / "eval_group_returns.png").exists():
            lines.append("![分组收益图](plots/eval_group_returns.png)")
            lines.append("")

    # 十档分层回测（动态净值）
    lines.extend(_decile_content(state, plots_dir))
    return lines


def _render_step5(state: PipelineState, plots_dir: Path) -> list[str]:
    result = state.step_results.get("step5")
    if not result or not result.metrics:
        return ["*无数据。*", ""]
    return _metrics_sub_table(result.metrics)


def _render_step6(state: PipelineState, plots_dir: Path) -> list[str]:
    result = state.step_results.get("step6")
    if not result or not result.metrics:
        return ["*无数据。*", ""]

    lines = _bt_metrics_table(result.metrics)
    lines.append("")
    # NAV plot was generated during step6 by _backtest_gate.
    # Copy it into the report plots dir if available.
    _copy_bt_nav_plot(state, "simple", plots_dir)
    if (plots_dir / "nav_simple.png").exists():
        lines.append("![简单回测净值曲线](plots/nav_simple.png)")
        lines.append("")
    return lines


def _render_step7(state: PipelineState, plots_dir: Path) -> list[str]:
    result = state.step_results.get("step7")
    if not result or not result.metrics:
        return ["*无数据。*", ""]

    lines = _bt_metrics_table(result.metrics)
    lines.append("")
    _copy_bt_nav_plot(state, "detailed", plots_dir)
    if (plots_dir / "nav_detailed.png").exists():
        lines.append("![详细回测净值曲线](plots/nav_detailed.png)")
        lines.append("")

    p = _plot_evaluation_report(state, plots_dir)
    if p:
        lines.append(f"![回测全景图](plots/{p.name})")
        lines.append("")

    step6 = state.step_results.get("step6")
    simple_ann = (step6.metrics or {}).get("annual_return", 0) or 0
    detailed_ann = result.metrics.get("annual_return", 0) or 0
    drag = simple_ann - detailed_ann
    lines.append(f"*成本侵蚀：简单回测年化 {_fmt(simple_ann, 'pct')} → "
                 f"详细回测年化 {_fmt(detailed_ann, 'pct')}，"
                 f"侵蚀 {_fmt(drag, 'pct')}*")
    lines.append("")
    return lines


def _render_step8(state: PipelineState, plots_dir: Path) -> list[str]:
    result = state.step_results.get("step8")
    if not result or not result.metrics:
        return ["*未执行。*", ""]

    r2 = result.metrics.get("r2")
    tier = result.metrics.get("tier")
    needs_residual = result.metrics.get("needs_residual", False)
    r2_stats = result.metrics.get("r2_stats", {})
    tier_names = {"pure_alpha": "纯 Alpha", "smart_beta": "Smart Beta", "reject": "风格克隆（需残差化）"}
    tier_cn = tier_names.get(tier, str(tier))

    lines = [
        f"- **方法**：每日截面 Ridge 回归（与 step9 一致），逐日 R² 取分布统计",
        f"- **R² 均值**：{_fmt(r2, 'f4')}（门控用）",
        f"- **R² 中位数**：{_fmt(r2_stats.get('median'), 'f4')}",
        f"- **R² P90**：{_fmt(r2_stats.get('p90'), 'f4')}",
        f"- **R² P95**：{_fmt(r2_stats.get('p95'), 'f4')}",
        f"- **R² P99**：{_fmt(r2_stats.get('p99'), 'f4')}",
        f"- **分档**：`{tier}`（{tier_cn}）",
        f"- **样本数**：{result.metrics.get('n_obs'):,}",
        "",
    ]
    if needs_residual:
        lines.append(
            f"> R² 超出阈值，因子与现有因子高度重叠。"
            f"不直接拒绝——交由 step9 检查残差预测力，"
            f"若残差 ICIR 通过则以**残差值**入库。"
        )
        lines.append("")
    else:
        lines.extend([
            "| 分档 | R² 范围 | 含义 |",
            "|------|---------|------|",
            "| `pure_alpha` | R² < 0.2 | 与现有风格正交 — 原值入库 |",
            "| `smart_beta` | 0.2 ≤ R² < 0.7 | 部分风格暴露 — 原值入库 |",
            "| `reject` | R² ≥ 0.7 | 委托 step9 残差 ICIR 检查 |",
            "",
        ])
    return lines


def _render_step9_residual_icir(state: PipelineState, plots_dir: Path) -> list[str]:
    """Residual ICIR incremental-information check."""
    result = state.step_results.get("step9")
    if not result or not result.metrics:
        return ["*未执行。*", ""]

    m = result.metrics
    annual_icirs = m.get("annual_icirs", {})
    rank_icirs = m.get("residual_rank_icirs", {})
    rank_means = m.get("residual_rank_ic_means", {})
    rank_stds = m.get("residual_rank_ic_stds", {})
    threshold = m.get("threshold", 0.0)
    n_regressors = m.get("n_regressors", 0)
    n_dates = m.get("n_dates", 0)
    passed = m.get("passed", False)
    admission_mode = m.get("admission_mode", "reject")

    mode_desc = {
        "raw": "原值入库（因子与现有因子正交）",
        "residual": "**残差入库**（剥离风格克隆部分，仅保留纯净 alpha）",
        "reject": "**拒绝**",
    }.get(admission_mode, str(admission_mode))

    lines = [
        f"- **方法**：每日截面 Ridge 回归取残差 → RankIC vs 远期收益",
        f"- **回归因子数**：{n_regressors}",
        f"- **有效日期数**：{n_dates}",
        f"- **年化阈值**：{threshold}",
        f"- **入库模式**：{mode_desc}",
        "",
        "| 周期 | 原始 RankICIR | 年化 RankICIR | RankIC 均值 | RankIC 标准差 |",
        "|------|---------------|---------------|-------------|---------------|",
    ]
    for h in sorted(annual_icirs.keys(), key=lambda k: int(k)):
        a_icir = annual_icirs.get(h, float("nan"))
        r_icir = rank_icirs.get(h, float("nan"))
        r_mean = rank_means.get(h, float("nan"))
        r_std = rank_stds.get(h, float("nan"))
        lines.append(
            f"| {h}D | {_fmt(r_icir, 'f4')} | {_fmt(a_icir, 'f4')} | "
            f"{_fmt(r_mean, 'f4')} | {_fmt(r_std, 'f4')} |"
        )
    lines.append("")
    return lines


def _render_step10(state: PipelineState, plots_dir: Path) -> list[str]:
    result = state.step_results.get("step10")
    if not result or not result.metrics:
        return ["*未执行。*", ""]
    lines = _metrics_sub_table(result.metrics)
    if state.status == "ready_for_review":
        lines.append(f"")
        lines.append(f"> 人工入库：`python -m backtest.factor.admission admit {state.config.factor_id}`")
    return lines


# ===========================================================================
# 十档分层回测
# ===========================================================================


def _decile_content(state: PipelineState, plots_dir: Path) -> list[str]:
    """十档分层回测（嵌入 step4 单调性）。

    Uses ``state.eval_result.decile_result`` populated by step4 — does not
    re-run ``evaluate()``.
    """
    lines = ["### 十档分层回测", ""]

    try:

        eval_result = state.eval_result
        if eval_result is None:
            lines.append("*无评估结果。*")
            lines.append("")
            return lines

        dr = getattr(eval_result, "decile_result", None)
        if dr is None:
            lines.append("*无结果。*")
            lines.append("")
            return lines

        # Decile plot was generated during step4; copy into report plots dir.
        config = state.config
        factor_eval = Path(config.results_root) / config.factor_id / "factor_eval"
        decile_png = factor_eval / "decile_backtest" / f"{config.factor_id}_decile.png"
        if decile_png.exists():
            dst = plots_dir / "decile_backtest.png"
            shutil.copy2(decile_png, dst)
            lines.append(f"![十档分层回测](plots/{dst.name})")
            lines.append("")

        ls = dr.ls_metrics
        lines.append(f"- **单调性**：{_fmt(dr.monotonicity_score, 'f4')}")
        lines.append(f"- **多空年化收益**：{_fmt(ls.get('annual_return'), 'pct')}")
        lines.append(f"- **多空 Sharpe**：{_fmt(ls.get('sharpe'), 'f3')}")
        lines.append(f"- **多空最大回撤**：{_fmt(ls.get('max_drawdown'), 'pct')}")
        lines.append("")
    except Exception as exc:
        lines.append(f"*暂不可用（{exc}）。*")
        lines.append("")

    return lines


def _bt_metrics_table(metrics: dict) -> list[str]:
    lines: list[str] = []

    # Main table: absolute + relative excess metrics
    core_rows = [
        ("年化收益", "annual_return", "excess_annual_return", "pct"),
        ("Sharpe", "sharpe", "excess_sharpe", "f3"),
        ("最大回撤", "max_drawdown", "excess_max_drawdown", "pct"),
        ("Calmar", "calmar", "excess_calmar", "f3"),
    ]
    aliases = [("hs300", "沪深300"), ("csi500", "中证500"), ("csi1000", "中证1000")]
    header = "| 指标 | 绝对 | " + " | ".join(f"相对{label}" for _, label in aliases) + " |"
    sep = "|------|------|" + "|".join("------" for _ in aliases) + "|"
    lines.extend([header, sep])

    for label, abs_key, rel_prefix, kind in core_rows:
        cells = [label]
        # absolute
        val = metrics.get(abs_key)
        cells.append(
            _fmt(val, kind)
            if val is not None and not (isinstance(val, float) and np.isnan(val))
            else "n/a"
        )
        # relative
        if rel_prefix:
            for alias, _ in aliases:
                key = f"{rel_prefix}_{alias}"
                val = metrics.get(key)
                cells.append(
                    _fmt(val, kind)
                    if val is not None and not (isinstance(val, float) and np.isnan(val))
                    else "n/a"
                )
        else:
            cells.extend(["—"] * len(aliases))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    # Supplementary table: metrics without relative versions
    extra_rows = [
        ("年化换手率", "annual_turnover", "f2"),
        ("日胜率", "daily_win_rate", "pct"),
        ("总交易笔数", "total_trades", "int"),
    ]
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    for label, key, kind in extra_rows:
        val = metrics.get(key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            lines.append(f"| {label} | {_fmt(val, kind)} |")

    return lines


def _copy_eval_plot(state: PipelineState, filename: str, plots_dir: Path) -> None:
    """Copy a pre-generated eval plot from factor_eval/plots/ to the report dir."""
    eval_plots = state.artifacts.get("eval_plots_dir")
    if eval_plots:
        src = Path(eval_plots) / filename
        if src.exists():
            shutil.copy2(src, plots_dir / filename)


def _copy_bt_nav_plot(state: PipelineState, tag: str, plots_dir: Path) -> None:
    """Copy the pre-generated backtest nav plot from the backtest output dir.

    The actual plot is generated during step6/step7 by
    ``_gen_backtest_nav_plot`` — the report only copies it.
    """
    art_key = f"{tag}_bt"
    bt_dir = state.artifacts.get(art_key)
    src = Path(bt_dir) / f"nav_{tag}.png" if bt_dir else None
    if src and src.exists():
        shutil.copy2(src, plots_dir / f"nav_{tag}.png")


def _plot_evaluation_report(state: PipelineState, plots_dir: Path) -> Path | None:
    try:
        from backtest.evaluation import evaluate
        bt_dir = state.artifacts.get("detailed_bt")
        if not bt_dir:
            return None
        src = Path(bt_dir) / "report.png"
        if not src.exists():
            evaluate(str(bt_dir), benchmark=state.config.benchmark, plot=True)
        if src.exists():
            dst = plots_dir / "bt_report.png"
            dst.write_bytes(src.read_bytes())
            return dst
        return None
    except Exception:
        return None


# ===========================================================================
# Helpers
# ===========================================================================


def _get_factor_formula(factor_id: str) -> list[str] | None:
    """Read factor source code from registry and format as code block.

    Only reads source when the function is already cached in memory;
    avoids triggering ``importlib.import_module`` side effects during
    read-only report generation.
    """
    try:
        import inspect
        from backtest.factor.registry import get_factor_meta, _FACTOR_FUNCTIONS

        meta = get_factor_meta(factor_id)
        name = meta.get("name", "")
        desc = meta.get("description", "")
        variant = meta.get("variant", "")
        sources = meta.get("data_sources", [])

        # Only use cached functions to avoid import side effects.
        func = _FACTOR_FUNCTIONS.get(factor_id)
        if func is not None:
            source = inspect.getsource(func)
            lines = [f"**因子名称**：{name}", ""]
            if desc:
                lines.append(f"> {desc}")
                lines.append("")
            lines.append(f"- 数据源：`{', '.join(sources)}`")
            lines.append(f"- 中性化：`{variant}`")
            lines.append("")
            lines.append("### 因子实现代码")
            lines.append("")
            lines.append("```python")
            lines.extend(source.strip().split("\n"))
            lines.append("```")
            lines.append("")
            return lines

        # Fallback: metadata only (function not in memory cache)
        params = meta.get("parameters", {})
        lines = [f"**因子名称**：{name}", ""]
        if desc:
            lines.append(f"> {desc}")
            lines.append("")
        lines.append(f"- 数据源：`{', '.join(sources)}`")
        lines.append(f"- 中性化：`{variant}`")
        if params:
            lines.append(f"- 参数：`{params}`")
        lines.append("")
        return lines
    except Exception:
        return None


def _find_rejection_reason(state: PipelineState) -> str | None:
    for step_key in _STEP_NAMES:
        result = state.step_results.get(step_key)
        if result and not result.passed and result.reason:
            return result.reason
    return None


def _find_rejection_step(state: PipelineState) -> str | None:
    for step_key in _STEP_NAMES:
        result = state.step_results.get(step_key)
        if result and not result.passed:
            return step_key
    return None
