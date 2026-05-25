"""生成中文诊断报告 + 全部图表。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        top_k = cfg.get("default_top_k")
        top_pct = cfg.get("default_top_pct")
        decay = cfg.get("default_decay", 5)
        rebalance = cfg.get("default_rebalance", "1D")
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
    tag_dir = Path(config.results_root) / config.factor_id / tag
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
        f"- **重试次数**：{state.retry_count}",
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
        lines.append("### IC 指标")
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

    # IC decay overview
    p = _plot_ic_decay(all_ic, plots_dir)
    if p:
        lines.append(f"![IC 衰减图](plots/{p.name})")
        lines.append("")

    # IC time series per major horizon — single evaluate() call, plot from cache
    _plot_ic_time_series_multi(state, plots_dir)
    for h in [1, 5, 20]:
        p = plots_dir / f"eval_ic_ts_h{h}.png"
        if p.exists():
            lines.append(f"![IC 时序图 (h={h})](plots/{p.name})")
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
        p = _plot_group_returns(group_rets, plots_dir)
        if p:
            lines.append(f"![分组收益图](plots/{p.name})")
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
    _plot_backtest_nav(state, tag="simple", plots_dir=plots_dir)
    lines.append("![简单回测净值曲线](plots/bt_simple_nav.png)")
    lines.append("")
    return lines


def _render_step7(state: PipelineState, plots_dir: Path) -> list[str]:
    result = state.step_results.get("step7")
    if not result or not result.metrics:
        return ["*无数据。*", ""]

    lines = _bt_metrics_table(result.metrics)
    lines.append("")
    _plot_backtest_nav(state, tag="detailed", plots_dir=plots_dir)
    lines.append("![详细回测净值曲线](plots/bt_detailed_nav.png)")
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
    tier_names = {"pure_alpha": "纯 Alpha", "smart_beta": "Smart Beta", "reject": "风格克隆"}
    tier_cn = tier_names.get(tier, str(tier))

    lines = [
        f"- **R²**：{_fmt(r2, 'f4')}",
        f"- **分档**：`{tier}`（{tier_cn}）",
        f"- **样本数**：{result.metrics.get('n_obs'):,}",
        "",
        "| 分档 | R² 范围 | 含义 |",
        "|------|---------|------|",
        "| `pure_alpha` | R² < 0.2 | 与现有风格正交 — 入库 |",
        "| `smart_beta` | 0.2 ≤ R² < 0.7 | 部分风格暴露 — 入库 |",
        "| `reject` | R² ≥ 0.7 | 风格克隆 — 拒绝 |",
        "",
    ]
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

    lines = [
        f"- **回归因子数**：{n_regressors}",
        f"- **有效日期数**：{n_dates}",
        f"- **年化阈值**：{threshold}",
        f"- **结论**：{'通过' if passed else '**拒绝**'}（增量信息检查）",
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
    """十档分层回测（嵌入 step4 单调性）。"""
    lines = ["### 十档分层回测", ""]

    try:
        from backtest.factor.evaluation import evaluate
        from backtest.simulation.decile import plot_decile_backtest

        config = state.config
        decile_png = Path(config.results_root) / config.factor_id / "decile_backtest" / f"{config.factor_id}_decile.png"

        result = evaluate(
            config.factor_id, config.start_date, config.end_date,
            horizons=[20], ret_type=config.ret_type,
            corr_top_k=0, exclude_limit_up=True, run_decile_backtest=True,
        )
        dr = result.decile_result
        if dr is None:
            lines.append("*无结果。*")
            lines.append("")
            return lines

        if not decile_png.exists():
            decile_png.parent.mkdir(parents=True, exist_ok=True)
            plot_decile_backtest(dr, str(decile_png))

        dst = plots_dir / "decile_backtest.png"
        dst.write_bytes(decile_png.read_bytes())
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


# ===========================================================================
# 图表生成
# ===========================================================================


def _plot_ic_decay(all_ic: dict, plots_dir: Path) -> Path | None:
    horizons = sorted(int(h) for h in all_ic)

    def _get(h: int, key: str):
        return (all_ic.get(h, {}) or all_ic.get(str(h), {})).get(key, np.nan)

    means = [_get(h, "ic_mean") for h in horizons]
    stds = [_get(h, "ic_std") for h in horizons]
    icirs = [_get(h, "icir") for h in horizons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_FIGSIZE_WIDE)
    ax1.errorbar(horizons, means, yerr=stds, marker="o", color="steelblue", capsize=4, linewidth=1.5)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("预测周期（天）")
    ax1.set_ylabel("IC")
    ax1.set_title("IC 均值 ± 标准差")
    ax1.grid(True, alpha=0.3)
    ax2.bar(horizons, icirs, color="darkorange", alpha=0.8)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("预测周期（天）")
    ax2.set_ylabel("ICIR")
    ax2.set_title("ICIR")
    ax2.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = plots_dir / "eval_ic_decay.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_ic_time_series_multi(state: PipelineState, plots_dir: Path) -> None:
    """Generate IC time series plots for h=1,5,20 in one evaluate() call."""
    try:
        from backtest.factor.evaluation import evaluate, plot_evaluation
        config = state.config
        result = evaluate(
            config.factor_id, config.start_date, config.end_date,
            horizons=[1, 5, 20], ret_type=config.ret_type,
            corr_top_k=0, exclude_limit_up=True, run_decile_backtest=False,
        )
        for h in [1, 5, 20]:
            if h in result.ic_series:
                out = plots_dir / f"eval_ic_ts_h{h}.png"
                plot_evaluation(result, horizon=h, output_path=str(out))
    except Exception:
        pass


def _plot_group_returns(group_rets: dict, plots_dir: Path) -> Path | None:
    groups = sorted(int(g) for g in group_rets)
    values = [group_rets[str(g)] if str(g) in group_rets else group_rets.get(g, 0) for g in groups]
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    colors = ["#d73027" if v < 0 else "#4575b4" for v in values]
    ax.bar(groups, values, color=colors, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("分位组")
    ax.set_ylabel("平均前瞻收益")
    ax.set_title("各分位组平均收益（h=1）")
    ax.set_xticks(groups)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = plots_dir / "eval_group_returns.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def _bt_metrics_table(metrics: dict) -> list[str]:
    rows = [
        ("年化收益", "annual_return", "pct"),
        ("Sharpe", "sharpe", "f3"),
        ("最大回撤", "max_drawdown", "pct"),
        ("Calmar", "calmar", "f3"),
        ("年化换手率", "annual_turnover", "f2"),
        ("日胜率", "daily_win_rate", "pct"),
        ("总交易笔数", "total_trades", "int"),
    ]
    lines = ["| 指标 | 数值 |", "|------|------|"]
    for label, key, kind in rows:
        val = metrics.get(key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            lines.append(f"| {label} | {_fmt(val, kind)} |")
    return lines


def _plot_backtest_nav(state: PipelineState, *, tag: str, plots_dir: Path) -> None:
    art_key = f"{tag}_bt"
    bt_dir = state.artifacts.get(art_key)
    nav_path = Path(bt_dir) / "nav.parquet" if bt_dir else None
    if nav_path is None or not nav_path.exists():
        _plot_backtest_summary_card(state, tag, plots_dir)
        return
    nav_df = pd.read_parquet(nav_path)
    if nav_df.empty or "nav" not in nav_df.columns:
        _plot_backtest_summary_card(state, tag, plots_dir)
        return
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_series = nav_df.set_index("date")["nav"].astype(float)
    nav_norm = nav_series / nav_series.iloc[0]
    drawdown = nav_series / nav_series.expanding().max() - 1.0
    title_map = {"simple": "简单回测", "detailed": "详细回测"}
    title = title_map.get(tag, tag)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
    ax1.plot(nav_norm.index, nav_norm.values, color="steelblue", linewidth=1.4)
    ax1.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
    ret_type = state.config.ret_type
    ret_label = "o2o" if ret_type == "open" else "c2c"
    ax1.set_ylabel("净值")
    ax1.set_title(f"{title} — 净值曲线 ({ret_label})")
    ax1.grid(True, alpha=0.3)
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    ax2.plot(drawdown.index, drawdown.values, color="red", linewidth=1.0)
    ax2.set_ylabel("回撤")
    ax2.set_xlabel("日期")
    ax2.set_title(f"{title} — 回撤曲线 ({ret_label})")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out = plots_dir / f"bt_{tag}_nav.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_backtest_summary_card(state: PipelineState, tag: str, plots_dir: Path) -> None:
    step_key = "step6" if tag == "simple" else "step7"
    step = state.step_results.get(step_key)
    metrics = step.metrics if step else {}
    title_map = {"simple": "简单回测", "detailed": "详细回测"}
    title = title_map.get(tag, tag)
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    card_lines = []
    for label, key, kind in [
        ("年化收益", "annual_return", "pct"),
        ("Sharpe", "sharpe", "f3"),
        ("最大回撤", "max_drawdown", "pct"),
        ("Calmar", "calmar", "f3"),
        ("年化换手", "annual_turnover", "f2"),
    ]:
        val = metrics.get(key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            card_lines.append(f"{label}: {_fmt(val, kind)}")
    ax.axis("off")
    ax.text(0.5, 0.5, "\n".join(card_lines), ha="center", va="center",
            transform=ax.transAxes, fontsize=14, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    ax.set_title(f"{title} 摘要", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = plots_dir / f"bt_{tag}_nav.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


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
