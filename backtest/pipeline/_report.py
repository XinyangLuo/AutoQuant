"""生成中文诊断报告 + 全部图表。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# CJK font support — try common macOS / Linux Chinese fonts in order.
for _font in ("PingFang SC", "Heiti SC", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"):
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

# ---------------------------------------------------------------------------
# Tag
# ---------------------------------------------------------------------------


def _build_tag(state: PipelineState) -> str:
    cfg = state.strategy_config
    if cfg is None:
        step5 = state.step_results.get("step5")
        if step5 and step5.metrics:
            top_pct = step5.metrics.get("top_pct", 0.1)
            decay = step5.metrics.get("decay", 5)
            rebalance = step5.metrics.get("rebalance", "1D")
            return f"top{int(round(top_pct * 100))}pct_{rebalance.lower()}_d{decay}"
        return "default"

    if isinstance(cfg, dict):
        top_pct = cfg.get("default_top_pct", 0.1)
        decay = cfg.get("default_decay", 5)
        rebalance = cfg.get("default_rebalance", "1D")
        return f"top{int(round(top_pct * 100))}pct_{rebalance.lower()}_d{decay}"

    sel = cfg.selection
    if sel.top_pct is not None:
        tag = f"top{int(round(sel.top_pct * 100))}pct"
    else:
        tag = f"top{sel.top_k}"
    decay = cfg.decay or 0
    return f"{tag}_{cfg.rebalance_freq.lower()}_d{decay}"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


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
        f"- **频率**：{config.frequency}",
        f"- **状态**：{state.status}",
        f"- **重试次数**：{state.retry_count}",
        "",
    ]

    lines.extend(_decision_banner(state))
    lines.extend(_step_summary_table(state))
    lines.extend(_factor_eval_section(state, tag_dir, plots_dir))
    lines.extend(_backtest_section(state, tag_dir, plots_dir))
    lines.extend(_decile_section(state, plots_dir))
    lines.extend(_ridge_section(state))
    lines.extend(_decision_detail(state))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# 决策横幅
# ---------------------------------------------------------------------------


def _decision_banner(state: PipelineState) -> list[str]:
    lines: list[str] = []
    if state.status == "admitted":
        lines.append("> **结果：已入库**")
    elif state.status == "ready_for_review":
        lines.append("> **结果：待人工审核**")
        lines.append(">")
        factor_id = state.config.factor_id
        lines.append(f"> 所有步骤通过，请查看报告后手动入库：")
        lines.append(f"> `python -m backtest.factor.admission admit {factor_id}`")
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


# ---------------------------------------------------------------------------
# 步骤汇总表
# ---------------------------------------------------------------------------


_STEP_NAMES = {
    "step1": "覆盖率",
    "step2": "中性化验证",
    "step3": "ICIR 门控",
    "step4": "单调性",
    "step5": "策略配置",
    "step6": "简单回测",
    "step7": "详细回测",
    "step8": "Ridge R² 分档",
    "step9": "报告生成",
}


def _step_label(step_key: str) -> str:
    return _STEP_NAMES.get(step_key, step_key)


def _step_summary_table(state: PipelineState) -> list[str]:
    lines = ["## 各步骤结果汇总", ""]
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
        "step3": ["annual_icir", "abs_ic", "tstat", "best_horizon"],
        "step4": ["spearman", "n_groups"],
        "step5": ["top_pct", "decay", "rebalance"],
        "step6": ["sharpe", "annual_return", "max_drawdown"],
        "step7": ["sharpe", "annual_return", "annual_turnover"],
        "step8": ["r2", "tier"],
        "step9": [],
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


# ---------------------------------------------------------------------------
# 因子评估
# ---------------------------------------------------------------------------


def _factor_eval_section(state: PipelineState, tag_dir: Path, plots_dir: Path) -> list[str]:
    lines = ["## 因子离线评估", ""]

    step3 = state.step_results.get("step3")
    if step3 is None:
        lines.append("*无评估数据。*")
        lines.append("")
        return lines

    # IC 指标表
    all_ic = step3.metrics.get("all_ic_metrics", {})
    if all_ic:
        lines.append("### 各周期 IC 指标")
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

    best_h = step3.metrics.get("best_horizon")
    if best_h:
        lines.append(f"**最优周期**：{best_h}天 "
                     f"（年化 ICIR={_fmt(step3.metrics.get('annual_icir'), 'f4')}，"
                     f"|IC|={_fmt(step3.metrics.get('abs_ic'), 'f4')}，"
                     f"t={_fmt(step3.metrics.get('tstat'), 'f4')}）")
        lines.append("")

    step4 = state.step_results.get("step4")
    if step4:
        lines.append(f"**单调性**：Spearman 相关系数 = {_fmt(step4.metrics.get('spearman'), 'f4')} "
                     f"（阈值 {state.config.thresholds.min_monotonicity}）")
        lines.append("")

    # IC 衰减图
    plot_path = _plot_ic_decay(all_ic, plots_dir)
    if plot_path:
        lines.append(f"![IC 衰减图](plots/{plot_path.name})")
        lines.append("")

    # IC 时序图（4 面板）
    ts_plot = _plot_ic_time_series(state, plots_dir)
    if ts_plot:
        lines.append(f"![IC 时序图](plots/{ts_plot.name})")
        lines.append("")

    # 分组收益图
    step4_metrics = (step4.metrics if step4 else {})
    group_rets = step4_metrics.get("group_mean_returns", {})
    if group_rets:
        plot_path = _plot_group_returns(group_rets, plots_dir)
        if plot_path:
            lines.append(f"![分组收益图](plots/{plot_path.name})")
            lines.append("")

    return lines


# ---------------------------------------------------------------------------
# 回测结果
# ---------------------------------------------------------------------------


def _backtest_section(state: PipelineState, tag_dir: Path, plots_dir: Path) -> list[str]:
    lines = ["## 策略回测", ""]

    step5 = state.step_results.get("step5")
    if step5 and step5.metrics:
        lines.append(f"**策略参数**：top_pct={step5.metrics.get('top_pct')}，"
                     f"decay={step5.metrics.get('decay')}，"
                     f"rebalance={step5.metrics.get('rebalance')}")
        lines.append("")

    step6 = state.step_results.get("step6")
    step7 = state.step_results.get("step7")

    # 简单回测
    if step6 and step6.metrics:
        lines.append("### 简单回测（向量化，无交易成本）")
        lines.append("")
        lines.extend(_metrics_table(step6.metrics))
        lines.append("")
        _plot_backtest_nav(state, tag="simple", plots_dir=plots_dir)
        lines.append("![简单回测净值曲线](plots/bt_simple_nav.png)")
        lines.append("")

    # 详细回测
    if step7 and step7.metrics:
        lines.append("### 详细回测（事件驱动，含佣金/印花税/过户费）")
        lines.append("")
        lines.extend(_metrics_table(step7.metrics))
        lines.append("")
        _plot_backtest_nav(state, tag="detailed", plots_dir=plots_dir)
        lines.append("![详细回测净值曲线](plots/bt_detailed_nav.png)")
        lines.append("")

        # 8 面板大图
        report_png = _plot_evaluation_report(state, plots_dir)
        if report_png:
            lines.append(f"![回测全景图](plots/{report_png.name})")
            lines.append("")

        # 成本侵蚀
        simple_ann = (step6.metrics or {}).get("annual_return", 0) or 0
        detailed_ann = step7.metrics.get("annual_return", 0) or 0
        drag = simple_ann - detailed_ann
        lines.append(f"**成本侵蚀**：简单回测年化 {_fmt(simple_ann, 'pct')} → "
                     f"详细回测年化 {_fmt(detailed_ann, 'pct')}，"
                     f"侵蚀 {_fmt(drag, 'pct')}")
        lines.append("")

    return lines


def _plot_evaluation_report(state: PipelineState, plots_dir: Path) -> Path | None:
    """调用 evaluation 模块生成 8 面板全景图（report.png）。"""
    try:
        from backtest.evaluation import evaluate
        bt_dir = state.artifacts.get("detailed_bt")
        if not bt_dir:
            return None
        src = Path(bt_dir) / "report.png"
        # Only re-run evaluate() if report.png doesn't already exist.
        if not src.exists():
            evaluate(str(bt_dir), benchmark=state.config.benchmark, plot=True)
        if src.exists():
            dst = plots_dir / "bt_report.png"
            dst.write_bytes(src.read_bytes())
            return dst
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 十档分层回测
# ---------------------------------------------------------------------------


def _decile_section(state: PipelineState, plots_dir: Path) -> list[str]:
    lines = ["## 十档分层回测", ""]

    try:
        from backtest.factor.evaluation import evaluate

        config = state.config

        # Check if decile plot already exists — skip re-computation if so.
        decile_png = Path(config.results_root) / config.factor_id / "decile_backtest" / f"{config.factor_id}_decile.png"
        if not decile_png.exists():
            evaluate(
                config.factor_id,
                config.start_date,
                config.end_date,
                horizons=[20],
                ret_type=config.ret_type,
                corr_top_k=0,
                exclude_limit_up=True,
                run_decile_backtest=True,
            )

        if decile_png.exists():
            dst = plots_dir / "decile_backtest.png"
            dst.write_bytes(decile_png.read_bytes())
            lines.append(f"![十档分层回测](plots/{dst.name})")
            lines.append("")

            # Metrics are embedded in the plot PNG; the image is self-contained.
            lines.append("*十档分层净值曲线见上图，含 D1~D10 分组 NAV 和多空对冲曲线。*")
            lines.append("")
    except Exception as exc:
        lines.append(f"*十档分层回测暂不可用（{exc}）。*")
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Ridge R²
# ---------------------------------------------------------------------------


def _ridge_section(state: PipelineState) -> list[str]:
    lines = ["## Ridge R² 风格分档", ""]

    step8 = state.step_results.get("step8")
    if step8 is None:
        lines.append("*未执行。*")
        lines.append("")
        return lines

    r2 = step8.metrics.get("r2")
    tier = step8.metrics.get("tier")
    tier_names = {"pure_alpha": "纯 Alpha", "smart_beta": "Smart Beta", "reject": "风格克隆"}
    tier_cn = tier_names.get(tier, str(tier))

    lines.append(f"- **R²**：{_fmt(r2, 'f4')}")
    lines.append(f"- **分档**：`{tier}`（{tier_cn}）")
    lines.append(f"- **样本数**：{step8.metrics.get('n_obs'):,}")
    lines.append("")
    lines.append("| 分档 | R² 范围 | 含义 |")
    lines.append("|------|---------|------|")
    lines.append("| `pure_alpha` | R² < 0.2 | 与现有风格正交 — 入库 |")
    lines.append("| `smart_beta` | 0.2 ≤ R² < 0.7 | 部分风格暴露 — 入库 |")
    lines.append("| `reject` | R² ≥ 0.7 | 风格克隆 — 拒绝 |")
    lines.append("")

    return lines


# ---------------------------------------------------------------------------
# 决策明细
# ---------------------------------------------------------------------------


def _decision_detail(state: PipelineState) -> list[str]:
    lines = ["## 各步骤明细", ""]

    for step_key in _STEP_NAMES:
        result = state.step_results.get(step_key)
        if result is None:
            continue
        status = "通过" if result.passed else "**拒绝**"
        name = _STEP_NAMES[step_key]
        lines.append(f"- **{name}**（{step_key}）：{status}")
        if result.metrics:
            for k, v in result.metrics.items():
                if k == "all_ic_metrics":
                    continue
                if isinstance(v, float):
                    lines.append(f"  - {k}: {v:.4f}")
                elif isinstance(v, dict):
                    flat = ", ".join(
                        f"{dk}={_fmt(dv, 'f4') if isinstance(dv, float) else dv}"
                        for dk, dv in list(v.items())[:5]
                    )
                    lines.append(f"  - {k}: {{{flat}}}")
                else:
                    lines.append(f"  - {k}: {v}")
        if result.reason:
            lines.append(f"  - *原因*: {result.reason}")
        lines.append("")

    return lines


# ===========================================================================
# 图表生成函数
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

    ax2.bar(horizons, icirs, color="darkorange", alpha=0.8)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("预测周期（天）")
    ax2.set_ylabel("ICIR")
    ax2.set_title("ICIR")

    fig.tight_layout()
    out = plots_dir / "eval_ic_decay.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_ic_time_series(state: PipelineState, plots_dir: Path) -> Path | None:
    step3 = state.step_results.get("step3")
    if step3 is None:
        return None
    best_h = step3.metrics.get("best_horizon")
    if best_h is None:
        return None

    try:
        from backtest.factor.evaluation import evaluate, plot_evaluation

        config = state.config
        result = evaluate(
            config.factor_id, config.start_date, config.end_date,
            horizons=[best_h], ret_type=config.ret_type,
            corr_top_k=0, exclude_limit_up=True, run_decile_backtest=False,
        )
        out = plots_dir / "eval_ic_ts.png"
        plot_evaluation(result, horizon=best_h, output_path=str(out))
        return out
    except Exception:
        return None


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

    fig.tight_layout()
    out = plots_dir / "eval_group_returns.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


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
    cummax = nav_series.expanding().max()
    drawdown = nav_series / cummax - 1.0

    title_map = {"simple": "简单回测", "detailed": "详细回测"}
    title = title_map.get(tag, tag)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))

    ax1.plot(nav_norm.index, nav_norm.values, color="steelblue", linewidth=1.4)
    ax1.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
    ax1.set_ylabel("净值")
    ax1.set_title(f"{title} — 净值曲线")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    ax2.plot(drawdown.index, drawdown.values, color="red", linewidth=1.0)
    ax2.set_ylabel("回撤")
    ax2.set_xlabel("日期")
    ax2.set_title(f"{title} — 回撤曲线")
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


def _metrics_table(metrics: dict) -> list[str]:
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
