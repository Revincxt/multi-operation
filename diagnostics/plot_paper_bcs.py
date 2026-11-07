"""把既有贪心与 BCS 测试结果绘制为论文图，不重新运行仿真。

使用方式:
    python3 -m diagnostics.plot_paper_bcs --output experiments/lstm_bcs_paper_figures

两版本共用坐标尺度，输出 600 dpi PNG、矢量 PDF/SVG 和逐项绘图数据。
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline.plot_paper_figures import _font_family


DEFAULT_INPUTS = (
    "experiments/lstm_greedy_vs_bcs_20261023",
    "experiments/lstm_joint_greedy_vs_bcs_mps1000_20261024",
)
POLICIES = ("greedy", "bcs")
COLORS = ("#0072B2", "#D55E00")
METRICS = (
    ("baseline_request_num", "Bubble requests", "=", "Requests", 1, ",.0f"),
    ("request_rate", "Order request rate", r"$\uparrow$", "Rate (%)", 100, ".2f"),
    ("driver_accept_rate", "Driver acceptance", r"$\uparrow$", "Rate (%)", 100, ".2f"),
    ("order_completion_rate", "Order completion", r"$\uparrow$", "Rate (%)", 100, ".2f"),
    ("mean_wait_to_match_seconds", "Mean waiting time", r"$\downarrow$", "Time (s)", 1, ".2f"),
    ("platform_revenue", "Platform revenue", r"$\uparrow$", r"Revenue ($10^3$ USD)", .001, ".3f"),
)


def _load_results(source):
    """核对配对种子和已发布均值，图中只采用真实仿真指标。"""
    audit_path = source / "comparison_manifest.json"
    reused_outcomes = audit_path.exists()
    if reused_outcomes:
        # 原版搜索选出的动作完全相同，沿用当时的仿真轨迹，不伪装成独立复测。
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        original_source = Path(audit["source"]).parent
        manifest_path = original_source / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (audit["action_differences"] != 0 or audit["predicted_global_revenue_gap"] != 0
                or audit["paired_seeds"] != len(manifest["seeds"])
                or audit["grid_num"] != manifest["config"]["grid_num"]
                or audit["driver_num"] != manifest["config"]["driver_num"]):
            raise ValueError
        original_runs = pd.read_csv(original_source / "run_summary.csv")
        original_runs = original_runs.loc[original_runs.policy == "lstm"]
        runs = pd.concat([original_runs.assign(policy=policy) for policy in POLICIES],
                         ignore_index=True)
        original_summary = pd.read_csv(original_source / "policy_summary.csv").set_index("policy")
        for metric, *_ in METRICS:
            for statistic in ("mean", "std"):
                if not np.isclose(original_summary.loc["lstm", f"{metric}_{statistic}"],
                                  audit["metrics_mean_std"][metric][statistic],
                                  rtol=1e-10, atol=1e-10):
                    raise ValueError
        summary = pd.DataFrame([original_summary.loc["lstm"] for _ in POLICIES], index=POLICIES)
        report = {"epochs_run": None, "best_epoch": manifest["model"]["best_epoch"]}
        checkpoint_sha256 = manifest["model"]["snapshot_sha256"]
        name, title = "original", f"Original LSTM (selected epoch {report['best_epoch']})"
        bcs_label = "BCS (identical actions)"
        sources = [audit_path, manifest_path, original_source / "run_summary.csv",
                   original_source / "policy_summary.csv"]
    else:
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics_path = Path(manifest["checkpoint"]).parent / "metrics.json"
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        runs = pd.read_csv(source / "run_summary.csv")
        runs = runs.loc[runs.policy.isin(POLICIES)]
        summary = pd.read_csv(source / "policy_summary.csv").set_index("policy")
        checkpoint_sha256 = manifest["checkpoint_sha256"]
        name = f"retrained_{report['epochs_run']}epochs"
        title = (f"Retrained LSTM: {report['epochs_run']:,} epochs "
                 f"(selected epoch {report['best_epoch']})")
        bcs_label = f"BCS (up to {manifest['max_bcs_sweeps']} sweeps)"
        sources = [manifest_path, metrics_path, source / "run_summary.csv",
                   source / "policy_summary.csv"]
    runs = runs.sort_values(["policy", "seed"])
    if (manifest["status"] != "complete" or runs.duplicated(["policy", "seed"]).any()
            or len(manifest["seeds"]) < 2
            or len(manifest["seeds"]) != len(set(manifest["seeds"]))):
        raise ValueError
    for policy in POLICIES:
        frame = runs.loc[runs.policy == policy]
        if (set(frame.seed) != set(manifest["seeds"])
                or frame.evaluation_start_minute.nunique() != 1
                or frame.evaluation_minutes.nunique() != 1):
            raise ValueError
        for metric, *_ in METRICS:
            values = frame[metric].to_numpy()
            if (not np.isfinite(values).all() or (values < 0).any()
                    or not np.isclose(values.mean(), summary.loc[policy, f"{metric}_mean"],
                                      rtol=1e-10, atol=1e-10)
                    or not np.isclose(values.std(ddof=1), summary.loc[policy, f"{metric}_std"],
                                      rtol=1e-10, atol=1e-10)):
                raise ValueError
    # 初始司机与预热阶段应相同；决策后允许各策略产生不同轨迹。
    for column in ("initial_driver_sha256", "warmup_metrics_sha256",
                   "evaluation_start_minute", "evaluation_minutes"):
        if (runs.groupby("seed")[column].nunique() != 1).any():
            raise ValueError
    return {
        "source": source, "runs": runs, "manifest": manifest, "report": report,
        "name": name, "title": title, "bcs_label": bcs_label, "sources": sources,
        "checkpoint_sha256": checkpoint_sha256, "reused_outcomes": reused_outcomes,
    }


def _shared_limits(results):
    limits = {}
    for metric, _, _, _, scale, _ in METRICS:
        upper = 0
        for result in results:
            for policy in POLICIES:
                values = result["runs"].loc[result["runs"].policy == policy, metric] * scale
                upper = max(upper, values.max(), values.mean() + values.std(ddof=1))
        # 柱状图从零开始，避免把很小的差异放大；两版本严格共用标尺。
        limits[metric] = max(upper * 1.24, 1)
        if metric.endswith("rate") and upper <= 100:
            limits[metric] = 110
    return limits


def _draw(result, limits):
    runs, report, config = result["runs"], result["report"], result["manifest"]["config"]
    seeds = sorted(result["manifest"]["seeds"])
    jitter = np.linspace(-.12, .12, len(seeds))
    figure, axes = plt.subplots(2, 3, figsize=(7.16, 5.05))
    figure.subplots_adjust(left=.085, right=.985, bottom=.12, top=.785,
                           wspace=.42, hspace=.46)
    records = []
    for index, (axis, spec) in enumerate(zip(axes.flat, METRICS)):
        metric, title, direction, unit, scale, number_format = spec
        for position, (policy, color) in enumerate(zip(POLICIES, COLORS)):
            raw = runs.loc[runs.policy == policy].set_index("seed").loc[seeds, metric]
            values = raw.to_numpy() * scale
            mean, std = float(values.mean()), float(values.std(ddof=1))
            axis.bar(position, mean, width=.52, color=color, alpha=.9,
                     edgecolor=color, linewidth=.55, zorder=2)
            axis.scatter(position + jitter, values, s=9, facecolors="white",
                         edgecolors="#555555", linewidths=.4, alpha=.82, zorder=4)
            axis.errorbar(position, mean, yerr=std, fmt="none", ecolor="#202020",
                          elinewidth=.75, capsize=2.5, capthick=.75, zorder=5)
            label = format(mean, number_format)
            if metric.endswith("rate"):
                label += "%"
            axis.text(position, max(values.max(), mean + std) + limits[metric] * .035,
                      label, ha="center", va="bottom", fontsize=8.3)
            records.append({
                "model_version": result["name"],
                "training_epochs": report["epochs_run"], "selected_epoch": report["best_epoch"],
                "policy": policy, "metric": metric, "n_seeds": len(seeds),
                "mean": float(raw.mean()), "std": float(raw.std(ddof=1)),
                "display_mean": mean, "display_std": std, "display_scale": scale,
                "y_min": 0, "y_max": limits[metric],
            })
        axis.set_title(f"({chr(97 + index)}) {title} {direction}", loc="left", pad=8)
        axis.set_ylabel(unit, labelpad=4)
        axis.set_xticks([0, 1], ["Greedy", "BCS"])
        axis.set_xlim(-.58, 1.58)
        axis.set_ylim(0, limits[metric])
        axis.grid(axis="y", color="#E2E5E8", linewidth=.45, zorder=0)
        axis.tick_params(axis="x", length=0, pad=5)
        axis.tick_params(axis="y", direction="out", length=2.5, width=.55, pad=3)
        if metric.endswith("rate"):
            axis.set_yticks([0, 25, 50, 75, 100])
        elif metric == "baseline_request_num":
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        else:
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.set_axisbelow(True)
    figure.suptitle(result["title"], fontsize=11, y=.987)
    figure.text(.5, .939,
                f"Low-density scenario  |  {config['grid_num']} grids  |  "
                f"{config['driver_num']} drivers  |  {len(seeds)} paired test seeds",
                ha="center", fontsize=8.6)
    figure.legend(handles=[
        Patch(facecolor=COLORS[0], edgecolor="none", label="One-pass greedy"),
        Patch(facecolor=COLORS[1], edgecolor="none", label=result["bcs_label"]),
        Line2D([], [], marker="o", linestyle="none", markersize=3.2,
               markerfacecolor="white", markeredgecolor="#555555", markeredgewidth=.5,
               label="Individual seed"),
    ], loc="upper center", bbox_to_anchor=(.52, .909), ncol=3, frameon=False,
       handlelength=1.2, columnspacing=1.5, handletextpad=.6)
    figure.text(.5, .043,
                r"Bars: mean $\pm$ 1 SD   |   $\uparrow$ Higher is better   "
                r"$\downarrow$ Lower is better   = Exogenous demand",
                ha="center", fontsize=7.7)
    return figure, records


def plot_paper_bcs(inputs=DEFAULT_INPUTS, output="experiments/lstm_bcs_original_vs_retrained_paper"):
    results = [_load_results(Path(source)) for source in inputs]
    if len({result["name"] for result in results}) != len(results):
        raise ValueError
    same_test_seeds = all(set(result["manifest"]["seeds"]) == set(results[0]["manifest"]["seeds"])
                          for result in results)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    limits = _shared_limits(results)
    family = _font_family()
    style = {
        "font.family": family, "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.3, "xtick.labelsize": 8.5, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "xtick.color": "#333333", "ytick.color": "#333333",
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "path", "axes.formatter.useoffset": False,
    }
    files, records, provenance, captions = [], [], [], []
    with plt.rc_context(style), PdfPages(output / "both_versions.pdf") as combined:
        for result in results:
            figure, values = _draw(result, limits)
            report, manifest = result["report"], result["manifest"]
            name = f"lstm_{result['name']}_greedy_vs_bcs"
            for extension in ("png", "pdf", "svg"):
                figure.savefig(output / f"{name}.{extension}", dpi=600, facecolor="white")
                files.append(f"{name}.{extension}")
            combined.savefig(figure, facecolor="white")
            plt.close(figure)
            records.extend(values)
            provenance.append({
                "source_directory": str(result["source"].resolve()),
                "model_version": result["name"],
                "training_epochs": report["epochs_run"], "selected_epoch": report["best_epoch"],
                "seeds": sorted(manifest["seeds"]),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "bcs_reuses_greedy_outcomes": result["reused_outcomes"],
                "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in result["sources"]},
            })
            start = result["runs"].evaluation_start_minute.iloc[0]
            duration = result["runs"].evaluation_minutes.iloc[0]
            training_note = ("The original model was not retrained for the BCS test. "
                             "The original search audit recorded zero action differences and zero "
                             "predicted-objective gap between greedy and BCS. Therefore both bars "
                             "reuse the same greedy simulation episodes, not independent reruns. "
                             if result["reused_outcomes"] else
                             f"The model was trained for {report['epochs_run']} epochs. ")
            seed_note = ("The two versions use the same test-seed set. " if same_test_seeds else
                         "The two versions use different test-seed sets; their absolute means "
                         "are not a paired comparison of checkpoints. ")
            captions.append(
                f"{name}: actual closed-loop simulator outcomes for one-pass greedy search and "
                f"block coordinate search using the same validation-selected epoch-{report['best_epoch']} "
                "LSTM checkpoint. " + training_note +
                f"The experiment used {len(manifest['seeds'])} paired seeds, "
                f"{manifest['config']['grid_num']} grids and {manifest['config']['driver_num']} drivers. "
                f"Each episode excludes a shared {start:g}-minute warm-up and scores the following "
                f"{duration:g} minutes (05:20–10:00 in these experiments). "
                "Bars show the arithmetic mean across seeds; error bars show one sample standard "
                "deviation (ddof=1), not confidence intervals. Open circles show individual seed "
                "observations. Panels report (a) exogenous bubble requests, (b) order request rate, "
                "(c) driver acceptance, (d) order completion, (e) mean matched-passenger waiting "
                "time to matching, and (f) platform revenue. These are simulator outcomes, not "
                "supervised prediction-error metrics. The optimization objective is platform "
                "revenue; arrows indicate each metric's preferred direction rather than a joint "
                "objective. Both versions share each panel's scale, with all bar axes starting at zero. "
                + seed_note + "\n")
    pd.DataFrame(records).to_csv(output / "plotted_values.csv", index=False)
    (output / "captions.txt").write_text("\n".join(captions), encoding="utf-8")
    figure_manifest = {
        "font_family": family, "figure_inches": [7.16, 5.05], "png_dpi": 600,
        "shared_axis_limits": limits, "error_bars": "sample standard deviation, ddof=1",
        "same_test_seed_set": same_test_seeds,
        "source_results": provenance, "files": files + ["both_versions.pdf"],
        "plot_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"论文版测试图已保存至：{output}")
    return figure_manifest


def main():
    parser = argparse.ArgumentParser(description="导出贪心与 BCS 的论文版测试图")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", default="experiments/lstm_bcs_original_vs_retrained_paper")
    plot_paper_bcs(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
