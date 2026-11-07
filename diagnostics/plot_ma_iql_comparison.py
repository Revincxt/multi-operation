"""绘制四策略的训练诊断与配对运营结果，图内只保留必要标记。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd
import torch

from baseline.compare_policies import METRICS
from offline_rl.data import save_json, sha256


ORDER = ("fixed_f3", "lstm_greedy", "ma_iql", "ma_iql_plus")
LABELS = dict(zip(ORDER, ("F3", "LSTM-G", "MA-IQL", "MA-IQL+")))
COLORS = dict(zip(ORDER, ("#9098A3", "#657FA8", "#CF986B", "#268F85")))
LINE_STYLES = dict(zip(ORDER, ("--", "-.", ":", "-")))
STYLE = {"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.direction": "out", "ytick.direction": "out",
    "legend.fontsize": 8, "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "text.color": "#26323F", "axes.labelcolor": "#26323F", "axes.edgecolor": "#55606D",
    "xtick.color": "#45515F", "ytick.color": "#45515F"}
METRIC_TITLES = ("Bubble requests =", "Order request rate ↑", "Driver acceptance rate ↑",
                 "Matching success rate ↑", "Matched-passenger wait (s) ↓", "Platform revenue (USD) ↑")
PLUS_METRICS = ("td_loss", "actor_loss", "q_auxiliary_loss", "actor_kl", "policy_q_gap", "joint_conservative_weight")
PLUS_TITLES = ("Q TD loss", "Policy loss", "Dynamics loss", "Policy KL", "Joint Q gap (USD)", "Penalty weight")
TRAINING = {
    "ma_iql": Path("models/iql_grid6_low_20261102/training_history.csv"),
    "ma_iql_plus": Path("models/conservative_policy_joint100k_20261103/train_seed74/training_history.csv"),
}


def export(fig, root, name, rect=None, w_pad=1.5, h_pad=1.6):
    directory = Path(root)/"figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=.8, w_pad=w_pad, h_pad=h_pad, rect=rect)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(directory/f"{name}.{ext}", dpi=600, facecolor="white")
    plt.close(fig)


def axis_style(ax):
    ax.grid(axis="y", color="#E2E6EB", linewidth=.55, alpha=.8)
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.margins(x=.025)


def draw_training(ax, frame, column, title, policy, selected_step=None):
    if policy == "ma_iql_plus":
        # 策略未更新阶段的占位零不作为实际损失或KL观测。
        frame = frame[frame.phase == "policy_improvement"] if column in ("actor_loss", "actor_kl") else frame[frame.phase != "behavior_initialization"]
    scale = 100 if column == "policy_q_gap" else 1
    ax.plot(frame.step, frame[column]*scale, color=COLORS[policy], lw=1.4, label=LABELS[policy])
    if selected_step is not None:
        ax.axvline(selected_step, color="#7D858F", lw=.75, ls="--", label="Selected checkpoint")
    if column == "policy_q_gap":
        ax.axhline(5., color="#7D858F", lw=.8, ls=":", label="Target")
        ax.legend(handles=[Line2D([], [], color="#7D858F", lw=.8, ls=":", label="Target")], loc="upper right")
    ax.set_title(title, loc="left", pad=7)
    ax.set_xlabel("Gradient updates")
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    axis_style(ax)


def training_figures(root):
    root = Path(root)
    data = root/"figure_data"
    data.mkdir(exist_ok=True)
    protocol = json.loads((root/"protocol.json").read_text())
    checkpoint = torch.load(protocol["models"]["ma_iql"], map_location="cpu", weights_only=True)
    selected_step = checkpoint["step"]
    with plt.rc_context(STYLE):
        original = pd.read_csv(TRAINING["ma_iql"])
        original[["step", "q_loss", "value_loss", "iql_actor_loss"]].to_csv(data/"ma_iql_training.csv", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.05))
        for index, (ax, column, title) in enumerate(zip(axes, ("q_loss", "value_loss", "iql_actor_loss"),
                                                     ("Q loss", "V loss", "Policy loss"))):
            draw_training(ax, original, column, f"({chr(97+index)}) {title}", "ma_iql", selected_step)
            ax.set_ylabel("Loss")
        axes[0].legend(loc="best")
        export(fig, root, "fig01_ma_iql_training")

        improved = pd.read_csv(TRAINING["ma_iql_plus"])
        improved[["step", "phase", *PLUS_METRICS]].to_csv(data/"ma_iql_plus_training.csv", index=False)
        fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6))
        for index, (ax, column, title) in enumerate(zip(axes.flat, PLUS_METRICS, PLUS_TITLES)):
            draw_training(ax, improved, column, f"({chr(97+index)}) {title}", "ma_iql_plus")
        axes.flat[0].legend(loc="best")
        export(fig, root, "fig02_ma_iql_plus_training")

        # 同一张图按方法分组，各目标保持原始数值与更新次数。
        fig, axes = plt.subplots(3, 3, figsize=(10.8, 7.9))
        for index, (ax, column, title) in enumerate(zip(axes[0], ("q_loss", "value_loss", "iql_actor_loss"),
                                                     ("Q loss", "V loss", "Policy loss"))):
            draw_training(ax, original, column, f"({chr(97+index)}) {title}", "ma_iql", selected_step)
        for index, (ax, column, title) in enumerate(zip(axes[1:].flat, PLUS_METRICS, PLUS_TITLES), start=3):
            draw_training(ax, improved, column, f"({chr(97+index)}) {title}", "ma_iql_plus")
        handles = [Line2D([], [], color=COLORS[name], lw=1.7, label=LABELS[name]) for name in ORDER[2:]]
        handles.append(Line2D([], [], color="#7D858F", lw=.8, ls="--", label="Selected checkpoint"))
        fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(.5, .997), columnspacing=2.5)
        export(fig, root, "fig06_training_combined", rect=(0, 0, 1, .952), h_pad=2.)


def bootstrap_weights(count, draws=10000):
    # 所有方法、指标和时间点共用同一组完整seed的bootstrap抽样。
    indices = np.random.default_rng(42).integers(count, size=(draws, count))
    weights = np.stack([np.bincount(row, minlength=count) for row in indices]).astype(float)/count
    return weights


def interval(values, weights):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all()
    samples = weights@values
    return values.mean(axis=0), np.quantile(samples, [.025, .975], axis=0)


def draw_metric(ax, statistics, metric, title, labels=True):
    scale = 100 if metric in ("request_rate", "driver_accept_rate", "order_completion_rate") else 1
    rows = statistics[statistics.metric == metric].set_index("policy").loc[list(ORDER)]
    centers = rows["mean"].to_numpy()*scale
    lowers, uppers = rows.ci95_low.to_numpy()*scale, rows.ci95_high.to_numpy()*scale
    ax.bar(np.arange(4), centers, width=.60, color=[COLORS[name] for name in ORDER],
        yerr=np.maximum(0, np.stack((centers-lowers, uppers-centers))),
        error_kw={"ecolor": "#36424E", "lw": .8, "capsize": 2.8, "capthick": .8})
    ax.set_title(title, loc="left", pad=7)
    ax.set_xticks(np.arange(4), [LABELS[name] for name in ORDER] if labels else [""]*4)
    ax.tick_params(axis="x", length=0 if not labels else 3)
    ax.set_ylim(0, min(100, float(uppers.max())*1.12) if scale == 100 else float(uppers.max())*1.12)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    axis_style(ax)
    if scale == 100:
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))


def draw_revenue_gain(ax, statistics, compact=False):
    ax.axvline(0, color=COLORS["fixed_f3"], lw=.85, ls="--", zorder=1)
    for row, name in enumerate(ORDER[1:]):
        item = statistics.set_index("policy").loc[name]
        center = item.mean_difference_usd
        ax.errorbar(center, row, xerr=np.maximum(0, np.array([[center-item.ci95_low], [item.ci95_high-center]])),
            fmt="o", ms=5., color=COLORS[name], lw=1.3, capsize=3, capthick=.9)
    ax.set_yticks(np.arange(3), [LABELS[name] for name in ORDER[1:]])
    ax.set_ylim(-.55, 2.55); ax.invert_yaxis()
    ax.set_xlabel("Difference (USD) →" if compact else "Revenue difference vs F3 (USD) →")
    ax.xaxis.set_major_locator(MaxNLocator(4 if compact else 6))
    ax.grid(axis="x", color="#E2E6EB", lw=.55, alpha=.8)


def operational_figures(root, frame, weights):
    seeds = sorted(frame.seed.unique())
    summaries = []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8))
        for index, (ax, metric, title) in enumerate(zip(axes.flat, METRICS, METRIC_TITLES)):
            for name in ORDER:
                values = frame.loc[frame.policy == name].set_index("seed").loc[seeds, metric].to_numpy()
                center, bounds = interval(values, weights)
                summaries.append({"policy": name, "label": LABELS[name], "metric": metric,
                    "mean": float(center), "ci95_low": float(bounds[0]), "ci95_high": float(bounds[1])})
            draw_metric(ax, pd.DataFrame(summaries), metric, f"({chr(97+index)}) {title}")
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        export(fig, root, "fig03_six_metrics")
        pd.DataFrame(summaries).to_csv(root/"metric_statistics.csv", index=False)

        revenue = frame.pivot(index="seed", columns="policy", values="platform_revenue").loc[seeds]
        records, differences = [], []
        fig, ax = plt.subplots(figsize=(6.7, 3.3))
        for name in ORDER[1:]:
            gap = (revenue[name]-revenue.fixed_f3).to_numpy()
            center, bounds = interval(gap, weights)
            records.append({"policy": name, "label": LABELS[name], "mean_difference_usd": float(center),
                "ci95_low": float(bounds[0]), "ci95_high": float(bounds[1]),
                "relative_percent": float(center/revenue.fixed_f3.mean()*100), "wins": int((gap > 1e-8).sum()),
                "ties": int((np.abs(gap) <= 1e-8).sum()), "seeds": len(seeds)})
            differences.extend({"seed": seed, "policy": name, "difference_usd": float(value)} for seed, value in zip(seeds, gap))
        draw_revenue_gain(ax, pd.DataFrame(records))
        export(fig, root, "fig04_revenue_gain")
        pd.DataFrame(records).to_csv(root/"paired_revenue_statistics.csv", index=False)
        pd.DataFrame(differences).to_csv(root/"paired_revenue_differences.csv", index=False)


def draw_timeline(ax, statistics, difference, title, compact=False):
    value = "mean_cumulative_difference" if difference else "mean_cumulative_revenue"
    low = "difference_ci95_low" if difference else "revenue_ci95_low"
    high = "difference_ci95_high" if difference else "revenue_ci95_high"
    for name in ORDER[1:] if difference else ORDER:
        part = statistics[statistics.policy == name].sort_values("time_hours")
        ax.plot(part.time_hours, part[value], color=COLORS[name], ls=LINE_STYLES[name], lw=1.55, label=LABELS[name])
        ax.fill_between(part.time_hours, part[low], part[high], color=COLORS[name], alpha=.11, lw=0)
    if difference:
        ax.axhline(0, color=COLORS["fixed_f3"], lw=.85, ls="--")
    start, end = statistics.time_hours.min(), statistics.time_hours.max()
    ax.set_title(title, loc="left", pad=7)
    ax.set_ylabel("Difference (USD)" if difference else "Revenue (USD)")
    ax.set_xlabel("Simulation time")
    ax.set_xlim(start, end)
    ax.set_xticks([start, 7., 8.5, end] if compact else [start, 6., 7., 8., 9., end])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{int(round(value*60))//60:02d}:{int(round(value*60))%60:02d}"))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    axis_style(ax)


def timeline_figure(root, frame, weights, protocol):
    seeds = sorted(frame.seed.unique())
    config = protocol["config"]
    decisions = np.arange(config["history_steps"], config["simulation_steps"]//config["decision_freq"])
    arrays, records, sources = {}, [], []
    for name in ORDER:
        episodes = []
        for seed in seeds:
            if name in protocol["reused_policies"]:
                directory = Path(protocol["source"])/"runs"/f"seed{seed}"/protocol["reused_policies"][name]
            else:
                directory = root/"supplemental/runs"/f"seed{seed}"/name
            path = directory/"decision_metrics.csv.gz"
            data = pd.read_csv(path)
            data = data[data.decision_step.isin(decisions)]
            assert not data.duplicated(["grid_id", "decision_step"]).any()
            assert data.groupby("decision_step").size().eq(config["grid_num"]).all()
            grouped = data.groupby("decision_step").platform_revenue.sum().reindex(decisions)
            assert grouped.notna().all()
            expected = frame.loc[(frame.seed == seed) & (frame.policy == name), "platform_revenue"].item()
            np.testing.assert_allclose(grouped.sum(), expected, atol=1e-7, rtol=1e-10)
            episodes.append(grouped.to_numpy())
            records.extend({"seed": seed, "policy": name, "decision_step": int(step), "revenue_usd": float(value)}
                           for step, value in zip(decisions, grouped))
            sources.append({"seed": seed, "policy": name, "path": str(path), "sha256": sha256(path)})
        arrays[name] = np.c_[np.zeros(len(seeds)), np.cumsum(episodes, axis=1)]
    pd.DataFrame(records).to_csv(root/"period_revenue.csv", index=False)
    pd.DataFrame(sources).to_csv(root/"trajectory_sources.csv", index=False)
    start = sum(int(value)*unit for value, unit in zip(config["start_time"].split(":"), (3600, 60)))
    time = (start+np.r_[decisions[0], decisions+1]*config["decision_freq"]*60)/3600
    summaries = []
    with plt.rc_context(STYLE):
        for name in ORDER:
            center, bounds = interval(arrays[name], weights)
            if name != "fixed_f3":
                gap, gap_bounds = interval(arrays[name]-arrays["fixed_f3"], weights)
            else:
                gap, gap_bounds = np.zeros(len(time)), np.zeros((2, len(time)))
            summaries.extend({"policy": name, "time_hours": float(t), "mean_cumulative_revenue": float(mean),
                "revenue_ci95_low": float(low), "revenue_ci95_high": float(high), "mean_cumulative_difference": float(delta),
                "difference_ci95_low": float(dlow), "difference_ci95_high": float(dhigh)}
                for t, mean, low, high, delta, dlow, dhigh in zip(time, center, bounds[0], bounds[1], gap, gap_bounds[0], gap_bounds[1]))
        statistics = pd.DataFrame(summaries)
        statistics.to_csv(root/"time_series_statistics.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
        draw_timeline(axes[0], statistics, False, "(a) Cumulative revenue ↑")
        draw_timeline(axes[1], statistics, True, "(b) Cumulative gain vs F3 ↑")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.))
        export(fig, root, "fig05_revenue_over_time", rect=(0, 0, 1, .90))


def results_combined(root):
    metrics = pd.read_csv(root/"metric_statistics.csv")
    revenue = pd.read_csv(root/"paired_revenue_statistics.csv")
    timeline = pd.read_csv(root/"time_series_statistics.csv")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(10.8, 8.5))
        titles = list(METRIC_TITLES)
        titles[4] = "Passenger wait (s) ↓"
        for index, (ax, metric, title) in enumerate(zip(axes[:2].flat, METRICS, titles)):
            draw_metric(ax, metrics, metric, f"({chr(97+index)}) {title}", labels=False)
        draw_revenue_gain(axes[2, 0], revenue, compact=True)
        axes[2, 0].set_title("(g) Revenue gain vs F3", loc="left", pad=7)
        draw_timeline(axes[2, 1], timeline, False, "(h) Cumulative revenue ↑", compact=True)
        draw_timeline(axes[2, 2], timeline, True, "(i) Cumulative gain vs F3 ↑", compact=True)
        handles = [Line2D([], [], color=COLORS[name], ls=LINE_STYLES[name], lw=1.8, label=LABELS[name]) for name in ORDER]
        fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(.5, .996), columnspacing=2.8)
        export(fig, root, "fig07_results_combined", rect=(0, 0, 1, .95), w_pad=1.9, h_pad=2.5)


def generate(root):
    root = Path(root)
    protocol = json.loads((root/"protocol.json").read_text())
    frame = pd.read_csv(root/"run_summary.csv")
    assert not frame.duplicated(["seed", "policy"]).any()
    assert len(frame) == len(protocol["seeds"])*len(ORDER)
    assert frame.groupby("seed").policy.nunique().eq(4).all()
    assert np.isfinite(frame[list(METRICS)].to_numpy()).all()
    weights = bootstrap_weights(len(protocol["seeds"]))
    training_figures(root)
    operational_figures(root, frame, weights)
    timeline_figure(root, frame, weights, protocol)
    results_combined(root)
    paths = list((root/"figures").glob("*"))
    save_json(root/"figure_manifest.json", {"labels": LABELS, "colors": COLORS, "line_styles": LINE_STYLES,
        "combined_figures": ["fig06_training_combined", "fig07_results_combined"],
        "statistic": "mean and paired-seed percentile bootstrap 95% intervals; resample whole episodes, 10000 draws, RNG42",
        "time_bands": "pointwise 95% intervals using identical episode bootstrap weights across all time points",
        "training_logs_sha256": {name: sha256(path) for name, path in TRAINING.items()},
        "plotting_source_sha256": sha256(__file__), "files_sha256": {str(path.relative_to(root)): sha256(path) for path in paths}})
    captions = [
        "图1：MA-IQL训练曲线。虚线为本次策略比较所用checkpoint；展示实际记录的Q、V和策略损失。",
        "图2：MA-IQL+所用模型的训练诊断。策略未更新阶段不绘制占位零；Q差以美元表示，虚线为训练约束目标。",
        "图3：四策略的六项运营指标。均值及95% seed bootstrap区间；冒泡数量为固定外生输入。",
        "图4：相对F3的配对平均平台收益差及95%区间，零线表示与F3持平。",
        "图5：累计平台收益及相对F3的累计收益差。阴影为逐时间点95%区间，完整轨迹按seed共同重采样。",
        "合图1（fig06）：MA-IQL与MA-IQL+训练曲线。第一行为MA-IQL，其余两行为MA-IQL+；各损失保留各自定义、原始数值和实际更新次数。",
        "合图2（fig07）：四策略运营结果总览。(a)—(f)为六项指标，柱从左到右依次为F3、LSTM-G、MA-IQL、MA-IQL+；(g)为配对平均收益差，(h)—(i)为累计收益与累计收益差。误差线、阴影为95%区间。",
        "条件：6网格、250司机，2015-05-18，统一预热20分钟，05:20—10:00评分；100个配对仿真种子。",
        "收益为平台佣金；匹配成功率分母含跨周期重复候选；等待只统计已匹配乘客排队。",
        "训练损失各自对应不同目标，不以不同方法损失的数值大小判断运营效果。",
        "MA-IQL+为项目中对升级方法的简称。所有权重固定，图中结果不用于重新选择模型。",
    ]
    (root/"captions.txt").write_text("\n".join(captions), encoding="utf-8")
    print(pd.read_csv(root/"paired_revenue_statistics.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="experiments/ma_iql_comparison_20261103")
    args = parser.parse_args()
    generate(args.root)
