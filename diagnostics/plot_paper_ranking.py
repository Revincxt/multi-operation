"""从既有配对诊断绘制 F3 与 BCS 的逐状态、分时刻及总体论文图。

使用方式:
    python3 -m diagnostics.plot_paper_ranking

只读取已保存的结果，不重新仿真；从共同贪心基准恢复 F3−BCS 的配对差。
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline.plot_paper_figures import _font_family


DEFAULT_INPUT = "experiments/lstm_joint_greedy_vs_bcs_mps1000_20261024"
PREDICTED, ACTUAL = "#0072B2", "#D55E00"
KEYS = ["seed", "decision_step"]
TRIAL_KEYS = KEYS + ["repeat"]
BOOTSTRAP_REPEATS = 10000


def load_diagnostic(source):
    """核对共同随机流，以逐次配对差计算实际均值和标准差。"""
    names = (
        "same_state_diagnostic.csv", "same_state_random_draws.csv",
        "fixed_reference/same_state_comparison.csv", "fixed_reference/random_draws.csv",
        "model_ranking_diagnostic.json", "manifest.json",
    )
    paths = [source / name for name in names]
    bcs, bcs_trials, fixed, fixed_trials = [pd.read_csv(path) for path in paths[:4]]
    reference = json.loads(paths[4].read_text(encoding="utf-8"))
    manifest = json.loads(paths[5].read_text(encoding="utf-8"))
    for frame, keys in ((bcs, KEYS), (fixed, KEYS),
                        (bcs_trials, TRIAL_KEYS), (fixed_trials, TRIAL_KEYS)):
        if frame.empty or frame.duplicated(keys).any():
            raise ValueError
    states = fixed.merge(bcs, on=KEYS, validate="one_to_one", how="outer", indicator=True)
    if (not states._merge.eq("both").all() or not states.repeats.eq(10).all()
            or set(states.seed) != set(manifest["seeds"])):
        raise ValueError
    states = states.drop(columns="_merge")
    states["predicted_gap"] = states.predicted_fixed_minus_greedy - states.predicted_gain
    trials = fixed_trials.merge(bcs_trials, on=TRIAL_KEYS, how="left",
                                suffixes=("_fixed", "_bcs"), validate="one_to_one")
    if not pd.MultiIndex.from_frame(bcs_trials[TRIAL_KEYS]).isin(
            pd.MultiIndex.from_frame(fixed_trials[TRIAL_KEYS])).all():
        raise ValueError
    trials = trials.merge(states[KEYS + ["changed_actions"]], on=KEYS, validate="many_to_one")
    changed = trials.changed_actions
    if (trials.loc[changed, "bcs_minus_greedy"].isna().any()
            or trials.loc[~changed, "bcs_minus_greedy"].notna().any()
            or not np.allclose(trials.loc[changed, "greedy_revenue_fixed"],
                               trials.loc[changed, "greedy_revenue_bcs"], rtol=0, atol=1e-9)):
        raise ValueError
    # 动作相同的 24 个状态无需额外仿真：任意共同随机流下 BCS−贪心严格为零。
    trials["bcs_equals_greedy"] = ~changed
    trials["greedy_revenue"] = trials.greedy_revenue_fixed
    trials["bcs_revenue"] = trials.bcs_revenue.where(changed, trials.greedy_revenue)
    trials["actual_gap"] = trials.fixed_f3_revenue - trials.bcs_revenue
    if not np.allclose(trials.actual_gap,
                       trials.fixed_minus_greedy - trials.bcs_minus_greedy.fillna(0),
                       rtol=0, atol=1e-9):
        raise ValueError
    actual = trials.groupby(KEYS).actual_gap.agg(
        actual_gap="mean", actual_gap_std="std", repeats_observed="count").reset_index()
    states = states.merge(actual, on=KEYS, validate="one_to_one")
    if (len(states) != reference["sampled_states"]
            or len(trials) != reference["paired_future_streams"]
            or not states.repeats_observed.eq(states.repeats).all()
            or not np.allclose(states.actual_gap,
                               states.actual_fixed_minus_greedy_mean - states.actual_gain_mean,
                               rtol=0, atol=1e-9)
            or not np.isfinite(states[["predicted_gap", "actual_gap", "actual_gap_std"]]).all().all()):
        raise ValueError
    decisions = sorted(states.decision_step.unique())
    if not all(sorted(group.decision_step) == decisions for _, group in states.groupby("seed")):
        raise ValueError
    for column, expected in (
            ("predicted_gap", reference["mean_model_predicted_f3_minus_bcs"]),
            ("actual_gap", reference["mean_actual_f3_minus_bcs"])):
        if not np.isclose(states[column].mean(), expected, rtol=0, atol=1e-9):
            raise ValueError
    if (int(states.predicted_gap.gt(0).sum()) != reference["model_prefers_f3_over_bcs_states"]
            or int(states.actual_gap.gt(0).sum()) != reference["actual_state_f3_wins"]):
        raise ValueError
    return states.sort_values(KEYS), trials.sort_values(TRIAL_KEYS), manifest, reference, paths


def summarize(states, manifest, reference):
    """按种子整组重采样，保留同一轨迹内六个状态之间的关联。"""
    decisions = sorted(states.decision_step.unique())
    seeds = sorted(states.seed.unique())
    indices = np.random.default_rng(42).integers(len(seeds), size=(BOOTSTRAP_REPEATS, len(seeds)))
    time_rows, overall_rows = [], []
    for column, label in (("predicted_gap", "predicted"), ("actual_gap", "simulated")):
        matrix = states.pivot(index="seed", columns="decision_step", values=column).loc[seeds, decisions]
        bootstrap = matrix.to_numpy()[indices].mean(axis=1)
        lower, upper = np.quantile(bootstrap, [.025, .975], axis=0)
        for index, step in enumerate(decisions):
            values = matrix[step].to_numpy()
            time_rows.append({"series": label, "decision_step": int(step),
                              "mean": float(values.mean()), "seed_std": float(values.std(ddof=1)),
                              "ci95_low": float(lower[index]), "ci95_high": float(upper[index]),
                              "f3_win_states": int((values > 0).sum()), "n_seeds": len(seeds)})
        mean_by_seed = matrix.mean(axis=1).to_numpy()
        # 与原诊断完全相同的种子级 bootstrap，用于复现已报告区间。
        overall_bootstrap = mean_by_seed[indices].mean(axis=1)
        lo, hi = np.quantile(overall_bootstrap, [.025, .975])
        values = states[column]
        overall_rows.append({"series": label, "mean": float(values.mean()),
                             "ci95_low": float(lo), "ci95_high": float(hi),
                             "f3_win_states": int(values.gt(0).sum()),
                             "bcs_win_states": int(values.lt(0).sum()),
                             "tie_states": int(values.eq(0).sum()), "n_states": len(states),
                             "n_seeds": len(seeds)})
    overall = pd.DataFrame(overall_rows)
    simulated = overall.loc[overall.series == "simulated"].iloc[0]
    if not np.allclose([simulated.ci95_low, simulated.ci95_high],
                       [reference["seed_cluster_bootstrap_95_lower"],
                        reference["seed_cluster_bootstrap_95_upper"]], rtol=0, atol=1e-9):
        raise ValueError
    start_hour, start_minute = map(int, manifest["config"]["start_time"].split(":"))
    minutes = start_hour * 60 + start_minute + np.array(decisions) * manifest["config"]["decision_freq"]
    times = [f"{minute // 60:02d}:{minute % 60:02d}" for minute in minutes]
    return pd.DataFrame(time_rows), overall, decisions, times


def _axes(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E1E5E8", linewidth=.45)
    axis.axhline(0, color="#656565", linewidth=.75, linestyle=(0, (4, 3)), zorder=1)
    axis.tick_params(direction="out", length=2.5, width=.55, pad=3)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _header(figure, title, description):
    figure.suptitle(title, fontsize=11, y=.987)
    # 用物理间距布置标题，避免短幅图的两行文字挤在一起。
    figure.text(.5, .987 - .31 / figure.get_figheight(), description, ha="center", fontsize=8.4)


def _legend(figure, simulated_label, anchor):
    figure.legend(handles=[
        Line2D([], [], color=PREDICTED, marker="s", markersize=3,
               linestyle="--", linewidth=.9, label="Model prediction"),
        Line2D([], [], color=ACTUAL, marker="o", markersize=3,
               linewidth=.9, label=simulated_label),
    ], loc="upper center", bbox_to_anchor=anchor, ncol=2, frameon=False,
       handlelength=2.3, columnspacing=2)


def _sign_note(figure, y=.018):
    figure.text(.5, y, r"Revenue gap: F3 $-$ BCS (USD / 5 min)   |   "
                r"$\uparrow$ Positive: F3 better   $\downarrow$ Negative: BCS better",
                ha="center", fontsize=7.7)


def draw_states(states, decisions, times):
    """每个种子单独显示六个抽样时刻；点为十组未来随机流的均值。"""
    seeds = sorted(states.seed.unique())
    rows = (len(seeds) + 1) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(7.16, rows * 1.32 + 1.15),
                                sharex=True, sharey=True, squeeze=False)
    figure.subplots_adjust(left=.09, right=.98, bottom=.09, top=.855, wspace=.23, hspace=.53)
    low = min(states.predicted_gap.min(), states.actual_gap.min())
    high = max(states.predicted_gap.max(), states.actual_gap.max())
    padding = (high - low) * .15
    x = np.arange(len(decisions))
    for index, (axis, seed) in enumerate(zip(axes.flat, seeds)):
        frame = states.loc[states.seed == seed].set_index("decision_step").loc[decisions]
        _axes(axis)
        axis.plot(x, frame.predicted_gap, color=PREDICTED, marker="s", markersize=3,
                  linestyle="--", linewidth=.9, zorder=3)
        axis.plot(x, frame.actual_gap, color=ACTUAL, marker="o", markersize=3,
                  linewidth=.95, zorder=4)
        axis.set_title(f"({chr(97 + index)}) Seed {seed}", loc="left", fontsize=8.7, pad=5)
        axis.set_ylim(low - padding, high + padding)
        axis.set_xlim(-.18, len(x) - .82)
        axis.set_xticks(x, times)
        if index >= (rows - 1) * 2:
            axis.set_xlabel("Decision time", labelpad=4)
    for axis in axes.flat[len(seeds):]:
        axis.set_visible(False)
    figure.text(.018, .47, "Revenue gap (USD / 5 min)", rotation=90, va="center", fontsize=9)
    _header(figure, "State-wise action-ranking diagnostic: F3 versus BCS",
            f"{len(seeds)} test seeds  |  {len(decisions)} sampled decision times per seed  |  F3: p = 1.6, r = 2 km")
    _legend(figure, "Simulation (10-stream mean)", (.52, .912))
    _sign_note(figure, .021)
    return figure


def draw_times(time_summary, decisions, times):
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.65))
    figure.subplots_adjust(left=.09, right=.98, bottom=.21, top=.71, wspace=.35)
    x = np.arange(len(decisions))
    axis = axes[0]
    _axes(axis)
    for label, color, marker, offset in (("predicted", PREDICTED, "s", -.045),
                                         ("simulated", ACTUAL, "o", .045)):
        data = time_summary.loc[time_summary.series == label].set_index("decision_step").loc[decisions]
        errors = np.vstack([data["mean"] - data.ci95_low, data.ci95_high - data["mean"]])
        axis.errorbar(x + offset, data["mean"], yerr=errors, color=color, marker=marker,
                       markersize=3.4, linewidth=.9, linestyle="--" if label == "predicted" else "-",
                       elinewidth=.65, capsize=2, capthick=.65)
    axis.set_title("(a) Mean gap and 95% CI", loc="left", pad=7)
    axis.set_ylabel("Revenue gap (USD / 5 min)")
    axis.set_xticks(x, times, rotation=35, ha="right")
    axis.set_xlabel("Decision time", labelpad=3)
    axis.set_xlim(-.35, len(x) - .65)
    axis = axes[1]
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E1E5E8", linewidth=.45)
    for label, color, offset in (("predicted", PREDICTED, -.17), ("simulated", ACTUAL, .17)):
        data = time_summary.loc[time_summary.series == label].set_index("decision_step").loc[decisions]
        counts = data.f3_win_states.to_numpy()
        axis.bar(x + offset, counts, width=.3, color=color, alpha=.9, zorder=2)
        for position, value in zip(x + offset, counts):
            axis.text(position, value + .22, str(value), ha="center", va="bottom", fontsize=7.5)
    axis.set_title("(b) States favoring F3", loc="left", pad=7)
    axis.set_ylabel("State count (out of 10)")
    axis.set_ylim(0, 11.4)
    axis.set_yticks([0, 2, 4, 6, 8, 10])
    axis.set_xticks(x, times, rotation=35, ha="right")
    axis.set_xlabel("Decision time", labelpad=3)
    _header(figure, "Decision-time aggregation: F3 versus BCS",
            "10 paired test seeds at each time  |  F3: p = 1.6, r = 2 km")
    _legend(figure, "Simulation", (.52, .866))
    _sign_note(figure, .018)
    return figure


def draw_overall(states, overall):
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 3.65))
    figure.subplots_adjust(left=.08, right=.985, bottom=.23, top=.75, wspace=.47)
    axis = axes[0]
    _axes(axis)
    for index, (row, color) in enumerate(zip(overall.itertuples(), (PREDICTED, ACTUAL))):
        axis.bar(index, row.mean, width=.52, color=color, alpha=.9, zorder=2)
        axis.errorbar(index, row.mean, yerr=[[row.mean - row.ci95_low], [row.ci95_high - row.mean]],
                       fmt="none", color="#222222", elinewidth=.7, capsize=2.5, capthick=.7, zorder=3)
        label_y = row.ci95_high + .4 if row.mean > 0 else row.ci95_low - .4
        axis.text(index, label_y, f"{row.mean:+.2f}", ha="center",
                  va="bottom" if row.mean > 0 else "top", fontsize=9)
    axis.set_title("(a) Overall mean gap", loc="left", pad=8)
    axis.set_ylabel("Revenue gap (USD / 5 min)")
    axis.set_xticks([0, 1], ["Predicted", "Simulated"])
    axis.set_xlim(-.55, 1.55)
    axis.set_ylim(min(overall.ci95_low) - 2.1, max(overall.ci95_high) + 2.1)
    axis = axes[1]
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E1E5E8", linewidth=.45)
    for index, (row, color) in enumerate(zip(overall.itertuples(), (PREDICTED, ACTUAL))):
        axis.bar(index, row.f3_win_states, width=.52, color=color, alpha=.9, zorder=2)
        axis.text(index, row.f3_win_states + 2, f"{row.f3_win_states}/{row.n_states}",
                  ha="center", va="bottom", fontsize=9)
    axis.set_title("(b) States favoring F3", loc="left", pad=8)
    axis.set_ylabel("State count")
    axis.set_xticks([0, 1], ["Predicted", "Simulated"])
    axis.set_xlim(-.55, 1.55)
    axis.set_ylim(0, 70)
    axis.set_yticks([0, 15, 30, 45, 60])
    axis = axes[2]
    _axes(axis)
    axis.axvline(0, color="#656565", linewidth=.75, linestyle=(0, (4, 3)))
    agreement = np.sign(states.predicted_gap) == np.sign(states.actual_gap)
    for mask, color in ((agreement, PREDICTED), (~agreement, ACTUAL)):
        axis.scatter(states.loc[mask, "predicted_gap"], states.loc[mask, "actual_gap"],
                     s=13, color=color, edgecolors="white", linewidths=.35, alpha=.8, zorder=3)
    axis.set_title("(c) State-wise ranking", loc="left", pad=8)
    axis.set_xlabel("Predicted gap (USD)", labelpad=4)
    axis.set_ylabel("Simulated mean gap (USD)")
    axis.set_xlim(min(states.predicted_gap) - 1, 1.3)
    axis.set_ylim(min(states.actual_gap) - 1.5, max(states.actual_gap) + 1.5)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    _header(figure, "Overall action-ranking diagnostic: F3 versus BCS",
            "60 states  |  10 paired future streams per state  |  Mean-gap error bars: seed-cluster 95% CI")
    figure.text(.5, .09, "(c) Blue: same ranking direction; orange: opposite ranking direction.",
                ha="center", fontsize=7.7)
    _sign_note(figure, .027)
    return figure


def plot_paper_ranking(input_dir=DEFAULT_INPUT, output_dir=None):
    source = Path(input_dir)
    output = Path(output_dir) if output_dir else source / "model_ranking_figures"
    states, trials, manifest, reference, sources = load_diagnostic(source)
    time_summary, overall, decisions, times = summarize(states, manifest, reference)
    output.mkdir(parents=True, exist_ok=True)
    states.to_csv(output / "state_comparison.csv", index=False)
    trials[TRIAL_KEYS + ["greedy_revenue", "bcs_revenue", "fixed_f3_revenue",
                        "actual_gap", "bcs_equals_greedy"]].to_csv(output / "paired_trials.csv", index=False)
    time_summary.to_csv(output / "decision_time_summary.csv", index=False)
    overall.to_csv(output / "overall_summary.csv", index=False)
    family = _font_family()
    style = {
        "font.family": family, "font.size": 8, "axes.titlesize": 8.7,
        "axes.labelsize": 8.2, "xtick.labelsize": 7.3, "ytick.labelsize": 7.5,
        "legend.fontsize": 8.3, "axes.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "xtick.color": "#333333", "ytick.color": "#333333", "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "axes.formatter.useoffset": False,
    }
    files = []
    with plt.rc_context(style), PdfPages(output / "all_ranking_figures.pdf") as combined:
        figures = (
            ("fig01_state_wise_gaps", draw_states(states, decisions, times)),
            ("fig02_decision_time_summary", draw_times(time_summary, decisions, times)),
            ("fig03_overall_statistics", draw_overall(states, overall)),
        )
        for name, figure in figures:
            for extension in ("png", "pdf", "svg"):
                figure.savefig(output / f"{name}.{extension}", dpi=600, facecolor="white")
                files.append(f"{name}.{extension}")
            combined.savefig(figure, facecolor="white")
            plt.close(figure)
    captions = (
        "Figure 1. State-wise predicted and simulated five-minute platform revenue gaps (F3 minus BCS) "
        "at six sampled decision times along each of ten greedy-policy trajectories. The sampled "
        "zero-based decision indices are 4, 14, 24, 34, 44 and 54 (05:20, 06:10, 07:00, 07:50, "
        "08:40 and 09:30). F3 uses price multiplier 1.6 and matching radius 2 km in all grids. "
        "Each simulated point is the average of ten paired future-stream differences at the same "
        "initial state and history. Predictions are deterministic for a given state and action. "
        "Figure 1 displays point estimates without within-state uncertainty intervals; the "
        "within-state paired sample SD is provided in state_comparison.csv. Lines connect sampled "
        "observations only and do not represent measurements at the intervening decision times.\n\n"
        "Figure 2. Aggregation by sampled decision time. (a) Mean predicted and simulated gaps across "
        "ten seeds, with percentile 95% confidence intervals from 10,000 seed-cluster bootstrap "
        "resamples (bootstrap RNG seed 42). The same seed resample is used across all six times. "
        "(b) Number of states with a strictly positive gap at each time. Simulated state preference "
        "is based on its ten-stream mean, not a per-state statistical significance test.\n\n"
        "Figure 3. Overall statistics for the same 60 sampled states. (a) Mean gaps with seed-cluster "
        "95% confidence intervals. (b) Counts of states preferring F3 based on predicted gaps and "
        "simulated mean gaps. (c) Each point is one state; blue indicates agreement in ranking "
        "direction and orange indicates disagreement. The simulated F3 advantage in 53 of 60 states "
        "does not imply 53 statistically significant individual-state wins.\n\n"
        "Methods. All simulated gaps compare the current five-minute interval from identical initial "
        "states, histories and paired future RNG initializations; they are not differences between "
        "separate closed-loop episodes. F3-minus-BCS differences are recovered as F3-minus-greedy "
        "minus BCS-minus-greedy, using exactly matching seed, decision index and future-stream ID. "
        "The 36 states where BCS differs from greedy have explicit BCS trial records; for the "
        "remaining 24 states BCS and greedy actions are identical, so their paired revenue difference "
        "is exactly zero. This reconstruction does not introduce additional simulator runs. "
        "The 600 paired comparisons share 60 states nested within ten trajectory seeds; they are "
        "not treated as 600 independent observations when computing overall intervals. Only one "
        "date and low-density scenario are covered. No smoothing, policy tuning, model retraining "
        "or checkpoint changes were performed. Positive gaps favor F3; negative gaps favor BCS.\n"
    )
    (output / "captions.txt").write_text(captions, encoding="utf-8")
    figure_manifest = {
        "source_directory": str(source.resolve()), "checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "plot_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sampled_states": len(states), "paired_comparisons": len(trials),
        "sampled_decisions": [int(value) for value in decisions], "sampled_times": times,
        "bcs_equals_greedy_states": int((~states.changed_actions).sum()),
        "gap_definition": "F3 minus BCS platform revenue, USD per five-minute period",
        "bootstrap_unit": "trajectory seed", "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": 42, "font_family": family, "png_dpi": 600,
        "overall_statistics": overall.to_dict("records"),
        "reference_statistics": reference, "files": files + ["all_ranking_figures.pdf"],
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"动作排序诊断图已保存至：{output}")
    return figure_manifest


def main():
    parser = argparse.ArgumentParser(description="绘制 F3 与 BCS 的论文版动作排序诊断图")
    parser.add_argument("--input", dest="input_dir", default=DEFAULT_INPUT)
    parser.add_argument("--output", dest="output_dir")
    plot_paper_ranking(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
