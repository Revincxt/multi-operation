"""统计既有 LSTM 闭环测试的时变动作，不重跑仿真、不修改策略。

使用方式:
    python3 -m diagnostics.analyze_lstm_actions --input experiments/lstm_fixed_random_20261009

统计排除共同预热；切换率仅比较同一 seed、同一网格的相邻模型决策。
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTIONS = ("price_multiplier", "matching_radius_km")
LABELS = ("Price multiplier", "Matching radius (km)")
GRAY = "#E3E6E8"


def clock_label(minute, position=None):
    return f"{int(round(minute)) // 60:02d}:{int(round(minute)) % 60:02d}"


def load_actions(source):
    """要求动作表完整，时间与清单一致，保留所有候选值（包括零次选择）。"""
    source = Path(source)
    report = json.loads((source / "manifest.json").read_text())
    config = report["config"]
    if report["status"] != "complete" or len(set(report["seeds"])) != len(report["seeds"]):
        raise ValueError
    starts = np.arange(0, config["simulation_steps"], config["decision_freq"])
    warmup = config["history_steps"] * config["decision_freq"]
    if warmup >= config["simulation_steps"]:
        raise ValueError
    hour, minute = map(int, config["start_time"].split(":"))
    initial = hour * 60 + minute
    expected = pd.MultiIndex.from_product([range(len(starts)), range(config["grid_num"])],
                                         names=["decision_step", "grid_id"])
    frames = []
    for seed in report["seeds"]:
        frame = pd.read_csv(source / "runs" / f"seed{seed}" / "lstm/action_plan.csv.gz")
        frame = frame.sort_values(["decision_step", "grid_id"]).reset_index(drop=True)
        index = pd.MultiIndex.from_frame(frame[["decision_step", "grid_id"]])
        if (not index.equals(expected) or frame.warmup.dtype != bool
                or frame[list(ACTIONS)].isna().any().any()
                or not np.array_equal(frame.decision_start_minute_step,
                                      np.repeat(starts, config["grid_num"]))
                or not np.array_equal(frame.decision_end_minute_step_exclusive,
                                      np.repeat(np.minimum(starts + config["decision_freq"],
                                                           config["simulation_steps"]), config["grid_num"]))
                or not frame.warmup.equals(frame.decision_start_minute_step.lt(warmup))):
            raise ValueError
        for action, key, warmup_key in zip(ACTIONS, ("price_action_values", "matching_radius_values"),
                                            ("warmup_price", "warmup_radius")):
            if (not frame.loc[~frame.warmup, action].isin(config[key]).all()
                    or not frame.loc[frame.warmup, action].eq(config[warmup_key]).all()):
                raise ValueError
        frame["seed"] = seed
        frame["clock_minute"] = initial + frame.decision_start_minute_step
        frame["clock_time"] = frame.clock_minute.map(clock_label)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), report


def action_statistics(history):
    """网格与种子等权，不按订单量加权，也不把跨网格差异算成切换。"""
    active = history.loc[~history.warmup].copy().sort_values(["seed", "grid_id", "decision_step"])
    previous = active.groupby(["seed", "grid_id"])[list(ACTIONS)].shift()
    valid = previous.notna().all(axis=1)
    for action in ACTIONS:
        active[f"{action}_changed"] = active[action].ne(previous[action]).astype(float).where(valid)
    active["joint_changed"] = active[list(ACTIONS)].ne(previous).any(axis=1).astype(float).where(valid)

    def summarize(keys):
        aggregations = {"decisions": ("grid_id", "size"), "transitions": ("joint_changed", "count"),
                        "joint_switch_rate": ("joint_changed", "mean")}
        for action in ACTIONS:
            for suffix, function in (("mean", "mean"), ("median", "median"), ("std", "std"),
                                     ("q25", lambda s: s.quantile(0.25)), ("q75", lambda s: s.quantile(0.75))):
                aggregations[f"{action}_{suffix}"] = (action, function)
            aggregations[f"{action}_switch_rate"] = (f"{action}_changed", "mean")
        return active.groupby(keys).agg(**aggregations).reset_index()

    active["hour"] = active.clock_minute // 60
    end = int((history.clock_minute + history.decision_end_minute_step_exclusive
               - history.decision_start_minute_step).max())
    hourly = summarize(["hour"])
    hourly.insert(1, "time_interval", [
        f"{clock_label(max(hour * 60, int(active.clock_minute.min())))}–{clock_label(min((hour + 1) * 60, end))}"
        for hour in hourly.hour])
    frequency = active.groupby(list(ACTIONS)).size().rename("decisions").reset_index()
    frequency["share"] = frequency.decisions / len(active)
    frequency = frequency.sort_values(["decisions", *ACTIONS], ascending=[False, True, True])
    return active, {"time_summary": summarize(["clock_minute", "clock_time"]),
                    "grid_time_summary": summarize(["grid_id", "clock_minute", "clock_time"]),
                    "grid_summary": summarize(["grid_id"]), "hourly_summary": hourly,
                    "joint_action_frequency": frequency}


def time_frequency(active, action, candidates):
    """每个时间步分别归一化，不能把均值/众数当作逐次实际动作。"""
    index = pd.MultiIndex.from_product([sorted(active.clock_minute.unique()), candidates],
                                      names=["clock_minute", action])
    result = active.groupby(["clock_minute", action]).size().reindex(index, fill_value=0).rename("decisions").reset_index()
    result["share"] = result.decisions / result.groupby("clock_minute").decisions.transform("sum")
    return result


def _font_family():
    for family in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
            return family
        except ValueError:
            continue
    raise ValueError


def _time_axis(ax, initial, end, warmup_end):
    ax.set_xlim(initial, end)
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.xaxis.set_major_formatter(FuncFormatter(clock_label))
    ax.axvline(warmup_end, color="#666666", linestyle="--", linewidth=0.7)
    ax.tick_params(length=2.5, width=0.5, labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def make_figures(history, active, tables, frequencies, report):
    config = report["config"]
    initial, end = int(history.clock_minute.min()), int(history.clock_minute.min() + config["simulation_steps"])
    warmup_end = initial + config["history_steps"] * config["decision_freq"]
    starts = sorted(history.clock_minute.unique())
    edges = np.array(starts + [end])
    sets = (config["price_action_values"], config["matching_radius_values"])
    figures = []

    # 全部网格、全部测试种子的分布趋势；阴影不是置信区间。
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.8), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.84, bottom=0.15, hspace=0.33)
    times = tables["time_summary"]
    x = np.append(times.clock_minute.to_numpy(), end)
    for i, (ax, action, label, candidates) in enumerate(zip(axes, ACTIONS, LABELS, sets)):
        mean = np.append(times[f"{action}_mean"], times[f"{action}_mean"].iloc[-1])
        q25 = np.append(times[f"{action}_q25"], times[f"{action}_q25"].iloc[-1])
        q75 = np.append(times[f"{action}_q75"], times[f"{action}_q75"].iloc[-1])
        ax.fill_between(x, q25, q75, step="post", color="#0072B2", alpha=0.18, label="25th–75th percentile")
        ax.step(x, mean, where="post", color="#0072B2", linewidth=1.1, label="Mean")
        ax.axvspan(initial, warmup_end, color=GRAY, zorder=0)
        ax.set_title(f"({chr(97 + i)}) {label}", loc="left", fontsize=9)
        ax.set_ylim(min(candidates) - 0.1, max(candidates) + 0.1)
        ax.grid(axis="y", color="#CCCCCC", linewidth=0.4, alpha=0.7)
        _time_axis(ax, initial, end, warmup_end)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.10, 1.01), fontsize=7, frameon=False, ncol=2)
    axes[-1].set_xlabel("Simulation time")
    fig.suptitle("LSTM action trends", fontsize=11, y=0.975)
    fig.text(0.5, 0.912, f"{len(report['seeds'])} seeds × {config['grid_num']} grids; equally weighted at each decision step",
             ha="center", fontsize=8)
    fig.text(0.5, 0.035, "Gray: fixed warm-up. Shading: action spread, not a confidence interval. Means are not executed actions.",
             ha="center", fontsize=7)
    figures.append(("lstm_action_trends", fig))

    # 动作候选的逐时选择频率，所有候选都显示，未选择记为 0%。
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 5.4), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.88, top=0.84, bottom=0.15, hspace=0.33)
    for i, (ax, action, label, candidates) in enumerate(zip(axes, ACTIONS, LABELS, sets)):
        values = frequencies[action].pivot(index=action, columns="clock_minute", values="share").reindex(index=candidates, columns=starts)
        im = ax.pcolormesh(edges, np.arange(len(candidates) + 1) - 0.5, np.ma.masked_invalid(values.to_numpy()),
                           cmap=plt.get_cmap("viridis").with_extremes(bad=GRAY), vmin=0, vmax=1,
                           shading="flat", rasterized=False, edgecolors="none")
        ax.set_yticks(range(len(candidates)), [f"{value:g}" for value in candidates])
        ax.set_title(f"({chr(97 + i)}) {label}", loc="left", fontsize=9)
        _time_axis(ax, initial, end, warmup_end)
        ax.tick_params(axis="y", labelsize=7)
    colorbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.03, aspect=40)
    colorbar.set_label("Share of grid–seed decisions (%)", fontsize=8)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    colorbar.ax.tick_params(labelsize=7)
    colorbar.solids.set_rasterized(False)
    axes[-1].set_xlabel("Simulation time")
    fig.suptitle("LSTM action-selection frequency over time", fontsize=11, y=0.975)
    fig.text(0.5, 0.912, f"{len(report['seeds']) * config['grid_num']} grid–seed decisions at each {config['decision_freq']}-min step",
             ha="center", fontsize=8)
    fig.text(0.5, 0.035, "Each time column sums to 100%. Gray: fixed warm-up, excluded. Candidate rows are categorical.",
             ha="center", fontsize=7)
    figures.append(("lstm_action_frequencies", fig))

    # 在观察结果前固定选择清单中的第一个 seed，不挑最好/最平滑的一次。
    example_seed = report["seeds"][0]
    sample = history.loc[history.seed == example_seed].copy()
    sample.loc[sample.warmup, list(ACTIONS)] = np.nan
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.8), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.90, top=0.84, bottom=0.15, hspace=0.35)
    for i, (ax, action, label, candidates) in enumerate(zip(axes, ACTIONS, LABELS, sets)):
        values = sample.pivot(index="grid_id", columns="clock_minute", values=action).reindex(
            index=range(config["grid_num"]), columns=starts)
        im = ax.pcolormesh(edges, np.arange(config["grid_num"] + 1) - 0.5, np.ma.masked_invalid(values.to_numpy()),
                           cmap=plt.get_cmap("magma" if i == 0 else "viridis").with_extremes(bad=GRAY),
                           vmin=min(candidates), vmax=max(candidates), shading="flat", rasterized=False)
        ax.set_yticks(range(config["grid_num"]))
        ax.set_ylabel("Grid")
        ax.set_title(f"({chr(97 + i)}) {label}", loc="left", fontsize=9)
        _time_axis(ax, initial, end, warmup_end)
        colorbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.solids.set_rasterized(False)
    axes[-1].set_xlabel("Simulation time")
    fig.suptitle(f"LSTM executed actions by grid — seed {example_seed}", fontsize=11, y=0.975)
    fig.text(0.5, 0.912, "One cell = one grid's executed action in one decision interval; no averaging", ha="center", fontsize=8)
    fig.text(0.5, 0.035, "Example: first seed in the evaluation manifest, not selected by performance. Gray: fixed warm-up.",
             ha="center", fontsize=7)
    figures.append(("lstm_actions_example", fig))
    return figures


def analyze(input_dir="experiments/lstm_fixed_random_20261009", output_dir=None):
    source = Path(input_dir)
    history, report = load_actions(source)
    active, tables = action_statistics(history)
    config = report["config"]
    sets = (config["price_action_values"], config["matching_radius_values"])
    frequencies = {action: time_frequency(active, action, candidates) for action, candidates in zip(ACTIONS, sets)}
    output = Path(output_dir) if output_dir else source / "lstm_action_analysis"
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    for action, frame in frequencies.items():
        frame.to_csv(output / f"{action}_time_frequency.csv", index=False)
    active.to_csv(output / "executed_actions.csv", index=False)
    family = _font_family()
    with plt.rc_context({"font.family": family, "font.size": 8, "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path"}), \
            PdfPages(output / "lstm_action_analysis.pdf") as combined:
        for name, fig in make_figures(history, active, tables, frequencies, report):
            for extension in ("pdf", "svg", "png"):
                fig.savefig(output / f"{name}.{extension}", dpi=600, facecolor="white")
            combined.savefig(fig, facecolor="white")
            plt.close(fig)
    paths = [source / "manifest.json"] + [source / "runs" / f"seed{seed}" / "lstm/action_plan.csv.gz"
                                         for seed in report["seeds"]]
    summary = {
        "seed_count": len(report["seeds"]), "grid_count": config["grid_num"],
        "decision_interval_minutes": config["decision_freq"], "example_seed": report["seeds"][0],
        "learned_decision_count": len(active), "decision_steps_per_seed": active.decision_step.nunique(),
        "transition_count": int(active.joint_changed.count()),
        "joint_switch_count": int(active.joint_changed.sum()), "joint_switch_rate": float(active.joint_changed.mean()),
        "price_switch_rate": float(active.price_multiplier_changed.mean()),
        "radius_switch_rate": float(active.matching_radius_km_changed.mean()),
        "top_joint_actions": tables["joint_action_frequency"].head(10).to_dict("records"),
        "source_sha256": {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "plot_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "font_family": family, "png_dpi": 600,
        "notes": ["Warm-up is excluded, including the warm-up-to-first-model-action transition.",
                  "Switches are computed within each seed and grid, never across grid or run boundaries.",
                  "Means and quantiles pool grid-seed decisions equally, not demand-weighted; no smoothing.",
                  "Hourly switching rates assign each transition to the hour of the later action.",
                  "Mean actions need not be candidate values. Frequency columns sum to 1 independently per time step.",
                  "An example trajectory is not the action of every test run. These summaries do not establish optimality."],
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(tables["hourly_summary"][["time_interval", "price_multiplier_mean", "matching_radius_km_mean", "joint_switch_rate"]].to_string(index=False))
    print(f"模型决策 {len(active)} 次，动作切换率 {summary['joint_switch_rate']:.2%}；输出：{output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="统计 LSTM 动作随时间和网格的变化")
    parser.add_argument("--input", dest="input_dir", default="experiments/lstm_fixed_random_20261009")
    parser.add_argument("--output", dest="output_dir")
    analyze(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
