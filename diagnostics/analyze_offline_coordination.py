"""从已完成的离线 RL 实验分析动作、收益位置与训练行为惯性。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.compare_policies import summarize_operation
from offline_rl.data import DEFAULT_REPLAY, save_json, sha256


POLICIES = ("fixed_f3", "bc9", "iql", "iql9")


def paired_interval(values):
    samples = np.random.default_rng(42).choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return np.quantile(samples, (.025, .975))


def analyze(input_dir, output, replay_root=DEFAULT_REPLAY):
    input_dir, output, replay_root = map(Path, (input_dir, output, replay_root))
    manifest = json.loads((input_dir / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError
    output.mkdir(parents=True, exist_ok=False)
    grid_rows, period_rows, action_rows = [], [], []
    for seed in manifest["seeds"]:
        for policy in POLICIES:
            directory = input_dir / "runs" / f"seed{seed}" / policy
            minutes = pd.read_csv(directory / "grid_minute_metrics.csv.gz")
            warmup = manifest["config"]["history_steps"] * manifest["config"]["decision_freq"]
            minutes = minutes[minutes.minute_step >= warmup]
            for grid, group in minutes.groupby("grid_id"):
                grid_rows.append(dict(seed=seed, policy=policy, grid_id=grid, **summarize_operation(group)))
            for decision, group in minutes.groupby("decision_step"):
                period_rows.append(dict(seed=seed, policy=policy, decision_step=decision,
                                        **summarize_operation(group)))
            actions = pd.read_csv(directory / "action_plan.csv.gz")
            actions = actions[~actions.warmup].copy()
            actions["seed"], actions["policy"] = seed, policy
            action_rows.append(actions[["seed", "policy", "grid_id", "decision_step",
                                        "price_multiplier", "matching_radius_km"]])
    grid = pd.DataFrame(grid_rows)
    periods = pd.DataFrame(period_rows)
    actions = pd.concat(action_rows, ignore_index=True)
    grid.to_csv(output / "grid_seed_metrics.csv", index=False)
    periods.to_csv(output / "period_seed_metrics.csv", index=False)
    actions.to_csv(output / "action_choices.csv", index=False)
    action_summary = []
    for (policy, region), group in actions.groupby(["policy", "grid_id"]):
        f3 = (group.price_multiplier == 1.6) & (group.matching_radius_km == 2)
        action_summary.append(dict(policy=policy, grid_id=region, count=len(group),
                                   mean_price=group.price_multiplier.mean(),
                                   mean_radius=group.matching_radius_km.mean(),
                                   f3_share=f3.mean(), low_price_share=group.price_multiplier.eq(1.4).mean()))
    action_summary = pd.DataFrame(action_summary)
    action_summary.to_csv(output / "grid_action_summary.csv", index=False)
    means = grid.groupby(["policy", "grid_id"]).mean(numeric_only=True)
    records = []
    for region, group in grid.groupby("grid_id"):
        revenue = group.pivot(index="seed", columns="policy", values="platform_revenue")
        difference = (revenue.iql9 - revenue.fixed_f3).to_numpy()
        low, high = paired_interval(difference)
        records.append(dict(grid_id=int(region), f3_revenue=revenue.fixed_f3.mean(),
                            iql9_revenue=revenue.iql9.mean(), difference_usd=difference.mean(),
                            ci95_low=low, ci95_high=high))
    grid_gaps = pd.DataFrame(records)
    grid_gaps.to_csv(output / "grid_revenue_gaps.csv", index=False)
    time_gaps = []
    for decision, group in periods.groupby("decision_step"):
        revenue = group.pivot(index="seed", columns="policy", values="platform_revenue")
        difference = (revenue.iql9 - revenue.fixed_f3).to_numpy()
        low, high = paired_interval(difference)
        time_gaps.append(dict(decision_step=int(decision), difference_usd=difference.mean(),
                              ci95_low=low, ci95_high=high))
    time_gaps = pd.DataFrame(time_gaps)
    time_gaps.to_csv(output / "decision_revenue_gaps.csv", index=False)
    assert np.isclose(grid_gaps.difference_usd.sum(), time_gaps.difference_usd.sum())

    metadata = json.loads((replay_root / "manifest.json").read_text())
    train = np.load(replay_root / "train/actions.npy", mmap_mode="r").reshape(
        -1, metadata["periods_per_run"], metadata["grid_count"])
    repeated = train[:, 1:] == train[:, :-1]
    steps = np.asarray(metadata["decision_steps"])[1:]
    valid_pairs = [i for i, (price, radius) in enumerate(metadata["action_pairs"])
                   if price in (1.4, 1.6, 1.8) and radius in (1.5, 2., 2.5)]
    allowed = np.isin(train, valid_pairs)
    usage = actions[actions.policy == "iql9"]
    data = {
        "input_dir": str(input_dir.resolve()), "input_manifest_sha256": sha256(input_dir / "manifest.json"),
        "seed_count": len(manifest["seeds"]), "config": manifest["config"],
        "grid_decisions_per_policy": len(usage),
        "iql9_price_1p4_share": float(usage.price_multiplier.eq(1.4).mean()),
        "iql9_radius_1p5_share": float(usage.matching_radius_km.eq(1.5).mean()),
        "iql9_f3_action_share": float(((usage.price_multiplier == 1.6) & (usage.matching_radius_km == 2)).mean()),
        "training_grid_action_repeat_share": float(repeated.mean()),
        "training_whole_city_action_repeat_share": float(repeated.all(axis=2).mean()),
        "training_repeat_at_preferred_hold_boundary": float(repeated[:, steps % 3 == 0].mean()),
        "training_repeat_elsewhere": float(repeated[:, steps % 3 != 0].mean()),
        "training_individual_action_in_nine_share": float(allowed.mean()),
        "training_joint_action_in_nine_share": float(allowed.all(axis=2).mean()),
        "training_joint_action_in_nine_count": int(allowed.all(axis=2).sum()),
        "grid_2_and_3_share_of_net_revenue_gap": float(grid_gaps.query("grid_id in [2,3]").difference_usd.sum() / grid_gaps.difference_usd.sum()),
        "limitations": [
            "网格与时段差异是闭环轨迹的观测分解，不是单个智能体的因果责任。",
            "同种子策略使用同一外生订单和初态，但策略分支后的随机数消耗可能不同。",
            "持续动作比例支持行为惯性假设；尚未做改变预热/移除历史动作的消融。",
            "测试集用于事后分析；下一版应只在验证集选模型，再用新留出 seed 确认。",
        ],
    }
    save_json(output / "summary.json", data)
    plot(output, actions, grid_gaps, time_gaps)
    return data


def plot(output, actions, grid_gaps, time_gaps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from baseline.plot_paper_figures import _font_family

    pairs = [(p, r) for p in (1.4, 1.6, 1.8) for r in (1.5, 2., 2.5)]
    choices = actions[actions.policy == "iql9"]
    table = choices.groupby(["grid_id", "price_multiplier", "matching_radius_km"]).size()
    heat = np.array([[table.get((g, p, r), 0) / len(choices[choices.grid_id == g])
                      for p, r in pairs] for g in range(6)])
    with plt.rc_context({"font.family": _font_family(), "font.size": 9, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.3))
        fig.subplots_adjust(left=.09, right=.97, top=.95, bottom=.15, hspace=.57, wspace=.35)
        ax = axes[0, 0]
        im = ax.imshow(heat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title("(a) IQL-9 action choices by grid", loc="left")
        ax.set_xticks(range(9), [f"{p:g}/{r:g}" for p, r in pairs], rotation=45, ha="right")
        ax.set_yticks(range(6), [f"Grid {g}" for g in range(6)])
        for g in range(6):
            for j in range(9):
                ax.text(j, g, f"{heat[g,j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if heat[g,j] > .6 else "black")
        ax.set_xlabel("Price multiplier / radius (km)")
        fig.colorbar(im, ax=ax, fraction=.04, pad=.02).ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax = axes[0, 1]
        ax.bar(grid_gaps.grid_id, grid_gaps.difference_usd, color="#D55E00", zorder=3)
        ax.errorbar(grid_gaps.grid_id, grid_gaps.difference_usd,
                    yerr=np.array([grid_gaps.difference_usd-grid_gaps.ci95_low,
                                   grid_gaps.ci95_high-grid_gaps.difference_usd]),
                    fmt="none", color="black", linewidth=.7, capsize=2)
        ax.axhline(0, color="#666666", linewidth=.6)
        ax.set_title("(b) Revenue difference by grid ↑", loc="left")
        ax.set_xlabel("Grid ID")
        ax.set_ylabel("IQL-9 − F3 (USD / episode)")
        ax = axes[1, 0]
        times = (5 * 60 + time_gaps.decision_step * 5).to_numpy() / 60
        ax.plot(times, time_gaps.difference_usd, color="#0072B2", linewidth=1)
        ax.fill_between(times, time_gaps.ci95_low, time_gaps.ci95_high, color="#0072B2", alpha=.15)
        ax.axhline(0, color="#666666", linewidth=.6)
        ax.set_title("(c) Revenue difference over time ↑", loc="left")
        ax.set_xlabel("Simulation hour")
        ax.set_ylabel("IQL-9 − F3 (USD / 5 min)")
        ax = axes[1, 1]
        policies = ("iql", "iql9", "bc9")
        shares = []
        for policy in policies:
            d = actions[actions.policy == policy]
            shares.append([(d.price_multiplier.eq(1.5) & d.matching_radius_km.eq(3)).mean(),
                           (d.price_multiplier.eq(1.6) & d.matching_radius_km.eq(2)).mean()])
        ax.bar(np.arange(3)-.18, np.array(shares)[:, 0], width=.36, label="Warmup: 1.5/3", color="#777777")
        ax.bar(np.arange(3)+.18, np.array(shares)[:, 1], width=.36, label="F3: 1.6/2", color="#009E73")
        ax.set_title("(d) Frequency of two reference actions", loc="left")
        ax.set_xticks(range(3), ("IQL", "IQL-9", "BC-9"))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, 1.16)
        ax.legend(fontsize=8, frameon=False)
        fig.text(.5, .025, "10 paired test seeds; 05:20–10:00. Bands/error bars: pointwise 95% seed bootstrap CIs.", ha="center", fontsize=8)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(output / f"action_coordination_diagnostic.{ext}", dpi=600, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析离线 RL 动作与跨网格收益分布，不重新训练")
    parser.add_argument("--input-dir", default="models/iql_grid6_low_20261102/test")
    parser.add_argument("--output", default="experiments/iql_coordination_analysis_20261102")
    parser.add_argument("--replay-root", default=DEFAULT_REPLAY)
    analyze(**vars(parser.parse_args()))
