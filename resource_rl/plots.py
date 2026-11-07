"""完整训练/验证记录的学术图；不平滑、不删除较差迭代。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STYLE = {"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
    "pdf.fonttype": 42, "svg.fonttype": "none"}
COLORS = {"flow": "#218A85", "window": "#C38D68"}
LABELS = {"flow": "Resource-Flow AC", "window": "Window-AC"}


def training(root):
    root = Path(root)
    files = list(root.glob("*_seed*/training_history.csv"))
    if not files:
        return
    history = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    validation_paths = list(root.glob("*_seed*/validation_history.csv"))
    validation = pd.concat([pd.read_csv(path) for path in validation_paths], ignore_index=True) if validation_paths else pd.DataFrame()
    history.to_csv(root/"training_history.csv", index=False)
    if len(validation):
        validation.to_csv(root/"validation_history.csv", index=False)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.1), layout="constrained")
        fields = ("revenue_mean_usd", "value_loss", "auxiliary_loss", "joint_kl", "entropy", "actor_effective_epochs")
        labels = ("Training return (USD) ↑", "State-value loss", "Resource prediction loss", "Full-city KL estimate",
                  "Conditional entropy", "Actor sample passes")
        for axis, field, label in zip(axes.flat, fields, labels):
            for variant, group in history.groupby("variant"):
                stats = group.groupby("episodes")[field].agg(["mean", "min", "max"])
                x = stats.index.to_numpy()/1000.
                axis.plot(x, stats["mean"], color=COLORS[variant], label=LABELS[variant], lw=1.1,
                          marker="o" if len(stats)<5 else None, ms=2)
                axis.fill_between(x, stats["min"], stats["max"], color=COLORS[variant], alpha=.13, lw=0)
            axis.set(xlabel="Training episodes per seed (×1000)", ylabel=label)
            axis.grid(axis="y", alpha=.18, lw=.5)
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="outside lower center", ncol=2, frameon=False)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(root/f"training_diagnostics.{ext}", dpi=600, facecolor="white")
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(7.0866, 3.5433), layout="constrained")
        for variant, group in history.groupby("variant"):
            stats = group.groupby("episodes").revenue_mean_usd.agg(["mean", "min", "max"])
            x = stats.index.to_numpy()/1000.
            axis.plot(x, stats["mean"], color=COLORS[variant], alpha=.55, lw=.9, label=LABELS[variant]+" (train)")
            axis.fill_between(x, stats["min"], stats["max"], color=COLORS[variant], alpha=.1, lw=0)
            if len(validation):
                selected = validation[validation.variant == variant]
                if len(selected):
                    stats = selected.groupby("episodes").revenue_mean_usd.mean()
                    axis.plot(stats.index.to_numpy()/1000., stats, color=COLORS[variant], lw=1.25,
                              marker="o" if len(stats)<10 else None, ms=2, label=LABELS[variant]+" (validation)")
        axis.set(xlabel="Training episodes per seed (×1000)", ylabel="Platform revenue (USD) ↑")
        axis.grid(axis="y", alpha=.18, lw=.5); axis.legend(frameon=False, fontsize=7, loc="lower right")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(root/f"training_curve.{ext}", dpi=600, facecolor="white")
        plt.close(fig)


def evaluation(root, frame, statistics):
    root = Path(root)
    order = ["fixed_f3", "original_ar_ppo", "window", "flow"]
    labels = {**LABELS, "fixed_f3": "F3", "original_ar_ppo": "Original AR-PPO"}
    colors = {**COLORS, "fixed_f3": "#969DA6", "original_ar_ppo": "#637FA3"}
    fields = ("baseline_request_num", "request_rate", "driver_accept_rate", "order_completion_rate", "mean_wait_to_match_seconds", "platform_revenue")
    names = ("Bubble requests", "Order request rate (%) ↑", "Driver acceptance rate (%) ↑", "Order completion rate (%) ↑", "Matched-passenger wait (s) ↓", "Platform revenue (USD) ↑")
    from resource_rl.evaluate import crossed_bootstrap
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), layout="constrained")
        for axis, field, name in zip(axes.flat, fields, names):
            factor = 100. if field.endswith("_rate") else 1.
            for position, variant in enumerate(order):
                part = frame[frame.variant == variant].pivot(index="training_seed", columns="seed", values=field)
                values = part.to_numpy()*factor
                mean = values.mean()
                low, high = np.quantile(crossed_bootstrap(values), [.025, .975])
                axis.bar(position, mean, color=colors[variant], width=.62)
                if high>low:
                    axis.errorbar(position, mean, yerr=[[max(mean-low, 0)], [max(high-mean, 0)]], fmt="none", color="#35434B", capsize=2, lw=.7)
            axis.set_ylabel(name); axis.set_xticks(range(4), ["F3", "Original", "Window", "Flow"])
            axis.set_ylim(bottom=0); axis.grid(axis="y", alpha=.18, lw=.5); axis.set_axisbelow(True)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[name]) for name in order]
        fig.legend(handles, [labels[name] for name in order], loc="outside lower center", frameon=False, ncol=2)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(root/f"six_metrics.{ext}", dpi=600, facecolor="white")
        plt.close(fig)
        selected = [row for row in statistics if row["metric"] == "platform_revenue"]
        fig, axis = plt.subplots(figsize=(6.5, 2.2), layout="constrained")
        for index, row in enumerate(selected):
            mean = row["mean_difference"]
            axis.errorbar(mean, index, xerr=[[max(mean-row["ci95_low"], 0)], [max(row["ci95_high"]-mean, 0)]],
                fmt="o", color=COLORS["flow"], ms=4, capsize=3, lw=1.1)
        axis.axvline(0., color="#87909A", ls="--", lw=.8)
        axis.set_yticks(range(len(selected)), [f"Resource-Flow AC − {labels[row['reference']]}" for row in selected])
        axis.set_xlabel("Paired platform revenue difference (USD) ↑"); axis.grid(axis="x", alpha=.18)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(root/f"paired_revenue.{ext}", dpi=600, facecolor="white")
        plt.close(fig)
