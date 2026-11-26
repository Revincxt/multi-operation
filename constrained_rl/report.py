"""汇总多指标试训；按共同环境种子重采样，保留服务指标的分子分母。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = {"fixed_f3": "F3", "frozen_ar_ppo": "Frozen AR-PPO",
          "revenue_control": "Revenue PPO", "constrained": "Constrained PPO"}
SHORT_LABELS = {"fixed_f3": "F3", "frozen_ar_ppo": "AR-PPO",
                "revenue_control": "PPO-R", "constrained": "PPO-L"}
COLORS = {"fixed_f3": "#929AA4", "frozen_ar_ppo": "#C49763",
          "revenue_control": "#56789C", "constrained": "#218C83"}
METRICS = {
    "platform_revenue": ("Platform revenue (USD) ↑", 1),
    "demand_satisfaction": ("Baseline demand coverage ↑", 1),
    "wait_per_demand_seconds": ("Queue wait / demand (s) ↓", -1),
    "baseline_request_num": ("Baseline requests", 0),
    "request_rate": ("Request rate ↑", 1),
    "driver_accept_rate": ("Driver acceptance ↑", 1),
    "order_completion_rate": ("Candidate-pool completion ↑", 1),
    "mean_wait_to_match_seconds": ("Matched-passenger wait (s) ↓", -1),
}
RATIOS = {"demand_satisfaction": ("unique_matched", "cohort_demand"),
          "wait_per_demand_seconds": ("queue_wait_seconds", "cohort_demand")}


def metric_values(frame, metric):
    """返回逐轨迹分子、分母；普通指标的分母为1。"""
    if metric in RATIOS:
        numerator, denominator = (frame[name].to_numpy(dtype=float) for name in RATIOS[metric])
        if np.any(denominator <= 0):
            raise ValueError
    else:
        numerator = frame[metric].to_numpy(dtype=float)
        denominator = np.ones(len(frame), dtype=float)
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
        raise ValueError
    return numerator, denominator


def estimate(numerator, denominator, samples=None):
    if samples is None:
        return float(numerator.sum() / denominator.sum())
    return numerator[samples].sum(1) / denominator[samples].sum(1)


def summarize(frame, draws=10000, seed=20261115):
    """同组种子配对bootstrap；新服务指标采用总分子/总分母。"""
    if (draws <= 0 or frame.empty or frame.duplicated(["policy", "seed"]).any()
            or not set(frame.policy).issubset(LABELS)):
        raise ValueError
    policies = [name for name in LABELS if name in set(frame.policy)]
    groups = {name: frame[frame.policy == name].sort_values("seed") for name in policies}
    seeds = groups[policies[0]].seed.to_numpy()
    if len(seeds) < 2 or any(not np.array_equal(group.seed.to_numpy(), seeds) for group in groups.values()):
        raise ValueError
    samples = np.random.default_rng(seed).integers(len(seeds), size=(draws, len(seeds)))
    summaries, comparisons, values = [], [], {}
    for policy, group in groups.items():
        for metric in METRICS:
            required = RATIOS.get(metric, (metric,))
            if not set(required).issubset(group.columns):
                continue
            numerator, denominator = metric_values(group, metric)
            point, boot = estimate(numerator, denominator), estimate(numerator, denominator, samples)
            values[policy, metric] = (point, boot, numerator / denominator)
            low, high = np.quantile(boot, [.025, .975])
            summaries.append({"policy": policy, "metric": metric, "scenarios": len(seeds),
                "estimate": point, "ci95_low": float(low), "ci95_high": float(high),
                "aggregation": "ratio_of_sums" if metric in RATIOS else "mean_of_episodes"})
    for policy in ("revenue_control", "constrained"):
        if policy not in groups:
            continue
        for reference in ("fixed_f3", "frozen_ar_ppo", "revenue_control"):
            if reference not in groups or reference == policy:
                continue
            for metric, (_, direction) in METRICS.items():
                if (policy, metric) not in values or (reference, metric) not in values:
                    continue
                point, boot, individual = values[policy, metric]
                base, base_boot, base_individual = values[reference, metric]
                low, high = np.quantile(boot - base_boot, [.025, .975])
                comparisons.append({"policy": policy, "reference": reference, "metric": metric,
                    "scenarios": len(seeds), "policy_estimate": point, "reference_estimate": base,
                    "difference": point - base, "relative_percent": 100 * (point-base) / base if base else np.nan,
                    "ci95_low": float(low), "ci95_high": float(high),
                    "favorable_scenarios": int(((individual-base_individual)*direction > 0).sum()) if direction else 0,
                    "ties": int(np.isclose(individual, base_individual, rtol=0., atol=1e-10).sum()),
                    "higher_is_better": direction})
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def feasibility(summary, constraints):
    result = []
    for policy in summary.policy.unique():
        rows = summary[summary.policy == policy].set_index("metric")
        if not set(RATIOS).issubset(rows.index):
            continue
        service, waiting = rows.loc["demand_satisfaction"], rows.loc["wait_per_demand_seconds"]
        floor, limit = constraints["service_floor"], constraints["wait_limit_seconds_per_demand"]
        result.append({"policy": policy, "service_floor": floor, "wait_limit_seconds_per_demand": limit,
            "service_margin": float(service.estimate - floor), "wait_margin_seconds": float(limit-waiting.estimate),
            "point_feasible": bool(service.estimate >= floor and waiting.estimate <= limit),
            "service_ci_above_floor": bool(service.ci95_low >= floor),
            "wait_ci_below_limit": bool(waiting.ci95_high <= limit)})
    return result


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    settings = {"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
                "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                "legend.fontsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none"}
    return plt, settings


def save_figure(fig, path):
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + suffix), dpi=600, facecolor="white", bbox_inches="tight", pad_inches=.04)


def plot_training(frame, constraints, output):
    plt, settings = plotting()
    with plt.rc_context(settings):
        fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.35), layout="constrained")
        for method in ("revenue_control", "constrained"):
            group = frame[frame.method == method].sort_values("iteration")
            if group.empty:
                continue
            for ax, column in zip(axes[0], ("revenue_mean_usd", "demand_satisfaction", "wait_per_demand_seconds")):
                ax.plot(group.episodes, group[column], color=COLORS[method], lw=1.35, label=LABELS[method])
            axes[1, 1].plot(group.episodes, group.joint_kl, color=COLORS[method], lw=1.2)
            if method == "constrained":
                for column, color, label in (("lambda_service", "#8A6B98", "Service"), ("lambda_wait", "#C49763", "Waiting")):
                    axes[1, 0].plot(group.episodes, group[column], color=color, lw=1.2, label=label)
                for column, color, label in (("cost_service_mean", "#8A6B98", "Service"), ("cost_wait_mean", "#C49763", "Waiting")):
                    axes[1, 2].plot(group.episodes, group[column], color=color, lw=1.2, label=label)
        axes[0, 1].axhline(constraints["service_floor"], color="#646A70", ls="--", lw=.9)
        axes[0, 2].axhline(constraints["wait_limit_seconds_per_demand"], color="#646A70", ls="--", lw=.9)
        axes[1, 2].axhline(0., color="#646A70", ls="--", lw=.9)
        titles = ("Return (USD) ↑", "Baseline demand coverage ↑", "Queue wait / demand (s) ↓",
                  "Lagrange multipliers", "Joint KL", "Normalized constraint excess")
        for ax, title in zip(axes.flat, titles):
            ax.set_title(title, loc="left", fontsize=8)
            ax.set_xlabel("Fine-tuning episodes")
            ax.grid(axis="y", alpha=.18, lw=.5)
        axes[0, 0].legend(frameon=False)
        axes[1, 0].legend(frameon=False)
        axes[1, 2].legend(frameon=False)
        save_figure(fig, output / "training_curves")
        plt.close(fig)


def plot_test(summary, constraints, output):
    plt, settings = plotting()
    policies = [name for name in LABELS if name in set(summary.policy)]
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65), layout="constrained")
        for ax, metric in zip(axes, list(METRICS)[:3]):
            values = summary[summary.metric == metric].set_index("policy").loc[policies]
            estimates, low, high = (values[key].to_numpy(dtype=float) for key in ("estimate", "ci95_low", "ci95_high"))
            if metric == "demand_satisfaction":
                estimates, low, high = estimates*100, low*100, high*100
            positions = np.arange(len(policies))
            for i, policy in enumerate(policies):
                ax.errorbar(i, estimates[i], yerr=[[max(0., estimates[i]-low[i])], [max(0., high[i]-estimates[i])]],
                            fmt="o", color=COLORS[policy], markersize=5, elinewidth=1.1, capsize=3)
            ax.set_xticks(positions, [SHORT_LABELS[name] for name in policies], fontsize=7)
            ax.set_xlim(-.5, len(policies)-.5)
            title = "Demand coverage (%) ↑" if metric == "demand_satisfaction" else METRICS[metric][0]
            ax.set_title(title, loc="left", fontsize=8)
            ax.grid(axis="y", alpha=.18, lw=.5)
            if metric == "demand_satisfaction":
                ax.axhline(100*constraints["service_floor"], color="#646A70", ls="--", lw=.9)
            elif metric == "wait_per_demand_seconds":
                ax.axhline(constraints["wait_limit_seconds_per_demand"], color="#646A70", ls="--", lw=.9)
        save_figure(fig, output / "test_comparison")
        plt.close(fig)


def run(output):
    output = Path(output)
    plan = json.loads((output / "protocol.json").read_text())
    constraints = plan["constraints"]
    training = output / "training_history.csv"
    if training.exists():
        frame = pd.read_csv(training)
        if len(frame):
            plot_training(frame, constraints, output)
    evaluation = output / "evaluation.csv"
    if not evaluation.exists():
        return {"status": "training"}
    frame = pd.read_csv(evaluation)
    test = frame[frame.phase == "test"]
    if test.empty:
        return {"status": "training"}
    expected = plan.get("seeds", {}).get("test")
    if (set(test.policy) != set(LABELS) or (expected is not None and any(
            set(group.seed) != set(expected) for _, group in test.groupby("policy")))):
        return {"status": "test_incomplete"}
    summary, paired = summarize(test)
    summary.to_csv(output / "policy_summary.csv", index=False)
    paired.to_csv(output / "paired_statistics.csv", index=False)
    result = {"status": "complete", "scope": "One frozen initialization; equal-budget fine-tuning pilot, not a from-scratch multi-seed benchmark.",
              "constraint_estimand": "E[served baseline demand lineages]/E[cohort baseline demand], E[all physical orders' queue wait seconds]/E[cohort baseline demand]",
              "uncertainty": "10000 paired environment-seed bootstrap draws; marginal 95% intervals, without multiplicity adjustment or training-seed uncertainty.",
              "constraints": constraints, "feasibility": feasibility(summary, constraints)}
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    plot_test(summary, constraints, output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(run(parser.parse_args().output), ensure_ascii=False, indent=2))
