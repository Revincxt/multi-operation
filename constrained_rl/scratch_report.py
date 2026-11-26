"""从随机初始化训练的约束PPO报告；保留原试训报告及其统计口径。"""

import argparse
import json
from pathlib import Path

import pandas as pd

from . import report


METHODS = ("revenue_control", "constrained")
SERVICE_COLORS = ("#8A6B98", "#C49763")


def plot_training(frame, constraints, output):
    """记录完整训练进程；单次更新也显示数据点。"""
    plt, settings = report.plotting()
    with plt.rc_context(settings):
        fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.35), layout="constrained")
        for method in METHODS:
            group = frame[frame.method == method].sort_values("iteration")
            if group.empty:
                continue
            marker = "o" if len(group) == 1 else None
            style = {"lw": 1.2, "marker": marker, "markersize": 3}
            for ax, column in zip(axes[0], ("revenue_mean_usd", "demand_satisfaction", "wait_per_demand_seconds")):
                ax.plot(group.episodes, group[column], color=report.COLORS[method],
                        label=report.LABELS[method], **style)
            axes[1, 1].plot(group.episodes, group.joint_kl, color=report.COLORS[method], **style)
            if method == "constrained":
                for ax, columns in ((axes[1, 0], ("lambda_service", "lambda_wait")),
                                    (axes[1, 2], ("cost_service_mean", "cost_wait_mean"))):
                    for column, color, label in zip(columns, SERVICE_COLORS, ("Service", "Waiting")):
                        ax.plot(group.episodes, group[column], color=color, label=label, **style)
        axes[0, 1].axhline(constraints["service_floor"], color="#646A70", ls="--", lw=.9)
        axes[0, 2].axhline(constraints["wait_limit_seconds_per_demand"], color="#646A70", ls="--", lw=.9)
        axes[1, 2].axhline(0., color="#646A70", ls="--", lw=.9)
        titles = ("Return (USD) ↑", "Baseline demand coverage ↑", "Queue wait / demand (s) ↓",
                  "Lagrange multipliers", "Joint KL", "Normalized constraint excess")
        for ax, title in zip(axes.flat, titles):
            ax.set_title(title, loc="left", fontsize=8)
            ax.set_xlabel("Training episodes")
            ax.ticklabel_format(axis="y", useOffset=False)
            ax.grid(axis="y", alpha=.18, lw=.5)
        for ax in (axes[0, 0], axes[1, 0], axes[1, 2]):
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False)
        report.save_figure(fig, output / "training_curves")
        plt.close(fig)


def validation_history(evaluation, training):
    """验证集只用于记录过程；服务指标沿用总分子/总分母。"""
    selected = evaluation[(evaluation.phase == "validation") & evaluation.policy.isin(METHODS)]
    if selected.empty:
        return pd.DataFrame()
    if selected.duplicated(["policy", "iteration", "seed"]).any():
        raise ValueError
    rows = []
    for (policy, iteration), group in selected.groupby(["policy", "iteration"], sort=True):
        progress = training[(training.method == policy) & (training.iteration == iteration)]
        if progress.empty:
            continue
        if len(progress) != 1:
            raise ValueError
        row = {"method": policy, "iteration": iteration,
               "episodes": int(progress.iloc[0].episodes), "scenarios": len(group)}
        for metric in ("platform_revenue", "demand_satisfaction", "wait_per_demand_seconds"):
            numerator, denominator = report.metric_values(group, metric)
            row[metric] = report.estimate(numerator, denominator)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_validation(frame, constraints, output):
    plt, settings = report.plotting()
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65), layout="constrained")
        for method in METHODS:
            group = frame[frame.method == method].sort_values("episodes")
            if group.empty:
                continue
            for ax, metric in zip(axes, list(report.METRICS)[:3]):
                ax.plot(group.episodes, group[metric], color=report.COLORS[method], lw=1.2,
                        marker="o" if len(group) == 1 else None, markersize=3,
                        label=report.LABELS[method])
        axes[1].axhline(constraints["service_floor"], color="#646A70", ls="--", lw=.9)
        axes[2].axhline(constraints["wait_limit_seconds_per_demand"], color="#646A70", ls="--", lw=.9)
        for ax, metric in zip(axes, list(report.METRICS)[:3]):
            ax.set_title(report.METRICS[metric][0], loc="left", fontsize=8)
            ax.set_xlabel("Training episodes")
            ax.ticklabel_format(axis="y", useOffset=False)
            ax.grid(axis="y", alpha=.18, lw=.5)
        axes[0].legend(frameon=False)
        report.save_figure(fig, output / "validation_curves")
        plt.close(fig)


def run(output):
    """封存测试种子及策略全部到齐后，生成最终配对统计。"""
    output = Path(output)
    plan = json.loads((output / "protocol.json").read_text())
    constraints = plan["constraints"]
    policies = plan.get("expected_test_policies", list(report.LABELS))
    seeds = plan.get("seeds", {}).get("test")
    if (not policies or len(set(policies)) != len(policies)
            or not set(policies).issubset(report.LABELS) or seeds is None
            or len(seeds) < 2 or len(set(seeds)) != len(seeds)):
        raise ValueError
    training = pd.DataFrame()
    path = output / "training_history.csv"
    if path.exists():
        training = pd.read_csv(path)
        if not training.empty:
            plot_training(training, constraints, output)
    path = output / "evaluation.csv"
    if not path.exists():
        return {"status": "training"}
    evaluation = pd.read_csv(path)
    if not training.empty:
        validation = validation_history(evaluation, training)
        if not validation.empty:
            validation.to_csv(output / "validation_history.csv", index=False)
            plot_validation(validation, constraints, output)
    test = evaluation[evaluation.phase == "test"]
    if test.empty:
        return {"status": "training"}
    if (set(test.policy) != set(policies) or test.duplicated(["policy", "seed"]).any()
            or any(set(group.seed) != set(seeds) for _, group in test.groupby("policy"))):
        return {"status": "test_incomplete"}
    summary, paired = report.summarize(test)
    summary.to_csv(output / "policy_summary.csv", index=False)
    paired.to_csv(output / "paired_statistics.csv", index=False)
    result = {
        "status": "complete",
        "scope": "From-scratch training of revenue and constrained PPO with equal episode budgets; one training seed per method, not a multi-seed training benchmark.",
        "initialization": "Both learned policies start from random weights. Frozen AR-PPO is a previously trained reference, not their initialization.",
        "checkpoint_selection": "Final fixed-budget checkpoints; validation and sealed test results are not used to select checkpoints.",
        "expected_test_policies": policies,
        "test_scenarios": len(seeds),
        "constraint_estimand": "E[served baseline demand lineages]/E[cohort baseline demand], E[all physical orders' queue wait seconds]/E[cohort baseline demand]",
        "uncertainty": "10000 paired environment-seed bootstrap draws; marginal 95% intervals, without multiplicity adjustment or training-seed uncertainty. Calibration thresholds are treated as fixed.",
        "constraints": constraints,
        "feasibility": report.feasibility(summary, constraints),
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    report.plot_test(summary, constraints, output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(run(parser.parse_args().output), ensure_ascii=False, indent=2))
