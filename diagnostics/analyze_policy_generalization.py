"""对照三个种子组，定位协同策略相对 F3 的收益、动作与排队差异。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from offline_rl.data import save_json, sha256


GROUPS = {"screen": ("validation/step005000", "conditional9"),
          "confirmation": ("confirmation_validation", "candidate_0"),
          "test": ("test", "optimized")}


def interval(values):
    values = np.asarray(values)
    means = np.random.default_rng(42).choice(values, (10000, len(values)), replace=True).mean(1)
    return np.quantile(means, [.025, .975])


def analyze(model_root, output):
    model_root, output = Path(model_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    outcomes, actions, summaries, episode_rows = [], [], {}, []
    for group, (relative, candidate) in GROUPS.items():
        folder = model_root / relative
        manifest = json.loads((folder / "manifest.json").read_text())
        frame = pd.read_csv(folder / "run_summary.csv")
        frame = frame[frame.policy.isin(["fixed_f3", candidate])].copy()
        frame["policy"] = frame.policy.replace({candidate: "candidate", "fixed_f3": "f3"})
        frame["group"] = group
        episode_rows.append(frame)
        paired = frame.pivot(index="seed", columns="policy", values="platform_revenue")
        difference = paired.candidate - paired.f3
        ci = interval(difference)
        summaries[group] = {
            "seed_count": len(paired), "candidate_mean": float(paired.candidate.mean()),
            "f3_mean": float(paired.f3.mean()), "difference_mean": float(difference.mean()),
            "difference_std": float(difference.std()), "ci95": ci.tolist(),
            "wins": int(difference.gt(0).sum()),
            "manifest_sha256": sha256(folder / "manifest.json"),
            "config": manifest["config"],
        }
        for seed in manifest["seeds"]:
            for name, label in ((candidate, "candidate"), ("fixed_f3", "f3")):
                directory = folder / "runs" / f"seed{seed}" / name
                d = pd.read_csv(directory / "decision_metrics.csv.gz")
                d = d[d.decision_start_minute_step >= 20].copy()
                observed = frame[(frame.seed == seed) & (frame.policy == label)].platform_revenue.iloc[0]
                assert len(d) == 336 and np.isclose(observed, d.platform_revenue.sum())
                d["group"], d["policy"], d["hour"] = group, label, 5 + d.decision_start_minute_step // 60
                outcomes.append(d)
                a = pd.read_csv(directory / "action_plan.csv.gz")
                a = a[~a.warmup].copy()
                a["group"], a["policy"], a["seed"] = group, label, seed
                actions.append(a)
    d, a, episodes = pd.concat(outcomes), pd.concat(actions), pd.concat(episode_rows)
    episodes.to_csv(output / "group_seed_metrics.csv", index=False)
    d.to_csv(output / "grid_period_metrics.csv.gz", index=False)
    a.to_csv(output / "action_choices.csv.gz", index=False)
    gaps = {}
    for key in ("grid_id", "hour"):
        rows = []
        for (group, value), part in d.groupby(["group", key]):
            pair = part.groupby(["seed", "policy"]).platform_revenue.sum().unstack()
            diff = pair.candidate - pair.f3
            ci = interval(diff)
            rows.append({"group": group, key: int(value), "difference_usd": diff.mean(),
                         "ci95_low": ci[0], "ci95_high": ci[1]})
        gaps[key] = pd.DataFrame(rows)
        gaps[key].to_csv(output / f"{key}_revenue_gaps.csv", index=False)
        for group in GROUPS:
            assert np.isclose(gaps[key].query("group == @group").difference_usd.sum(), summaries[group]["difference_mean"])
    frequency = a.groupby(["group", "policy", "price_multiplier", "matching_radius_km"]).size().rename("count").reset_index()
    frequency["share"] = frequency["count"] / frequency.groupby(["group", "policy"])["count"].transform("sum")
    frequency.to_csv(output / "action_frequency.csv", index=False)
    # 每个种子先将六区域状态相加，再对周期与种子求平均。
    city_queue = d.groupby(["group", "policy", "seed", "decision_step"]).start_waiting_request_num.sum().reset_index()
    city_queue["hour"] = 5 + city_queue.decision_step * 5 // 60
    city_queue.groupby(["group", "policy", "hour"]).start_waiting_request_num.mean().to_csv(output / "hour_queue.csv")
    for group in GROUPS:
        selected = a[(a.group == group) & (a.policy == "candidate")]
        summaries[group].update(
            mean_price=float(selected.price_multiplier.mean()), mean_radius=float(selected.matching_radius_km.mean()),
            radius_1p5_share=float(selected.matching_radius_km.eq(1.5).mean()),
            price_1p4_share=float(selected.price_multiplier.eq(1.4).mean()),
            f3_action_share=float((selected.price_multiplier.eq(1.6) & selected.matching_radius_km.eq(2.)).mean()))
    m = episodes[episodes.group == "test"].groupby("policy").mean(numeric_only=True)
    n0, n1 = m.loc["f3", "matched_request_num"], m.loc["candidate", "matched_request_num"]
    r0, r1 = m.loc["f3", "platform_revenue"] / n0, m.loc["candidate", "platform_revenue"] / n1
    decomposition = {"quantity_component_usd": float((n1 - n0) * r0),
                     "revenue_per_match_component_usd": float(n1 * (r1 - r0)),
                     "f3_revenue_per_match_usd": float(r0), "candidate_revenue_per_match_usd": float(r1),
                     "interpretation": "Accounting identity only; does not identify causal price or radius effects."}
    summary = {"groups": summaries, "test_metrics": m.to_dict(orient="index"),
               "test_revenue_accounting": decomposition,
               "limitations": ["Screen confidence interval is not adjusted for candidate selection.",
                    "Grid/hour outcome differences are descriptive, not causal agent attribution.",
                    "All groups use the same small-scale scenario and order date, with different seeds.",
                    "Earlier test seeds are diagnostic only; subsequent model choices use validation seeds."]}
    save_json(output / "summary.json", summary)
    plot(output, summaries, gaps, city_queue)
    return summary


def plot(output, summaries, gaps, queue):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from baseline.plot_paper_figures import _font_family

    colors = {"screen": "#009E73", "confirmation": "#E69F00", "test": "#0072B2"}
    labels = {"screen": "Selection (5 seeds)", "confirmation": "Confirmation (20)", "test": "Test (20)"}
    with plt.rc_context({"font.family": _font_family(), "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
        fig.subplots_adjust(left=.09, right=.98, bottom=.13, top=.95, hspace=.42, wspace=.32)
        ax = axes[0, 0]
        for i, (name, s) in enumerate(summaries.items()):
            mean, ci = s["difference_mean"], s["ci95"]
            ax.bar(i, mean, color=colors[name], width=.6)
            ax.errorbar(i, mean, yerr=[[mean-ci[0]], [ci[1]-mean]], color="black", fmt="none", capsize=3, linewidth=.8)
        ax.set_xticks(range(3), ["Selection\n5 seeds", "Confirmation\n20 seeds", "Test\n20 seeds"])
        ax.set_title("(a) Revenue difference across seed groups ↑", loc="left")
        ax.set_ylabel("Candidate − F3 (USD / episode)")
        for key, ax, title in (("grid_id", axes[0, 1], "(b) Revenue difference by grid ↑"),
                               ("hour", axes[1, 0], "(c) Revenue difference by time ↑")):
            values = sorted(gaps[key][key].unique())
            for i, group in enumerate(GROUPS):
                rows = gaps[key].query("group == @group").set_index(key).loc[values]
                ax.bar(np.arange(len(values)) + (i-1)*.24, rows.difference_usd, width=.23,
                       color=colors[group], label=labels[group])
            ax.set_xticks(range(len(values)), [str(v) if key == "grid_id" else f"{v:02d}–{v+1:02d}" for v in values])
            ax.set_title(title, loc="left")
            ax.set_xlabel("Grid ID" if key == "grid_id" else "Simulation hour")
            ax.set_ylabel("Candidate − F3 (USD)")
        axes[0, 1].legend(frameon=False, fontsize=7)
        ax = axes[1, 1]
        for policy, label, color in (("f3", "F3", "#777777"), ("candidate", "Candidate", "#0072B2")):
            part = queue[(queue.group == "test") & (queue.policy == policy)].groupby("decision_step").start_waiting_request_num.mean()
            ax.plot((300 + part.index*5)/60, part, color=color, label=label, linewidth=1.1)
        ax.set_title("(d) Pre-decision queue in new tests ↓", loc="left")
        ax.set(xlabel="Simulation hour", ylabel="Citywide waiting requests")
        ax.legend(frameon=False, fontsize=8)
        for ax in axes.flat:
            ax.grid(axis="y", color="#DDDDDD", linewidth=.4)
            if ax is not axes[1, 1]:
                ax.axhline(0, color="#555555", linewidth=.6)
        fig.text(.5, .025, "Same low-density scenario; paired seeds. Panel (a): pointwise 95% bootstrap CIs, unadjusted for selection.", ha="center", fontsize=8)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(output / f"generalization_diagnostic.{ext}", dpi=600, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="协同策略在不同种子组中的优势稳定性诊断")
    parser.add_argument("--model-root", default="models/iql_coordination_20261102")
    parser.add_argument("--output", default="experiments/iql_generalization_analysis_20261102")
    args = parser.parse_args()
    result = analyze(args.model_root, args.output)
    print(json.dumps({name: {k: v for k, v in value.items() if k != "config"}
                      for name, value in result["groups"].items()}, ensure_ascii=False, indent=2))
