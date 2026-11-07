"""汇总同状态分支回放，分别比较两种后续策略下的动作收益差。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from offline_rl.data import save_json, sha256


ROOT = Path("experiments/rl_research_diagnosis_20261103")
KEYS = ["seed", "minute_step", "repetition"]
CONTRASTS = {"js_minus_f3": ("js", "f3"), "eq_minus_f3": ("eq", "f3"),
             "radius2_minus_js": ("same_price_radius2", "js"),
             "single_radius2_minus_js": ("single_grid_radius2", "js")}
LABELS = {"js_minus_f3": "JS − F3", "eq_minus_f3": "EQ − F3",
          "radius2_minus_js": "All radii 2 km − JS", "single_radius2_minus_js": "One radius 2 km − JS"}


def bootstrap(values):
    """先抽完整历史种子，再抽每个状态的未来流；三个时刻保留在同一簇内。"""
    values = values.sort_index()
    seeds = sorted(values.index.get_level_values("seed").unique())
    steps = sorted(values.index.get_level_values("minute_step").unique())
    streams = sorted(values.index.get_level_values("repetition").unique())
    expected = pd.MultiIndex.from_product((seeds, steps, streams), names=KEYS)
    array = values.reindex(expected).to_numpy().reshape(len(seeds), len(steps), len(streams))
    assert np.isfinite(array).all()
    rng = np.random.default_rng(7300704)
    outer = rng.integers(len(seeds), size=(10000, len(seeds)))
    inner = rng.integers(len(streams), size=(10000, len(seeds), len(steps), len(streams)))
    samples = array[outer[:, :, None, None], np.arange(len(steps))[None, None, :, None], inner].mean((1, 2, 3))
    return {"mean": float(array.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist()}


def contrasts(frame):
    rows = []
    for continuation, subset in frame.groupby("continuation"):
        wide = subset.pivot(index=KEYS, columns="candidate")
        for name, (candidate, reference) in CONTRASTS.items():
            row = wide.xs(candidate, axis=1, level="candidate").select_dtypes("number")-wide.xs(reference, axis=1, level="candidate").select_dtypes("number")
            row["continuation"], row["contrast"] = continuation, name
            rows.append(row.reset_index())
    return pd.concat(rows, ignore_index=True)


def plot_results(paired, summary, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"stq": "#0072B2", "f3": "#D55E00", "js": "#009E73"}
    style = {"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
             "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"}

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(root/f"{name}.{ext}", dpi=600, facecolor="white")
        plt.close(fig)

    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
        subset = paired[paired.contrast == "js_minus_f3"]
        order = sorted(set(zip(subset.minute_step, subset.seed)))
        for ax, continuation, title in zip(axes, ("f3", "js"), ("(a) F3 continuation", "(b) JS continuation")):
            part = subset[subset.continuation == continuation]
            grouped = part.groupby(["minute_step", "seed"])
            means = grouped.return_usd.mean().reindex(order)
            sem = grouped.return_usd.sem().reindex(order)
            predicted = grouped.stq_q_usd.first().reindex(order)
            x = np.arange(len(order))
            ax.axhline(0, color=".4", lw=.8)
            ax.errorbar(x, means, yerr=2.571*sem, fmt="o", ms=3, color=colors[continuation],
                        lw=.7, capsize=2, label="Paired simulator return (95% t interval)")
            ax.plot(x, predicted, "s--", ms=3, lw=.9, color=colors["stq"], label="STQ predicted Q difference")
            ax.set_title(title, loc="left")
            ax.set_xticks([2.5, 8.5, 14.5], ["05:20", "07:00", "08:40"])
            ax.set_xlabel("Snapshot time (six past seeds per time)")
            ax.grid(axis="y", alpha=.2)
            ax.legend(frameon=False, fontsize=7, loc="best")
        axes[0].set_ylabel("JS − F3 remaining revenue (USD) ↑")
        save(fig, "value_ranking_by_state")

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
        x = np.arange(len(CONTRASTS))
        for ax, continuation, title in zip(axes, ("f3", "js"), ("(a) F3 continuation", "(b) JS continuation")):
            rows = summary["contrasts"][continuation]
            for offset, column, label, color in ((-.17, "stq_q_usd", "Predicted STQ Q gap", colors["stq"]),
                    (.17, "return_usd", "Paired simulator return gap", colors[continuation])):
                center = np.array([rows[name][column]["mean"] for name in CONTRASTS])
                bounds = np.array([rows[name][column]["ci95"] for name in CONTRASTS])
                ax.bar(x+offset, center, width=.32, color=color, alpha=.85, label=label)
                ax.errorbar(x+offset, center, yerr=np.stack((center-bounds[:, 0], bounds[:, 1]-center)),
                            fmt="none", color=".2", lw=.8, capsize=3)
            ax.axhline(0, color=".4", lw=.8)
            ax.set_xticks(x, [LABELS[name].replace(" − ", "\n− ") for name in CONTRASTS], fontsize=8)
            ax.set_title(title, loc="left")
            ax.grid(axis="y", alpha=.2)
            ax.legend(frameon=False, fontsize=7)
        axes[0].set_ylabel("Mean remaining revenue difference (USD) ↑")
        save(fig, "value_ranking_summary")

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
        for ax, model, empirical, title in zip(axes, ("stq", "eq"), ("return_usd", "return_discount99_usd"),
                ("(a) STQ: undiscounted return", "(b) EQ: discounted return, γ = 0.99")):
            for continuation, marker in (("f3", "o"), ("js", "s")):
                part = paired[(paired.continuation == continuation) & paired.contrast.isin(CONTRASTS)]
                grouped = part.groupby(["seed", "minute_step", "contrast"])
                ax.scatter(grouped[f"{model}_q_usd"].first(), grouped[empirical].mean(), s=16,
                           alpha=.65, marker=marker, color=colors[continuation], label=f"{continuation.upper()} continuation")
            ax.axvline(0, color=".6", lw=.6)
            ax.axhline(0, color=".6", lw=.6)
            ax.axline((0, 0), slope=1, color=".4", lw=.8, ls="--")
            ax.set_xlabel("Predicted action-value difference (USD)")
            ax.set_ylabel("Paired simulator return difference (USD)")
            ax.set_title(title, loc="left")
            ax.grid(alpha=.15)
            ax.legend(frameon=False, fontsize=7)
        save(fig, "value_gap_calibration")


def run(root=ROOT):
    root = Path(root)
    status = json.loads((root/"status.json").read_text())
    assert status["status"] == "complete" and status["checkpoints_unchanged"]
    frame = pd.read_csv(root/"branch_returns.csv")
    protocol = json.loads((root/"protocol.json").read_text())
    assert not frame.duplicated(KEYS+["candidate", "continuation"]).any()
    assert len(frame) == len(protocol["seeds"])*len(protocol["minute_steps"])*protocol["future_streams_per_state"]*10
    numerical = frame.select_dtypes("number")
    assert np.isfinite(numerical.to_numpy()).all()
    paired = contrasts(frame)
    paired.to_csv(root/"paired_contrasts.csv", index=False)
    summary = {"scope": protocol["scope"], "observed_states": status["states"],
        "independent_past_seeds": len(protocol["seeds"]), "future_streams_per_state": protocol["future_streams_per_state"],
        "inference": "paired gaps; nested bootstrap resamples entire past-seed clusters, then future streams within each state; only six independent past seeds",
        "warning": "Q is the IQL implicit value, not the Q of either continuation. This study tests deployment-relevant ranking and sensitivity; it does not isolate a single cause.",
        "contrasts": {}, "ranking": {}, "continuation_dependence": {}, "exploratory_interactions": {}, "ensemble_diagnostic": {}}
    metrics = ["stq_q_usd", "eq_q_usd", "return_usd", "return_discount99_usd", "first5_usd", "first20_usd", "pickup_cancellations", "matched_requests"]
    for continuation, subset in paired.groupby("continuation"):
        summary["contrasts"][continuation] = {}
        summary["ranking"][continuation] = {}
        for name, rows in subset.groupby("contrast"):
            summary["contrasts"][continuation][name] = {metric: bootstrap(rows.set_index(KEYS)[metric]) for metric in metrics}
            states = rows.groupby(["seed", "minute_step"])[metrics].mean()
            summary["ranking"][continuation][name] = {f"{model}_sign_agreement": int((np.sign(states[f"{model}_q_usd"])*np.sign(states[empirical]) > 0).sum())
                for model, empirical in (("stq", "return_usd"), ("eq", "return_discount99_usd"))}
            summary["ranking"][continuation][name].update({"states": len(states),
                "empirical_tied_states": int(np.isclose(states.return_usd, 0, atol=1e-8).sum()),
                "positive_empirical_gap_states": int((states.return_usd > 1e-8).sum())})
    for name, subset in paired.groupby("contrast"):
        wide = subset.pivot(index=KEYS, columns="continuation", values="return_usd")
        summary["continuation_dependence"][name] = bootstrap(wide.js-wide.f3)
    snapshots = json.loads((root/"snapshots.json").read_text())
    members = pd.read_csv(root/"critic_members.csv")
    for name, part in members.groupby("model"):
        wide = part.pivot(index=["seed", "minute_step", "member"], columns="candidate", values="q_usd")
        gaps = wide.js-wide.f3
        stats = gaps.groupby(["seed", "minute_step"]).agg(["mean", "std"])
        stats["std"] *= np.sqrt(2/3)
        summary["ensemble_diagnostic"][name] = {"mean_js_minus_f3": float(stats["mean"].mean()),
            "mean_member_std_of_gap": float(stats["std"].mean()),
            "all_members_agree_on_gap_sign_states": int(gaps.groupby(["seed", "minute_step"]).apply(lambda x: (x > 0).all() or (x < 0).all()).sum()),
            "note": "Three fitted critics are a heuristic ensemble; their dispersion is not a calibrated confidence interval."}
    for continuation, subset in paired.groupby("continuation"):
        wide = subset.pivot(index=KEYS, columns="contrast", values="return_usd")
        estimated = wide.radius2_minus_js-6*wide.single_radius2_minus_js
        summary["exploratory_interactions"][continuation] = {"random_grid_simulator_estimate": bootstrap(estimated),
            "predicted_stq_mean": float(np.mean([s["predicted_interactions"]["stq"]["synergy_usd"] for s in snapshots])),
            "note": "All-grid effect minus six times one random grid effect is an unbiased randomized-grid estimator in expectation. With 18 random grid assignments, its variance is high; not a per-state interaction ground truth."}
    first = frame.pivot(index=KEYS+["candidate"], columns="continuation", values="first5_usd")
    assert np.array_equal(first.f3.to_numpy(), first.js.to_numpy())
    config = protocol["config"]
    pairs = [[p, r] for p in config["price_action_values"] for r in config["matching_radius_values"]]
    for (seed, step), rows in frame[frame.candidate == "js"].groupby(["seed", "minute_step"]):
        path = Path("models/structured_joint_search_20261103/validation_new/runs")/f"seed{seed}/joint_start/action_plan.csv.gz"
        plan = pd.read_csv(path)
        expected = plan[plan.decision_step == step//config["decision_freq"]].sort_values("grid_id")
        indices = json.loads(rows.action_vector.iloc[0])
        np.testing.assert_allclose(np.asarray(pairs)[indices], expected[["price_multiplier", "matching_radius_km"]].to_numpy(), atol=1e-6, rtol=0)
    checks = {"complete_balanced_paired_design": True, "no_nonfinite_results": True,
        "snapshot_prefixes_verified": all(row["prefix_replay_verified"] for row in snapshots),
        "snapshot_files_unchanged": all(sha256(row["path"]) == row["sha256"] for row in snapshots),
        "current_js_decisions_equal_recorded_decisions": True,
        "identical_first_period_across_continuations": True,
        "checkpoints_unchanged": status["checkpoints_unchanged"],
        "offline_sources_unchanged": all(sha256(path) == value for path, value in protocol["source_sha256"].items()),
        "simulator_sources_unchanged": all(sha256(path) == value for path, value in protocol["simulator_sources_sha256"].items()),
        "diagnostic_source_unchanged": sha256("diagnostics/diagnose_offline_value.py") == protocol["diagnostic_sha256"],
        "analysis_source_sha256": sha256(__file__)}
    assert all(value for key, value in checks.items() if key != "analysis_source_sha256")
    save_json(root/"verification.json", checks)
    save_json(root/"diagnosis_statistics.json", summary)
    plot_results(paired, summary, root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
