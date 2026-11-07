"""检验当前动作未改变物理状态时，策略是否仍被记录中的历史动作影响。"""

from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

import diagnostics.diagnose_offline_value as experiment
from diagnostics.analyze_offline_value import bootstrap, KEYS
from offline_rl.data import save_json, sha256


def plot_results(root, frame, summary, predicted):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
        grouped = frame.groupby("seed").next_joint_action_differs.sum()
        axes[0].bar(np.arange(6), grouped, color="#0072B2", width=.6)
        axes[0].axhline(6, color=".5", ls="--", lw=.8)
        axes[0].set_xticks(np.arange(6), [f"S{i+1}" for i in range(6)])
        axes[0].set_ylim(0, 6.6)
        axes[0].set_yticks(np.arange(7))
        axes[0].set_xlabel("Past seed (six future streams each)")
        axes[0].set_ylabel("Different next joint actions / 6")
        axes[0].set_title("(a) Same physical state; different radius history", loc="left", fontsize=10)
        entries = summary["remaining_js_minus_f3_gap"]
        means = np.array([predicted["mean"], entries["f3"]["mean"], entries["js"]["mean"]])
        bounds = np.array([predicted["ci95"], entries["f3"]["ci95"], entries["js"]["ci95"]])
        axes[1].bar(np.arange(3), means, color=["#0072B2", "#D55E00", "#009E73"], width=.6)
        axes[1].errorbar(np.arange(3), means, yerr=np.stack((means-bounds[:, 0], bounds[:, 1]-means)),
                         fmt="none", color=".2", capsize=4, lw=1)
        axes[1].axhline(0, color=".4", lw=.8)
        axes[1].set_xticks(np.arange(3), ["STQ prediction", "F3 continuation", "JS continuation"], fontsize=8)
        axes[1].set_ylabel("JS − F3 remaining revenue (USD) ↑")
        axes[1].set_title("(b) Effect after the common physical transition", loc="left", fontsize=10)
        for ax in axes:
            ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(root/f"policy_memory_effect.{ext}", dpi=600, facecolor="white")
        plt.close(fig)


def assert_equal(a, b):
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b, check_exact=True)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert pickle.dumps(a, protocol=5) == pickle.dumps(b, protocol=5)


def run():
    root = experiment.OUTPUT
    protocol = json.loads((root/"protocol.json").read_text())
    experiment.initialize(protocol["config"])
    records = []
    for entry in json.loads((root/"snapshots.json").read_text()):
        if entry["minute_step"] != 20:
            continue
        with Path(entry["path"]).open("rb") as stream:
            snapshot = pickle.load(stream)
        for repetition in range(protocol["future_streams_per_state"]):
            future_seed = int(np.random.SeedSequence([7300702, entry["seed"], 20, repetition]).generate_state(1)[0])
            branches = []
            for candidate in ("js", "f3"):
                sim = deepcopy(snapshot["sim"])
                sim.rng = np.random.RandomState(future_seed)
                recent = experiment.period_step(sim, experiment.action_mapping(snapshot["actions"][candidate]), snapshot["metadata"])
                periods = pd.concat((snapshot["periods"], recent), ignore_index=True)
                state = experiment.observation(periods, sim)
                with torch.no_grad():
                    next_action = experiment._POLICIES["stq"].actor.act(state)[0].tolist()
                branches.append((sim, state.numpy(), next_action))
            (js, js_state, js_next), (f3, f3_state, f3_next) = branches
            ignored = {"grid_action", "_radius_by_grid", "evaluate_df", "evaluate_table"}
            assert vars(js).keys() == vars(f3).keys()
            for key in vars(js).keys()-ignored:
                assert_equal(getattr(js, key), getattr(f3, key))
            columns = [name for name in js.evaluate_df.columns if name != "matching_radius_km"]
            pd.testing.assert_frame_equal(js.evaluate_df[columns], f3.evaluate_df[columns], check_exact=True)
            indices = [js.evaluate_df.columns.get_loc(name) for name in columns]
            np.testing.assert_array_equal(js.evaluate_table[:, :, indices], f3.evaluate_table[:, :, indices])
            changed = np.flatnonzero(js_state[0] != f3_state[0])
            expected = np.arange(6)*4*24+3*24+23
            np.testing.assert_array_equal(changed, expected)
            records.append({"seed": entry["seed"], "minute_step": 20, "repetition": repetition,
                "same_physical_state_and_rng": True, "only_six_last_radius_inputs_differ": True,
                "next_joint_action_differs": js_next != f3_next,
                "next_grid_actions_differ": int(np.sum(np.array(js_next) != f3_next)),
                "js_history_next_action": json.dumps(js_next), "f3_history_next_action": json.dumps(f3_next)})
    frame = pd.DataFrame(records)
    frame.to_csv(root/"memory_interventions.csv", index=False)
    paired = pd.read_csv(root/"paired_contrasts.csv")
    paired = paired[(paired.contrast == "js_minus_f3") & (paired.minute_step == 20)]
    summary = {"scope": "mechanism diagnostic on six development-validation warmup states and six future streams each; not an intervention banning historical actions",
        "paired_first_periods": len(frame), "identical_physics_and_rng_pairs": int(frame.same_physical_state_and_rng.sum()),
        "only_last_radius_history_differs_pairs": int(frame.only_six_last_radius_inputs_differ.sum()),
        "next_joint_action_differs_pairs": int(frame.next_joint_action_differs.sum()),
        "different_next_grid_actions": int(frame.next_grid_actions_differ.sum()),
        "total_next_grid_actions": len(frame)*6,
        "remaining_js_minus_f3_gap": {c: bootstrap(part.set_index(KEYS).return_usd) for c, part in paired.groupby("continuation")},
        "interpretation": "At these states both actions produce identical physical transitions. Their recorded past radii change the next controller input and can change later actions. This identifies a controller-memory path, not a direct physical radius effect, and does not by itself prove that all use of past actions is undesirable.",
        "source_sha256": sha256(__file__)}
    save_json(root/"memory_diagnostic.json", summary)
    predicted = bootstrap(paired[paired.continuation == "js"].set_index(KEYS).stq_q_usd)
    plot_results(root, frame, summary, predicted)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
