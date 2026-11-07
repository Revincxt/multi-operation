"""在已验证物理等价的历史对上检查新策略；不用于模型选择。"""

from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

import diagnostics.diagnose_offline_value as diagnostic
from diagnostics.diagnose_policy_memory import assert_equal
from offline_rl.data import save_json, sha256
from offline_rl.evaluate import Policy
from offline_rl.train_conservative import ROOT


def prepare_states():
    source = diagnostic.OUTPUT
    protocol = json.loads((source/"protocol.json").read_text())
    diagnostic.initialize(protocol["config"])
    rows, states = [], []
    for entry in json.loads((source/"snapshots.json").read_text()):
        if entry["minute_step"] != 20:
            continue
        with Path(entry["path"]).open("rb") as stream:
            snapshot = pickle.load(stream)
        for repetition in range(6):
            future = int(np.random.SeedSequence([7300702, entry["seed"], 20, repetition]).generate_state(1)[0])
            branches = []
            for name in ("js", "f3"):
                sim = deepcopy(snapshot["sim"])
                sim.rng = np.random.RandomState(future)
                recent = diagnostic.period_step(sim, diagnostic.action_mapping(snapshot["actions"][name]), snapshot["metadata"])
                periods = pd.concat((snapshot["periods"], recent), ignore_index=True)
                state = diagnostic.observation(periods, sim).numpy()[0]
                branches.append((sim, state))
            for key in vars(branches[0][0]).keys()-{"grid_action", "_radius_by_grid", "evaluate_df", "evaluate_table"}:
                assert_equal(getattr(branches[0][0], key), getattr(branches[1][0], key))
            np.testing.assert_array_equal(np.flatnonzero(branches[0][1] != branches[1][1]), np.arange(6)*96+95)
            states.append(np.stack((branches[0][1], branches[1][1])))
            rows.append({"seed": entry["seed"], "repetition": repetition})
    return torch.from_numpy(np.stack(states)), rows


def run(root=ROOT):
    root = Path(root)
    torch.set_num_threads(1)
    states, rows = prepare_states()
    definitions = {"joint_start": diagnostic.CHECKPOINTS["stq"],
                   **{f"ac_seed{seed}": root/f"train_seed{seed}/policy.pt" for seed in (72, 73, 74)}}
    records, checks = [], {}
    for name, path in definitions.items():
        original_hash = sha256(path)
        policy = Policy(path)
        with torch.no_grad():
            actions = policy.actor.act(states.flatten(0, 1)).reshape(36, 2, 6).numpy()
        if name != "joint_start":
            saved = torch.load(path.parent/"latest.pt", map_location="cpu", weights_only=True)
            for key, value in policy.actor.state_dict().items():
                torch.testing.assert_close(value, saved["learner"]["models"]["target_actor"][key], atol=0, rtol=0)
        for index, row in enumerate(rows):
            records.append({"policy": name, **row, "next_joint_action_differs": bool((actions[index, 0] != actions[index, 1]).any()),
                "next_grid_actions_differ": int((actions[index, 0] != actions[index, 1]).sum()),
                "js_history_next_action": json.dumps(actions[index, 0].tolist()),
                "f3_history_next_action": json.dumps(actions[index, 1].tolist())})
        checks[name] = {"checkpoint_unchanged": sha256(path) == original_hash,
                        "export_equals_td_target_actor": name != "joint_start"}
    frame = pd.DataFrame(records)
    frame.to_csv(root/"memory_diagnostic.csv", index=False)
    summary = frame.groupby("policy").agg(pairs=("seed", "size"), changed_joint_actions=("next_joint_action_differs", "sum"),
                                         changed_grid_actions=("next_grid_actions_differ", "sum"))
    assert int(summary.loc["joint_start", "changed_joint_actions"]) == 25
    save_json(root/"memory_diagnostic.json", {"scope": "same 36 development diagnostic pairs; not policy selection or fresh generalization test",
        "caution": "lower history sensitivity alone is not better control; a constant policy also has zero sensitivity",
        "results": summary.to_dict("index"), "checks": checks, "source_sha256": sha256(__file__)})
    print(summary.to_string())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    run(parser.parse_args().root)
