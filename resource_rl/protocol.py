"""冻结信息边界、结构对照、随机种子和交互预算。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from resource_rl.observation import ResourceObservation
from src.env.observation import load_neighbor_adjacency
from src.env.simulator_pattern import input_data_paths


REFERENCE_PROTOCOL = Path("experiments/online_ppo_100000_20261104/protocol.json")
REFERENCE_CHECKPOINT = Path("models/online_ppo_100000_20261104/seed81/latest.pt")


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def source_files():
    paths = []
    for name in ("resource_rl", "online_rl", "offline_rl", "src", "baseline", "data_pipeline"):
        paths.extend(Path(name).rglob("*.py"))
    return sorted(paths)


def make_protocol(output, episodes=100000, pilot=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reference = json.loads(REFERENCE_PROTOCOL.read_text())
    config = reference["config"]
    encoder = ResourceObservation(config, load_neighbor_adjacency(config["data_dir"], config["grid_num"]))
    metadata = encoder.metadata()
    count, batch = (32, 16) if pilot else (episodes, 64)
    if count<=0:
        raise ValueError
    rng = np.random.default_rng(2026110515 if pilot else 2026110516)
    base, stride, width = (2_000_000_000, 10_000_000, 1_000_000) if pilot else (1_700_000_000, 100_000_000, 100_000_000)
    seeds = {name: (rng.choice(width, n, replace=False)+base+offset*stride).tolist() for name, n, offset in
             (("train", count, 0), ("validation", 4 if pilot else 30, 1), ("test", 4 if pilot else 100, 2))}
    excluded = set().union(*(set(values) for values in reference["seeds"].values()))
    from offline_rl.train_joint import known_seeds
    excluded |= known_seeds()
    if any(excluded.intersection(values) for values in seeds.values()):
        raise ValueError
    model = dict(grid_count=config["grid_num"], node_size=len(metadata["node_columns"]), edge_size=len(metadata["edge_columns"]),
        driver_size=metadata["driver_size"], order_size=metadata["order_size"],
        price_count=len(config["price_action_values"]), radius_count=len(config["matching_radius_values"]),
        hidden_size=64, heads=4, layers=1, window=4)
    data_paths = list(input_data_paths(config["date"], config["data_dir"]))
    data_paths += [Path(config["data_dir"])/name for name in ("grid_info.csv", "grid_adjacency.csv", "grid_manifest.json")]
    snapshot = output/"checkpoints"/"original_ar_ppo.pt"
    snapshot.parent.mkdir()
    shutil.copy2(REFERENCE_CHECKPOINT, snapshot)
    original_paths = [REFERENCE_PROTOCOL, REFERENCE_CHECKPOINT, Path("models/online_ppo_100000_20261104/progress.json")]
    plan = {"schema_version": "resource_recurrent_online_v1", "stage": "pilot" if pilot else "main",
        "created_at": datetime.now(timezone.utc).isoformat(), "config": config, "model": model, "observation": metadata,
        "variants": ["flow", "window"], "training_seeds": [201] if pilot else [201, 202, 203],
        "training_episodes": count, "episodes_per_update": batch, "seeds": seeds,
        "algorithm": dict(warmup_steps=config["history_steps"], gamma=1., gae_lambda=.95, reward_scale=100.,
            clip=.1, target_kl=.02, actor_epochs=4, critic_epochs=4, sequence_batch_size=8,
            learning_rate=3e-4, value_weight=.5, auxiliary_weight=.1, entropy_weight=.001, max_grad_norm=.5),
        "validation_interval_updates": 16, "figure_interval_updates": 4,
        "initialization": "random weights; no weights/scalers fitted from offline trajectories; pilot weights never initialize main study",
        "method": "Resource-Flow Recurrent Actor-Critic, centralized training and centralized execution (CTCE)",
        "control": "Window-AC: identical current actor observations, individual critic tokens, decoder, auxiliary labels, optimizers and budgets; four-cycle window with spatial attention replaces persistent flow-graph recurrence; not the original AR-PPO checkpoint",
        "recurrent_training": "all four fixed-policy warmup observations followed by 56 scored decisions; current parameters recompute both memories from episode start; shuffle episodes only",
        "update": "full joint-action PPO ratio; full-rollout joint KL checked between actor epochs; independent critic optimizer completes all its epochs",
        "auxiliary": "same head/loss for both variants; next idle/queue counts plus actual completed-driver OD counts; current actions enter auxiliary head only",
        "reward": "same undiscounted platform commission in USD, 05:20--10:00; no local reward or terminal shaping",
        "checkpoint_rule": "fixed final interaction budget; validation only monitors; sealed new test requires all planned runs complete",
        "pairing": "same sampled orders, initial drivers, warmup and policy RNG keys across paired structures; simulator random draw counts may diverge after different actions",
        "references": {"fixed_f3": [config["fixed_price"], config["fixed_radius"]],
            "original_ar_ppo": str(snapshot), "original_ar_ppo_sha256": sha256(snapshot),
            "original_observation_metadata": reference["observation_metadata"]},
        "original_files_sha256": {str(path): sha256(path) for path in original_paths},
        "source_sha256": {str(path): sha256(path) for path in source_files()},
        "data_sha256": {str(path): sha256(path) for path in data_paths},
        "metrics": "six existing metrics; completion denominator includes repeated matching candidates; wait conditions on retained matches",
        "sources": ["https://arxiv.org/abs/2110.05038", "https://arxiv.org/abs/2105.11674",
                    "https://arxiv.org/abs/2103.01955", "https://arxiv.org/abs/1611.05397"],
        "scope": "new problem-specific combination; approximate learned histories/state representations; no exact belief-state, unbiased-PPO, or monotonic-improvement claim"}
    save_json(output/"protocol.json", plan)
    snapshot_root = output/"code_snapshot"
    for path in source_files():
        destination = snapshot_root/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return plan


def load_protocol(path):
    plan = json.loads(Path(path).read_text())
    if plan["schema_version"] != "resource_recurrent_online_v1":
        raise ValueError
    for name in ("source_sha256", "data_sha256"):
        if not all(Path(p).is_file() and sha256(p) == digest for p, digest in plan[name].items()):
            raise ValueError
    reference = plan["references"]
    if sha256(reference["original_ar_ppo"]) != reference["original_ar_ppo_sha256"]:
        raise ValueError
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    plan = make_protocol(args.output, args.episodes, args.pilot)
    print(json.dumps({key: plan[key] for key in ("stage", "training_episodes", "training_seeds", "variants")}))
