"""先确定研究条件、交互预算和独立种子，再开始训练。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from offline_rl.data import save_json, sha256
from src.env.simulator_pattern import input_data_paths


REPLAY = Path("datasets_grid6_low_mixed100k_20261103/offline_rl")
REFERENCE = Path("experiments/ma_iql_comparison_20261103/protocol.json")


def source_hashes():
    paths = list(Path("src").rglob("*.py"))
    for directory in ("online_rl", "offline_rl", "baseline", "data_pipeline"):
        paths.extend(Path(directory).glob("*.py"))
    return {str(path): sha256(path) for path in sorted(paths)}


def make_protocol(output, pilot=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reference = json.loads(REFERENCE.read_text())
    metadata = json.loads((REPLAY/"manifest.json").read_text())
    config = reference["config"]
    if config["grid_num"] != metadata["grid_count"]:
        raise ValueError
    episodes = 16 if pilot else 4096
    rng = np.random.default_rng(20261104)
    # 三个不相交的命名空间；不使用已有训练轨迹或旧测试种子。
    blocks = {name: rng.choice(high-low, size=count, replace=False).astype("int64")+low
              for name, count, low, high in (
                  ("train", episodes, 1_100_000_000, 1_200_000_000),
                  ("validation", 4 if pilot else 30, 1_300_000_000, 1_400_000_000),
                  ("test", 8 if pilot else 100, 1_500_000_000, 1_600_000_000))}
    from offline_rl.train_joint import known_seeds
    historical = known_seeds()
    if any(historical.intersection(map(int, seeds)) for seeds in blocks.values()):
        raise ValueError
    data_root = Path(config["data_dir"])
    data_paths = list(input_data_paths(config["date"], data_root))
    data_paths += [data_root/"grid_adjacency.csv", data_root/"grid_manifest.json"]
    model = dict(grid_count=metadata["grid_count"], history_steps=metadata["preprocessor"]["history_steps"],
        history_features=len(metadata["preprocessor"]["history_features"]),
        context_size=len(metadata["state_context_indices"]),
        price_count=len(config["price_action_values"]), radius_count=len(config["matching_radius_values"]),
        hidden_size=64, memory_size=32, heads=4, layers=1, autoregressive=True)
    active = json.loads(Path("offline_rl/active_policy.json").read_text())
    references = {"fixed_f3": None, "lstm_greedy": reference["models"]["lstm_greedy"],
                  "ma_iql": reference["models"]["ma_iql"], "ma_iql_plus": active["checkpoint"]}
    plan = {"schema_version": "online_joint_ppo_v1", "stage": "pilot" if pilot else "main",
        "created_at": datetime.now(timezone.utc).isoformat(), "config": config, "model": model,
        "algorithm": dict(gamma=1., gae_lambda=.95, clip=.1, epochs=5, batch_size=256,
            learning_rate=3e-4, value_weight=.5, entropy_weight=.001, target_kl=.02,
            max_grad_norm=.5, reward_scale=100.),
        "episodes_per_update": 8 if pilot else 16, "training_episodes": episodes,
        "training_seeds": [81] if pilot else [81, 82, 83, 84, 85],
        "seeds": {name: values.tolist() for name, values in blocks.items()},
        "observation_metadata": metadata, "preprocessing": "frozen shared training-only feature scalers; no offline weight initialization",
        "primary_endpoint": "undiscounted platform commission in USD, 05:20--10:00",
        "execution": "stochastic ancestral sampling; fixed grid order; all six actions applied simultaneously",
        "method": "MAT-inspired autoregressive joint-policy PPO with a four-period LSTM window; CTCE",
        "ppo_ratio": "one full joint-action likelihood ratio per city decision; not per-agent clipping",
        "structural_control": "same network and PPO settings with previous-action inputs disabled (--factorized)",
        "scope": "project adaptation, not a verbatim MAT reproduction; no monotonic-improvement guarantee",
        "checkpoint_rule": "fixed final interaction budget; validation monitors learning; no test-based selection",
        "reference_policies": references,
        "reference_sha256": {name: sha256(path) for name, path in references.items() if path},
        "source_sha256": source_hashes(), "replay_manifest_sha256": sha256(REPLAY/"manifest.json"),
        "data_sha256": {str(path): sha256(path) for path in data_paths},
        "seed_design": "same exogenous scenarios across training initializations; independent policy RNG; distinct train/validation/test namespaces",
        "metrics": "existing six metrics; matching denominator includes repeated candidates; wait conditions on retained matches",
        "sources": ["https://arxiv.org/abs/2205.14953", "https://arxiv.org/abs/1707.06347", "https://arxiv.org/abs/1506.02438"]}
    save_json(output/"protocol.json", plan)
    return plan


def load_protocol(path):
    plan = json.loads(Path(path).read_text())
    if plan["schema_version"] != "online_joint_ppo_v1":
        raise ValueError
    if not all(Path(path).is_file() and sha256(path) == value for path, value in plan["source_sha256"].items()):
        raise ValueError
    if not all(sha256(path) == value for path, value in plan["data_sha256"].items()):
        raise ValueError
    return plan


def policy_seed(training_seed, scenario_seed, namespace=918):
    return int(np.random.SeedSequence([namespace, training_seed, scenario_seed]).generate_state(1)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    plan = make_protocol(args.output, args.pilot)
    print(json.dumps({key: plan[key] for key in ("stage", "training_episodes", "training_seeds")}, ensure_ascii=False))
