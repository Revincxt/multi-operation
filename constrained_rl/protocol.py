"""在校准与训练前固定小试预算、种子、输入和原版权重。"""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from offline_rl.data import sha256, save_json
from online_rl.protocol import source_hashes


def make_protocol(output, episodes=256, calibration_runs=32, test_runs=50, warmup_runs=32, warmup_epochs=50):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = json.loads(Path("checkpoints/online_observation.json").read_text())
    original = torch.load("checkpoints/ar_ppo.pt", map_location="cpu", weights_only=True)
    old = json.loads(Path("experiments/online_ppo_100000_20261104/protocol.json").read_text())
    rng = np.random.default_rng(20261115)
    sizes = {"calibration": calibration_runs, "critic_warmup": warmup_runs, "train": episodes,
             "validation": 16, "test": test_runs}
    seeds = {name: (rng.choice(1_000_000, count, replace=False) + 3_700_000_000 + i*10_000_000).tolist()
             for i, (name, count) in enumerate(sizes.items())}
    flat = [seed for group in seeds.values() for seed in group]
    if len(set(flat)) != len(flat) or max(flat) >= 2**32 or episodes <= 0:
        raise ValueError
    sources = source_hashes()
    sources.update({str(p): sha256(p) for p in Path("constrained_rl").glob("*.py")})
    plan = {"schema_version": "constrained_joint_ppo_pilot_v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "training_seed": 411,
        "config": settings["config"], "observation_metadata": settings["observation_metadata"],
        "model": original["model_config"], "methods": ["revenue_control", "constrained"],
        "training_episodes": episodes, "episodes_per_update": 16, "seeds": seeds,
        "validation_interval_updates": 8,
        "initial_checkpoint": "checkpoints/ar_ppo.pt", "initial_sha256": sha256("checkpoints/ar_ppo.pt"),
        "algorithm": {"gamma": 1., "gae_lambda": .95, "clip": .1, "epochs": 5, "batch_size": 256,
            "learning_rate": 1e-4, "value_weight": .5, "cost_value_weight": .5, "entropy_weight": .001,
            "target_kl": .02, "max_grad_norm": .5, "dual_learning_rate": 5.,
            "cost_warmup_epochs": warmup_epochs, "cost_warmup_learning_rate": .001,
            "value_scale_floor": .05, "cost_scale_floor": .01},
        "initial_multipliers": [0., 0.], "constraints": None,
        "constraint_rule": "F3 calibration ratio of total unique baseline cohorts covered to total cohort demand; total queue wait seconds divided by total cohort demand; zero margin",
        "reward_rule": "Commission divided by fixed F3 calibration mean episode commission; original V output rescaled from its USD/100 units",
        "cost_rule": "(coverage_floor*new_cohort_demand-new_covered_cohorts)/calibration_mean_demand; (queue_wait_seconds-wait_limit*new_cohort_demand)/(calibration_mean_demand*100 seconds)",
        "coverage_rule": "Initial post-warmup backlog cohorts plus new baseline parents. A parent expanded into several physical orders counts as covered once. All physical-order revenue and waiting remain counted. Final unmatched cohorts fail coverage; queue waiting stops at the finite horizon.",
        "initialization": f"Common frozen AR-PPO actor/critic checkpoint; fresh identical optimizers; fit only two additional cost heads on {warmup_runs} separate training episodes before cloning both arms",
        "execution": "Stochastic ancestral sampling, same action RNG namespace for paired scenarios, all 110 actions legal",
        "checkpoint_rule": "Final fixed budget, no test-based selection; one initialization pilot, not a from-scratch method benchmark",
        "scope": "Expected cohort-weighted constraints; no per-episode feasibility guarantee. Old six reported metrics retained without changing their definitions.",
        "source_sha256": sources, "data_sha256": old["data_sha256"],
        "protected_files_sha256": {str(p): sha256(p) for p in [
            Path("models/online_ppo_100000_20261104/progress.json"),
            Path("models/online_ppo_100000_20261104/seed81/latest.pt"),
            Path("models/resource_flow_100k_20261105/progress.json")]},
        "sources": ["https://proceedings.mlr.press/v70/achiam17a.html", "https://proceedings.mlr.press/v119/stooke20a.html"]}
    save_json(output/"protocol.json", plan)
    return plan


def load_protocol(path):
    plan = json.loads(Path(path).read_text())
    if plan["schema_version"] != "constrained_joint_ppo_pilot_v1":
        raise ValueError
    for group in ("source_sha256", "data_sha256", "protected_files_sha256"):
        for name, expected in plan[group].items():
            if not Path(name).is_file() or sha256(name) != expected:
                raise ValueError(name)
    if sha256(plan["initial_checkpoint"]) != plan["initial_sha256"]:
        raise ValueError
    return plan
