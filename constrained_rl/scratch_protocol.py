"""从随机权重训练的配对研究协议；复用冻结校准，不继承任何策略权重。"""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

import numpy as np

from offline_rl.data import save_json, sha256
from online_rl.protocol import source_hashes


REFERENCE = Path("models/constrained_ppo_pilot_20261115/protocol.json")
VERSION = "constrained_joint_ppo_from_scratch_v1"


def _check_hashes(expected):
    for name, digest in expected.items():
        if not Path(name).is_file() or sha256(name) != digest:
            raise ValueError(name)


def _calibration(reference, path):
    """保留 F3 的累计分子/分母，避免把逐轮平均误用为总体约束。"""
    evaluation = path.parent / "evaluation.csv"
    with evaluation.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["phase"] == "calibration" and row["policy"] == "fixed_f3"]
    if len(rows) != len(reference["seeds"]["calibration"]) or not rows:
        raise ValueError
    demand = sum(float(row["cohort_demand"]) for row in rows)
    matched = sum(float(row["unique_matched"]) for row in rows)
    waiting = sum(float(row["queue_wait_seconds"]) for row in rows)
    revenue = sum(float(row["platform_revenue"]) for row in rows)
    if demand <= 0:
        raise ValueError
    checks = ((matched / demand, reference["constraints"]["service_floor"]),
              (waiting / demand, reference["constraints"]["wait_limit_seconds_per_demand"]),
              (demand / len(rows), reference["constraints"]["demand_scale"]),
              (revenue / len(rows), reference["algorithm"]["reward_scale"]))
    if not all(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
               for actual, expected in checks):
        raise ValueError
    return {"protocol": str(path), "protocol_sha256": sha256(path),
            "evaluation": str(evaluation), "evaluation_sha256": sha256(evaluation),
            "policy": "fixed_f3", "episodes": len(rows),
            "seeds": reference["seeds"]["calibration"],
            "cohort_demand": demand, "unique_matched": matched,
            "queue_wait_seconds": waiting, "revenue_mean_usd": revenue / len(rows),
            "aggregation": "sum of numerators divided by sum of cohort demand; zero margin",
            "reuse": "Frozen calibration only; no new calibration or pretrained initialization"}


def make_protocol(output, episodes=100000, training_seed=412, smoke=False):
    if (not isinstance(episodes, int) or isinstance(episodes, bool) or not 0 < episodes <= 10_000_000
            or not isinstance(training_seed, int) or isinstance(training_seed, bool)
            or not 0 <= training_seed < 2**32):
        raise ValueError
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    reference = json.loads(REFERENCE.read_text())
    calibration = _calibration(reference, REFERENCE)
    _check_hashes(reference["data_sha256"])
    _check_hashes(reference["protected_files_sha256"])
    checkpoint = reference["initial_checkpoint"]
    if sha256(checkpoint) != reference["initial_sha256"]:
        raise ValueError

    # 两个训练臂共享场景与初始化；验证、测试和 smoke 使用互斥的命名空间。
    rng = np.random.default_rng(20261116)
    offset = 60_000_000 if smoke else 0
    specifications = (("train", episodes, 3_800_000_000),
                      ("validation", 2 if smoke else 32, 3_820_000_000),
                      ("test", 2 if smoke else 100, 3_840_000_000))
    seeds = {name: (rng.choice(10_000_000, count, replace=False) + low + offset).tolist()
             for name, count, low in specifications}
    algorithm = {key: value for key, value in reference["algorithm"].items()
                 if not key.startswith("cost_warmup")}
    algorithm["learning_rate"] = 3e-4
    sources = source_hashes()
    sources.update({str(path): sha256(path) for path in sorted(Path("constrained_rl").rglob("*.py"))})
    protected = dict(reference["protected_files_sha256"])
    protected.update({str(path): sha256(path) for path in (
        REFERENCE, REFERENCE.parent / "evaluation.csv", REFERENCE.parent / "progress.json",
        REFERENCE.parent / "revenue_control/latest.pt", REFERENCE.parent / "constrained/latest.pt")})
    protected[checkpoint] = reference["initial_sha256"]

    plan = {"schema_version": VERSION, "stage": "smoke" if smoke else "main",
        "training_regime": "from_scratch", "created_at": datetime.now(timezone.utc).isoformat(),
        "training_seed": training_seed, "config": reference["config"],
        "observation_metadata": reference["observation_metadata"], "model": reference["model"],
        "methods": ["revenue_control", "constrained"],
        "expected_test_policies": ["fixed_f3", "frozen_ar_ppo", "revenue_control", "constrained"],
        "training_episodes": episodes, "episodes_per_update": 16,
        "validation_interval_updates": 1 if smoke else 64, "early_validation_updates": [1, 4, 16],
        "seeds": seeds, "seed_generator": 20261116,
        "algorithm": algorithm, "initial_multipliers": [0., 0.],
        "constraints": reference["constraints"], "baseline_calibration_provenance": calibration,
        "reference_checkpoint": checkpoint, "reference_sha256": reference["initial_sha256"],
        "reference_checkpoint_usage": "Frozen AR-PPO comparison at final test only; never loaded for initialization, training or validation",
        "initialization": "One randomly initialized actor and critic network with two zero-initialized cost outputs, cloned identically into both arms with fresh optimizers; no pretrained weight loading or cost-head warmup",
        "preprocessing": "Frozen shared feature scalers copied from the reference protocol; no network weights inherited",
        "constraint_rule": reference["constraint_rule"], "cost_rule": reference["cost_rule"],
        "coverage_rule": reference["coverage_rule"], "execution": reference["execution"],
        "reward_rule": "Commission divided by the frozen F3 calibration mean episode commission",
        "checkpoint_rule": "Final fixed interaction budget; validation monitors learning only; no validation/test-based checkpoint selection",
        "scope": "Single initialization paired from-scratch experiment, not a multi-training-seed benchmark. Expected cohort-weighted constraints, not per-episode guarantees; original six reported metric definitions retained.",
        "source_sha256": sources, "source_snapshot": "source_snapshot",
        "data_sha256": reference["data_sha256"], "protected_files_sha256": protected,
        "sources": reference["sources"]}
    output.mkdir(parents=True, exist_ok=False)
    for name in sources:
        destination = output / plan["source_snapshot"] / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
    save_json(output / "protocol.json", plan)
    return plan


def load_protocol(path):
    path = Path(path)
    plan = json.loads(path.read_text())
    if plan["schema_version"] != VERSION or plan["training_regime"] != "from_scratch":
        raise ValueError
    for group in ("source_sha256", "data_sha256", "protected_files_sha256"):
        _check_hashes(plan[group])
    _check_hashes({plan["reference_checkpoint"]: plan["reference_sha256"]})
    snapshots = {str(path.parent / plan["source_snapshot"] / name): digest
                 for name, digest in plan["source_sha256"].items()}
    _check_hashes(snapshots)
    return plan
