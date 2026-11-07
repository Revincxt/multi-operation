"""固定预算终点的配对闭环评估；同时输出运营指标、动作和交叉重采样区间。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from resource_rl.environment import ResourceEnvironment
from resource_rl.policy import ResourcePolicy
from resource_rl.protocol import load_protocol, save_json, sha256
from resource_rl.rollout import action_seed, run_episode


METRICS = ("baseline_request_num", "request_rate", "driver_accept_rate", "order_completion_rate",
           "mean_wait_to_match_seconds", "platform_revenue")
_PLAN = _ENVIRONMENT = _MODELS = _LEGACY = _ENCODER = None


def initialize_worker(plan, checkpoints):
    global _PLAN, _ENVIRONMENT, _MODELS, _LEGACY, _ENCODER
    from online_rl.policy import JointPolicy
    from online_rl.environment import ObservationEncoder
    torch.set_num_threads(1)
    _PLAN, _ENVIRONMENT, _MODELS = plan, ResourceEnvironment(plan["config"]), {}
    for path in checkpoints:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = ResourcePolicy(**saved["model_config"]).eval()
        model.load_state_dict(saved["model_state_dict"])
        name = f"{saved['variant']}_seed{saved['training_seed']}"
        _MODELS[name] = (model, saved["training_seed"])
    saved = torch.load(plan["references"]["original_ar_ppo"], map_location="cpu", weights_only=True)
    _LEGACY = JointPolicy(**saved["model_config"]).eval()
    _LEGACY.load_state_dict(saved["model_state_dict"])
    _ENCODER = ObservationEncoder(plan["references"]["original_observation_metadata"])


def reference_episode(seed, legacy):
    from src.env.observation import capture_decision_state
    env, c = _ENVIRONMENT, _PLAN["config"]
    env.reset(seed)
    fixed = env.pairs.index((c["fixed_price"], c["fixed_radius"]))
    # 原模型的策略采样键保持其已有评估约定；不冒充配对新初始化。
    legacy_seed = int(np.random.SeedSequence([918, 81, seed]).generate_state(1)[0])
    generator = torch.Generator().manual_seed(legacy_seed)
    terminal = False
    while not terminal:
        if legacy:
            state = _ENCODER(env.periods, env.sim.current_step//c["decision_freq"], env.sim.time,
                             c["decision_freq"], capture_decision_state(env.sim, env.adjacency))
            with torch.inference_mode():
                chosen = _LEGACY.sample(torch.from_numpy(state[None]), generator)[0][0].numpy()
        else:
            chosen = np.full(c["grid_num"], fixed, dtype="int64")
        _, _, terminal = env.step(chosen)
    return env.summary()


def evaluate_seed(job):
    seed, output = job
    rows = []
    directory = Path(output)/"runs"/f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"summary.csv").exists():
        return pd.read_csv(directory/"summary.csv").to_dict("records")
    policies = [(name, model, train_seed) for name, (model, train_seed) in _MODELS.items()]
    policies += [("fixed_f3", None, -1), ("original_ar_ppo", None, -1)]
    for name, model, train_seed in policies:
        if model is None:
            summary = reference_episode(seed, name == "original_ar_ppo")
            variant = name
        else:
            summary = run_episode(_ENVIRONMENT, model, seed, train_seed, "evaluation")["summary"]
            variant = model.variant
        target = directory/name
        target.mkdir(exist_ok=True)
        pd.DataFrame(_ENVIRONMENT.rows).to_csv(target/"action_plan.csv.gz", index=False)
        _ENVIRONMENT.minutes.to_csv(target/"grid_minute_metrics.csv.gz", index=False)
        _ENVIRONMENT.periods.to_csv(target/"decision_metrics.csv.gz", index=False)
        record = {**summary, "policy": name, "variant": variant, "training_seed": train_seed}
        save_json(target/"summary.json", record)
        rows.append(record)
    for field in ("sampled_orders_sha256", "initial_driver_sha256", "warmup_metrics_sha256", "baseline_request_num"):
        if len({row[field] for row in rows}) != 1:
            raise AssertionError
    pd.DataFrame(rows).to_csv(directory/"summary.csv", index=False)
    print(f"evaluation scenario={seed} policies={len(rows)} complete", flush=True)
    return rows


def crossed_bootstrap(values, draws=10000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError
    rng = np.random.default_rng(20261105)
    rows = rng.integers(values.shape[0], size=(draws, values.shape[0], 1))
    columns = rng.integers(values.shape[1], size=(draws, 1, values.shape[1]))
    return values[rows, columns].mean((1, 2))


def paired_statistics(frame):
    result = []
    for metric in METRICS:
        first = frame[frame.variant == "flow"].pivot(index="training_seed", columns="seed", values=metric).sort_index().sort_index(axis=1)
        for reference in ("window", "fixed_f3", "original_ar_ppo"):
            other = frame[frame.variant == reference].pivot(index="training_seed", columns="seed", values=metric).sort_index().sort_index(axis=1)
            if list(first.columns) != list(other.columns) or (len(other)>1 and list(first.index) != list(other.index)):
                raise ValueError
            difference = first.to_numpy()-other.to_numpy()
            low, high = np.quantile(crossed_bootstrap(difference), [.025, .975])
            result.append({"policy": "flow", "reference": reference, "metric": metric,
                "mean_difference": float(difference.mean()), "ci95_low": float(low), "ci95_high": float(high),
                "training_initializations": len(first), "environment_seeds": len(first.columns),
                "per_initialization_differences": difference.mean(1).tolist(),
                "relative_percent": float(100*difference.mean()/other.to_numpy().mean()) if other.to_numpy().mean() else 0.})
    return result


def run(args):
    plan = load_protocol(args.protocol)
    checkpoints = []
    for seed in plan["training_seeds"]:
        for variant in plan["variants"]:
            root = Path(args.models)/f"{variant}_seed{seed}"
            path = root/"policy.pt" if (root/"policy.pt").exists() else root/"latest.pt"
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["protocol_sha256"] != sha256(args.protocol):
                raise ValueError
            if args.split == "test" and saved["episodes"] != plan["training_episodes"]:
                raise ValueError
            checkpoints.append(str(path))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"protocol_sha256": sha256(args.protocol), "split": args.split,
                "checkpoint_sha256": {path: sha256(path) for path in checkpoints}}
    if (output/"manifest.json").exists():
        prior = json.loads((output/"manifest.json").read_text())
        if any(prior[name] != value for name, value in identity.items()):
            raise ValueError
    save_json(output/"manifest.json", {**identity, "status": "running", "seeds": plan["seeds"][args.split], "pairing": plan["pairing"]})
    jobs = [(seed, str(output)) for seed in plan["seeds"][args.split]]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
            initializer=initialize_worker, initargs=(plan, checkpoints)) as pool:
        records = list(pool.map(evaluate_seed, jobs))
    frame = pd.DataFrame([row for group in records for row in group]).sort_values(["seed", "policy"])
    frame.to_csv(output/"run_summary.csv", index=False)
    frame.groupby("variant")[list(METRICS)].mean().to_csv(output/"policy_means.csv")
    statistics = paired_statistics(frame)
    pd.DataFrame(statistics).to_csv(output/"paired_statistics.csv", index=False)
    from resource_rl.plots import evaluation
    evaluation(output, frame, statistics)
    save_json(output/"manifest.json", {**identity, "status": "complete", "seeds": plan["seeds"][args.split],
        "episodes": len(frame), "paired_initial_conditions_verified": True, "pairing": plan["pairing"]})
    print(frame.groupby("variant")[list(METRICS)].mean().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
