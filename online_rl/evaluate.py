"""冻结在线策略，使用预留种子与现有策略进行配对闭环比较。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.compare_policies import METRICS, run_episode, sample_common_orders
from baseline.lstm import LSTMPredictor
from data_pipeline.collect_trajectories import simulation_window
from offline_rl.data import save_json, sha256
from offline_rl.evaluate import Policy
from online_rl.environment import ObservationEncoder, frame_hash
from online_rl.policy import JointPolicy
from online_rl.protocol import load_protocol, policy_seed
from src.env.actions import GridAction
from src.env.simulator_pattern import SimulatorPattern


_PLAN = _PATTERN = _MODELS = _ENCODER = _REFERENCES = None


def initialize_worker(plan, checkpoints, references):
    global _PLAN, _PATTERN, _MODELS, _ENCODER, _REFERENCES
    torch.set_num_threads(1)
    _PLAN = plan
    _PATTERN = SimulatorPattern(plan["config"]["date"], data_dir=plan["config"]["data_dir"])
    _ENCODER = ObservationEncoder(plan["observation_metadata"])
    _MODELS, _REFERENCES = {}, {}
    for path in checkpoints:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = JointPolicy(**saved["model_config"]).eval()
        model.load_state_dict(saved["model_state_dict"])
        prefix = "AR-PPO" if saved["model_config"]["autoregressive"] else "Factorized-PPO"
        name = f"{prefix}_seed{saved['training_seed']}"
        if name in _MODELS:
            raise ValueError
        _MODELS[name] = (model, saved["training_seed"])
    for name in references:
        path = plan["reference_policies"][name]
        _REFERENCES[name] = (LSTMPredictor.load(path) if name == "lstm_greedy" else Policy(path)) if path else None


def evaluate_seed(job):
    seed, output = job
    config = _PLAN["config"]
    _, start, end = simulation_window(config["start_time"], config["simulation_steps"])
    requests, sampled = sample_common_orders(_PATTERN.request_all, start, end, config["order_sample_ratio"], seed)
    directory = Path(output)/"runs"/f"seed{seed}"
    directory.mkdir(parents=True)
    sampled.to_csv(directory/"sampled_orders.csv.gz", index=False)
    pairs, records = _PLAN["observation_metadata"]["action_pairs"], []
    for name, (model, training_seed) in _MODELS.items():
        generator = torch.Generator().manual_seed(policy_seed(training_seed, seed))

        def action_policy(periods, decision, time, interval, current_state):
            state = _ENCODER(periods, decision, time, interval, current_state)
            with torch.inference_mode():
                actions = model.sample(torch.from_numpy(state[None]), generator)[0][0].numpy()
            return {grid: GridAction(*pairs[int(index)]) for grid, index in enumerate(actions)}

        record = run_episode(config, name, seed, requests, _PATTERN.driver_info, None, directory/name, action_policy=action_policy)
        records.append(record)
    for name, reference in _REFERENCES.items():
        predictor = reference if name == "lstm_greedy" else None
        action_policy = reference if name not in ("lstm_greedy", "fixed_f3") else None
        records.append(run_episode(config, name, seed, requests, _PATTERN.driver_info, predictor, directory/name,
                                   action_policy=action_policy))
    for row in records:
        row["sampled_orders_sha256"] = frame_hash(sampled)
    for field in ("initial_driver_sha256", "warmup_metrics_sha256", "baseline_request_num", "sampled_orders_sha256"):
        if len({row[field] for row in records}) != 1:
            raise AssertionError
    print(f"evaluation seed={seed} completed", flush=True)
    return [{key: value for key, value in row.items() if key != "full_window"} for row in records]


def paired_statistics(frame, references, draws=10000):
    pivot = frame.pivot(index="seed", columns="policy", values="platform_revenue").sort_index()
    samples = np.random.default_rng(42).integers(len(pivot), size=(draws, len(pivot)))
    result = []
    for policy in pivot.columns:
        if not policy.startswith(("AR-PPO_", "Factorized-PPO_")):
            continue
        paired_references = list(references)
        counterpart = policy.replace("AR-PPO_", "Factorized-PPO_")
        if policy.startswith("AR-PPO_") and counterpart in pivot:
            paired_references.append(counterpart)
        for reference in paired_references:
            differences = (pivot[policy]-pivot[reference]).to_numpy()
            low, high = np.quantile(differences[samples].mean(1), [.025, .975])
            result.append({"policy": policy, "reference": reference, "seeds": len(pivot),
                "mean_difference_usd": float(differences.mean()),
                "relative_percent": float(100*differences.mean()/pivot[reference].mean()),
                "ci95_low": float(low), "ci95_high": float(high), "wins": int((differences>0).sum())})
    # 多初始化与同一批环境种子构成完全交叉设计；共同重采样行、列。
    for prefix in ("AR-PPO_", "Factorized-PPO_"):
        names = sorted(name for name in pivot.columns if name.startswith(prefix))
        if len(names) < 2:
            continue
        rng = np.random.default_rng(43)
        train_indices = rng.integers(len(names), size=(draws, len(names), 1))
        environment_indices = rng.integers(len(pivot), size=(draws, 1, len(pivot)))
        for reference in references:
            differences = np.stack([(pivot[name]-pivot[reference]).to_numpy() for name in names])
            bootstrap = differences[train_indices, environment_indices].mean((1, 2))
            low, high = np.quantile(bootstrap, [.025, .975])
            result.append({"policy": prefix.rstrip("_"), "reference": reference, "seeds": len(pivot),
                "training_initializations": len(names), "mean_difference_usd": float(differences.mean()),
                "relative_percent": float(100*differences.mean()/pivot[reference].mean()),
                "ci95_low": float(low), "ci95_high": float(high),
                "inference": "two-way bootstrap over crossed training and environment seeds"})
    ar = sorted(name for name in pivot if name.startswith("AR-PPO_"))
    if len(ar) >= 2 and all(name.replace("AR-PPO_", "Factorized-PPO_") in pivot for name in ar):
        differences = np.stack([(pivot[name]-pivot[name.replace("AR-PPO_", "Factorized-PPO_")]).to_numpy() for name in ar])
        rng = np.random.default_rng(44)
        train_indices = rng.integers(len(ar), size=(draws, len(ar), 1))
        environment_indices = rng.integers(len(pivot), size=(draws, 1, len(pivot)))
        low, high = np.quantile(differences[train_indices, environment_indices].mean((1, 2)), [.025, .975])
        result.append({"policy": "AR-PPO", "reference": "Factorized-PPO", "seeds": len(pivot),
            "training_initializations": len(ar), "mean_difference_usd": float(differences.mean()),
            "ci95_low": float(low), "ci95_high": float(high),
            "inference": "matched initializations and two-way bootstrap over crossed seeds"})
    return result


def run(args):
    plan = load_protocol(args.protocol)
    checkpoints = [str(Path(path).resolve()) for path in args.checkpoints]
    if args.split == "test" and plan["stage"] == "main":
        # 主实验必须提交五个完整初始化，避免只挑最好的模型。
        for variant in {torch.load(path, map_location="cpu", weights_only=True)["model_config"]["autoregressive"] for path in checkpoints}:
            members = [torch.load(path, map_location="cpu", weights_only=True) for path in checkpoints]
            members = [row for row in members if row["model_config"]["autoregressive"] == variant]
            if (sorted(row["training_seed"] for row in members) != sorted(plan["training_seeds"])
                    or any(row["episodes"] != plan["training_episodes"] for row in members)):
                raise ValueError
    for path in checkpoints:
        if torch.load(path, map_location="cpu", weights_only=True)["protocol_sha256"] != sha256(args.protocol):
            raise ValueError
    references = args.references or list(plan["reference_policies"])
    if not set(references).issubset(plan["reference_policies"]):
        raise ValueError
    for name in references:
        if plan["reference_policies"][name] and sha256(plan["reference_policies"][name]) != plan["reference_sha256"][name]:
            raise ValueError
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/"manifest.json", {"stage": plan["stage"], "split": args.split, "status": "running",
        "protocol_sha256": sha256(args.protocol), "checkpoints_sha256": {path: sha256(path) for path in checkpoints},
        "seeds": plan["seeds"][args.split], "references": references, "execution": plan["execution"],
        "pairing": "shared sampled orders, initial drivers and warmup; simulator RNG draw counts may diverge after actions"})
    initialization = plan, checkpoints, references
    jobs = [(seed, str(output)) for seed in plan["seeds"][args.split]]
    if args.workers == 1:
        initialize_worker(*initialization)
        rows = [evaluate_seed(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=initialize_worker, initargs=initialization) as pool:
            rows = list(pool.map(evaluate_seed, jobs))
    frame = pd.DataFrame([row for episode in rows for row in episode]).sort_values(["seed", "policy"])
    frame.to_csv(output/"run_summary.csv", index=False)
    frame.groupby("policy")[list(METRICS)].agg(["mean", "std"]).to_csv(output/"policy_summary.csv")
    statistics = paired_statistics(frame, references)
    pd.DataFrame(statistics).to_csv(output/"paired_revenue_statistics.csv", index=False)
    manifest = json.loads((output/"manifest.json").read_text())
    manifest["status"] = "complete"
    save_json(output/"manifest.json", manifest)
    print(pd.DataFrame(statistics).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--references", nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
