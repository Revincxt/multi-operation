"""相同冻结起点的收益对照与约束PPO试训；完整轨迹、独立校准与封存测试。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from constrained_rl.policy import ConstrainedPolicy
from constrained_rl.ppo import constraint_costs, update, warm_cost_heads
from constrained_rl.protocol import make_protocol, load_protocol
from constrained_rl.rollout import collect, initialize_worker
from offline_rl.data import save_json, sha256
from online_rl.train import atomic_save, cpu_weights


STOP = False


def stop(*_):
    global STOP
    STOP = True


def cost_arrays(episodes, constraints):
    for row in episodes:
        row["costs"] = constraint_costs(row["demand"], row["matches"], row["wait_seconds"],
            service_floor=constraints["service_floor"],
            wait_limit_seconds_per_demand=constraints["wait_limit_seconds_per_demand"],
            demand_scale=constraints["demand_scale"], wait_scale_seconds=constraints["wait_scale_seconds"])


def aggregate(episodes):
    summaries = [row["summary"] for row in episodes]
    denominator = sum(row["cohort_demand"] for row in summaries)
    return {"revenue_mean_usd": float(np.mean([row["platform_revenue"] for row in summaries])),
            "revenue_std_usd": float(np.std([row["platform_revenue"] for row in summaries])),
            "demand_satisfaction": sum(row["unique_matched"] for row in summaries)/denominator,
            "wait_per_demand_seconds": sum(row["queue_wait_seconds"] for row in summaries)/denominator}


def run(args):
    global STOP
    STOP = False
    root = Path(args.output)
    if not root.exists():
        make_protocol(root, args.episodes, args.calibration_runs, args.test_runs, args.warmup_runs, args.warmup_epochs)
    lock = (root/".train.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = load_protocol(root/"protocol.json")
    if args.workers <= 0:
        raise ValueError
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    torch.set_num_threads(1 if args.device == "mps" else 2)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    device = args.device
    original = torch.load(plan["initial_checkpoint"], map_location="cpu", weights_only=True)
    torch.manual_seed(plan["training_seed"])
    initial = ConstrainedPolicy(**plan["model"]).to(device).eval()
    initial.load_original(original["model_state_dict"], reward_scale=100.)
    history = pd.read_csv(root/"training_history.csv").to_dict("records") if (root/"training_history.csv").exists() else []
    evaluations = pd.read_csv(root/"evaluation.csv").to_dict("records") if (root/"evaluation.csv").exists() else []
    members = []
    started = perf_counter()

    def progress(status, **extra):
        save_json(root/"progress.json", {"status": status, "pid": os.getpid(),
            "completed_episodes": sum(m["episodes"] for m in members),
            "target_episodes": len(plan["methods"])*plan["training_episodes"],
            "runs": [{k:m[k] for k in ("method", "episodes", "iteration")} for m in members],
            "elapsed_this_process_seconds": perf_counter()-started, **extra})

    def record_eval(episodes, phase, policy, iteration=0):
        evaluations.extend({"phase": phase, "policy": policy, "iteration": iteration, **row["summary"]} for row in episodes)
        pd.DataFrame(evaluations).to_csv(root/"evaluation.csv", index=False)

    def save_member(member):
        target = root/member["method"]
        target.mkdir(exist_ok=True)
        atomic_save(target/"latest.pt", {"model_state_dict": cpu_weights(member["model"]),
            "optimizer": member["optimizer"].state_dict(), "multipliers": member["multipliers"].tolist(),
            "numpy_rng": member["rng"].bit_generator.state, "episodes": member["episodes"],
            "iteration": member["iteration"], "model_config": plan["model"], "method": member["method"],
            "protocol_sha256": sha256(root/"protocol.json"),
            "history": [r for r in history if r["method"] == member["method"]]})

    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                initializer=initialize_worker, initargs=(plan["config"], plan["observation_metadata"], plan["model"])) as pool:
            if plan["constraints"] is None:
                progress("calibrating", current="fixed_f3")
                episodes = collect(pool, initial, plan["seeds"]["calibration"], plan["training_seed"], "fixed_f3")
                record_eval(episodes, "calibration", "fixed_f3")
                totals = aggregate(episodes)
                plan["constraints"] = {"service_floor": totals["demand_satisfaction"],
                    "wait_limit_seconds_per_demand": totals["wait_per_demand_seconds"],
                    "demand_scale": float(np.mean([r["summary"]["cohort_demand"] for r in episodes])),
                    "wait_scale_seconds": 100.}
                plan["algorithm"]["reward_scale"] = totals["revenue_mean_usd"]
                save_json(root/"protocol.json", plan)
                print("calibration="+json.dumps({**plan["constraints"], "reward_scale": totals["revenue_mean_usd"]}), flush=True)
                del episodes
            if STOP:
                progress("paused"); return
            initial.load_original(original["model_state_dict"], reward_scale=plan["algorithm"]["reward_scale"])
            common = root/"initial.pt"
            if common.exists():
                initial.load_state_dict(torch.load(common, map_location=device, weights_only=True)["model_state_dict"])
            else:
                progress("cost_critic_warmup")
                episodes = collect(pool, initial, plan["seeds"]["critic_warmup"], plan["training_seed"], "frozen_ar_ppo")
                cost_arrays(episodes, plan["constraints"])
                warm = warm_cost_heads(initial, episodes, plan["algorithm"], np.random.default_rng(411), device)
                save_json(root/"cost_warmup.json", warm)
                pd.DataFrame([r["summary"] for r in episodes]).to_csv(root/"cost_warmup_episodes.csv", index=False)
                atomic_save(common, {"model_state_dict": cpu_weights(initial), "model_config": plan["model"],
                    "source_sha256": plan["initial_sha256"], "protocol_sha256": sha256(root/"protocol.json")})
                del episodes
            if STOP:
                progress("paused"); return
            for method in plan["methods"]:
                model = ConstrainedPolicy(**plan["model"]).to(device)
                model.load_state_dict(initial.state_dict())
                optimizer = torch.optim.Adam(model.parameters(), lr=plan["algorithm"]["learning_rate"], eps=1e-5)
                member = {"method": method, "model": model, "optimizer": optimizer,
                    "multipliers": np.array(plan["initial_multipliers"], dtype=float),
                    "rng": np.random.default_rng(plan["training_seed"]), "episodes": 0, "iteration": 0}
                checkpoint = root/method/"latest.pt"
                if checkpoint.exists():
                    saved = torch.load(checkpoint, map_location=device, weights_only=True)
                    if saved["protocol_sha256"] != sha256(root/"protocol.json"):
                        raise ValueError
                    model.load_state_dict(saved["model_state_dict"]); optimizer.load_state_dict(saved["optimizer"])
                    member["multipliers"][:] = saved["multipliers"]
                    member["rng"].bit_generator.state = saved["numpy_rng"]
                    member["episodes"], member["iteration"] = saved["episodes"], saved["iteration"]
                    history = [r for r in history if r["method"] != method] + saved["history"]
                members.append(member)
                save_member(member)
            while not STOP and any(m["episodes"] < plan["training_episodes"] for m in members):
                for member in members:
                    if STOP: break
                    if member["episodes"] >= plan["training_episodes"]: continue
                    method, version = member["method"], member["iteration"]
                    progress("collecting", current=method, iteration=version+1)
                    begin = perf_counter()
                    seeds = plan["seeds"]["train"][member["episodes"]:member["episodes"]+plan["episodes_per_update"]]
                    episodes = collect(pool, member["model"], seeds, plan["training_seed"], method)
                    cost_arrays(episodes, plan["constraints"])
                    collected = perf_counter()
                    progress("updating", current=method, iteration=version+1)
                    diagnostic = update(member["model"], member["optimizer"], episodes, plan["algorithm"],
                        member["rng"], device, member["multipliers"], constrained=method == "constrained")
                    member["episodes"] += len(episodes); member["iteration"] += 1
                    record = {"method": method, "iteration": member["iteration"], "episodes": member["episodes"],
                        "environment_steps": member["episodes"]*len(episodes[0]["rewards"]), **aggregate(episodes),
                        "collection_seconds": collected-begin, "learning_seconds": perf_counter()-collected,
                        **diagnostic, "lambda_service": float(member["multipliers"][0]),
                        "lambda_wait": float(member["multipliers"][1]),
                        "cost_service_mean": float(np.mean([r["costs"][:, 0].sum() for r in episodes])),
                        "cost_wait_mean": float(np.mean([r["costs"][:, 1].sum() for r in episodes]))}
                    history.append(record)
                    save_member(member)
                    pd.DataFrame(history).to_csv(root/"training_history.csv", index=False)
                    pd.DataFrame([r["summary"] for r in episodes]).to_csv(root/method/f"episodes_{member['iteration']:04d}.csv", index=False)
                    print(f"{method} update={member['iteration']} episodes={member['episodes']} revenue={record['revenue_mean_usd']:.2f} "
                          f"coverage={record['demand_satisfaction']:.4f} wait={record['wait_per_demand_seconds']:.2f} "
                          f"lambda={member['multipliers'].round(4).tolist()} KL={record['joint_kl']:.5f} collect={record['collection_seconds']:.1f}s", flush=True)
                    if not STOP and member["iteration"]%plan["validation_interval_updates"] == 0:
                        progress("validating", current=method)
                        evaluated = collect(pool, member["model"], plan["seeds"]["validation"], plan["training_seed"], method)
                        record_eval(evaluated, "validation", method, member["iteration"])
                    if member["iteration"]%4 == 0:
                        from constrained_rl.report import run as report
                        report(root)
            if STOP:
                progress("paused"); return
            policies = {"fixed_f3": initial, "frozen_ar_ppo": initial, **{m["method"]: m["model"] for m in members}}
            for name, model in policies.items():
                if STOP: break
                if len([r for r in evaluations if r["phase"] == "test" and r["policy"] == name]) == len(plan["seeds"]["test"]): continue
                progress("testing", current=name)
                evaluated = collect(pool, model, plan["seeds"]["test"], plan["training_seed"], name)
                record_eval(evaluated, "test", name)
            if STOP:
                progress("paused"); return
            from constrained_rl.report import run as report
            report(root)
            load_protocol(root/"protocol.json")
            progress("complete")
            print("trial_complete", flush=True)
    except BaseException as error:
        progress("failed", error=type(error).__name__)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="models/constrained_ppo_pilot_20261115")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--calibration-runs", type=int, default=32)
    parser.add_argument("--test-runs", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=32)
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    run(parser.parse_args())
