"""随机初始化的新研究队列；六个独立运行交替采集、序列更新、保存完整断点。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
import fcntl
from pathlib import Path
import signal
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from resource_rl.policy import ResourcePolicy
from resource_rl.ppo import update
from resource_rl.protocol import load_protocol, save_json, sha256
from resource_rl.rollout import collect, cpu_weights, initialize_worker, save_batch


STOP = False


def stop(*_):
    global STOP
    STOP = True


def atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def prepare_members(plan, output, device, protocol_hash):
    members = []
    for seed in plan["training_seeds"]:
        for variant in plan["variants"]:
            root = output/f"{variant}_seed{seed}"
            root.mkdir(exist_ok=True)
            (root/"rollouts").mkdir(exist_ok=True)
            torch.manual_seed(seed)
            model = ResourcePolicy(**plan["model"], variant=variant).to(device).eval()
            actor = torch.optim.Adam(model.actor_parameters(), lr=plan["algorithm"]["learning_rate"], eps=1e-5)
            critic = torch.optim.Adam(model.critic_parameters(), lr=plan["algorithm"]["learning_rate"], eps=1e-5)
            rng = np.random.default_rng(seed)
            identity = {"protocol_sha256": protocol_hash, "model_config": dict(plan["model"], variant=variant), "training_seed": seed, "variant": variant}
            history, validation, completed, iteration = [], [], 0, 0
            if (root/"latest.pt").exists():
                saved = torch.load(root/"latest.pt", map_location=device, weights_only=True)
                if any(saved[name] != value for name, value in identity.items()):
                    raise ValueError
                model.load_state_dict(saved["model_state_dict"])
                actor.load_state_dict(saved["actor_optimizer"]); critic.load_state_dict(saved["critic_optimizer"])
                rng.bit_generator.state = saved["numpy_rng"]
                torch.set_rng_state(saved["torch_rng"].cpu())
                history, validation = saved["history"], saved["validation"]
                completed, iteration = saved["episodes"], saved["iteration"]
            else:
                atomic_save(root/"initial.pt", {**identity, "model_state_dict": cpu_weights(model), "episodes": 0,
                    "initialization": "torch random seed; no existing model weights loaded"})
            save_json(root/"run_config.json", {**identity, "device": device,
                "parameters": sum(p.numel() for p in model.parameters()), "initialization": plan["initialization"]})
            members.append(dict(root=root, model=model, actor=actor, critic=critic, rng=rng, identity=identity,
                history=history, validation=validation, completed=completed, iteration=iteration))
    return members


def save_member(member):
    payload = {**member["identity"], "model_state_dict": cpu_weights(member["model"]),
        "actor_optimizer": member["actor"].state_dict(), "critic_optimizer": member["critic"].state_dict(),
        "numpy_rng": member["rng"].bit_generator.state, "torch_rng": torch.get_rng_state(),
        "history": member["history"], "validation": member["validation"],
        "episodes": member["completed"], "iteration": member["iteration"]}
    atomic_save(member["root"]/"latest.pt", payload)
    pd.DataFrame(member["history"]).to_csv(member["root"]/"training_history.csv", index=False)
    if member["validation"]:
        pd.DataFrame(member["validation"]).to_csv(member["root"]/"validation_history.csv", index=False)


def progress(root, plan, members, state, current=None):
    save_json(root/"progress.json", {"status": state, "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(), "stage": plan["stage"],
        "completed_episodes": sum(member["completed"] for member in members),
        "target_episodes": plan["training_episodes"]*len(members), "episodes_per_run": plan["training_episodes"],
        "current": current, "runs": [{**member["identity"], "episodes": member["completed"], "iteration": member["iteration"]} for member in members]})


def validate(member, pool, plan, output, members, record=None):
    iteration = member["iteration"]
    due = iteration == 1 or iteration%plan["validation_interval_updates"] == 0
    done = any(row["iteration"] == iteration for row in member["validation"])
    if iteration == 0 or not due or done:
        return
    progress(output, plan, members, "validating", record or member["identity"])
    identity = member["identity"]
    evaluated = collect(pool, member["model"], plan["seeds"]["validation"], identity["training_seed"], iteration)
    member["validation"].append({"iteration": iteration, "episodes": member["completed"],
        "variant": identity["variant"], "training_seed": identity["training_seed"],
        "revenue_mean_usd": float(np.mean([row["summary"]["platform_revenue"] for row in evaluated]))})
    pd.DataFrame([row["summary"] for row in evaluated]).to_csv(member["root"]/f"validation_{iteration:05d}.csv", index=False)
    save_member(member)


def run(args):
    plan = load_protocol(args.protocol)
    output = Path(args.output)
    output.mkdir(exist_ok=True, parents=True)
    lock = (output/".train.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.workers<=0 or (args.max_updates is not None and args.max_updates<=0):
        raise ValueError
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError
    torch.set_num_threads(1 if device == "mps" else 4)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    members = prepare_members(plan, output, device, sha256(args.protocol))
    progress(output, plan, members, "starting")
    timer = perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                initializer=initialize_worker, initargs=(plan["config"], plan["model"], plan["variants"])) as pool:
            for member in members:
                if not STOP:
                    validate(member, pool, plan, output, members)
            while not STOP:
                pending = [member for member in members if member["completed"] < plan["training_episodes"]
                           and (args.max_updates is None or member["iteration"] < args.max_updates)]
                if not pending:
                    break
                for member in pending:
                    if STOP:
                        break
                    before, version = member["completed"], member["iteration"]
                    seeds = plan["seeds"]["train"][before:before+plan["episodes_per_update"]]
                    identity = member["identity"]
                    begun = perf_counter()
                    progress(output, plan, members, "collecting", {**identity, "iteration": version+1, "batch_episodes": len(seeds)})
                    atomic_save(member["root"]/"rollouts"/f"behavior_{version:05d}.pt", cpu_weights(member["model"]))
                    episodes = collect(pool, member["model"], seeds, identity["training_seed"], version)
                    if any(row["policy_version"] != version for row in episodes):
                        raise ValueError
                    collected = perf_counter()
                    save_batch(member["root"]/"rollouts"/f"batch_{version:05d}.npz", episodes)
                    pd.DataFrame([row["summary"] for row in episodes]).to_csv(
                        member["root"]/"rollouts"/f"episodes_{version:05d}.csv", index=False)
                    progress(output, plan, members, "updating", {**identity, "iteration": version+1})
                    diagnostics = update(member["model"], member["actor"], member["critic"], episodes,
                        plan["algorithm"], member["rng"], device)
                    member["completed"] += len(episodes); member["iteration"] += 1
                    revenue = [row["summary"]["platform_revenue"] for row in episodes]
                    record = {"iteration": member["iteration"], "episodes": member["completed"],
                        "environment_steps": member["completed"]*(plan["config"]["simulation_steps"]//plan["config"]["decision_freq"]-plan["config"]["history_steps"]),
                        "revenue_mean_usd": float(np.mean(revenue)), "revenue_std_usd": float(np.std(revenue)),
                        "collection_seconds": collected-begun, "learning_seconds": perf_counter()-collected,
                        "variant": identity["variant"], "training_seed": identity["training_seed"], **diagnostics}
                    member["history"].append(record)
                    save_member(member)
                    print(f"{identity['variant']} seed={identity['training_seed']} update={member['iteration']} "
                        f"episodes={member['completed']}/{plan['training_episodes']} return={record['revenue_mean_usd']:.2f} "
                        f"KL={record['joint_kl']:.5f} actor_passes={record['actor_effective_epochs']:.1f} "
                        f"critic_passes={record['critic_effective_epochs']:.1f} seconds={perf_counter()-begun:.1f}", flush=True)
                    if not STOP:
                        validate(member, pool, plan, output, members, record)
                    progress(output, plan, members, "training", record)
                    if member["iteration"] == 1 or member["iteration"]%plan["figure_interval_updates"] == 0:
                        from resource_rl.plots import training
                        training(output)
            state = "complete" if all(member["completed"] == plan["training_episodes"] for member in members) else "paused" if STOP else "checkpointed"
            progress(output, plan, members, state)
        from resource_rl.plots import training
        training(output)
        if state == "complete":
            for member in members:
                atomic_save(member["root"]/"policy.pt", {**member["identity"], "model_state_dict": cpu_weights(member["model"]),
                    "episodes": member["completed"], "iteration": member["iteration"]})
            if args.evaluate_at_end:
                progress(output, plan, members, "evaluating")
                from resource_rl.evaluate import run as evaluate
                evaluate(argparse.Namespace(protocol=args.protocol, models=str(output), output=str(output/"test"),
                                             split="test", workers=args.workers))
                progress(output, plan, members, "complete")
        print(f"queue_status={state} elapsed_seconds={perf_counter()-timer:.1f}", flush=True)
    except BaseException as exception:
        progress(output, plan, members, "failed", {"exception_type": type(exception).__name__})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--evaluate-at-end", action="store_true")
    run(parser.parse_args())
