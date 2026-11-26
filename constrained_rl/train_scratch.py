"""从同一随机初始化训练收益对照与约束PPO，按完整仿真更新并保存断点。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import fcntl
import multiprocessing as mp
import os
from pathlib import Path
import signal
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from constrained_rl.policy import ConstrainedPolicy
from constrained_rl.ppo import update
from constrained_rl.rollout import collect, initialize_worker
from constrained_rl.scratch_protocol import make_protocol, load_protocol
from constrained_rl.train import aggregate, cost_arrays
from offline_rl.data import save_json, sha256
from online_rl.train import atomic_save, cpu_weights


STOP = False


def stop(*_):
    global STOP
    STOP = True


def save_csv(rows, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def random_initial(plan, path, device):
    """仅接受本协议的随机初始点；不导入旧模型，也不拟合任何初始参数。"""
    torch.manual_seed(plan["training_seed"])
    model = ConstrainedPolicy(**plan["model"]).eval()
    identity = {"initialization": "random", "training_seed": plan["training_seed"],
                "protocol_sha256": sha256(Path(path).parent / "protocol.json")}
    if Path(path).exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError
        # 同种子重新构造的参数必须与保存的初始点一致，防止误用微调权重。
        for key, value in model.state_dict().items():
            if not torch.equal(value, saved["model_state_dict"][key]):
                raise ValueError
    else:
        atomic_save(path, {**identity, "model_config": plan["model"], "model_state_dict": cpu_weights(model)})
    return model.to(device)


def create_member(method, initial, plan):
    model = deepcopy(initial)
    return {"method": method, "model": model, "model_config": plan["model"],
            "optimizer": torch.optim.Adam(model.parameters(), lr=plan["algorithm"]["learning_rate"], eps=1e-5),
            "multipliers": np.zeros(2, dtype=float), "rng": np.random.default_rng(plan["training_seed"]),
            "episodes": 0, "iteration": 0, "history": []}


def save_member(member, root, protocol_hash, initial_hash):
    target = Path(root) / member["method"]
    target.mkdir(exist_ok=True)
    atomic_save(target / "latest.pt", {
        "initialization": "random", "method": member["method"], "protocol_sha256": protocol_hash,
        "initial_sha256": initial_hash, "model_config": member["model_config"],
        "model_state_dict": cpu_weights(member["model"]),
        "optimizer": member["optimizer"].state_dict(), "multipliers": member["multipliers"].tolist(),
        "numpy_rng": member["rng"].bit_generator.state, "episodes": member["episodes"],
        "iteration": member["iteration"], "history": member["history"]})


def restore_member(member, root, protocol_hash, initial_hash):
    checkpoint = Path(root) / member["method"] / "latest.pt"
    if not checkpoint.exists():
        return
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {"initialization": "random", "method": member["method"],
                "protocol_sha256": protocol_hash, "initial_sha256": initial_hash}
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError
    member["model"].load_state_dict(saved["model_state_dict"])
    member["optimizer"].load_state_dict(saved["optimizer"])
    member["multipliers"][:] = saved["multipliers"]
    member["rng"].bit_generator.state = saved["numpy_rng"]
    for key in ("episodes", "iteration", "history"):
        member[key] = saved[key]
    if len(member["history"]) != member["iteration"]:
        raise ValueError
    if member["history"] and member["history"][-1]["episodes"] != member["episodes"]:
        raise ValueError


def critic_diagnostics(episodes, reward_scale):
    """用本批真实整段回报检验采样时价值预测；不使用拟合后的训练内误差。"""
    signals = [np.stack([row["rewards"] for row in episodes]) / reward_scale]
    signals += [np.stack([row["costs"][:, k] for row in episodes]) for k in range(2)]
    predictions = [np.stack([row["values"] for row in episodes])]
    predictions += [np.stack([row["cost_values"][:, k] for row in episodes]) for k in range(2)]
    result = {}
    for name, signal_values, prediction in zip(("revenue", "service", "waiting"), signals, predictions):
        target = np.cumsum(signal_values[:, ::-1], axis=1)[:, ::-1]
        variance = float(np.var(target))
        result[name + "_mc_explained_variance"] = 1. - float(np.var(target - prediction)) / variance if variance > 1e-12 else 0.
        result[name + "_mc_mse"] = float(np.mean((target - prediction) ** 2))
    return result


def run(args):
    global STOP
    STOP = False
    root = Path(args.output)
    if not root.exists():
        make_protocol(root, episodes=args.episodes, training_seed=args.training_seed, smoke=args.smoke)
    lock = (root / ".train.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = load_protocol(root / "protocol.json")
    if args.workers <= 0 or (args.device == "mps" and not torch.backends.mps.is_available()):
        raise ValueError
    torch.set_num_threads(1 if args.device == "mps" else 2)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    initial = random_initial(plan, root / "initial.pt", args.device)
    protocol_hash, initial_hash = sha256(root / "protocol.json"), sha256(root / "initial.pt")
    members = [create_member(method, initial, plan) for method in plan["methods"]]
    for member in members:
        restore_member(member, root, protocol_hash, initial_hash)
        save_member(member, root, protocol_hash, initial_hash)
    evaluations = pd.read_csv(root / "evaluation.csv").to_dict("records") if (root / "evaluation.csv").exists() else []
    started = perf_counter()

    def write_history():
        rows = [row for member in members for row in member["history"]]
        if rows:
            save_csv(sorted(rows, key=lambda row: (row["iteration"], row["method"])), root / "training_history.csv")

    def progress(status, **extra):
        save_json(root / "progress.json", {"status": status, "pid": os.getpid(), "initialization": "random",
            "completed_episodes": sum(member["episodes"] for member in members),
            "target_episodes": len(members) * plan["training_episodes"],
            "runs": [{key: member[key] for key in ("method", "episodes", "iteration")} for member in members],
            "elapsed_this_process_seconds": perf_counter() - started, **extra})

    def evaluate(pool, policy, model, phase, iteration):
        done = {int(row["seed"]) for row in evaluations
                if row["phase"] == phase and row["policy"] == policy and row["iteration"] == iteration}
        pending = [seed for seed in plan["seeds"][phase] if seed not in done]
        if pending:
            progress("validating" if phase == "validation" else "testing", current=policy, iteration=iteration)
            rows = collect(pool, model, pending, plan["training_seed"], policy)
            evaluations.extend({"phase": phase, "policy": policy, "iteration": iteration, **row["summary"]} for row in rows)
            save_csv(evaluations, root / "evaluation.csv")
            print(f"{phase} policy={policy} update={iteration} " +
                  " ".join(f"{key}={value:.4f}" for key, value in aggregate(rows).items()), flush=True)

    def validation_due(member):
        return member["iteration"] > 0 and (
            member["iteration"] in plan["early_validation_updates"]
            or member["iteration"] % plan["validation_interval_updates"] == 0
            or member["episodes"] == plan["training_episodes"])

    def validate_member(pool, member):
        evaluate(pool, member["method"], member["model"], "validation", member["iteration"])
        target = root / member["method"] / "checkpoints"
        target.mkdir(exist_ok=True)
        atomic_save(target / f"episode_{member['episodes']:08d}.pt", {
            "model_state_dict": cpu_weights(member["model"]), "model_config": plan["model"],
            "protocol_sha256": protocol_hash, "initialization": "random", "episodes": member["episodes"]})

    write_history()  # checkpoint是恢复依据，补回可能因中断未写入CSV的最后一批。
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                initializer=initialize_worker, initargs=(plan["config"], plan["observation_metadata"], plan["model"])) as pool:
            if not any(member["episodes"] for member in members):
                evaluate(pool, "fixed_f3", initial, "validation", 0)
                if not STOP:
                    evaluate(pool, "random_initial", initial, "validation", 0)
            # 若上次在保存权重后、验证前暂停，先补齐该节点的验证再继续更新。
            for member in members:
                if validation_due(member) and not STOP:
                    validate_member(pool, member)
            while not STOP and any(member["episodes"] < plan["training_episodes"] for member in members):
                for member in members:
                    if STOP:
                        break
                    if member["episodes"] >= plan["training_episodes"]:
                        continue
                    method = member["method"]
                    progress("collecting", current=method, iteration=member["iteration"] + 1)
                    begin = perf_counter()
                    seeds = plan["seeds"]["train"][member["episodes"]:member["episodes"] + plan["episodes_per_update"]]
                    episodes = collect(pool, member["model"], seeds, plan["training_seed"], method)
                    cost_arrays(episodes, plan["constraints"])
                    collected = perf_counter()
                    progress("updating", current=method, iteration=member["iteration"] + 1)
                    fit = critic_diagnostics(episodes, plan["algorithm"]["reward_scale"])
                    diagnostic = update(member["model"], member["optimizer"], episodes, plan["algorithm"],
                        member["rng"], args.device, member["multipliers"], constrained=method == "constrained")
                    member["episodes"] += len(episodes)
                    member["iteration"] += 1
                    row = {"method": method, "iteration": member["iteration"], "episodes": member["episodes"],
                        "environment_steps": member["episodes"] * len(episodes[0]["rewards"]), **aggregate(episodes),
                        "collection_seconds": collected - begin, "learning_seconds": perf_counter() - collected,
                        **diagnostic, **fit, "lambda_service": float(member["multipliers"][0]),
                        "lambda_wait": float(member["multipliers"][1]),
                        "cost_service_mean": float(np.mean([e["costs"][:, 0].sum() for e in episodes])),
                        "cost_wait_mean": float(np.mean([e["costs"][:, 1].sum() for e in episodes]))}
                    member["history"].append(row)
                    save_member(member, root, protocol_hash, initial_hash)
                    write_history()
                    print(f"{method} update={member['iteration']} episodes={member['episodes']} "
                          f"revenue={row['revenue_mean_usd']:.2f} coverage={row['demand_satisfaction']:.4f} "
                          f"wait={row['wait_per_demand_seconds']:.2f} lambda={member['multipliers'].round(4).tolist()} "
                          f"KL={row['joint_kl']:.5f} entropy={row['final_entropy']:.3f} collect={row['collection_seconds']:.1f}s", flush=True)
                    validate = validation_due(member)
                    if validate and not STOP:
                        validate_member(pool, member)
                    if validate or member["iteration"] % 16 == 0:
                        from constrained_rl.scratch_report import run as report
                        report(root)
            if STOP:
                progress("paused")
                return
            # 旧模型仅在训练预算完成后读取，作为独立参考，不参与初始化或采样。
            frozen = ConstrainedPolicy(**plan["model"]).to(args.device).eval()
            reference = torch.load(plan["reference_checkpoint"], map_location="cpu", weights_only=True)
            frozen.load_original(reference["model_state_dict"], reward_scale=plan["algorithm"]["reward_scale"])
            policies = {"fixed_f3": initial, "frozen_ar_ppo": frozen,
                        **{member["method"]: member["model"] for member in members}}
            for name in plan["expected_test_policies"]:
                if STOP:
                    break
                evaluate(pool, name, policies[name], "test", 0)
            if STOP:
                progress("paused")
                return
            from constrained_rl.scratch_report import run as report
            report(root)
            load_protocol(root / "protocol.json")
            progress("complete")
            print("training_complete", flush=True)
    except BaseException as error:
        progress("failed", error=type(error).__name__)
        raise


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="models/constrained_ppo_scratch100k_20261116")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--training-seed", type=int, default=412)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(arguments())
