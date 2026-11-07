"""按冻结的交互预算训练；保存完整断点和每批行为策略、联合转移。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
import multiprocessing as mp
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from offline_rl.data import save_json, sha256
from online_rl.policy import JointPolicy
from online_rl.ppo import update
from online_rl.protocol import load_protocol
from online_rl.rollout import collect, initialize_worker


def cpu_weights(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def plot_training(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.read_csv(root/"training_history.csv")
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.7))
        for ax, column, title in zip(axes.flat,
            ("revenue_mean_usd", "policy_loss", "value_loss", "entropy", "joint_kl", "effective_epochs"),
            ("Return (USD) ↑", "Policy loss", "Value loss", "Conditional entropy", "Joint KL", "Effective epochs")):
            ax.plot(frame.environment_steps, frame[column], color="#268F85", lw=1.35, marker="o" if len(frame)<10 else None, ms=3)
            ax.set_title(title, loc="left"); ax.set_xlabel("City decision transitions")
            ax.grid(axis="y", alpha=.2)
        if (root/"validation_history.csv").exists():
            validation = pd.read_csv(root/"validation_history.csv")
            axes[0, 0].plot(validation.environment_steps, validation.revenue_mean_usd, color="#657FA8", label="Validation")
            axes[0, 0].legend(frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(root/f"training_curves.{ext}", dpi=600, facecolor="white")
        plt.close(fig)


def run(args):
    plan = load_protocol(args.protocol)
    if (args.training_seed not in plan["training_seeds"] or args.workers <= 0
            or (args.max_updates is not None and args.max_updates <= 0)):
        raise ValueError
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=args.resume)
    (root/"rollouts").mkdir(exist_ok=True)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(1 if device == "mps" else 4)
    torch.manual_seed(args.training_seed)
    config = dict(plan["model"], autoregressive=not args.factorized)
    model = JointPolicy(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=plan["algorithm"]["learning_rate"], eps=1e-5)
    rng = np.random.default_rng(args.training_seed)
    history, validation, completed, iteration, previous_seconds = [], [], 0, 0, 0.
    identity = {"protocol_sha256": sha256(args.protocol), "model_config": config, "training_seed": args.training_seed}
    latest = root/"latest.pt"
    if args.resume:
        saved = torch.load(latest, map_location=device, weights_only=True)
        if any(saved[key] != value for key, value in identity.items()):
            raise ValueError
        model.load_state_dict(saved["model_state_dict"]); optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["numpy_rng"]
        history, validation = saved["history"], saved["validation"]
        completed, iteration, previous_seconds = saved["episodes"], saved["iteration"], saved["elapsed_seconds"]
    save_json(root/"run_config.json", {**identity, "protocol": str(Path(args.protocol).resolve()),
        "device": device, "workers": args.workers, "parameters": sum(p.numel() for p in model.parameters()),
        "variant": "Factorized-PPO" if args.factorized else "AR-PPO", "initialization": "random weights"})
    timer = perf_counter()
    periods = plan["config"]["simulation_steps"]//plan["config"]["decision_freq"]-plan["config"]["history_steps"]
    worker_args = plan["config"], plan["observation_metadata"], config
    if args.workers == 1:
        initialize_worker(*worker_args)
        context = nullcontext(None)
    else:
        context = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
            initializer=initialize_worker, initargs=worker_args)
    with context as pool:
        while completed < plan["training_episodes"]:
            if args.max_updates is not None and iteration >= args.max_updates:
                break
            seeds = plan["seeds"]["train"][completed:completed+plan["episodes_per_update"]]
            weights = cpu_weights(model)
            version = iteration
            atomic_save(root/"rollouts"/f"behavior_{version:05d}.pt", weights)
            episodes = collect(pool, weights, seeds, args.training_seed, version)
            if any(row["policy_version"] != version for row in episodes):
                raise ValueError
            arrays = {name: np.stack([row[name] for row in episodes]) for name in
                      ("states", "actions", "log_probs", "values", "rewards", "terminals")}
            np.savez_compressed(root/"rollouts"/f"batch_{version:05d}.npz", **arrays,
                scenario_seeds=np.asarray(seeds), action_seeds=np.asarray([row["action_seed"] for row in episodes]))
            diagnostics = update(model, optimizer, episodes, plan["algorithm"], rng, device)
            completed += len(episodes); iteration += 1
            environment_steps = completed*arrays["rewards"].shape[1]
            revenue = [row["summary"]["platform_revenue"] for row in episodes]
            record = {"iteration": iteration, "episodes": completed, "environment_steps": environment_steps,
                "revenue_mean_usd": float(np.mean(revenue)), "revenue_std_usd": float(np.std(revenue)),
                "elapsed_seconds": previous_seconds+perf_counter()-timer, **diagnostics}
            history.append(record)
            pd.DataFrame(history).to_csv(root/"training_history.csv", index=False)
            pd.DataFrame([row["summary"] for row in episodes]).to_csv(root/"rollouts"/f"episodes_{version:05d}.csv", index=False)
            # 固定最终预算选checkpoint；验证曲线仅监测，完整主实验每16轮记录一次。
            if plan["stage"] == "main" and iteration%16 == 0:
                evaluated = collect(pool, cpu_weights(model), plan["seeds"]["validation"], args.training_seed, iteration)
                validation.append({"iteration": iteration, "environment_steps": environment_steps,
                    "revenue_mean_usd": float(np.mean([row["summary"]["platform_revenue"] for row in evaluated]))})
                pd.DataFrame(validation).to_csv(root/"validation_history.csv", index=False)
            payload = {**identity, "model_state_dict": cpu_weights(model), "optimizer": optimizer.state_dict(),
                "numpy_rng": rng.bit_generator.state, "history": history, "validation": validation,
                "episodes": completed, "iteration": iteration, "environment_steps": environment_steps,
                "elapsed_seconds": previous_seconds+perf_counter()-timer, "execution": plan["execution"]}
            atomic_save(latest, payload)
            state = "complete" if completed == plan["training_episodes"] else "training"
            save_json(root/"status.json", {"status": state, "stage": plan["stage"], **record})
            print(f"update={iteration} episodes={completed}/{plan['training_episodes']} city_steps={environment_steps} "
                  f"return={record['revenue_mean_usd']:.2f} KL={record['joint_kl']:.5f} reuse={record['effective_epochs']:.2f}", flush=True)
    if history:
        plot_training(root)
    if completed == plan["training_episodes"]:
        atomic_save(root/"policy.pt", {**identity, "model_state_dict": cpu_weights(model), "episodes": completed,
            "environment_steps": completed*periods, "execution": plan["execution"]})
    else:
        save_json(root/"status.json", {"status": "checkpointed", "episodes": completed, "iteration": iteration})
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-seed", type=int, default=81)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--factorized", action="store_true")
    run(parser.parse_args())
