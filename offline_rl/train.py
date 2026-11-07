"""离线 IQL 与 BC 试验；只用训练转移更新，验证仿真选模型，测试独立。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from baseline.data import DEFAULT_MODEL
from offline_rl.data import DEFAULT_REPLAY, Replay, save_json, sha256
from offline_rl.evaluate import evaluate
from offline_rl.iql import IQL


LOSS_COLUMNS = ("q_loss", "value_loss", "iql_actor_loss", "bc_loss", "advantage", "weight", "q_mean")


def atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def evaluation_output(path):
    """恢复时保留已完成或中断的评估记录，新尝试写入独立目录。"""
    path = Path(path)
    candidate, attempt = path, 0
    while candidate.exists():
        attempt += 1
        candidate = path.with_name(f"{path.name}_retry{attempt:03d}")
    return candidate


def export_policy(learner, config, metadata, name, step, path):
    narrow = name.endswith("9")
    actor = learner.bc if name.startswith("bc") else learner.actor
    prices = sorted({pair[0] for pair in metadata["action_pairs"]})
    radii = sorted({pair[1] for pair in metadata["action_pairs"]})
    atomic_save(path, {
        "schema_version": "multiagent_iql_policy_v1", "algorithm": "BC" if name.startswith("bc") else "IQL",
        "step": step, "model_config": config["model"], "replay_metadata": metadata,
        "actor_state_dict": {key: value.detach().cpu() for key, value in actor.state_dict().items()},
        "allowed_price_indices": [prices.index(x) for x in (1.4, 1.6, 1.8)] if narrow else None,
        "allowed_radius_indices": [radii.index(x) for x in (1.5, 2., 2.5)] if narrow else None,
        "selection": "maximum mean platform revenue on held-out validation seeds",
    })


def plot_training(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from baseline.plot_paper_figures import _font_family
    from offline_rl.evaluate import LABELS

    output = Path(output)
    history = pd.read_csv(output / "training_history.csv")
    validation = pd.read_csv(output / "validation_history.csv")
    with plt.rc_context({"font.family": _font_family(), "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(8, 5.7))
        fig.subplots_adjust(left=.1, right=.98, bottom=.1, top=.94, hspace=.45, wspace=.3)
        for ax, column, title in zip(axes.flat[:2], ("q_loss", "value_loss"),
                                     ("(a) Twin-Q Bellman loss", "(b) Expectile value loss")):
            ax.plot(history.step, history[column], color="#0072B2", linewidth=.9)
            ax.set_title(title, loc="left")
            ax.set_yscale("log")
        ax = axes[1, 0]
        ax.plot(history.step, history.iql_actor_loss, label="IQL (weighted)", color="#D55E00", linewidth=.9)
        ax.plot(history.step, history.bc_loss, label="BC (unweighted)", color="#0072B2", linewidth=.9)
        ax.set_title("(c) Policy losses (different objectives)", loc="left")
        ax.legend(frameon=False, fontsize=8)
        ax = axes[1, 1]
        colors = ("#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#777777")
        for name, color in zip(("bc", "bc9", "iql", "iql9", "fixed_f3"), colors):
            values = validation[validation.policy == name]
            ax.plot(values.step, values.revenue_mean, "o-", markersize=3, linewidth=.9,
                    color=color, label=LABELS[name])
        ax.set_title("(d) Validation platform revenue ↑", loc="left")
        ax.set_ylabel("USD / 280 min")
        ax.legend(frameon=False, fontsize=7, ncol=3)
        for ax in axes.flat:
            ax.set_xlabel("Gradient updates")
            ax.grid(color="#DDDDDD", linewidth=.4)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"training_curves.{extension}", dpi=600, facecolor="white")
        plt.close(fig)


def train(args):
    output, replay_root = Path(args.output), Path(args.replay_root)
    if min(args.steps, args.batch_size, args.eval_every, args.validation_runs, args.threads) <= 0:
        raise ValueError
    replay = Replay(replay_root, "train")
    metadata = replay.metadata
    device = args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pairs = np.asarray(metadata["action_pairs"])
    model_config = dict(state_size=metadata["state_size"], grid_count=metadata["grid_count"],
                        price_count=len(np.unique(pairs[:, 0])), radius_count=len(np.unique(pairs[:, 1])),
                        local_indices=metadata["local_state_indices"], hidden_size=args.hidden_size, embedding_size=16)
    algorithm = dict(learning_rate=3e-4, discount=.99, expectile=.7, temperature=3.,
                     target_tau=.005, reward_scale=100.)
    config = {"model": model_config, "algorithm": algorithm, "steps": args.steps,
              "batch_size": args.batch_size, "seed": args.seed, "device": device,
              "threads": args.threads, "eval_every": args.eval_every, "validation_runs": args.validation_runs,
              "validation_selection_seed": 1771, "test_selection_seed": 20261102,
              "replay_root": str(replay_root.resolve()), "replay_manifest_sha256": sha256(replay_root / "manifest.json"),
              "training_transitions": replay.count, "training_trajectories": metadata["splits"]["train"]["runs"],
              "actor": "six independent local-observation agents, parallel price/radius heads",
              "critic": "centralized twin Q with all six grids' histories, pre-action states and joint actions",
              "method": "IQL losses adapted to a product of 12 categorical distributions; not an official multi-agent implementation",
              "reward": "global platform commission in USD / 100; finite 10:00 horizon; discount 0.99 per 5 min",
              "model_selection": "held-out validation simulation revenue, never test labels or returns",
              "sources": ["https://arxiv.org/abs/2110.06169", "https://github.com/ikostrikov/implicit_q_learning"],
              "source_sha256": {str(p): sha256(p) for p in sorted(Path("offline_rl").glob("*.py"))}}
    learner = IQL(model_config, device=device, **algorithm)
    history, validation_history, best, first = [], [], {}, 0
    if args.resume:
        previous = json.loads((output / "training_config.json").read_text())
        if any(config[key] != previous[key] for key in (
                "model", "algorithm", "steps", "batch_size", "seed", "replay_manifest_sha256",
                "eval_every", "validation_runs")):
            raise ValueError
        checkpoint = torch.load(output / "latest.pt", map_location=device, weights_only=True)
        learner.load_state_dict(checkpoint["learner"])
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        first, history, validation_history, best = (checkpoint[k] for k in
                                                  ("step", "history", "validation_history", "best"))
        config = previous
        completed = json.loads((output / "training_status.json").read_text())
        if first == args.steps and completed["status"] == "complete":
            print("该训练及测试已经完成，保留现有结果。", flush=True)
            return completed
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "checkpoints").mkdir()
        save_json(output / "training_config.json", config)
    status = {"status": "running", "device": device, "step": first, "target_steps": args.steps,
              "started_at": datetime.now(timezone.utc).isoformat(), "best": best}
    save_json(output / "training_status.json", status)
    started = perf_counter()
    accumulated = torch.zeros(len(LOSS_COLUMNS), device=device)
    last_logged = first

    def checkpoint(step):
        atomic_save(output / "latest.pt", {"step": step, "learner": learner.state_dict(),
                    "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "history": history, "validation_history": validation_history, "best": best})

    print(f"开始训练：{device}, 六智能体×双动作头, {replay.count:,} 条训练转移，{args.steps:,} 次更新", flush=True)
    try:
        for step in range(first + 1, args.steps + 1):
            indices = rng.integers(replay.count, size=args.batch_size)
            accumulated += learner.update(replay.batch(indices))
            # 与官方实现一致，策略学习率余弦衰减；Q/V 学习率保持不变。
            multiplier = .5 * (1 + np.cos(np.pi * step / args.steps))
            for name in ("actor", "bc"):
                for group in learner.optimizers[name].param_groups:
                    group["lr"] = float(algorithm["learning_rate"] * multiplier)
            if step % 250 == 0 or step == args.steps:
                statistics = (accumulated / (step - last_logged)).cpu().numpy()
                if not np.isfinite(statistics).all():
                    raise FloatingPointError
                history.append({"step": step, **dict(zip(LOSS_COLUMNS, map(float, statistics))),
                                "elapsed_seconds": perf_counter() - started})
                pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
                accumulated.zero_()
                last_logged = step
                status.update(step=step, elapsed_seconds=perf_counter() - started)
                save_json(output / "training_status.json", status)
                print(f"update={step:,}/{args.steps:,} Q={statistics[0]:.4f} V={statistics[1]:.4f} "
                      f"IQL={statistics[2]:.3f} BC={statistics[3]:.3f} elapsed={perf_counter()-started:.1f}s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                definitions = {"fixed_f3": None}
                for name in ("bc", "bc9", "iql", "iql9"):
                    path = output / "checkpoints" / f"step{step:06d}_{name}.pt"
                    export_policy(learner, config, metadata, name, step, path)
                    definitions[name] = str(path)
                validation_output = evaluation_output(output / "validation" / f"step{step:06d}")
                summary = evaluate(replay_root, definitions, validation_output,
                                   split="validation", runs=args.validation_runs,
                                   selection_seed=config["validation_selection_seed"], workers=args.workers, plot=False)
                for name in definitions:
                    revenue = float(summary.loc[name, "platform_revenue_mean"])
                    validation_history.append({"step": step, "policy": name, "revenue_mean": revenue})
                    if name != "fixed_f3" and revenue > best.get(name, {}).get("validation_revenue", -float("inf")):
                        shutil.copy2(definitions[name], output / f"best_{name}.pt")
                        best[name] = {"step": step, "validation_revenue": revenue}
                pd.DataFrame(validation_history).to_csv(output / "validation_history.csv", index=False)
                status["best"] = best
                torch.set_num_threads(args.threads)
            if step % 1000 == 0 or step == args.steps:
                checkpoint(step)
        plot_training(output)
        status.update(status="trained", step=args.steps, training_seconds=perf_counter() - started, best=best)
        save_json(output / "training_status.json", status)
        if args.test_runs:
            definitions = {"fixed_f3": None, "random": None,
                           **{name: str(output / f"best_{name}.pt") for name in ("bc", "bc9", "iql", "iql9")},
                           "lstm_greedy": str(Path(DEFAULT_MODEL) / "best_model.pt"),
                           "lstm_bcs": str(Path(DEFAULT_MODEL) / "best_model.pt")}
            test_output = evaluation_output(output / "test")
            evaluate(replay_root, definitions, test_output, runs=args.test_runs,
                     selection_seed=config["test_selection_seed"], workers=args.workers)
            status["test_output"] = str(test_output)
        status.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                      total_seconds=perf_counter() - started)
        save_json(output / "training_status.json", status)
        return status
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, best=best)
        save_json(output / "training_status.json", status)
        raise


def parser():
    p = argparse.ArgumentParser(description="六智能体双动作头 IQL 离线训练与闭环对比")
    p.add_argument("--replay-root", default=DEFAULT_REPLAY)
    p.add_argument("--output", default="models/iql_grid6_low_20261102")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--validation-runs", type=int, default=5)
    p.add_argument("--test-runs", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
