"""冻结100k价值集成，训练小型时序注意力策略并独立检验运营收益。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from offline_rl.cooperative_policy import CooperativeActor, CooperativeDistillation
from offline_rl.coordination import make_actor
from offline_rl.data import Replay, save_json, sha256
from offline_rl.evaluate import evaluate, selected_seeds
from offline_rl.optimize import paired_statistics
from offline_rl.train import atomic_save
from offline_rl.train_joint import known_seeds


ROOT = Path("models/cooperative_full100k_20261103")
TEACHER = Path("models/value_guided_full100k_20261103/eq_soft.pt")
REPLAY = "datasets_grid6_low_mixed100k_20261103/offline_rl"
LABELS = {"fixed_f3": "F3", "eq_joint": "EQ-Joint", "learned": "CQPI actor", "refined": "CQPI refined"}


def export(root, actor, evaluator, teacher, config):
    common = {"schema_version": "multiagent_iql_policy_v2", "step": config["updates"],
        "algorithm": "IQL-critic-guided cooperative policy learning; not standard InSPO",
        "replay_metadata": teacher["replay_metadata"], "allowed_price_indices": None, "allowed_radius_indices": None}
    states = {name: value.detach().cpu() for name, value in actor.state_dict().items()}
    atomic_save(root / "learned.pt", {**common, "actor_architecture": "cooperative_distill",
        "model_config": config["actor"], "actor_state_dict": states})
    combined = {name: value.detach().cpu() for name, value in evaluator.state_dict().items()}
    combined.update({"proposal." + name: value for name, value in states.items()})
    atomic_save(root / "refined.pt", {**common, "actor_architecture": "cooperative_refined",
        "model_config": {**teacher["model_config"], "proposal_config": config["actor"]},
        "actor_state_dict": combined})


def plot_training(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = pd.read_csv(root / "training_history.csv")
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        warm = data[data.phase == "behavior_warmup"]
        value = data[data.phase == "value_improvement"]
        for ax, frame, column, title in ((axes[0], warm, "loss", "(a) Behavior initialization loss"),
            (axes[1], value, "loss", "(b) Value-target cross entropy"),
            (axes[2], value, "target_entropy", "(c) Soft target entropy")):
            ax.plot(frame.step, frame[column], color="#0072B2", lw=.9)
            ax.set_title(title, loc="left", fontsize=10)
            ax.set_xlabel("Gradient updates")
            ax.grid(alpha=.25)
        fig.tight_layout()
        for extension in ("png", "pdf", "svg"):
            fig.savefig(root / f"training_curves.{extension}", dpi=600, facecolor="white")
        plt.close(fig)


def train(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(2)
    torch.manual_seed(52)
    replay = Replay(REPLAY, "train")
    teacher = torch.load(TEACHER, map_location="cpu", weights_only=True)
    evaluator = make_actor(teacher["model_config"], teacher["actor_architecture"])
    evaluator.load_state_dict(teacher["actor_state_dict"])
    evaluator = evaluator.to(args.device)
    metadata = replay.metadata
    actor_config = {"state_size": metadata["state_size"], "grid_count": 6, "history_steps": 4,
        "history_features": 24, "context_size": 14,
        "price_values": sorted({pair[0] for pair in metadata["action_pairs"]}),
        "radius_values": sorted({pair[1] for pair in metadata["action_pairs"]}),
        "hidden_size": 64, "memory_size": 48}
    actor = CooperativeActor(**actor_config).to(args.device)
    learner = CooperativeDistillation(actor, evaluator)
    configuration = {"actor": actor_config, "updates": args.steps, "warmup_updates": args.warmup,
        "batch_size": args.batch_size, "seed": 52, "device": args.device,
        "method": "frozen IQL ensemble; exact 110-action local soft policy targets with lagged teammates and shared actor representation",
        "scope": "projected cooperative policy improvement; shared encoder changes multiple heads, no strict InSPO monotonic-improvement claim",
        "objective": {"behavior_kl_coefficient": .0001, "entropy_coefficient": .002, "uncertainty_coefficient": .25},
        "teacher_sha256": sha256(TEACHER), "replay_manifest_sha256": sha256(Path(REPLAY) / "manifest.json"),
        "source_sha256": {str(path): sha256(path) for path in Path("offline_rl").glob("*.py")},
        "inference": "learned own-joint MAP; or all-legal value refinement with learned third start, maximum 4621 joint scores",
        "model_selection": "fixed final update; select learned/refined inference with 20 validation seeds; new 100-seed final test"}
    rng, history, first, previous_elapsed = np.random.default_rng(52), [], 0, 0.
    seen = np.zeros(metadata["splits"]["train"]["runs"], dtype=bool)
    if args.resume:
        old = json.loads((root / "training_config.json").read_text())
        assert old == configuration
        saved = torch.load(root / "latest.pt", map_location=args.device, weights_only=True)
        learner.load_state_dict(saved["learner"])
        rng.bit_generator.state = saved["numpy_rng"]
        first, history = saved["step"], saved["history"]
        seen = saved["seen_runs"].cpu().numpy()
        previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.
    else:
        save_json(root / "training_config.json", configuration)
        for name in configuration["source_sha256"]:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
    timer, total, sums, count = perf_counter(), args.warmup + args.steps, torch.zeros(2, device=args.device), 0
    for step in range(first + 1, total + 1):
        indices = rng.integers(replay.count, size=args.batch_size)
        seen[indices // replay.periods] = True
        states = torch.as_tensor(np.array(replay.arrays["states"][indices]), device=args.device)
        if step <= args.warmup:
            actions = torch.as_tensor(np.array(replay.arrays["actions"][indices], dtype="int64"), device=args.device)
            result = learner.update(states, actions=actions)
        else:
            result = learner.update(states, grid=(step-args.warmup-1) % 6)
        sums += result
        count += 1
        rate = 3e-4 * (.1 + .9 * .5 * (1 + np.cos(np.pi * max(0, step-args.warmup) / args.steps)))
        for group in learner.optimizer.param_groups:
            group["lr"] = float(rate)
        if step % 250 == 0 or step == total:
            values = (sums / count).cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError
            phase = "behavior_warmup" if step <= args.warmup else "value_improvement"
            record = {"step": step, "phase": phase, "loss": float(values[0]), "target_entropy": float(values[1]),
                "elapsed_seconds": previous_elapsed+perf_counter()-timer, "training_trajectories_visited": int(seen.sum())}
            history.append(record)
            pd.DataFrame(history).to_csv(root / "training_history.csv", index=False)
            save_json(root / "status.json", {"status": "training", "step": step, "target_steps": total, **record})
            print(record, flush=True)
            sums.zero_(); count = 0
        if step % 500 == 0 or step == total:
            atomic_save(root / "latest.pt", {"step": step, "learner": learner.state_dict(), "numpy_rng": rng.bit_generator.state,
                "history": history, "seen_runs": torch.from_numpy(seen.copy())})
    export(root, actor, evaluator, teacher, configuration)
    plot_training(root)
    save_json(root / "training_complete.json", {"updates": total, "parameters": sum(p.numel() for p in actor.parameters()),
        "training_trajectories_visited": int(seen.sum()), "elapsed_seconds": previous_elapsed+perf_counter()-timer})
    del learner, evaluator, actor
    if args.device == "mps":
        torch.mps.empty_cache()


def run_evaluation(args):
    root = Path(args.output)
    # 验证沿用前一轮的验证池；这些样本不属于任何最终test池。
    old = json.loads(Path("models/value_guided_full100k_20261103/protocol.json").read_text())
    validation = old["blocks"]["validation"]
    protocol_path = root / "evaluation_protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        assert all(sha256(root / f"{name}.pt") == value for name, value in protocol["checkpoint_sha256"].items())
    else:
        used = sorted(known_seeds())
        test_ids, test_seeds = selected_seeds(REPLAY, "test", 100, 7300402, used)
        protocol = {"validation": validation, "test": {"run_ids": test_ids, "seeds": test_seeds,
            "selection_seed": 7300402, "excluded": used}, "selection_uses_test": False,
            "checkpoint_sha256": {name: sha256(root / f"{name}.pt") for name in ("learned", "refined")},
            "adoption": "require positive paired 95% CI versus F3 and EQ-Joint on test and a fresh 50-seed confirmation"}
        save_json(protocol_path, protocol)
    validation, used = protocol["validation"], protocol["test"]["excluded"]
    definitions = {"fixed_f3": None, "eq_joint": str(TEACHER),
                   "learned": str(root / "learned.pt"), "refined": str(root / "refined.pt")}
    save_json(root / "status.json", {"status": "evaluating_validation", "runs": 20})
    summary = evaluate_stage(REPLAY, definitions, root / "validation", split="validation", runs=20,
        selection_seed=validation["selection_seed"], exclude_seeds=validation["excluded"], workers=args.workers, labels=LABELS)
    candidates = {name: float(summary.loc[name, "platform_revenue_mean"]) for name in ("learned", "refined")}
    selected = max(candidates, key=candidates.get)
    save_json(root / "selection.json", {"selected": selected, "validation_revenues": candidates, "selection_uses_test": False})
    save_json(root / "status.json", {"status": "evaluating_test", "selected": selected, "runs": 100})
    evaluate_stage(REPLAY, {name: path for name, path in definitions.items() if name in ("fixed_f3", "eq_joint", selected)},
        root / "test", split="test", runs=100, selection_seed=7300402, exclude_seeds=used,
        workers=args.workers, labels=LABELS)
    frame = pd.read_csv(root / "test/run_summary.csv")
    statistics = [paired_statistics(frame, selected, name) for name in ("fixed_f3", "eq_joint")]
    save_json(root / "test/comparisons.json", statistics)
    save_json(root / "status.json", {"status": "complete", "selected": selected,
        "completed_at": datetime.now(timezone.utc).isoformat(), "test_comparisons": statistics})
    print(json.dumps(statistics, indent=2), flush=True)


def evaluate_stage(replay, definitions, output, **kwargs):
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("status") == "complete":
        return pd.read_csv(output / "policy_summary.csv", index_col=0)
    if output.exists():
        output.rename(output.with_name(output.name + "_interrupted_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")))
    return evaluate(replay, definitions, output, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT))
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (args.resume and (Path(args.output) / "training_complete.json").exists()):
        train(args)
    run_evaluation(args)
