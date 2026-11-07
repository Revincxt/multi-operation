"""五初始化、全动作支持的独立/自回归IQL成对实验。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from baseline.compare_policies import METRICS
from offline_rl.data import DEFAULT_REPLAY, Replay, save_json, sha256
from offline_rl.evaluate import evaluate, plot_comparison, selected_seeds
from offline_rl.joint_policy import PairedIQL
from offline_rl.train import atomic_save


DEFAULT_OUTPUT = "models/iql_joint_ablation_20261102"
REFERENCE = "models/iql_grid6_low_20261102/best_iql9.pt"
LABELS = {"fixed_f3": "F3", "independent": "Ind-IQL", "autoregressive": "AR-IQL",
          "bc_independent": "Ind-BC", "bc_autoregressive": "AR-BC", "incumbent": "Old IQL-9"}


def hierarchical_statistics(frame, policy, reference, draws=10000):
    """先重采样训练初始化，再在初始化内重采样配对仿真种子。"""
    pivot = frame.pivot(index=["training_seed", "seed"], columns="policy", values="platform_revenue")
    blocks = [(part[policy] - part[reference]).to_numpy() for _, part in pivot.groupby(level=0)]
    if not blocks or len({len(x) for x in blocks}) != 1:
        raise ValueError
    values = np.stack(blocks)
    rng = np.random.default_rng(42)
    train_indices = rng.integers(len(values), size=(draws, len(values), 1))
    environment_indices = rng.integers(values.shape[1], size=(draws, len(values), values.shape[1]))
    bootstrap = values[train_indices, environment_indices].mean((1, 2))
    return {"policy": policy, "reference": reference, "training_initializations": len(values),
            "paired_simulations": int(values.size), "mean_difference_usd": float(values.mean()),
            "relative_percent": float(100 * values.mean() / pivot[reference].mean()),
            "ci95_low": float(np.quantile(bootstrap, .025)), "ci95_high": float(np.quantile(bootstrap, .975)),
            "wins": int((values > 1e-8).sum()), "training_seed_mean_differences": values.mean(1).tolist(),
            "design": "balanced nested training/environment seeds; not a fully crossed design"}


def known_seeds():
    """排除既有闭环回放中的种子；不读取训练标签来选择测试种子。"""
    found = set()
    for directory in (Path("experiments"), Path("models")):
        for path in directory.rglob("manifest.json"):
            record = json.loads(path.read_text())
            values = record.get("seeds", [])
            if isinstance(values, list):
                found.update(int(x) for x in values if isinstance(x, (int, float)))
    return found


def make_protocol(args, metadata):
    excluded = known_seeds()
    blocks = {}
    for split, count, namespace in (("validation", args.validation_per_seed, 41000),
                                    ("test", args.test_per_seed, 42000)):
        used, planned = set(excluded), []
        for seed in args.training_seeds:
            excluded_here = sorted(used)
            run_ids, seeds = selected_seeds(args.replay_root, split, count, namespace + seed, excluded_here)
            planned.append({"training_seed": seed, "selection_seed": namespace + seed,
                            "excluded_seeds": excluded_here, "run_ids": run_ids, "seeds": seeds})
            used.update(seeds)
        blocks[split] = planned
    prices = sorted({p for p, _ in metadata["action_pairs"]})
    radii = sorted({r for _, r in metadata["action_pairs"]})
    return {"schema_version": "paired_joint_iql_ablation_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args), "model": {"state_size": metadata["state_size"], "grid_count": metadata["grid_count"],
            "price_count": len(prices), "radius_count": len(radii), "price_values": prices, "radius_values": radii,
            "hidden_size": args.hidden_size, "embedding_size": 16},
        "algorithm": {"discount": .99, "expectile": .7, "temperature": 3., "reward_scale": 100.,
                      "learning_rate": 3e-4, "target_tau": .005, "critic_hidden_size": 256},
        "selection": "fixed final update budget; no checkpoint search or test-based tuning",
        "seed_blocks": blocks, "critic": "same newly trained unrestricted joint twin Q/V for both actors within each initialization",
        "actor_control": "same global observation, parameter count, initialization, batches and advantage weights; conditional own price/radius in both",
        "prefix_control": "three non-learned features per grid: normalized price, radius, known-prefix flag; all zero in independent control",
        "support": "all 11 prices x 10 radii, no masks or F3 fallback",
        "execution": "centralized joint information; autoregressive computational decoding followed by simultaneous simulator action",
        "scope": "IQL policy-representation adaptation; not an InSPO implementation or a recurrent-policy experiment",
        "primary_comparisons": ["autoregressive vs independent", "autoregressive vs fixed_f3"],
        "deployment_gate": "validation selected architecture >= 0.5% over F3 and positive hierarchical CI; test positive hierarchical CIs vs F3 and incumbent; then separate 50-seed single-checkpoint confirmation",
        "replay_manifest_sha256": sha256(Path(args.replay_root) / "manifest.json"),
        "reference_checkpoint_sha256": sha256(REFERENCE),
        "source_sha256": {str(p): sha256(p) for p in sorted(Path("offline_rl").glob("*.py"))},
        "sources": ["https://arxiv.org/abs/2110.06169", "https://ojs.aaai.org/index.php/AAAI/article/view/34099"]}


def export(learner, protocol, metadata, seed, output):
    for name, actor in learner.actors.items():
        architecture = "autoregressive" if name.endswith("autoregressive") else "global_conditional"
        atomic_save(output / f"{name}.pt", {"schema_version": "multiagent_iql_policy_v2",
            "algorithm": "BC" if name.startswith("bc_") else "IQL", "actor_architecture": architecture,
            "model_config": protocol["model"], "replay_metadata": metadata, "training_seed": seed,
            "step": protocol["args"]["steps"],
            "actor_state_dict": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
            "allowed_price_indices": None, "allowed_radius_indices": None})


def train_seed(args, protocol, replay, seed, root):
    output = root / f"train_seed{seed}"
    if (output / "complete.json").exists():
        return
    output.mkdir(exist_ok=True)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    learner = PairedIQL(protocol["model"], device=args.device, **protocol["algorithm"])
    rng, first, history = np.random.default_rng(seed), 0, []
    if (output / "latest.pt").exists():
        state = torch.load(output / "latest.pt", map_location=args.device, weights_only=True)
        learner.load_state_dict(state["learner"])
        rng.bit_generator.state = state["numpy_rng"]
        first, history = state["step"], state["history"]
    sums = torch.zeros(len(learner.metrics), device=args.device)
    timer, logged = perf_counter(), first
    previous_seconds = history[-1]["elapsed_seconds"] if history else 0.
    validation = Replay(args.replay_root, "validation")
    offline_scores = json.loads((output / "offline_validation.json").read_text()) if (output / "offline_validation.json").exists() else []
    for step in range(first + 1, args.steps + 1):
        batch = {k: torch.as_tensor(v, device=args.device) for k, v in replay.batch(
            rng.integers(replay.count, size=args.batch_size)).items()}
        sums += learner.update(batch)
        rate = float(3e-4 * .5 * (1 + np.cos(np.pi * step / args.steps)))
        for name in learner.names:
            for group in learner.optimizers[name].param_groups:
                group["lr"] = rate
        if step % 250 == 0 or step == args.steps:
            values = (sums / (step - logged)).cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError
            elapsed = previous_seconds + perf_counter() - timer
            history.append({"step": step, "elapsed_seconds": elapsed, **dict(zip(learner.metrics, map(float, values)))})
            pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
            save_json(root / "status.json", {"status": "training", "training_seed": seed,
                "step": step, "target_steps": args.steps, "completed_training_seeds": args.training_seeds.index(seed),
                "total_training_seeds": len(args.training_seeds), "seed_elapsed_seconds": elapsed})
            print(f"train_seed={seed} update={step}/{args.steps} Q={values[0]:.4f} Ind/AR={values[2]:.3f}/{values[3]:.3f} elapsed={elapsed:.1f}s", flush=True)
            sums.zero_()
            logged = step
        if step % 5000 == 0 or step == args.steps:
            scoring_rng = np.random.default_rng(901)
            scores = {name: 0. for name in learner.names}
            with torch.no_grad():
                for _ in range(10):
                    sample = validation.batch(scoring_rng.integers(validation.count, size=512))
                    s = torch.as_tensor(sample["states"], device=args.device)
                    a = torch.as_tensor(sample["actions"], device=args.device)
                    for name, actor in learner.actors.items():
                        scores[name] += float(-actor.log_prob(s, a).mean().cpu()) / 10
            offline_scores.append({"step": step, **scores})
            save_json(output / "offline_validation.json", offline_scores)
        if step % 1000 == 0 or step == args.steps:
            atomic_save(output / "latest.pt", {"step": step, "learner": learner.state_dict(),
                        "numpy_rng": rng.bit_generator.state, "history": history})
    export(learner, protocol, replay.metadata, seed, output)
    save_json(output / "complete.json", {"step": args.steps, "training_seed": seed,
        "elapsed_seconds": history[-1]["elapsed_seconds"],
        "actor_parameters_each": sum(p.numel() for p in learner.actors["independent"].parameters()),
        "checkpoint_sha256": {name: sha256(output / f"{name}.pt") for name in learner.names}})
    del learner
    if args.device == "mps":
        torch.mps.empty_cache()


def evaluate_blocks(args, protocol, root, split):
    stage = root / split
    stage.mkdir(exist_ok=True)
    frames = []
    for block in protocol["seed_blocks"][split]:
        seed = block["training_seed"]
        output = stage / f"train_seed{seed}"
        record = output / "manifest.json"
        if not record.exists() or json.loads(record.read_text())["status"] != "complete":
            if output.exists():
                suffix = 1
                while output.with_name(output.name + f"_incomplete{suffix}").exists():
                    suffix += 1
                output.rename(output.with_name(output.name + f"_incomplete{suffix}"))
            definitions = {"fixed_f3": None, **{name: str(root / f"train_seed{seed}" / f"{name}.pt")
                            for name in PairedIQL.names}, "incumbent": REFERENCE}
            save_json(root / "status.json", {"status": f"evaluating_{split}", "training_seed": seed,
                      "runs_per_training_seed": len(block["seeds"])})
            evaluate(args.replay_root, definitions, output, split=split, runs=len(block["seeds"]),
                     selection_seed=block["selection_seed"], workers=args.workers, plot=False,
                     exclude_seeds=block["excluded_seeds"], labels=LABELS)
        assert json.loads(record.read_text())["seeds"] == block["seeds"]
        frame = pd.read_csv(output / "run_summary.csv")
        frame["training_seed"] = seed
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    assert not frame.duplicated(["policy", "seed"]).any()
    frame.to_csv(stage / "run_summary.csv", index=False)
    summary = frame.groupby("policy")[list(METRICS)].agg(["mean", "std"]).reindex(LABELS)
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv(stage / "policy_summary.csv")
    save_json(stage / "manifest.json", {"schema_version": "nested_training_seed_evaluation_v1", "status": "complete",
        "split": split, "policies": dict.fromkeys(LABELS), "policy_labels": LABELS,
        "training_seeds": args.training_seeds,
        "seeds": sorted(frame.seed.unique().tolist()), "source_blocks": protocol["seed_blocks"][split],
        "plot_caption": f"{len(args.training_seeds)} training initializations; {frame.seed.nunique()} disjoint paired simulation seeds; 05:20–10:00. Means ± SD."})
    comparisons = []
    for policy in ("independent", "autoregressive"):
        for reference in ("fixed_f3", "incumbent", f"bc_{policy}",
                          "autoregressive" if policy == "independent" else "independent"):
            comparisons.append(hierarchical_statistics(frame, policy, reference))
    save_json(stage / "hierarchical_comparisons.json", comparisons)
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, list)} for row in comparisons]).to_csv(
        stage / "paired_revenue_comparisons.csv", index=False)
    plot_comparison(stage)
    return frame, comparisons


def plot_training(root, training_seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from baseline.plot_paper_figures import _font_family

    frames = []
    for seed in training_seeds:
        f = pd.read_csv(root / f"train_seed{seed}" / "training_history.csv")
        f["training_seed"] = seed
        frames.append(f)
    history = pd.concat(frames)
    history.to_csv(root / "training_history.csv", index=False)
    with plt.rc_context({"font.family": _font_family(), "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
        for ax, columns, title in ((axes[0], ["q_loss"], "(a) Joint-Q Bellman loss ↓"),
                                   (axes[1], ["independent_loss", "autoregressive_loss"], "(b) IQL actor objective ↓"),
                                   (axes[2], ["bc_independent_loss", "bc_autoregressive_loss"], "(c) BC joint NLL ↓")):
            for column, color in zip(columns, ["#0072B2", "#D55E00"]):
                means = history.groupby("step")[column].agg(["mean", "std"])
                ax.plot(means.index, means["mean"], color=color, linewidth=1,
                        label=column.removesuffix("_loss").replace("bc_", ""))
                ax.fill_between(means.index, means["mean"] - means["std"], means["mean"] + means["std"],
                                color=color, alpha=.15)
            ax.set_title(title, loc="left")
            ax.set_xlabel("Gradient updates")
            ax.grid(color="#DDDDDD", linewidth=.4)
            if len(columns) > 1:
                ax.legend(frameon=False, fontsize=8)
        axes[0].set_yscale("log")
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(root / f"training_curves.{ext}", dpi=600, facecolor="white")
        plt.close(fig)


def confirm_deployment(args, protocol, root, selection, validation_comparisons, test_comparisons):
    name = selection["architecture"]
    val = next(x for x in validation_comparisons if x["policy"] == name and x["reference"] == "fixed_f3")
    tests = [x for x in test_comparisons if x["policy"] == name and x["reference"] in ("fixed_f3", "incumbent")]
    result = {"selected": selection, "adopted": False,
              "architecture_passed": val["relative_percent"] >= .5 and val["ci95_low"] > 0
                and len(tests) == 2 and all(x["ci95_low"] > 0 for x in tests)}
    if result["architecture_passed"]:
        excluded = known_seeds()
        definitions = {"fixed_f3": None, "selected": selection["checkpoint"], "incumbent": REFERENCE}
        output = root / "deployment_confirmation"
        evaluate(args.replay_root, definitions, output, runs=50, selection_seed=43001,
                 workers=args.workers, exclude_seeds=sorted(excluded),
                 labels={"fixed_f3": "F3", "selected": LABELS[name], "incumbent": LABELS["incumbent"]})
        from offline_rl.optimize import paired_statistics
        frame = pd.read_csv(output / "run_summary.csv")
        stats = [paired_statistics(frame, "selected", ref) for ref in ("fixed_f3", "incumbent")]
        result["single_checkpoint_confirmation"] = stats
        result["adopted"] = all(x["ci95_low"] > 0 for x in stats)
        if result["adopted"]:
            save_json(Path("offline_rl/active_policy.json"), {"checkpoint": selection["checkpoint"],
                "sha256": selection["sha256"], "variant": "joint_iql_" + name,
                "adoption_record": str(root / "adoption.json")})
    save_json(root / "adoption.json", result)
    return result


def run(args):
    root = Path(args.output)
    if root.exists() and not args.resume:
        raise FileExistsError
    if args.resume and (root / "status.json").exists() and json.loads((root / "status.json").read_text())["status"] == "complete":
        print("本次成对实验已经完成，保留现有结果。", flush=True)
        return
    root.mkdir(exist_ok=True)
    replay = Replay(args.replay_root, "train")
    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    if (root / "protocol.json").exists():
        protocol = json.loads((root / "protocol.json").read_text())
        keys = ("steps", "batch_size", "training_seeds", "hidden_size", "replay_root")
        if any(protocol["args"][k] != vars(args)[k] for k in keys):
            raise ValueError
        if any(sha256(path) != digest for path, digest in protocol["source_sha256"].items()):
            raise ValueError
    else:
        protocol = make_protocol(args, replay.metadata)
        save_json(root / "protocol.json", protocol)
        for name in protocol["source_sha256"]:
            destination = root / "code_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, destination)
    timer = perf_counter()
    for seed in args.training_seeds:
        train_seed(args, protocol, replay, seed, root)
    plot_training(root, args.training_seeds)
    validation, validation_stats = evaluate_blocks(args, protocol, root, "validation")
    gains = {name: next(x["mean_difference_usd"] for x in validation_stats
                       if x["policy"] == name and x["reference"] == "fixed_f3")
             for name in ("independent", "autoregressive")}
    architecture = max(gains, key=gains.get)
    paired = validation.pivot(index=["training_seed", "seed"], columns="policy", values="platform_revenue")
    seed = int((paired[architecture] - paired.fixed_f3).groupby(level=0).mean().idxmax())
    checkpoint = root / f"train_seed{seed}" / f"{architecture}.pt"
    selection = {"architecture": architecture, "training_seed": seed, "checkpoint": str(checkpoint),
                 "sha256": sha256(checkpoint), "selection_uses_test": False, "validation_gains_usd": gains}
    save_json(root / "selection.json", selection)
    _, test_stats = evaluate_blocks(args, protocol, root, "test")
    adoption = confirm_deployment(args, protocol, root, selection, validation_stats, test_stats)
    save_json(root / "status.json", {"status": "complete", "training_seeds": args.training_seeds,
        "updates_per_seed": args.steps, "elapsed_this_invocation_seconds": perf_counter() - timer,
        "adopted": adoption["adopted"], "completed_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"selection": selection, "adoption": adoption, "test_comparisons": test_stats},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全动作支持、五初始化的联合策略结构消融")
    parser.add_argument("--replay-root", default=DEFAULT_REPLAY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--training-seeds", type=lambda x: [int(v) for v in x.split(",")], default=[42, 43, 44, 45, 46])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--validation-per-seed", type=int, default=10)
    parser.add_argument("--test-per-seed", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.validation_per_seed, args.test_per_seed, args.workers) <= 0 or len(set(args.training_seeds)) != len(args.training_seeds):
        raise ValueError
    run(args)
