"""依次比较状态共享、条件双头和局部信用分配，通过配对验证才采用。"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from baseline.data import DEFAULT_MODEL
from offline_rl.coordination import OMIGA, broadcast_indices, make_actor
from offline_rl.data import DEFAULT_REPLAY, Replay, save_json, sha256
from offline_rl.evaluate import evaluate, selected_seeds
from offline_rl.iql import TwinQ, advantage_weights, mlp
from offline_rl.train import atomic_save, evaluation_output


DEFAULT_OUTPUT = "models/iql_coordination_20261102"
INCUMBENT = "models/iql_grid6_low_20261102"
NAMES = {"refit": "IQL-refit", "shared": "IQL-shared", "conditional": "IQL-cond",
         "bc_shared": "BC-cond", "omiga": "OMIGA-adapt"}
METRICS = ("refit_loss", "shared_loss", "conditional_loss", "bc_loss", "omiga_q_loss",
           "omiga_value_loss", "omiga_actor_loss", "omiga_weight", "omiga_q_mean")


def paired_statistics(frame, policy, reference):
    paired = frame.pivot(index="seed", columns="policy", values="platform_revenue")
    diff = (paired[policy] - paired[reference]).to_numpy()
    intervals = np.random.default_rng(42).choice(diff, (10000, len(diff)), replace=True).mean(axis=1)
    return {"policy": policy, "reference": reference, "seed_count": len(diff),
            "mean_difference_usd": float(diff.mean()),
            "relative_percent": float(100 * diff.mean() / paired[reference].mean()),
            "ci95_low": float(np.quantile(intervals, .025)), "ci95_high": float(np.quantile(intervals, .975)),
            "wins": int((diff > 1e-8).sum())}


def passes_gate(validation, testing):
    """按预先确定的门槛采用；测试失败时不再改选其他候选。"""
    return (validation["relative_percent"] >= .5 and testing["mean_difference_usd"] > 0
            and testing["ci95_low"] > 0)


def export_actor(actor, metadata, model_config, architecture, kind, step, path, narrow):
    prices = sorted({pair[0] for pair in metadata["action_pairs"]})
    radii = sorted({pair[1] for pair in metadata["action_pairs"]})
    atomic_save(path, {"schema_version": "multiagent_iql_policy_v2", "algorithm": kind,
                      "actor_architecture": architecture, "model_config": model_config,
                      "step": step, "replay_metadata": metadata,
                      "actor_state_dict": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
                      "allowed_price_indices": [prices.index(x) for x in (1.4, 1.6, 1.8)] if narrow else None,
                      "allowed_radius_indices": [radii.index(x) for x in (1.5, 2., 2.5)] if narrow else None})


def plot_history(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from baseline.plot_paper_figures import _font_family

    output = Path(output)
    history = pd.read_csv(output / "training_history.csv")
    validation = pd.read_csv(output / "validation_history.csv")
    colors = ("#777777", "#0072B2", "#D55E00", "#009E73", "#CC79A7")
    with plt.rc_context({"font.family": _font_family(), "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        fig.subplots_adjust(left=.1, right=.98, top=.94, bottom=.1, wspace=.32, hspace=.45)
        ax = axes[0, 0]
        for name, color in zip(("refit", "shared", "conditional"), colors):
            ax.plot(history.step, history[f"{name}_loss"], color=color, label=NAMES[name], linewidth=.9)
        ax.set_title("(a) IQL extraction losses", loc="left")
        ax.legend(frameon=False, fontsize=8)
        ax = axes[0, 1]
        ax.plot(history.step, history.omiga_q_loss, color="#CC79A7", linewidth=.9)
        ax.set_yscale("log")
        ax.set_title("(b) OMIGA global Bellman loss", loc="left")
        for ax, suffix, title in ((axes[1, 0], "", "(c) Validation revenue: full actions ↑"),
                                   (axes[1, 1], "9", "(d) Validation revenue: nine actions ↑")):
            for (name, label), color in zip(NAMES.items(), colors):
                rows = validation[validation.policy == name + suffix]
                ax.plot(rows.step, rows.revenue_mean, "o-", color=color, label=label,
                        linewidth=.9, markersize=2)
            ref = validation[validation.policy == "fixed_f3"]
            ax.plot(ref.step, ref.revenue_mean, "--", color="black", label="F3", linewidth=.8)
            ax.set_title(title, loc="left")
            ax.set_ylabel("USD / 280 min")
        axes[1, 1].legend(frameon=False, fontsize=7, ncol=2)
        for ax in axes.flat:
            ax.set_xlabel("Gradient updates")
            ax.grid(color="#DDDDDD", linewidth=.4)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(output / f"training_curves.{ext}", dpi=600, facecolor="white")
        plt.close(fig)


def train_screen(args):
    root, output = Path(args.replay_root), Path(args.output)
    replay = Replay(root, "train")
    meta = replay.metadata
    reference_path = Path(INCUMBENT) / "latest.pt"
    reference_config = json.loads((Path(INCUMBENT) / "training_config.json").read_text())
    if reference_config["replay_manifest_sha256"] != sha256(root / "manifest.json"):
        raise ValueError
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    device = args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(2)
    frozen_q = TwinQ(**reference_config["model"]).to(device)
    frozen_q.load_state_dict(reference["learner"]["models"]["target_q"])
    frozen_v = mlp(meta["state_size"], 256, 1).to(device)
    frozen_v.load_state_dict(reference["learner"]["models"]["value"])
    frozen_q.requires_grad_(False).eval()
    frozen_v.requires_grad_(False).eval()
    del reference
    base_config = reference_config["model"]
    shared_config = {**base_config, "local_indices": broadcast_indices(meta)}
    configs = {"refit": base_config, **dict.fromkeys(("shared", "conditional", "bc_shared", "omiga"), shared_config)}
    architecture = {name: "conditional" if name in ("conditional", "bc_shared", "omiga") else "parallel"
                    for name in NAMES}
    actors = {}
    for name in ("refit", "shared", "conditional"):
        torch.manual_seed(args.seed)
        actors[name] = make_actor(configs[name], architecture[name]).to(device)
    actors["bc_shared"] = deepcopy(actors["conditional"])
    optimizers = {name: torch.optim.Adam(actor.parameters(), lr=3e-4) for name, actor in actors.items()}
    torch.manual_seed(args.seed)
    omiga = OMIGA(shared_config, device=device, alpha=.3)
    omiga.actor.load_state_dict(actors["conditional"].state_dict())
    rng = np.random.default_rng(args.seed)
    config = {"steps": args.steps, "batch_size": args.batch_size, "seed": args.seed, "device": device,
              "model_configs": configs, "architectures": architecture,
              "replay_manifest_sha256": sha256(root / "manifest.json"), "reference_checkpoint_sha256": sha256(reference_path),
              "reference_checkpoint": str(reference_path.resolve()), "validation_selection_seed": 1771,
              "validation_runs": args.screen_runs, "eval_every": args.eval_every,
              "extra_eval_steps": [1000, 2500],
              "iql": "policy extraction with the same frozen 20k-update training-only Q/V for all three actor ablations",
              "omiga": {"alpha": .3, "discount": .99, "reward_scale": 100., "lr": 3e-4, "tau": .005,
                         "local_critic": "shared parameters with grid ID", "actor": "six independent conditional dual-head agents",
                         "source": "https://github.com/ZhengYinan-AIR/OMIGA/blob/master/algos/OMIGA.py"},
              "adoption_gate": "validation uplift >= 0.5%; fresh-test paired revenue difference CI95 lower > 0; no retuning after test",
              "source_sha256": {str(p): sha256(p) for p in sorted(Path("offline_rl").glob("*.py"))}}
    first, history, validation_history, best = 0, [], [], {}
    if args.resume:
        saved_config = json.loads((output / "training_config.json").read_text())
        if any(config[k] != saved_config[k] for k in ("steps", "batch_size", "seed", "model_configs",
                                                      "replay_manifest_sha256", "reference_checkpoint_sha256")):
            raise ValueError
        saved = torch.load(output / "latest.pt", map_location=device, weights_only=True)
        for name, actor in actors.items():
            actor.load_state_dict(saved["actors"][name])
            optimizers[name].load_state_dict(saved["optimizers"][name])
        omiga.load_state_dict(saved["omiga"])
        rng.bit_generator.state = saved["numpy_rng"]
        first, history, validation_history, best = (saved[k] for k in ("step", "history", "validation_history", "best"))
        config = saved_config
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "checkpoints").mkdir()
        save_json(output / "training_config.json", config)
    status = {"status": "screening", "step": first, "target_steps": args.steps,
              "started_at": datetime.now(timezone.utc).isoformat(), "best": best}
    save_json(output / "status.json", status)
    accumulated = torch.zeros(len(METRICS), device=device)
    start, logged = perf_counter(), first
    try:
        for step in range(first + 1, args.steps + 1):
            raw = replay.batch(rng.integers(replay.count, size=args.batch_size))
            batch = {name: torch.as_tensor(value, device=device) for name, value in raw.items()}
            with torch.no_grad():
                advantage = frozen_q.minimum(batch["states"], batch["actions"]) - frozen_v(batch["states"]).squeeze(-1)
                weights = advantage_weights(advantage, 3.)
            losses = []
            for name, actor in actors.items():
                logp = actor.log_prob(batch["states"], batch["actions"])
                loss = -logp.mean() if name == "bc_shared" else -(weights * logp).mean()
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[name].step()
                losses.append(loss.detach())
            omiga_statistics = omiga.update(batch)
            accumulated += torch.cat((torch.stack(losses), omiga_statistics))
            rate = float(3e-4 * .5 * (1 + np.cos(np.pi * step / args.steps)))
            for optimizer in [*optimizers.values(), omiga.optimizers["actor"]]:
                for group in optimizer.param_groups:
                    group["lr"] = rate
            if step % 250 == 0 or step == args.steps:
                values = (accumulated / (step - logged)).cpu().numpy()
                if not np.isfinite(values).all():
                    raise FloatingPointError
                history.append({"step": step, "elapsed_seconds": perf_counter() - start,
                                **dict(zip(METRICS, map(float, values)))})
                pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
                accumulated.zero_()
                logged = step
                status.update(step=step, elapsed_seconds=perf_counter() - start)
                save_json(output / "status.json", status)
                print(f"update={step}/{args.steps} local/shared/cond={values[:3]} OMIGA Q={values[4]:.4f} elapsed={perf_counter()-start:.1f}s", flush=True)
            if step in (1000, 2500) or step % args.eval_every == 0 or step == args.steps:
                definitions, labels = {"fixed_f3": None}, {"fixed_f3": "F3"}
                for name in NAMES:
                    actor = omiga.actor if name == "omiga" else actors[name]
                    for narrow in (False, True):
                        key = name + ("9" if narrow else "")
                        path = output / "checkpoints" / f"step{step:06d}_{key}.pt"
                        export_actor(actor, meta, configs[name], architecture[name], name, step, path, narrow)
                        definitions[key] = str(path)
                        labels[key] = NAMES[name] + ("-9" if narrow else "")
                validation_output = evaluation_output(output / "validation" / f"step{step:06d}")
                summary = evaluate(root, definitions, validation_output, split="validation", runs=args.screen_runs,
                                   selection_seed=1771, workers=args.workers, plot=False, labels=labels)
                for key in definitions:
                    revenue = float(summary.loc[key, "platform_revenue_mean"])
                    validation_history.append({"step": step, "policy": key, "revenue_mean": revenue})
                    if key != "fixed_f3" and revenue > best.get(key, {}).get("validation_revenue", -float("inf")):
                        shutil.copy2(definitions[key], output / f"best_{key}.pt")
                        best[key] = {"step": step, "validation_revenue": revenue,
                                     "checkpoint": str(output / f"best_{key}.pt")}
                pd.DataFrame(validation_history).to_csv(output / "validation_history.csv", index=False)
                status["best"] = best
                torch.set_num_threads(2)
            if step % 1000 == 0 or step == args.steps:
                atomic_save(output / "latest.pt", {"step": step,
                            "actors": {name: actor.state_dict() for name, actor in actors.items()},
                            "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
                            "omiga": omiga.state_dict(), "numpy_rng": rng.bit_generator.state,
                            "history": history, "validation_history": validation_history, "best": best})
        plot_history(output)
        status.update(status="screened", elapsed_seconds=perf_counter() - start)
        save_json(output / "status.json", status)
        return status
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__)
        save_json(output / "status.json", status)
        raise


def confirm_and_adopt(args, status):
    output, root = Path(args.output), Path(args.replay_root)
    ranking = sorted(status["best"].items(), key=lambda x: x[1]["validation_revenue"], reverse=True)
    candidates, seen = [], set()
    for name, info in ranking:
        architecture = name.removesuffix("9")
        if architecture == "bc_shared" or architecture in seen:
            continue
        candidates.append((name, info))
        seen.add(architecture)
        if len(candidates) == 2:
            break
    bc_key, bc_info = next((name, info) for name, info in ranking if name.startswith("bc_shared"))
    _, screen_seeds = selected_seeds(root, "validation", args.screen_runs, 1771)
    definitions = {"fixed_f3": None, "incumbent": str(Path(INCUMBENT) / "best_iql9.pt"),
                   "bc_old": str(Path(INCUMBENT) / "best_bc9.pt"), "bc_new": bc_info["checkpoint"],
                   **{f"candidate_{i}": info["checkpoint"] for i, (_, info) in enumerate(candidates)}}
    labels = {"fixed_f3": "F3", "incumbent": "Original IQL-9", "bc_old": "Original BC-9", "bc_new": "New BC"}
    labels.update({f"candidate_{i}": name for i, (name, _) in enumerate(candidates)})
    status["status"] = "confirming_validation"
    save_json(output / "status.json", status)
    validation_output = evaluation_output(output / "confirmation_validation")
    summary = evaluate(root, definitions, validation_output, split="validation", runs=args.confirm_runs,
                       selection_seed=20071, exclude_seeds=screen_seeds, workers=args.workers, labels=labels)
    selected = max((f"candidate_{i}" for i in range(len(candidates))), key=lambda n: summary.loc[n, "platform_revenue_mean"])
    index = int(selected.rsplit("_", 1)[1])
    name, info = candidates[index]
    validation_frame = pd.read_csv(validation_output / "run_summary.csv")
    validation_stats = paired_statistics(validation_frame, selected, "incumbent")
    selection = {"selected_variant": name, "checkpoint": info["checkpoint"],
                 "checkpoint_sha256": sha256(info["checkpoint"]), "validation_comparison": validation_stats,
                 "selection_uses_test": False, "bc_control": bc_key,
                 "confirmation_validation": str(validation_output)}
    save_json(output / "selection.json", selection)
    old_test = json.loads((Path(INCUMBENT) / "test/manifest.json").read_text())["seeds"]
    lstm_test_record = Path("experiments/lstm_mixed47k_bcs_retest_20261029/manifest.json")
    if lstm_test_record.exists():
        old_test = sorted(set(old_test) | set(json.loads(lstm_test_record.read_text())["seeds"]))
    testing = {"fixed_f3": None, "incumbent": definitions["incumbent"], "bc_old": definitions["bc_old"],
               "bc_new": definitions["bc_new"], "optimized": info["checkpoint"],
               "lstm_bcs": str(Path(DEFAULT_MODEL) / "best_model.pt")}
    test_labels = {k: v for k, v in labels.items() if k in testing}
    test_labels.update(optimized=name, lstm_bcs="LSTM-BCS")
    status["status"] = "testing"
    save_json(output / "status.json", status)
    test_output = evaluation_output(output / "test")
    evaluate(root, testing, test_output, runs=args.test_runs, selection_seed=20072,
             exclude_seeds=old_test, workers=args.workers, labels=test_labels)
    frame = pd.read_csv(test_output / "run_summary.csv")
    statistics = [paired_statistics(frame, "optimized", reference) for reference in testing if reference != "optimized"]
    pd.DataFrame(statistics).to_csv(test_output / "paired_optimized_comparisons.csv", index=False)
    test_stats = next(x for x in statistics if x["reference"] == "incumbent")
    adopted = passes_gate(validation_stats, test_stats) and not args.no_adopt
    active_checkpoint = info["checkpoint"] if adopted else definitions["incumbent"]
    decision = {**selection, "adopted": adopted, "test_comparisons": statistics,
                "active_checkpoint": active_checkpoint, "active_checkpoint_sha256": sha256(active_checkpoint),
                "test_output": str(test_output),
                "scope": "default offline RL policy; F3 remains an explicitly reported reference",
                "gate": "validation improvement >= 0.5%, fresh-test mean difference > 0 and CI95 lower > 0"}
    save_json(output / "adoption.json", decision)
    if not args.no_adopt:
        save_json(Path("offline_rl/active_policy.json"), {
            "checkpoint": active_checkpoint, "sha256": sha256(active_checkpoint),
            "variant": name if adopted else "original_iql9", "adoption_record": str(output / "adoption.json")})
    status.update(status="complete", adopted=adopted, active_checkpoint=active_checkpoint,
                  completed_at=datetime.now(timezone.utc).isoformat())
    save_json(output / "status.json", status)
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="协同策略消融、验证确认和默认模型采用")
    parser.add_argument("--replay-root", default=DEFAULT_REPLAY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--screen-runs", type=int, default=5)
    parser.add_argument("--confirm-runs", type=int, default=20)
    parser.add_argument("--test-runs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-adopt", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.screen_runs, args.confirm_runs, args.test_runs, args.workers) <= 0:
        raise ValueError
    existing_status = Path(args.output) / "status.json"
    if args.resume and existing_status.exists() and json.loads(existing_status.read_text())["status"] == "complete":
        print("本次优化与验证已经完成，保留现有采用决定。", flush=True)
    else:
        confirm_and_adopt(args, train_screen(args))
