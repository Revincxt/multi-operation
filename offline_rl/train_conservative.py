"""在完整100k轨迹上训练保守actor-critic，并用独立种子检验闭环收益。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from offline_rl.compact_replay import CompactReplay
from offline_rl.conservative_policy import ConservativeActorCritic
from offline_rl.data import Replay, save_json, sha256
from offline_rl.evaluate import selected_seeds
from offline_rl.optimize import paired_statistics
from offline_rl.train import atomic_save
from offline_rl.train_cooperative import evaluate_stage
from offline_rl.train_joint import known_seeds
from offline_rl.train_structured import BatchPrefetch


REPLAY = Path("datasets_grid6_low_mixed100k_20261103/offline_rl")
SOURCE = Path("models/structured_iql_full100k_20261103")
ROOT = Path("models/conservative_policy_full100k_20261103")
REFERENCES = {"fixed_f3": None, "eq_joint": "models/value_guided_full100k_20261103/eq_soft.pt",
              "joint_start": "models/structured_joint_search_20261103/joint_start.pt"}


def configuration(metadata):
    critic = json.loads((SOURCE/"protocol.json").read_text())["critic"]
    actor = {key: value for key, value in critic.items() if key not in ("basis_size", "rank")}
    actor["memory_size"] = 48
    features, context = metadata["preprocessor"]["history_features"], metadata["state_context_columns"]
    grids, steps = metadata["grid_count"], metadata["preprocessor"]["history_steps"]
    boundary = grids*steps*len(features)
    indices = [boundary+g*len(context)+context.index(f"start_{name}") for g in range(grids)
               for name in ("waiting_request_num", "idle_driver_num", "busy_driver_num", "repositioning_driver_num")]
    indices += [g*steps*len(features)+(steps-1)*len(features)+features.index("pickup_cancel_num") for g in range(grids)]
    return {"actor_config": actor, "critic_config": critic, "auxiliary_indices": indices}


def protocol(args, metadata):
    used = sorted(known_seeds())
    blocks = {}
    for split, count, namespace in (("validation", 30, 7300801), ("test", 100, 7300802)):
        if split == "validation" and args.validation_protocol:
            blocks[split] = json.loads(Path(args.validation_protocol).read_text())["blocks"]["validation"]
            seeds = blocks[split]["seeds"]
        else:
            ids, seeds = selected_seeds(REPLAY, split, count, namespace, used)
            blocks[split] = {"run_ids": ids, "seeds": seeds, "selection_seed": namespace, "excluded": used.copy()}
        used = sorted(set(used).union(seeds))
    return {"created_at": datetime.now(timezone.utc).isoformat(), "args": vars(args),
        "model": configuration(metadata), "training_seeds": [72, 73, 74], "source_training_seeds": [62, 63, 64],
        "algorithm": {"discount": 1., "reward_scale": 100., "conservative_weight": .05,
            "conservative_temperature": .1, "behavior_weight": .0001, "entropy_weight": .002,
            "auxiliary_weight": .02, "critic_learning_rate": 1e-4, "actor_learning_rate": 3e-4, "target_tau": .005},
        "method": "Deterministic target-policy evaluation, counterfactual and policy-action conservative Q regularization, projected regional policy improvement and physical-dynamics auxiliary tasks",
        "scope": "CFCQL-inspired project adaptation; shared actor encoder means regional updates are not exact independent-coordinate InSPO updates; no monotonic-improvement theorem claimed",
        "execution": "export target actor joint MAP exactly as used by TD; no post-hoc Q search or action masks",
        "projection": "entropy and behavior KL regularize the actor projection only; Bellman target remains undiscounted commission under the deterministic target actor",
        "behavior_prior": "learned factorized regional joint-price/radius likelihood; no claim to represent residual cross-grid behavior correlations",
        "history": "existing four-step windows retained; auxiliary objectives added to both Q and actor encoders; no persistent recurrent burn-in in this trial",
        "auxiliary": "normalized next pre-action own counts, current-period pickup cancellation observable in next history, and immediate total revenue/100; terminal next-state labels masked",
        "budget": "one random permutation of six region targets per six updates; each Q regularizer enumerates 110 own actions with recorded teammates",
        "joint_regularization": "CQL-style current/target deterministic policy-action expectation minus logged Q; adaptive nonnegative Lagrange weight targets the configured gap; no exact lower-bound guarantee claimed",
        "selection": "fixed final budget; choose training initialization using new 30-seed validation revenue; final test never selects checkpoint or changes parameters",
        "adoption": "positive paired 95% CI versus F3 and EQ-Joint on 100 new test seeds, then fresh 50-seed confirmation",
        "blocks": blocks, "replay_manifest_sha256": sha256(REPLAY/"manifest.json"),
        "validation_protocol_sha256": sha256(args.validation_protocol) if args.validation_protocol else None,
        "critic_initial_sha256": {str(SOURCE/f"train_seed{seed}/latest.pt"): sha256(SOURCE/f"train_seed{seed}/latest.pt") for seed in (62, 63, 64)},
        "reference_sha256": {name: sha256(path) for name, path in REFERENCES.items() if path},
        "source_sha256": {str(path): sha256(path) for path in Path("offline_rl").glob("*.py")},
        "simulator_sources_sha256": {str(path): sha256(path) for path in Path("src").rglob("*.py")},
        "sources": ["https://arxiv.org/abs/2309.12696", "https://arxiv.org/abs/2412.07639", "https://arxiv.org/abs/2006.04779"]}


def initialize(plan, index, device):
    torch.manual_seed(plan["training_seeds"][index])
    learner = ConservativeActorCritic(**plan["model"], **plan["algorithm"], device=device)
    path = SOURCE/f"train_seed{plan['source_training_seeds'][index]}/latest.pt"
    assert sha256(path) == plan["critic_initial_sha256"][str(path)]
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    result = learner.q.load_state_dict(checkpoint["learner"]["models"]["target_q"], strict=False)
    assert not result.unexpected_keys and all(key.startswith("auxiliary.") for key in result.missing_keys)
    learner.target_q.load_state_dict(learner.q.state_dict())
    return learner


def train_seed(args, plan, replay, index, root):
    seed = plan["training_seeds"][index]
    output = root/f"train_seed{seed}"
    if (output/"complete.json").exists():
        return
    output.mkdir(exist_ok=True)
    learner = initialize(plan, index, args.device)
    rng, history, first, elapsed = np.random.default_rng(seed), [], 0, 0.
    seen = np.zeros(replay.count, dtype=bool)
    order = np.arange(6)
    if (output/"latest.pt").exists():
        saved = torch.load(output/"latest.pt", map_location=args.device, weights_only=True)
        learner.load_state_dict(saved["learner"])
        rng.bit_generator.state = saved["numpy_rng"]
        first, history, order = saved["step"], saved["history"], saved["grid_order"].cpu().numpy()
        seen = saved["seen_transitions"].cpu().numpy()
        elapsed = history[-1]["elapsed_seconds"] if history else 0.
    total = args.warmup+args.evaluation_warmup+args.steps+args.settle
    timer, sums, count = perf_counter(), torch.zeros(len(learner.metrics), device=args.device), 0
    # 网格顺序使用独立RNG，避免预取队列改变训练批次序列。
    order_rng = np.random.default_rng(seed+7300800)
    if first:
        order_rng.bit_generator.state = saved["order_rng"]
    loader = BatchPrefetch(replay, rng, args.batch_size, workers=2, depth=4)
    for step in range(first+1, total+1):
        indices, arrays = loader.next()
        seen[indices] = True
        batch = {name: torch.from_numpy(value).to(args.device) for name, value in arrays.items()}
        if (step-1)%6 == 0:
            order = order_rng.permutation(6)
        grid = int(order[(step-1)%6])
        if step <= args.warmup:
            result = torch.zeros(len(learner.metrics), device=args.device)
            warm = learner.warmup(batch)
            result[3], result[4] = warm[0], warm[1]
            phase = "behavior_initialization"
            if step == args.warmup:
                learner.finish_warmup()
        else:
            improve = args.warmup+args.evaluation_warmup < step <= total-args.settle
            progress = np.clip((step-args.warmup-args.evaluation_warmup)/args.steps, 0., 1.)
            for group in learner.optimizers["actor"].param_groups:
                group["lr"] = float(plan["algorithm"]["actor_learning_rate"]*(.1+.45*(1+np.cos(np.pi*progress))))
            result = learner.update(batch, grid, improve=improve)
            phase = "policy_improvement" if improve else ("fixed_final_policy_evaluation" if step > total-args.settle else "initial_policy_evaluation")
        sums += result
        count += 1
        if step%100 == 0 or step == total:
            values = (sums/count).cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError
            row = {"step": step, "phase": phase, "elapsed_seconds": elapsed+perf_counter()-timer,
                "training_transitions_visited": int(seen.sum()), "training_trajectories_visited": int(seen.reshape(-1, replay.periods).any(1).sum()),
                **dict(zip(learner.metrics, map(float, values)))}
            history.append(row)
            pd.DataFrame(history).to_csv(output/"training_history.csv", index=False)
            save_json(root/"status.json", {"status": "training", "training_seed": seed,
                "completed_updates": index*total+step, "target_updates": 3*total, **row})
            print({"training_seed": seed, **row}, flush=True)
            sums.zero_(); count = 0
        if step%500 == 0 or step == total:
            atomic_save(output/"latest.pt", {"step": step, "learner": learner.state_dict(), "numpy_rng": rng.bit_generator.state,
                "order_rng": order_rng.bit_generator.state, "grid_order": torch.from_numpy(order.copy()),
                "history": history, "seen_transitions": torch.from_numpy(seen.copy())})
    loader.close()
    checkpoint = output/"policy.pt"
    atomic_save(checkpoint, {"schema_version": "multiagent_iql_policy_v2", "actor_architecture": "cooperative_distill",
        "algorithm": plan["method"], "model_config": plan["model"]["actor_config"], "replay_metadata": replay.metadata,
        "actor_state_dict": {key: value.detach().cpu() for key, value in learner.target_actor.state_dict().items()},
        "step": total, "training_seed": seed, "allowed_price_indices": None, "allowed_radius_indices": None})
    save_json(output/"complete.json", {"checkpoint_sha256": sha256(checkpoint), "updates": total,
        "training_transitions_visited": int(seen.sum()), "training_trajectories_visited": int(seen.reshape(-1, replay.periods).any(1).sum()),
        "elapsed_seconds": history[-1]["elapsed_seconds"], "actor_parameters": sum(p.numel() for p in learner.actor.parameters()),
        "critic_parameters": sum(p.numel() for p in learner.q.parameters())})
    del learner
    if args.device == "mps":
        torch.mps.empty_cache()


def plot_training(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
        for seed, color in zip((72, 73, 74), ("#0072B2", "#D55E00", "#009E73")):
            frame = pd.read_csv(root/f"train_seed{seed}/training_history.csv")
            frame = frame[frame.phase != "behavior_initialization"]
            for ax, metric in zip(axes, ("td_loss", "cf_gap", "q_auxiliary_loss", "actor_kl")):
                ax.plot(frame.step, frame[metric], lw=.8, color=color, label=f"Seed {seed}")
        for ax, title in zip(axes, ("(a) Joint-Q TD loss", "(b) Counterfactual Q gap", "(c) Dynamics auxiliary loss", "(d) Policy KL to behavior")):
            ax.set_title(title, fontsize=9, loc="left")
            ax.set_xlabel("Updates per initialization")
            ax.grid(alpha=.2)
        axes[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(root/f"training_curves.{ext}", dpi=600, facecolor="white")
        plt.close(fig)


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(2)
    source = Replay(REPLAY, "train")
    if args.resume:
        plan = json.loads((root/"protocol.json").read_text())
        assert plan["args"] == {**vars(args), "resume": False}
        assert all(sha256(path) == value for path, value in plan["source_sha256"].items())
    else:
        plan = protocol(args, source.metadata)
        plan["algorithm"].update(policy_gap_limit=args.policy_gap_limit, dual_learning_rate=.003)
        save_json(root/"protocol.json", plan)
        for path in plan["source_sha256"]:
            target = root/"code_snapshot"/path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    assert sha256(REPLAY/"manifest.json") == plan["replay_manifest_sha256"]
    if not all((root/f"train_seed{seed}/complete.json").exists() for seed in plan["training_seeds"]):
        timer = perf_counter()
        replay = CompactReplay(source)
        save_json(root/"replay_loading.json", {"seconds": perf_counter()-timer, "bytes": replay.bytes, "all_overlap_windows_verified": True})
        for index in range(3):
            train_seed(args, plan, replay, index, root)
        del replay
    plot_training(root)
    definitions = {**REFERENCES, **{f"ac_seed{seed}": str(root/f"train_seed{seed}/policy.pt") for seed in plan["training_seeds"]}}
    labels = {"fixed_f3": "F3", "eq_joint": "EQ-Joint", "joint_start": "STQ-JS",
              **{f"ac_seed{seed}": f"Conservative AC ({seed})" for seed in plan["training_seeds"]}}
    block = plan["blocks"]["validation"]
    save_json(root/"status.json", {"status": "evaluating_validation", "runs": 30})
    summary = evaluate_stage(REPLAY, definitions, root/"validation", split="validation", runs=30,
        selection_seed=block["selection_seed"], exclude_seeds=block["excluded"], workers=args.workers, labels=labels)
    selected = max((f"ac_seed{seed}" for seed in plan["training_seeds"]), key=lambda name: summary.loc[name, "platform_revenue_mean"])
    save_json(root/"selection.json", {"selected": selected, "uses_test": False})
    block = plan["blocks"]["test"]
    save_json(root/"status.json", {"status": "evaluating_test", "runs": 100, "selected": selected})
    evaluate_stage(REPLAY, definitions, root/"test", split="test", runs=100,
        selection_seed=block["selection_seed"], exclude_seeds=block["excluded"], workers=args.workers, labels=labels)
    frame = pd.read_csv(root/"test/run_summary.csv")
    statistics = [paired_statistics(frame, selected, name) for name in REFERENCES]
    save_json(root/"test/comparisons.json", statistics)
    checks = {"simulator_sources_unchanged": all(sha256(path) == value for path, value in plan["simulator_sources_sha256"].items()),
              "reference_weights_unchanged": all(sha256(REFERENCES[name]) == value for name, value in plan["reference_sha256"].items()),
              "replay_manifest_unchanged": sha256(REPLAY/"manifest.json") == plan["replay_manifest_sha256"]}
    assert all(checks.values())
    save_json(root/"verification.json", checks)
    save_json(root/"status.json", {"status": "complete", "selected": selected, "test_comparisons": statistics,
        "adopted": False, "adoption_note": "confirmation required before changing the default policy"})
    print(json.dumps(statistics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--evaluation-warmup", type=int, default=2000)
    parser.add_argument("--settle", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--policy-gap-limit", type=float, default=.05)
    parser.add_argument("--validation-protocol")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
