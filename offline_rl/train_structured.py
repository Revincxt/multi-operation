"""重训有限时域时空联合Q/V，保持全动作搜索并独立闭环检验。"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from offline_rl.data import Replay, save_json, sha256
from offline_rl.compact_replay import CompactReplay
from offline_rl.evaluate import selected_seeds
from offline_rl.optimize import paired_statistics
from offline_rl.structured_value import StructuredIQL, StructuredValueActor
from offline_rl.train import atomic_save
from offline_rl.train_cooperative import evaluate_stage
from offline_rl.train_joint import known_seeds


REPLAY = Path("datasets_grid6_low_mixed100k_20261103/offline_rl")
REFERENCE = Path("models/value_guided_full100k_20261103/eq_soft.pt")
OUTPUT = Path("models/structured_iql_full100k_20261103")
SETTINGS = {"structured_soft": (.25, .0001), "structured_regular": (1., .001), "structured_conservative": (2., .003)}
LABELS = {"fixed_f3": "F3", "eq_joint": "EQ-Joint", "structured_soft": "STQ soft",
          "structured_regular": "STQ regular", "structured_conservative": "STQ conservative"}


class BatchPrefetch:
    """预读不消耗训练RNG；只有已使用批次推进RNG，断点无需保存队列。"""

    def __init__(self, replay, rng, batch_size, workers=4, depth=8):
        self.replay, self.rng, self.batch_size = replay, rng, batch_size
        self.future_rng = np.random.default_rng()
        self.future_rng.bit_generator.state = rng.bit_generator.state
        self.pool, self.queue = ThreadPoolExecutor(max_workers=workers), deque()
        for _ in range(depth):
            self._submit()

    def _submit(self):
        indices = self.future_rng.integers(self.replay.count, size=self.batch_size)
        self.queue.append((indices, self.pool.submit(self.replay.batch, indices)))

    def next(self):
        indices, future = self.queue.popleft()
        values = future.result()
        actual = self.rng.integers(self.replay.count, size=self.batch_size)
        assert np.array_equal(indices, actual)
        self._submit()
        return indices, values

    def close(self):
        self.pool.shutdown(wait=True)


def protocol(args, metadata):
    used = sorted(known_seeds())
    blocks = {}
    for split, count, namespace in (("validation", 30, 7300501), ("test", 100, 7300502)):
        ids, seeds = selected_seeds(REPLAY, split, count, namespace, used)
        blocks[split] = {"run_ids": ids, "seeds": seeds, "selection_seed": namespace, "excluded": used.copy()}
        used = sorted(set(used).union(seeds))
    config = dict(state_size=metadata["state_size"], grid_count=6, history_steps=4, history_features=24,
        context_size=14, price_values=sorted({p for p, r in metadata["action_pairs"]}),
        radius_values=sorted({r for p, r in metadata["action_pairs"]}), hidden_size=64, memory_size=32,
        basis_size=16, rank=8)
    return {"args": vars(args), "critic": config, "training_seeds": [62, 63, 64],
        "algorithm": {"discount": 1., "expectile": .7, "target_tau": .005, "reward_scale": 100., "learning_rate": 3e-4},
        "method": "Finite-horizon IQL Q/V with temporal attention, numerical/factorized action basis and nonadditive joint heads; value-guided extraction",
        "scope": "combined critic/target/budget change, not a component ablation; not policy-conditioned actor-critic or InSPO",
        "candidate_settings": SETTINGS, "blocks": blocks,
        "checkpoint_selection": "fixed final target Q for all three initializations; choose soft regularization only on new validation seeds",
        "adoption": "positive paired 95% CI versus F3 and EQ-Joint on test and a fresh 50-seed confirmation",
        "replay_manifest_sha256": sha256(REPLAY / "manifest.json"), "reference_sha256": sha256(REFERENCE),
        "source_sha256": {str(path): sha256(path) for path in Path("offline_rl").glob("*.py")}}


def offline_validation(learner, replay, device):
    rng = np.random.default_rng(7300500)
    sums = torch.zeros(3, device=device)
    with torch.no_grad():
        for _ in range(4):
            data = {name: torch.as_tensor(value, device=device) for name, value in replay.batch(rng.integers(replay.count, size=512)).items()}
            target = data["rewards"]/100 + data["masks"]*learner.value(data["next_states"])
            predictions = learner.q(data["states"], data["actions"])
            difference = predictions-target[None]
            sums += torch.stack((difference.square().mean(), difference.abs().mean(), predictions.mean()))
    return dict(zip(("td_mse", "td_mae", "q_mean"), map(float, (sums/4).cpu())))


def train_seed(args, plan, replay, validation, seed, seed_index, root):
    target = root / f"train_seed{seed}"
    if (target / "complete.json").exists():
        return
    target.mkdir(exist_ok=True)
    torch.manual_seed(seed)
    learner = StructuredIQL(plan["critic"], device=args.device, **plan["algorithm"])
    rng, first, records, previous = np.random.default_rng(seed), 0, [], 0.
    seen = np.zeros(replay.metadata["splits"]["train"]["runs"], dtype=bool)
    latest = target / "latest.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location=args.device, weights_only=True)
        learner.load_state_dict(checkpoint["learner"])
        rng.bit_generator.state = checkpoint["numpy_rng"]
        first, records = checkpoint["step"], checkpoint["history"]
        seen = checkpoint["seen_runs"].cpu().numpy()
        previous = records[-1]["elapsed_seconds"] if records else 0.
    timer, sums, count = perf_counter(), torch.zeros(5, device=args.device), 0
    offline = json.loads((target / "offline_validation.json").read_text()) if (target / "offline_validation.json").exists() else []
    loader = BatchPrefetch(replay, rng, args.batch_size, workers=2, depth=4)
    for step in range(first+1, args.steps+1):
        indices, arrays = loader.next()
        seen[indices//replay.periods] = True
        batch = {name: torch.from_numpy(value).to(args.device, non_blocking=True) for name, value in arrays.items()}
        sums += learner.update(batch)
        count += 1
        rate = float(3e-4*(.1+.9*.5*(1+np.cos(np.pi*step/args.steps))))
        for optimizer in learner.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = rate
        if step % 250 == 0 or step == args.steps:
            values = (sums/count).cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError
            row = {"step": step, "elapsed_seconds": previous+perf_counter()-timer,
                "training_trajectories_visited": int(seen.sum()),
                **dict(zip(("q_loss", "value_loss", "td_mae", "target_q_mean", "target_mean"), map(float, values)))}
            records.append(row)
            pd.DataFrame(records).to_csv(target / "training_history.csv", index=False)
            save_json(root / "status.json", {"status": "training", "training_seed": seed,
                "completed_updates": seed_index*args.steps+step, "target_updates": 3*args.steps, **row})
            print({"training_seed": seed, **row}, flush=True)
            sums.zero_(); count = 0
        if step % 2500 == 0 or step == args.steps:
            offline.append({"step": step, **offline_validation(learner, validation, args.device)})
            save_json(target / "offline_validation.json", offline)
        if step % 1000 == 0 or step == args.steps:
            atomic_save(latest, {"step": step, "learner": learner.state_dict(), "numpy_rng": rng.bit_generator.state,
                                "seen_runs": torch.from_numpy(seen.copy()), "history": records})
    loader.close()
    save_json(target / "complete.json", {"step": args.steps, "seed": seed, "elapsed_seconds": records[-1]["elapsed_seconds"],
        "training_trajectories_visited": int(seen.sum()), "checkpoint_sha256": sha256(latest),
        "q_parameters": sum(p.numel() for p in learner.q.parameters()), "value_parameters": sum(p.numel() for p in learner.value.parameters())})
    del learner
    if args.device == "mps":
        torch.mps.empty_cache()


def export(root, plan):
    teacher = torch.load(REFERENCE, map_location="cpu", weights_only=True)
    config = {"behavior_config": teacher["model_config"]["behavior_config"], "critic_config": plan["critic"], "ensemble_size": 3,
              "uncertainty_weight": .25, "behavior_weight": .0001, "sweeps": 2, "multistart": True}
    model = StructuredValueActor(**config)
    model.behavior.load_state_dict({key.removeprefix("behavior."): value for key, value in teacher["actor_state_dict"].items() if key.startswith("behavior.")})
    for critic, seed in zip(model.critics, plan["training_seeds"]):
        checkpoint = torch.load(root / f"train_seed{seed}/latest.pt", map_location="cpu", weights_only=True)
        critic.load_state_dict(checkpoint["learner"]["models"]["target_q"])
    state = model.state_dict()
    for name, (uncertainty, behavior) in SETTINGS.items():
        atomic_save(root / f"{name}.pt", {"schema_version": "multiagent_iql_policy_v2", "actor_architecture": "structured_value",
            "algorithm": plan["method"], "model_config": {**config, "uncertainty_weight": uncertainty, "behavior_weight": behavior},
            "actor_state_dict": state, "replay_metadata": teacher["replay_metadata"],
            "allowed_price_indices": None, "allowed_radius_indices": None, "step": plan["args"]["steps"]})
    save_json(root / "export.json", {"checkpoint_sha256": {name: sha256(root / f"{name}.pt") for name in SETTINGS}})


def plot_training(root, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
        for seed, color in zip(seeds, ("#0072B2", "#D55E00", "#009E73")):
            train = pd.read_csv(root / f"train_seed{seed}/training_history.csv")
            valid = pd.DataFrame(json.loads((root / f"train_seed{seed}/offline_validation.json").read_text()))
            for ax, frame, column in ((axes[0], train, "q_loss"), (axes[1], train, "value_loss"), (axes[2], valid, "td_mae")):
                ax.plot(frame.step, frame[column], color=color, lw=.9, label=f"Seed {seed}")
        for ax, title in zip(axes, ("(a) Joint-Q TD loss ↓", "(b) Expectile value loss ↓", "(c) Validation TD MAE ↓")):
            ax.set_title(title, loc="left", fontsize=10)
            ax.set_xlabel("Gradient updates per critic initialization")
            ax.grid(alpha=.2)
            ax.set_yscale("log")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(root / f"training_curves.{ext}", dpi=600, facecolor="white")
        plt.close(fig)


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(2)
    replay, validation = Replay(REPLAY, "train"), Replay(REPLAY, "validation")
    if args.resume:
        plan = json.loads((root / "protocol.json").read_text())
        assert all(sha256(path) == value for path, value in plan["source_sha256"].items())
        assert plan["args"] == {**vars(args), "resume": False}
        assert sha256(REPLAY / "manifest.json") == plan["replay_manifest_sha256"]
    else:
        plan = protocol(args, replay.metadata)
        save_json(root / "protocol.json", plan)
        for path in plan["source_sha256"]:
            target = root / "code_snapshot" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    assert sha256(REFERENCE) == plan["reference_sha256"]
    if not (root / "export.json").exists():
        timer = perf_counter()
        replay = CompactReplay(replay)
        save_json(root / "compact_replay.json", {"bytes": replay.bytes, "materialization_seconds": perf_counter()-timer,
            "all_overlap_windows_verified": True, "dtype": "float32", "source_data_unchanged": True})
        for index, seed in enumerate(plan["training_seeds"]):
            train_seed(args, plan, replay, validation, seed, index, root)
        del replay
        export(root, plan)
        plot_training(root, plan["training_seeds"])
    definitions = {"fixed_f3": None, "eq_joint": str(REFERENCE), **{name: str(root / f"{name}.pt") for name in SETTINGS}}
    val = plan["blocks"]["validation"]
    save_json(root / "status.json", {"status": "evaluating_validation", "runs": 30})
    summary = evaluate_stage(REPLAY, definitions, root / "validation", split="validation", runs=30,
        selection_seed=val["selection_seed"], exclude_seeds=val["excluded"], workers=args.workers, labels=LABELS)
    revenues = {name: float(summary.loc[name, "platform_revenue_mean"]) for name in SETTINGS}
    selected = max(revenues, key=revenues.get)
    save_json(root / "selection.json", {"selected": selected, "validation_revenues": revenues, "selection_uses_test": False})
    test = plan["blocks"]["test"]
    save_json(root / "status.json", {"status": "evaluating_test", "selected": selected, "runs": 100})
    evaluate_stage(REPLAY, {key: path for key, path in definitions.items() if key in ("fixed_f3", "eq_joint", selected)},
        root / "test", split="test", runs=100, selection_seed=test["selection_seed"], exclude_seeds=test["excluded"], workers=args.workers, labels=LABELS)
    frame = pd.read_csv(root / "test/run_summary.csv")
    statistics = [paired_statistics(frame, selected, name) for name in ("fixed_f3", "eq_joint")]
    save_json(root / "test/comparisons.json", statistics)
    save_json(root / "status.json", {"status": "complete", "selected": selected,
        "completed_at": datetime.now(timezone.utc).isoformat(), "test_comparisons": statistics})
    print(json.dumps(statistics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
