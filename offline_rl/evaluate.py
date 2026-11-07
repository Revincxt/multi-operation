"""离线学得策略的配对闭环评估；测试期不更新参数。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.compare_policies import (
    METRICS, _joint_candidate_frame, check_collection_revision, run_episode, sample_common_orders,
)
from baseline.data import DEFAULT_MODEL
from baseline.lstm import LSTMPredictor
from baseline.sequence import SequencePreprocessor
from data_pipeline.collect_trajectories import simulation_window
from offline_rl.data import DEFAULT_REPLAY, encode_state, save_json, sha256
from offline_rl.iql import Actor
from offline_rl.coordination import make_actor
from src.env.actions import GridAction
from src.env.simulator_pattern import SimulatorPattern


LABELS = {"fixed_f3": "F3", "random": "Random", "bc": "BC", "bc9": "BC-9",
          "iql": "IQL", "iql9": "IQL-9", "lstm_greedy": "LSTM-G", "lstm_bcs": "LSTM-BCS"}
_PATTERN = None


def active_checkpoint():
    """默认策略由已完成的验证采用记录指定，首次使用沿用原 IQL-9。"""
    project = Path(__file__).resolve().parents[1]
    path = Path(__file__).with_name("active_policy.json")
    if not path.exists():
        return project / "models/iql_grid6_low_20261102/best_iql9.pt"
    record = json.loads(path.read_text())
    checkpoint = Path(record["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = project / checkpoint
    if sha256(checkpoint) != record["sha256"]:
        raise ValueError
    return checkpoint


class Policy:
    """复用同一观测编码，按checkpoint定义读取局部或全城信息。"""

    def __init__(self, checkpoint=None):
        checkpoint = active_checkpoint() if checkpoint is None else checkpoint
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["schema_version"] not in ("multiagent_iql_policy_v1", "multiagent_iql_policy_v2"):
            raise ValueError
        self.actor = make_actor(payload["model_config"], payload.get("actor_architecture", "parallel")).eval()
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.metadata = payload["replay_metadata"]
        self.processor = SequencePreprocessor.from_state_dict(self.metadata["preprocessor"])
        self.allowed_prices = payload.get("allowed_price_indices")
        self.allowed_radii = payload.get("allowed_radius_indices")
        self.pairs = self.metadata["action_pairs"]

    def __call__(self, periods, decision, start, interval, current_state):
        rows, grids, action_columns = _joint_candidate_frame(
            periods, self.processor, decision, start, interval, current_state)
        # 占位动作仅满足公共编码器列约定，随后全部从状态中剔除。
        frame = pd.DataFrame([{**row, "price_multiplier": 0., "matching_radius_km": 0.,
                               **dict.fromkeys(action_columns, 0.)} for row in rows])
        history, context, _ = self.processor.transform_inputs(frame)
        state = encode_state(history[None], context[None], self.metadata["state_context_indices"])
        selected = self.actor.act(torch.from_numpy(state), self.allowed_prices, self.allowed_radii)[0].numpy()
        return {int(grid): GridAction(*self.pairs[int(index)]) for grid, index in zip(grids, selected)}


def selected_seeds(replay_root, split, runs, selection_seed, exclude_seeds=()):
    run_ids = json.loads((Path(replay_root) / split / "run_ids.json").read_text())
    excluded = set(exclude_seeds)
    run_ids = [run for run in run_ids if int(run.split("_simseed", 1)[1].split("_", 1)[0]) not in excluded]
    if not 0 < runs <= len(run_ids):
        raise ValueError
    selected = np.random.default_rng(selection_seed).choice(sorted(run_ids), runs, replace=False).tolist()
    return selected, [int(run.split("_simseed", 1)[1].split("_", 1)[0]) for run in selected]


def _run_seed(config, seed, definitions, output):
    global _PATTERN
    torch.set_num_threads(1)
    if _PATTERN is None:
        _PATTERN = SimulatorPattern(config["date"], data_dir=config["data_dir"])
    _, start, end = simulation_window(config["start_time"], config["simulation_steps"])
    requests, sampled = sample_common_orders(_PATTERN.request_all, start, end,
                                            config["order_sample_ratio"], seed)
    target = Path(output) / "runs" / f"seed{seed}"
    target.mkdir(parents=True)
    sampled.to_csv(target / "sampled_order_ids.csv.gz", index=False)
    order_hash = hashlib.sha256(sampled.to_csv(index=False).encode()).hexdigest()
    results = []
    for name, checkpoint in definitions.items():
        policy, predictor, optimizer = None, None, "greedy"
        if name in ("lstm_greedy", "lstm_bcs"):
            predictor = LSTMPredictor.load(checkpoint)
            optimizer = "bcs" if name == "lstm_bcs" else "greedy"
        elif checkpoint is not None:
            policy = Policy(checkpoint)
        row = run_episode(config, name, seed, requests, _PATTERN.driver_info, predictor,
                          target / name, action_optimizer=optimizer, action_policy=policy)
        row.update(sampled_orders_sha256=order_hash, sampled_order_count=len(sampled))
        save_json(target / name / "summary.json", row)
        results.append(row)
        print(f"seed={seed} {name}: revenue={row['platform_revenue']:.2f}", flush=True)
    for key in ("sampled_orders_sha256", "initial_driver_sha256", "warmup_metrics_sha256",
                "baseline_request_num", "generated_baseline_requests"):
        if len({row[key] for row in results}) != 1:
            raise AssertionError
    return results


def evaluate(replay_root, definitions, output, split="test", runs=10,
             selection_seed=20261102, workers=4, plot=True, exclude_seeds=(), labels=None):
    """完整轨迹隔离；仅验证 seed 用于训练期间选择 checkpoint。"""
    if split not in ("validation", "test") or workers <= 0:
        raise ValueError
    replay_root, output = Path(replay_root), Path(output)
    metadata = json.loads((replay_root / "manifest.json").read_text())
    collection_root = Path(metadata["collection_root"])
    source = json.loads((collection_root / "collection_config.json").read_text())
    if sha256(collection_root / "collection_config.json") != metadata["collection_config_sha256"]:
        raise ValueError
    check_collection_revision(collection_root, source)
    selected, seeds = selected_seeds(replay_root, split, runs, selection_seed, exclude_seeds)
    train_runs = set(json.loads((replay_root / "train/run_ids.json").read_text()))
    if train_runs.intersection(selected):
        raise ValueError
    config = {key: source[key] for key in (
        "date", "data_dir", "start_time", "simulation_steps", "grid_num", "driver_num",
        "order_sample_ratio", "decision_freq", "price_response_model", "price_action_values",
        "matching_radius_values")}
    config.update(fixed_price=1.6, fixed_radius=2., warmup_price=1.5, warmup_radius=3.,
                  history_steps=metadata["preprocessor"]["history_steps"])
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "split": split, "selected_run_ids": selected, "seeds": seeds,
              "excluded_seeds": list(exclude_seeds), "policy_labels": labels or LABELS,
              "selection_seed": selection_seed, "config": config, "policies": definitions,
              "checkpoint_sha256": {name: sha256(path) for name, path in definitions.items() if path},
              "replay_manifest_sha256": sha256(replay_root / "manifest.json"),
              "selection": "validation-only simulator revenue; test never updates weights",
              "created_at": datetime.now(timezone.utc).isoformat()}
    save_json(output / "manifest.json", report)
    results = []
    if workers == 1:
        for seed in seeds:
            results.extend(_run_seed(config, seed, definitions, output))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(_run_seed, config, seed, definitions, output) for seed in seeds]
            for job in as_completed(jobs):
                results.extend(job.result())
    frame = pd.DataFrame([{key: value for key, value in row.items() if key != "full_window"}
                          for row in results]).sort_values(["seed", "policy"])
    frame.to_csv(output / "run_summary.csv", index=False)
    summary = frame.groupby("policy")[list(METRICS)].agg(["mean", "std"]).reindex(definitions)
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv(output / "policy_summary.csv")
    paired = frame.pivot(index="seed", columns="policy", values="platform_revenue")
    comparisons = []
    for name in ("iql", "iql9"):
        if name not in definitions:
            continue
        for reference in definitions:
            if reference == name:
                continue
            diff = (paired[name] - paired[reference]).to_numpy()
            bootstrap = np.random.default_rng(42).choice(diff, (10000, len(diff)), replace=True).mean(axis=1)
            comparisons.append({"policy": name, "reference": reference, "seed_count": len(diff),
                                "mean_difference_usd": diff.mean(), "relative_percent": 100 * diff.mean() / paired[reference].mean(),
                                "ci95_low": np.quantile(bootstrap, .025), "ci95_high": np.quantile(bootstrap, .975),
                                "wins": int((diff > 1e-8).sum()), "ties": int((np.abs(diff) <= 1e-8).sum())})
    pd.DataFrame(comparisons).to_csv(output / "paired_revenue_comparisons.csv", index=False)
    report.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
    save_json(output / "manifest.json", report)
    if plot:
        plot_comparison(output)
    return summary


def plot_comparison(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from baseline.plot_paper_figures import _font_family

    output = Path(output)
    frame = pd.read_csv(output / "run_summary.csv")
    report = json.loads((output / "manifest.json").read_text())
    order = list(report["policies"])
    labels = report.get("policy_labels", LABELS)
    titles = ("Bubble requests =", "Order request rate ↑", "Driver acceptance rate ↑",
              "Order completion rate ↑", "Matched-passenger wait (s) ↓", "Platform revenue (USD) ↑")
    colors = ["#888888", "#BBBBBB", "#56B4E9", "#0072B2", "#E69F00", "#D55E00", "#009E73", "#CC79A7"]
    with plt.rc_context({"font.family": _font_family(), "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
        fig.subplots_adjust(left=.07, right=.99, bottom=.15, top=.95, hspace=.48, wspace=.3)
        for index, (ax, metric, title) in enumerate(zip(axes.flat, METRICS, titles)):
            values = [frame.loc[frame.policy == name, metric] for name in order]
            scale = 100 if metric in ("request_rate", "driver_accept_rate", "order_completion_rate") else 1
            ax.bar(range(len(order)), [x.mean() * scale for x in values],
                   yerr=[x.std(ddof=1) * scale for x in values], color=[colors[i % len(colors)] for i in range(len(order))],
                   error_kw={"linewidth": .7, "capsize": 2}, width=.7, zorder=3)
            ax.set_title(f"({chr(97 + index)}) {title}", loc="left", fontsize=10)
            ax.set_xticks(range(len(order)), [labels.get(n, n) for n in order], rotation=40, ha="right")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", color="#DDDDDD", linewidth=.45, zorder=0)
            if scale == 100:
                ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        caption = report.get("plot_caption", "Means ± SD over paired held-out simulation seeds; 05:20–10:00.  -9: 3 prices × 3 radii.")
        fig.text(.5, .025, caption, ha="center", fontsize=9)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"policy_comparison.{extension}", dpi=600, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线 RL、BC、F3 与 LSTM 配对闭环评估")
    parser.add_argument("--replay-root", default=DEFAULT_REPLAY)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.model is None:
        definitions = {"fixed_f3": None, "active": str(active_checkpoint())}
    else:
        model = Path(args.model)
        definitions = {"fixed_f3": None, "random": None,
                       **{name: str(model / f"best_{name}.pt") for name in ("bc", "bc9", "iql", "iql9")},
                       "lstm_greedy": str(Path(DEFAULT_MODEL) / "best_model.pt"),
                       "lstm_bcs": str(Path(DEFAULT_MODEL) / "best_model.pt")}
    evaluate(args.replay_root, definitions, args.output, runs=args.runs, workers=args.workers)
