"""DL-MRD closed-loop test on the 100 paired seeds from the policy benchmark."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / "experiments/ma_iql_comparison_20261103/protocol.json"
BASELINE_SUMMARY = ROOT / "experiments/ma_iql_comparison_20261103/run_summary.csv"
OUT = Path(os.environ.get("DLMRD_EVAL_OUTPUT", HERE / "operational")).resolve()
MODEL = Path(os.environ.get(
    "DLMRD_EVAL_MODEL",
    ROOT / "checkpoints/dlmrd/best_model.pt",
)).resolve()
POLICIES = ("fixed_f3", "lstm_greedy", "ma_iql_plus", "dlmrd_bcs5")
METRICS = (
    "baseline_request_num", "request_rate", "driver_accept_rate",
    "order_completion_rate", "mean_wait_to_match_seconds", "platform_revenue",
)
_WORKER = {}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_reference(protocol):
    for name, expected in protocol["simulator_sources_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"simulator source differs from benchmark: {name}")
    for name, expected in protocol["evaluation_sources_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"evaluation source differs from benchmark: {name}")
    for name, relative in protocol["models"].items():
        if sha256(ROOT / relative) != protocol["checkpoint_sha256"][name]:
            raise ValueError(f"benchmark checkpoint changed: {name}")


def worker_init(config, model_path, output_dir, references):
    import torch
    from baseline.dlmrd.model import DLMRDPredictor
    from src.env.simulator_pattern import SimulatorPattern

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER["config"] = config
    _WORKER["output"] = Path(output_dir)
    _WORKER["references"] = references
    _WORKER["pattern"] = SimulatorPattern(config["date"], data_dir=config["data_dir"])
    _WORKER["predictor"] = DLMRDPredictor.load(model_path, device="cpu")


def run_seed(seed):
    from baseline.compare_policies import run_episode, sample_common_orders
    from data_pipeline.collect_trajectories import simulation_window

    config = _WORKER["config"]
    pattern = _WORKER["pattern"]
    reference = _WORKER["references"][int(seed)]
    _, start, end = simulation_window(config["start_time"], config["simulation_steps"])
    requests, sampled = sample_common_orders(
        pattern.request_all, start, end, config["order_sample_ratio"], int(seed))
    sample_hash = hashlib.sha256(sampled.to_csv(index=False).encode()).hexdigest()
    if sample_hash != reference["sampled_orders_sha256"]:
        raise ValueError(f"sampled orders differ for seed {seed}")

    run_dir = _WORKER["output"] / "runs" / f"seed{seed}" / "dlmrd_bcs5"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    result = run_episode(
        config, "dlmrd_bcs5", int(seed), requests, pattern.driver_info,
        _WORKER["predictor"], run_dir, action_optimizer="bcs", bcs_max_sweeps=5)
    if (result["initial_driver_sha256"] != reference["initial_driver_sha256"]
            or result["warmup_metrics_sha256"] != reference["warmup_metrics_sha256"]):
        raise ValueError(f"driver initialization or warm-up differs for seed {seed}")
    return {key: value for key, value in result.items() if key != "full_window"}


def bootstrap_mean(values, rng, repeats=20000):
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def aggregate_and_plot(dlmrd_rows, baseline, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    dlmrd = pd.DataFrame(dlmrd_rows)
    policies = baseline[baseline["policy"].isin(POLICIES[:-1])].copy()
    combined = pd.concat([policies, dlmrd], ignore_index=True).sort_values(["seed", "policy"])
    expected = set(report["paired_seeds"])
    if any(set(frame.seed.astype(int)) != expected for _, frame in combined.groupby("policy")):
        raise ValueError("paired seed set is incomplete")
    combined.to_csv(OUT / "run_summary.csv", index=False)

    rng = np.random.default_rng(42025)
    stats = []
    for policy in POLICIES:
        frame = combined[combined.policy == policy]
        for metric in METRICS:
            values = frame[metric].to_numpy(dtype=float)
            low, high = bootstrap_mean(values, rng)
            stats.append({"policy": policy, "metric": metric, "mean": float(values.mean()),
                          "std": float(values.std(ddof=1)), "ci95_low": low, "ci95_high": high,
                          "seeds": len(values)})
    stats = pd.DataFrame(stats)
    stats.to_csv(OUT / "metric_statistics.csv", index=False)

    wide = combined.pivot(index="seed", columns="policy", values=list(METRICS))
    paired = []
    for metric in METRICS:
        for baseline_policy in ("fixed_f3", "lstm_greedy", "ma_iql_plus"):
            differences = (wide[(metric, "dlmrd_bcs5")] -
                           wide[(metric, baseline_policy)]).to_numpy(dtype=float)
            low, high = bootstrap_mean(differences, rng)
            paired.append({"metric": metric, "comparison": f"dlmrd_bcs5_minus_{baseline_policy}",
                           "mean_difference": float(differences.mean()), "ci95_low": low,
                           "ci95_high": high, "seeds": len(differences)})
    pd.DataFrame(paired).to_csv(OUT / "paired_metric_differences.csv", index=False)

    selected_rows = []
    for seed in sorted(expected):
        run_dir = OUT / "runs" / f"seed{seed}" / "dlmrd_bcs5"
        actions = pd.read_csv(run_dir / "action_plan.csv.gz")
        actions = actions[~actions.warmup][[
            "decision_step", "grid_id", "price_multiplier", "matching_radius_km",
            "predicted_platform_revenue"]].copy()
        actions["seed"] = seed
        observed = pd.read_csv(run_dir / "decision_metrics.csv.gz",
                               usecols=["seed", "decision_step", "grid_id", "platform_revenue"])
        observed = observed[observed.decision_step >= 4]
        selected_rows.append(actions.merge(observed, on=["seed", "decision_step", "grid_id"],
                                           validate="one_to_one"))
    selected = pd.concat(selected_rows, ignore_index=True)
    action_summary = selected.groupby(
        ["grid_id", "price_multiplier", "matching_radius_km"], as_index=False).agg(
            decisions=("seed", "size"),
            mean_predicted_revenue_per_5min=("predicted_platform_revenue", "mean"),
            mean_realized_revenue_per_5min=("platform_revenue", "mean"))
    action_summary["mean_prediction_error"] = (
        action_summary.mean_predicted_revenue_per_5min - action_summary.mean_realized_revenue_per_5min)
    action_summary.to_csv(OUT / "action_selection_diagnostic.csv", index=False)

    names = ("F3", "LSTM-G", "MA-IQL+", "DL-MRD-BCS")
    colors = ("#8796A5", "#D08B58", "#4C918A", "#745A91")
    titles = ("Bubble requests (input)", "Order request rate", "Driver acceptance rate ↑",
              "Order completion rate ↑", "Mean wait (s) ↓", "Platform revenue (USD) ↑")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.9))
    for ax, metric, title in zip(axes.flat, METRICS, titles):
        frame = stats[stats.metric == metric].set_index("policy").loc[list(POLICIES)]
        means = frame["mean"].to_numpy()
        errors = np.vstack((means - frame["ci95_low"].to_numpy(),
                            frame["ci95_high"].to_numpy() - means))
        bars = ax.bar(np.arange(4), means, yerr=errors, capsize=2.5,
                      color=colors, width=.67, linewidth=0)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(np.arange(4), names, fontsize=6.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", color="#dedede", linewidth=.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if metric.endswith("rate"):
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        if metric != "baseline_request_num":
            for bar, value, upper in zip(bars, means, frame["ci95_high"].to_numpy()):
                label = f"{value:.1%}" if metric.endswith("rate") else (
                    f"{value:.1f}" if abs(value) < 100 else f"{value:,.0f}")
                ax.annotate(label, (bar.get_x() + bar.get_width()/2, upper),
                            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=6)
    fig.suptitle("Closed-loop results | 100 paired seeds | mean with 95% bootstrap CI", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, .97), h_pad=1.5, w_pad=1.2)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"operational_comparison.{suffix}", dpi=400 if suffix == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return combined, stats, pd.DataFrame(paired)


def finalize(completed, baseline, report):
    combined, stats, paired = aggregate_and_plot(completed, baseline, report)
    report.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
                   "completed_seeds": len(completed), "run_count": len(combined),
                   "outputs": ["run_summary.csv", "metric_statistics.csv",
                               "paired_metric_differences.csv", "action_selection_diagnostic.csv",
                               "operational_comparison.png", "operational_comparison.pdf"]})
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "partial_run_summary.csv").unlink(missing_ok=True)
    print("\nMean operational metrics:")
    print(stats.pivot(index="metric", columns="policy", values="mean").to_string())
    print("\nPaired revenue differences:")
    print(paired[paired.metric == "platform_revenue"].to_string(index=False))


def main():
    import concurrent.futures

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    verify_reference(protocol)
    source_rows = pd.read_csv(BASELINE_SUMMARY)
    seeds = [int(seed) for seed in protocol["seeds"]]
    if OUT.exists():
        partial = OUT / "partial_run_summary.csv"
        if partial.is_file():
            completed_frame = pd.read_csv(partial)
            if (len(completed_frame) == len(seeds)
                    and set(completed_frame.seed.astype(int)) == set(seeds)):
                report = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
                baseline = source_rows[source_rows.policy.isin(("fixed_f3", "lstm_greedy", "ma_iql_plus"))]
                finalize(completed_frame.to_dict(orient="records"), baseline, report)
                return
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    fixed = source_rows[source_rows.policy == "fixed_f3"].set_index("seed")
    if set(fixed.index.astype(int)) != set(seeds):
        raise ValueError("F3 benchmark seed set differs from protocol")
    references = {
        int(seed): {key: fixed.loc[seed, key] for key in (
            "sampled_orders_sha256", "initial_driver_sha256", "warmup_metrics_sha256")}
        for seed in seeds
    }
    config = dict(protocol["config"])
    model_path = str(MODEL.resolve())
    output_dir = str(OUT.resolve())
    workers = min(4, os.cpu_count() or 1)
    import torch
    model_checkpoint = torch.load(MODEL, map_location="cpu", weights_only=True)
    model_training = json.loads((MODEL.parent / "training_config.json").read_text(encoding="utf-8"))
    model_metrics = json.loads((MODEL.parent / "metrics.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": "dlmrd_closed_loop_v1", "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(), "paired_seeds": seeds,
        "workers": workers, "policy": "DL-MRD predicts six interval metrics; BCS maximizes summed predicted grid revenue, up to 5 coordinate sweeps per decision.",
        "benchmark_protocol": str(PROTOCOL.relative_to(ROOT)), "config": config,
        "dlmrd_checkpoint": str(MODEL.relative_to(ROOT)), "dlmrd_checkpoint_sha256": sha256(MODEL),
        "dlmrd_model_name": model_training["model_name"],
        "dlmrd_model_schema": model_checkpoint["schema_version"],
        "dlmrd_training_config": model_training,
        "dlmrd_best_epoch": int(model_checkpoint["best_epoch"]),
        "requested_training_epochs": int(model_training["epochs"]),
        "training_epochs_completed": int(model_metrics["epochs_run"]),
        "comparison_scope": "same 100 paired seeds and fixed simulator inputs as F3, LSTM-G, and MA-IQL+ benchmark",
        "metric_definitions": {
            "request_rate": "submitted / baseline requests",
            "driver_accept_rate": "accepted offers / driver offers",
            "order_completion_rate": "retained matches / (candidate attempts + unmatched cancellations)",
            "mean_wait_to_match_seconds": "matched-count weighted wait",
            "platform_revenue": "retained matched fare times 0.25 commission; not net profit",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    completed, start = [], time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init,
                             initargs=(config, model_path, output_dir, references)) as pool:
        futures = {pool.submit(run_seed, seed): seed for seed in seeds}
        for future in as_completed(futures):
            row = future.result()
            completed.append(row)
            completed.sort(key=lambda item: item["seed"])
            pd.DataFrame(completed).to_csv(OUT / "partial_run_summary.csv", index=False)
            elapsed = time.time() - start
            print(f"[{len(completed)}/{len(seeds)}] seed={row['seed']} "
                  f"revenue={row['platform_revenue']:.2f} "
                  f"completion={row['order_completion_rate']:.3f} "
                  f"wait={row['mean_wait_to_match_seconds']:.1f}s "
                  f"elapsed={elapsed/60:.1f}min", flush=True)
            progress = {"completed": len(completed), "total": len(seeds),
                        "elapsed_seconds": elapsed,
                        "mean_revenue_so_far": float(np.mean([r["platform_revenue"] for r in completed]))}
            (OUT / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")

    baseline = source_rows[source_rows.policy.isin(("fixed_f3", "lstm_greedy", "ma_iql_plus"))]
    finalize(completed, baseline, report)


if __name__ == "__main__":
    import torch  # ensure PyTorch is imported before worker initialization
    main()
