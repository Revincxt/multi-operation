"""用真实闭环仿真比较新 LSTM、固定联合动作与可选的随机策略，不重新训练。

使用方式:
    python3 -m baseline.compare_lstm_fixed --runs 10 --workers 4
    python3 -m baseline.compare_lstm_fixed --fixed-actions 1:2 1.5:3 1.6:2
    python3 -m baseline.compare_lstm_fixed --include-random --runs 10 --workers 4

固定组合按「价格乘子:匹配半径公里」给出；测试结果不参与选择模型或固定动作。
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from baseline.compare_policies import (
    METRICS, RANDOM_ACTION_SEED_NAMESPACE, check_collection_revision, run_episode, sample_common_orders,
)
from baseline.data import DEFAULT_COLLECTION, DEFAULT_MODEL
from baseline.lstm import LSTMPredictor
from data_pipeline.collect_density import config_signature, save_json, sha256
from data_pipeline.collect_trajectories import simulation_window
from src.env.observation import state_columns
from src.env.simulator_pattern import SimulatorPattern


DEFAULT_ACTIONS = ((1.0, 2.0), (1.5, 3.0), (1.6, 2.0), (1.6, 3.0), (1.9, 2.5))
_PATTERN = None


def policy_definitions(actions, include_random=False):
    """动作在仿真前确定，重复动作或非正数直接拒绝。"""
    pairs = [tuple(map(float, action)) for action in actions]
    if (not pairs or any(len(pair) != 2 for pair in pairs) or len(set(pairs)) != len(pairs)
            or not np.isfinite(pairs).all() or np.any(np.asarray(pairs) <= 0)):
        raise ValueError
    fixed = [{"policy": f"fixed_{index + 1}", "price": price, "radius": radius}
             for index, (price, radius) in enumerate(pairs)]
    return fixed + ([{"policy": "random"}] if include_random else []) + [{"policy": "lstm"}]


def select_test_runs(training, collection, runs, selection_seed=42):
    """仅从已留出的测试 run 抽样，不从训练或验证 seed 中抽取。"""
    split = training["split_run_ids"]
    test = sorted(split["test"])
    if (runs <= 0 or runs > len(test) or len(set(test)) != len(test)
            or set(test) & (set(split["train"]) | set(split["validation"]))):
        raise ValueError
    selected = np.random.default_rng(selection_seed).choice(test, runs, replace=False).tolist()
    root = Path(collection)
    source = json.loads((root / "collection_config.json").read_text())
    signature = config_signature(source)
    seeds = []
    for run_id in selected:
        paths = list(root.glob(f"batch_*/{run_id}/manifest.json"))
        if len(paths) != 1:
            raise ValueError
        manifest = json.loads(paths[0].read_text())
        if (manifest["planned_split"] != "test" or manifest["collection_config_sha256"] != signature
                or manifest["quality"]["minute_metric_rows"] != source["simulation_steps"] * source["grid_num"]
                or not manifest.get("file_sha256")
                or any(sha256(paths[0].parent / name) != digest
                       for name, digest in manifest["file_sha256"].items())):
            raise ValueError
        seeds.append(manifest["simulator_seed"])
    if len(set(seeds)) != len(seeds):
        raise ValueError
    return selected, seeds


def run_seed(config, definitions, seed, output, model_path):
    """每个进程单线程计算，一组 seed 的策略使用同一份外生输入。"""
    global _PATTERN
    torch.set_num_threads(1)
    if _PATTERN is None:
        _PATTERN = SimulatorPattern(config["date"], data_dir=config["data_dir"])
    _, start, end = simulation_window(config["start_time"], config["simulation_steps"])
    requests, sampled = sample_common_orders(_PATTERN.request_all, start, end,
                                            config["order_sample_ratio"], seed)
    root = Path(output) / "runs" / f"seed{seed}"
    root.mkdir(parents=True, exist_ok=False)
    sampled.to_csv(root / "sampled_order_ids.csv.gz", index=False)
    order_hash = hashlib.sha256(sampled.to_csv(index=False).encode()).hexdigest()
    predictor = LSTMPredictor.load(model_path)
    results = []
    for definition in definitions:
        policy = definition["policy"]
        run_config = {**config, "fixed_price": definition.get("price", config["warmup_price"]),
                      "fixed_radius": definition.get("radius", config["warmup_radius"])}
        result = run_episode(run_config, policy, seed, requests, _PATTERN.driver_info,
                             predictor if policy == "lstm" else None, root / policy)
        result.update(sampled_orders_sha256=order_hash, sampled_order_count=len(sampled))
        save_json(root / policy / "summary.json", result)
        results.append(result)
        print(f"seed={seed} {policy}: revenue={result['platform_revenue']:.2f}, "
              f"{result['elapsed_seconds']:.1f}s", flush=True)
    for key in ("sampled_orders_sha256", "initial_driver_sha256", "warmup_metrics_sha256",
                "baseline_request_num", "generated_baseline_requests"):
        if len({row[key] for row in results}) != 1:
            raise AssertionError
    if any(row["generated_baseline_requests"] != len(sampled) for row in results):
        raise AssertionError
    return results


def summarize_results(frame, definitions):
    """保留六指标均值/标准差，并计算配对收益差的 bootstrap 区间。"""
    policies = [item["policy"] for item in definitions]
    if frame.duplicated(["seed", "policy"]).any():
        raise ValueError
    if any(set(group.policy) != set(policies) for _, group in frame.groupby("seed")):
        raise ValueError
    summary = frame.groupby("policy")[list(METRICS)].agg(["mean", "std"]).reindex(policies)
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = pd.DataFrame(definitions).set_index("policy").join(summary)
    paired = frame.pivot(index="seed", columns="policy", values="platform_revenue")
    rng = np.random.default_rng(42)
    comparisons = []
    for definition in definitions[:-1]:
        name = definition["policy"]
        differences = (paired.lstm - paired[name]).to_numpy()
        samples = rng.choice(differences, (10000, len(differences)), replace=True).mean(axis=1)
        baseline_mean = paired[name].mean()
        comparisons.append({
            **definition, "paired_seed_count": len(differences),
            "lstm_revenue_mean": paired.lstm.mean(), "baseline_revenue_mean": baseline_mean,
            "mean_revenue_difference": differences.mean(),
            "improvement_percent": 100 * differences.mean() / baseline_mean if baseline_mean else np.nan,
            "bootstrap_95_lower": np.quantile(samples, 0.025) if len(differences) > 1 else np.nan,
            "bootstrap_95_upper": np.quantile(samples, 0.975) if len(differences) > 1 else np.nan,
            "lstm_win_count": int((differences > 0).sum()),
        })
        paired[f"lstm_minus_{name}"] = differences
    return summary, paired, pd.DataFrame(comparisons)


def plot_results(output):
    """六指标论文图：真实仿真均值，误差线为跨 seed 标准差。"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output = Path(output)
    report = json.loads((output / "manifest.json").read_text())
    definitions, config = report["policies"], report["config"]
    summary = pd.read_csv(output / "policy_summary.csv").set_index("policy")
    policies = [item["policy"] for item in definitions]
    has_random = "random" in policies
    labels = [("Rand." if name == "random" else "LSTM" if name == "lstm"
               else "F" + name.removeprefix("fixed_")) for name in policies]
    family = "DejaVu Serif"
    for candidate in ("Times New Roman", "Liberation Serif"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=candidate), fallback_to_default=False)
            family = candidate
            break
        except ValueError:
            continue
    names = ("Bubble requests =", r"Order request rate $\uparrow$", r"Driver acceptance $\uparrow$",
             r"Order completion $\uparrow$", r"Mean waiting time $\downarrow$", r"Platform revenue $\uparrow$")
    colors = [{"random": "#E69F00", "lstm": "#2878B5"}.get(name, "#BBC5D0") for name in policies]
    with plt.rc_context({"font.family": family, "font.size": 8, "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.25))
        fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.28, hspace=0.52, wspace=0.48)
        for index, (ax, metric) in enumerate(zip(axes.flat, METRICS)):
            mean = summary.loc[policies, f"{metric}_mean"].to_numpy()
            deviation = summary.loc[policies, f"{metric}_std"].fillna(0).to_numpy()
            scale = 1000 if metric in ("baseline_request_num", "platform_revenue") else 1
            x = np.arange(len(policies))
            ax.bar(x, mean / scale, yerr=deviation / scale, capsize=2, width=0.7,
                   color=colors, error_kw={"elinewidth": 0.65, "capthick": 0.65})
            for position, value, error in zip(x, mean, deviation):
                label = f"{value:.0%}" if metric.endswith("rate") else f"{value / scale:.2f}" if scale == 1000 else f"{value:.1f}"
                ax.annotate(label, (position, (value + error) / scale), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=6)
            ax.set_xticks(x, labels, fontsize=7)
            if has_random:
                plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
            ax.set_title(f"({chr(97 + index)}) {names[index]}", loc="left", fontsize=8, pad=6)
            ax.set_ylim(0, max(float(np.nanmax(mean + deviation)) / scale, 1e-9) * 1.28)
            ax.grid(axis="y", alpha=0.2, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(length=2, width=0.5, labelsize=7)
            if metric.endswith("rate"):
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            elif metric == "mean_wait_to_match_seconds":
                ax.set_ylabel("Seconds")
            else:
                ax.set_ylabel(r"Requests ($10^3$)" if metric == "baseline_request_num" else r"Revenue ($10^3$ USD)")
        fig.suptitle("LSTM vs. fixed and random joint actions" if has_random else
                     "LSTM vs. fixed joint actions", fontsize=11, y=0.978)
        fig.text(0.5, 0.924, f"{config['grid_num']} grids | {config['driver_num']} drivers | "
                 f"{len(report['seeds'])} paired test seeds", ha="center", fontsize=8)
        warmup = config['history_steps'] * config['decision_freq']
        fig.text(0.5, 0.883, f"Mean ± 1 SD; shared {warmup}-min warm-up excluded", ha="center", fontsize=8)
        fixed_labels = [f"F{i + 1}: p = {d['price']:g}, r = {d['radius']:g} km"
                        for i, d in enumerate(definitions) if d["policy"].startswith("fixed_")]
        for row in range((len(fixed_labels) + 2) // 3):
            fig.text(0.5, 0.19 - row * 0.034, "     ".join(fixed_labels[row * 3:(row + 1) * 3]),
                     ha="center", fontsize=7)
        fig.text(0.5, 0.098, "LSTM: per-grid, per-step predicted-revenue maximization", ha="center", fontsize=8)
        if has_random:
            fig.text(0.5, 0.063, "Rand.: uniform joint-action sampling, independently per grid and step",
                     ha="center", fontsize=7)
        fig.text(0.5, 0.025 if has_random else 0.058,
                 r"$\uparrow$ Higher is better   $\downarrow$ Lower is better   = Controlled input", ha="center", fontsize=8)
        for extension in ("pdf", "svg", "png"):
            fig.savefig(output / f"policy_comparison.{extension}", dpi=600, facecolor="white")
        plt.close(fig)
    # 只更新重绘指纹，保留当时执行仿真的代码指纹。
    report["plot_code_sha256"] = sha256(__file__)
    report["plot_generated_at"] = datetime.now(timezone.utc).isoformat()
    save_json(output / "manifest.json", report)
    return report


def compare(model=DEFAULT_MODEL, collection_root=DEFAULT_COLLECTION, output=None,
            runs=10, workers=4, fixed_actions=DEFAULT_ACTIONS, include_random=False):
    """锁定模型/动作/测试种子后再评估，所有输出写入新的目录。"""
    if runs <= 0 or workers <= 0:
        raise ValueError
    started = perf_counter()
    torch.set_num_threads(1)
    model, collection_root = Path(model), Path(collection_root)
    training = json.loads((model / "training_config.json").read_text())
    source = json.loads((collection_root / "collection_config.json").read_text())
    project = Path(__file__).resolve().parents[1]
    check_collection_revision(collection_root, source)
    dataset_manifest = Path(training["input_dir"]) / "dataset_manifest.json"
    if sha256(dataset_manifest) != training["source_sha256"]["manifest"] or source["enable_reposition"]:
        raise ValueError
    dataset = json.loads(dataset_manifest.read_text())
    selected, seeds = select_test_runs(training, collection_root, runs)
    if any(dataset["source_run_splits"][run_id] != "test" for run_id in selected):
        raise ValueError
    definitions = policy_definitions(fixed_actions, include_random)
    checkpoint = torch.load(model / "best_model.pt", map_location="cpu", weights_only=True)
    predictor = LSTMPredictor.load(model / "best_model.pt")
    config = {key: source[key] for key in (
        "date", "data_dir", "start_time", "simulation_steps", "grid_num", "driver_num",
        "order_sample_ratio", "decision_freq", "price_response_model", "price_action_values", "matching_radius_values")}
    config.update(history_steps=predictor.preprocessor.history_steps, warmup_price=1.5, warmup_radius=3.0)
    if (predictor.preprocessor.grid_ids != list(range(config["grid_num"]))
            or state_columns(predictor.preprocessor.context_columns) != source["state_feature_columns"]
            or dataset["neighbor_adjacency_sha256"] != source["neighbor_adjacency_sha256"]
            or config["simulation_steps"] <= config["history_steps"] * config["decision_freq"]
            or any(d["price"] not in config["price_action_values"] or d["radius"] not in config["matching_radius_values"]
                   for d in definitions if d["policy"].startswith("fixed_"))):
        raise ValueError
    output = Path(output or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("experiments/lstm_vs_fixed_%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "lstm_snapshot.pt"
    snapshot.write_bytes((model / "best_model.pt").read_bytes())
    code_paths = [*project.joinpath("src").rglob("*.py"), Path(__file__),
                  project / "baseline/compare_policies.py", project / "baseline/lstm.py",
                  project / "baseline/sequence.py", project / "data_pipeline/build_supervised_dataset.py",
                  project / "data_pipeline/collect_trajectories.py"]
    report = {
        "schema_version": "lstm_fixed_comparison_v1", "status": "running", "config": config,
        "policies": definitions, "seeds": seeds, "source_test_run_ids": selected, "selection_seed": 42,
        "created_at": datetime.now(timezone.utc).isoformat(), "workers": min(workers, runs),
        "model": {"path": str(model.resolve()), "snapshot_sha256": sha256(snapshot),
                  "best_epoch": checkpoint["best_epoch"], "validation_mse": checkpoint["validation_mse"]},
        "source_sha256": source["source_sha256"],
        "provenance_sha256": {str(p): sha256(p) for p in (
            model / "training_config.json", dataset_manifest, collection_root / "collection_config.json")},
        "code_sha256": {str(p.relative_to(project)): sha256(p) for p in code_paths},
        "objective": "argmax predicted next-period platform revenue independently per grid",
        "completed_seeds": 0, "total_episodes": runs * len(definitions),
        "limitations": [
            "Single date and low-density scenario; a subset of held-out test seeds, not all test runs.",
            "Fixed actions chosen before this evaluation; no model or action tuning on the resulting test scores.",
            "Input orders, initial drivers and warm-up are paired; behavioral RNG paths may diverge after policies branch.",
            "Shared warm-up excluded from primary scores; full-window scores retained in each episode summary.",
            "Request rate may exceed 1 due to demand resampling. Completion denominator counts repeated matching attempts.",
            "Wait measures matched-passenger queueing only, not pickup or unserved passengers; revenue is commission, not profit.",
            "Independent per-grid greedy action search is not multi-step or joint spatial optimization; distribution shift is possible.",
            "Paired bootstrap intervals describe seed variation only, not parameter uncertainty or a multiple-comparison test.",
        ],
    }
    if include_random:
        report["random_policy"] = {
            "sampling": "iid uniform with replacement over the same price-radius Cartesian product as LSTM",
            "decision_unit": "each grid at each decision step after shared warm-up",
            "candidate_count": len(config["price_action_values"]) * len(config["matching_radius_values"]),
            "action_seed": "numpy SeedSequence([simulator_seed, seed_namespace])",
            "seed_namespace": RANDOM_ACTION_SEED_NAMESPACE,
            "independent_of_order_and_simulator_rng": True,
            "action_replicates_per_simulator_seed": 1,
        }
    save_json(output / "manifest.json", report)
    all_results = []

    def record(results):
        all_results.extend(results)
        frame = pd.DataFrame([{k: v for k, v in row.items() if k != "full_window"} for row in all_results])
        frame.sort_values(["seed", "policy"]).to_csv(output / "run_summary.csv", index=False)
        report.update(completed_seeds=len(all_results) // len(definitions),
                      elapsed_seconds=perf_counter() - started)
        save_json(output / "manifest.json", report)
        print(f"完成 {report['completed_seeds']}/{runs} 组配对 seed，{len(all_results)}/{report['total_episodes']} 次仿真", flush=True)

    print(f"模型 epoch={checkpoint['best_epoch']}；输出：{output}；共 {report['total_episodes']} 次仿真", flush=True)
    try:
        if workers == 1:
            for seed in seeds:
                record(run_seed(config, definitions, seed, output, snapshot))
        else:
            with ProcessPoolExecutor(max_workers=min(workers, runs)) as pool:
                jobs = [pool.submit(run_seed, config, definitions, seed, output, snapshot) for seed in seeds]
                for future in as_completed(jobs):
                    record(future.result())
        summary, paired, differences = summarize_results(pd.read_csv(output / "run_summary.csv"), definitions)
        summary.to_csv(output / "policy_summary.csv")
        paired.to_csv(output / "paired_revenue.csv")
        differences.to_csv(output / "lstm_vs_baselines.csv", index=False)
        differences[differences.policy.str.startswith("fixed_")].rename(
            columns={"baseline_revenue_mean": "fixed_revenue_mean"}).to_csv(output / "lstm_vs_fixed.csv", index=False)
        if include_random:
            differences[differences.policy == "random"].to_csv(output / "lstm_vs_random.csv", index=False)
        report = plot_results(output)
        report.update(status="complete", elapsed_seconds=perf_counter() - started,
                      completed_at=datetime.now(timezone.utc).isoformat())
        save_json(output / "manifest.json", report)
    except BaseException:
        report.update(status="interrupted_or_failed", elapsed_seconds=perf_counter() - started)
        save_json(output / "manifest.json", report)
        raise
    print(differences.to_string(index=False), flush=True)
    print(f"策略对比已完成：{output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="LSTM 与多组固定价格/半径的配对闭环仿真")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--collection-root", default=DEFAULT_COLLECTION)
    parser.add_argument("--output")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-random", action="store_true",
                        help="加入每网格、每周期均匀随机的联合动作基线")
    parser.add_argument("--fixed-actions", nargs="+", type=lambda value: tuple(map(float, value.split(":"))),
                        default=DEFAULT_ACTIONS, metavar="PRICE:RADIUS")
    compare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
