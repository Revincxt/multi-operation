"""在同一批测试 seed 上比较固定策略、单轮坐标贪心和多轮 BCS。

使用方式:
    python3 -m baseline.compare_bcs --dataset datasets_grid6_low_mixed47k_20261029 \
      --checkpoint models/lstm_grid6_low_mixed47k_mps100_20261029/best_model.pt --runs 10
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.compare_policies import (
    METRICS, run_episode, sample_common_orders,
)
from baseline.data import DEFAULT_COLLECTION
from baseline.lstm import LSTMPredictor
from baseline.sequence import CONTEXT_COLUMNS
from data_pipeline.collect_density import (
    config_signature, make_config, sha256, validate_execution_revision,
)
from data_pipeline.collect_trajectories import simulation_window
from src.env.simulator_pattern import SimulatorPattern


POLICIES = ("fixed", "greedy", "bcs")
COLORS = ("#94a3b8", "#2563eb", "#8b5cf6")


def _training_source_matches(dataset_dir, checkpoint):
    manifest_path = dataset_dir / "dataset_manifest.json"
    data_path = dataset_dir / "supervised_dataset.csv"
    config_path = checkpoint.parent / "training_config.json"
    if not config_path.is_file():
        raise ValueError
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_hashes = {"csv": sha256(data_path), "manifest": sha256(manifest_path)}
    if (config.get("input_dir") != str(dataset_dir.resolve())
            or config.get("source_sha256") != expected_hashes
            or not manifest.get("joint_action_context")):
        raise ValueError
    predictor = LSTMPredictor.load(checkpoint)
    expected_context = list(CONTEXT_COLUMNS) + manifest.get("additional_context_columns", [])
    if (not predictor.preprocessor.has_joint_actions
            or predictor.preprocessor.context_columns != expected_context
            or config.get("model", {}).get("context_size") != len(expected_context)
            or set(predictor.preprocessor.grid_ids) != set(range(6))):
        raise ValueError
    return manifest, predictor


def _validate_collection(root, source):
    current = make_config(
        source["scenario"], source["run_count"], source["seed_start"],
        source["batch_size"], source["data_dir"], source["date"],
        source["start_time"], source["simulation_steps"])
    if ({key: value for key, value in source.items() if key != "code_sha256"}
            != {key: value for key, value in current.items() if key != "code_sha256"}):
        raise ValueError
    # 新增分析/特征文件可以存在；已有源码只接受原版或已验证的精确修订。
    if any(current["code_sha256"].get(path) != digest
           for path, digest in source.get("code_sha256", {}).items()):
        revision = json.loads((Path(root) / "execution_revision.json").read_text())
        approved = revision.get("code_sha256", {})
        if not set(source["code_sha256"]).issubset(approved):
            raise ValueError
        checked = {**current, "code_sha256": {
            path: current["code_sha256"].get(path) for path in approved}}
        validate_execution_revision(root, source, checked)
    return config_signature(source)


def plot_bcs_results(output):
    """输出六指标配对策略总览图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output = Path(output)
    frame = pd.read_csv(output / "run_summary.csv")
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10}):
        figure, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
        for axis, (metric, (title, direction)) in zip(axes.flat, METRICS.items()):
            stats = frame.groupby("policy")[metric].agg(["mean", "std"]).reindex(POLICIES)
            bars = axis.bar(POLICIES, stats["mean"], yerr=stats["std"].fillna(0),
                            capsize=4, color=COLORS)
            axis.bar_label(
                bars,
                labels=[f"{value:.1%}" if metric.endswith("rate") else f"{value:,.1f}"
                        for value in stats["mean"]],
                padding=8, fontsize=9)
            axis.set_title(f"{title}\n{direction}")
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
            upper = (stats["mean"] + stats["std"].fillna(0)).max()
            axis.set_ylim(0, max(float(upper), 1e-9) * 1.25)
            if metric.endswith("rate"):
                axis.yaxis.set_major_formatter(PercentFormatter(1))
        report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        figure.suptitle(
            f"Fixed vs one-pass greedy vs BCS | {len(report['seeds'])} paired seeds\n"
            f"{report['config']['grid_num']} grids, {report['config']['driver_num']} drivers; "
            "bars: seed mean, error bars: ±1 SD")
        figure.savefig(output / "bcs_comparison.png", dpi=180)
        figure.savefig(output / "bcs_comparison.pdf")
        plt.close(figure)


def compare_bcs(dataset, checkpoint, collection_root=DEFAULT_COLLECTION,
                output=None, runs=10, max_sweeps=5,
                fixed_price=1.5, fixed_radius=3.0):
    """使用模型训练时保留的 test runs 比较固定策略、贪心和 BCS。"""
    if (runs <= 0 or max_sweeps <= 0
            or not np.isfinite([fixed_price, fixed_radius]).all()
            or min(fixed_price, fixed_radius) <= 0):
        raise ValueError
    dataset_dir, checkpoint, collection_root = map(Path, (dataset, checkpoint, collection_root))
    dataset_manifest, predictor = _training_source_matches(dataset_dir, checkpoint)
    source_config = json.loads((collection_root / "collection_config.json").read_text(
        encoding="utf-8"))
    if source_config["enable_reposition"]:
        raise ValueError
    signature = _validate_collection(collection_root, source_config)
    test_runs = sorted(run_id for run_id, split in dataset_manifest["source_run_splits"].items()
                       if split == "test")
    if runs > len(test_runs):
        raise ValueError
    selected = np.random.default_rng(42).choice(test_runs, runs, replace=False).tolist()
    seeds = []
    for run_id in selected:
        paths = list(collection_root.glob(f"batch_*/{run_id}/manifest.json"))
        if len(paths) != 1:
            raise ValueError
        run_manifest = json.loads(paths[0].read_text(encoding="utf-8"))
        if (run_manifest["planned_split"] != "test"
                or run_manifest["collection_config_sha256"] != signature):
            raise ValueError
        seeds.append(int(run_manifest["simulator_seed"]))
    if len(set(seeds)) != len(seeds):
        raise ValueError

    config = {key: source_config[key] for key in (
        "date", "data_dir", "start_time", "simulation_steps", "grid_num", "driver_num",
        "order_sample_ratio", "decision_freq", "price_response_model", "price_action_values",
        "matching_radius_values")}
    config.update(fixed_price=fixed_price, fixed_radius=fixed_radius,
                  history_steps=predictor.preprocessor.history_steps)
    if config["grid_num"] != len(predictor.preprocessor.grid_ids):
        raise ValueError

    output = Path(output or datetime.now().strftime("experiments/bcs_comparison_%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "joint_lstm_bcs_comparison_v1", "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
        "dataset": str(dataset_dir.resolve()), "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint), "seeds": seeds,
        "execution_revision_sha256": (sha256(collection_root / "execution_revision.json")
                                      if (collection_root / "execution_revision.json").is_file()
                                      else None),
        "source_test_run_ids": selected, "selection_seed": 42,
        "max_bcs_sweeps": max_sweeps,
        "objective": "sum of six grids' predicted current-period platform revenue",
        "policies": {
            "fixed": f"p={fixed_price:g}, r={fixed_radius:g} km",
            "greedy": "one ordered coordinate sweep, same joint-action LSTM",
            "bcs": f"coordinate sweeps until stable or {max_sweeps} sweeps",
        },
        "limitations": [
            "测试 seed 与监督数据集 test split 一致；按 seed 配对比较。",
            "BCS 是坐标局部搜索，不保证找到全局联合最优。",
            "BCS 与贪心共用一个联合动作模型；模拟结果是闭环评估，不是模型预测值。",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    all_results = []
    for seed in seeds:
        pattern = SimulatorPattern(config["date"], data_dir=config["data_dir"])
        _, start, end = simulation_window(config["start_time"], config["simulation_steps"])
        requests, sampled = sample_common_orders(
            pattern.request_all, start, end, config["order_sample_ratio"], seed)
        seed_root = output / "runs" / f"seed{seed}"
        seed_root.mkdir(parents=True)
        sampled.to_csv(seed_root / "sampled_order_ids.csv.gz", index=False)
        results = []
        for policy, optimizer, sweeps, model in (
                ("fixed", "greedy", 1, None),
                ("greedy", "greedy", 1, predictor),
                ("bcs", "bcs", max_sweeps, predictor)):
            row = run_episode(
                config, policy, seed, requests, pattern.driver_info, model,
                seed_root / policy, action_optimizer=optimizer, bcs_max_sweeps=sweeps)
            results.append(row)
            print(f"seed={seed} {policy}: revenue={row['platform_revenue']:.2f}", flush=True)
        if (len({row["initial_driver_sha256"] for row in results}) != 1
                or len({row["baseline_request_num"] for row in results}) != 1):
            raise AssertionError
        all_results.extend(results)

    frame = pd.DataFrame([{key: value for key, value in row.items() if key != "full_window"}
                          for row in all_results]).sort_values(["seed", "policy"])
    frame.to_csv(output / "run_summary.csv", index=False)
    summary = frame.groupby("policy")[list(METRICS)].agg(["mean", "std"]).reindex(POLICIES)
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv(output / "policy_summary.csv")
    paired = frame.pivot(index="seed", columns="policy", values="platform_revenue")
    paired["bcs_minus_greedy"] = paired["bcs"] - paired["greedy"]
    paired["bcs_improvement_percent"] = 100 * (
        paired["bcs"] / paired["greedy"].replace(0, np.nan) - 1)
    paired.to_csv(output / "paired_revenue.csv")
    report["status"] = "complete"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["code_sha256"] = sha256(__file__)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    plot_bcs_results(output)
    print(summary.to_string(), flush=True)
    print(f"BCS 对比结果已保存至：{output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="联合动作 LSTM 上比较单轮贪心和 BCS")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--collection-root", default=DEFAULT_COLLECTION)
    parser.add_argument("--output")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-sweeps", type=int, default=5)
    parser.add_argument("--fixed-price", type=float, default=1.5)
    parser.add_argument("--fixed-radius", type=float, default=3.0)
    compare_bcs(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
