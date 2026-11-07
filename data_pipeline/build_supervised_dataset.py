"""将仿真器分钟级原始数据构建为监督学习数据集。

核心逻辑:
1. 把每分钟指标聚合到决策周期
2. 按 ``run_id + grid_id`` 构建 t-1 到 t-T 的历史指标和历史动作
3. 拼接 t 时刻决策前的本区/邻区状态、价格乘子和匹配半径
4. 使用当前动作下观测到的 t 周期网格指标作为标签

积压订单保留发单时的价格，因此标签表示历史状态和历史动作在当前
联合动作下产生的观测结果，不表示所有完成订单都使用当前价格。

使用方式:
    PYTHONPATH=. python3 -m data_pipeline.build_supervised_dataset \
        --input trajectories --history-steps 4 --output datasets
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.env.observation import STATE_SCHEMA, state_columns, validate_decision_states
from src.utils.joint_context import add_joint_context


DATASET_SCHEMA_VERSION = "grid_supervised_dataset"

SUM_COLUMNS = (
    "matched_request_num",
    "gbv",
    "platform_revenue",
    "baseline_request_num",
    "submitted_request_num",
    "driver_offer_num",
    "driver_accept_num",
    "unmatched_cancel_num",
    "pickup_cancel_num",
)

MEAN_COLUMNS = (
    "candidate_request_num",
    "idle_driver_num",
    "busy_driver_num",
    "repositioning_driver_num",
    "vehicle_occupancy_rate",
)

PERIOD_METRIC_COLUMNS = (
    "candidate_request_num",
    "matched_request_num",
    "gbv",
    "platform_revenue",
    "baseline_request_num",
    "submitted_request_num",
    "request_rate",
    "driver_offer_num",
    "driver_accept_num",
    "driver_accept_rate",
    "unmatched_cancel_num",
    "pickup_cancel_num",
    "unmatched_cancel_rate",
    "pickup_cancel_rate",
    "order_completion_rate",
    "mean_wait_to_match_seconds",
    "mean_pickup_time_seconds",
    "mean_pickup_distance_km",
    "idle_driver_num",
    "busy_driver_num",
    "repositioning_driver_num",
    "vehicle_occupancy_rate",
)

HISTORY_ACTION_COLUMNS = (
    "price_multiplier",
    "matching_radius_km",
)


def _safe_ratio(numerator, denominator):
    """分母为零时返回 0。"""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.zeros(len(numerator), dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def aggregate_decision_metrics(action_plan, minute_metrics, run_metadata):
    """将一次仿真的分钟级指标聚合到决策周期。"""
    current_state_columns = state_columns(action_plan.columns)
    if run_metadata.get("decision_state_schema") is not None:
        if (run_metadata["decision_state_schema"] != STATE_SCHEMA
                or run_metadata.get("state_feature_columns") != current_state_columns):
            raise ValueError
        validate_decision_states(action_plan, run_metadata["neighbor_adjacency"])
    required = set(SUM_COLUMNS + MEAN_COLUMNS + (
        "minute_step", "sim_time", "decision_step", "grid_id",
        "mean_wait_to_match_seconds", "mean_pickup_time_seconds",
        "mean_pickup_distance_km"))
    if not required.issubset(minute_metrics.columns):
        raise ValueError
    if minute_metrics.duplicated(["minute_step", "grid_id"]).any():
        raise ValueError

    keys = ["decision_step", "grid_id"]
    grouped = minute_metrics.groupby(keys, sort=True)
    period = grouped.size().rename("interval_minutes").reset_index()
    period["decision_start_sim_time"] = grouped["sim_time"].min().to_numpy()

    time_by_step = minute_metrics.groupby("minute_step")["sim_time"]
    if not time_by_step.nunique().eq(1).all():
        raise ValueError
    time_by_step = time_by_step.first().sort_index()
    minute_steps = time_by_step.index.to_numpy(dtype=float)
    sim_times = time_by_step.to_numpy(dtype=float)
    if len(time_by_step) < 2:
        raise ValueError
    step_seconds = np.diff(sim_times) / np.diff(minute_steps)
    if (np.any(step_seconds <= 0)
            or not np.allclose(step_seconds, step_seconds[0])):
        raise ValueError
    period["decision_end_sim_time_exclusive"] = (
        grouped["sim_time"].max().to_numpy() + step_seconds[0])

    for column in SUM_COLUMNS:
        period[column] = grouped[column].sum().to_numpy()
    for column in MEAN_COLUMNS:
        period[column] = grouped[column].mean().to_numpy()

    matched = minute_metrics["matched_request_num"].to_numpy(dtype=float)
    weighted_columns = (
        "mean_wait_to_match_seconds",
        "mean_pickup_time_seconds",
        "mean_pickup_distance_km",
    )
    for column in weighted_columns:
        weighted = minute_metrics.assign(
            _weighted=minute_metrics[column].to_numpy(dtype=float) * matched
        ).groupby(keys, sort=True)["_weighted"].sum().to_numpy()
        period[column] = _safe_ratio(weighted, period["matched_request_num"])

    period["request_rate"] = _safe_ratio(
        period["submitted_request_num"], period["baseline_request_num"])
    period["driver_accept_rate"] = _safe_ratio(
        period["driver_accept_num"], period["driver_offer_num"])
    pre_cancel_request_num = (
        grouped["candidate_request_num"].sum().to_numpy()
        + period["unmatched_cancel_num"])
    period["unmatched_cancel_rate"] = _safe_ratio(
        period["unmatched_cancel_num"], pre_cancel_request_num)
    period["pickup_cancel_rate"] = _safe_ratio(
        period["pickup_cancel_num"],
        period["matched_request_num"] + period["pickup_cancel_num"])
    period["order_completion_rate"] = _safe_ratio(
        period["matched_request_num"], pre_cancel_request_num)

    plan_columns = [
        "decision_step", "grid_id", "decision_start_minute_step",
        "decision_end_minute_step_exclusive", "joint_action_id",
        "price_multiplier", "matching_radius_km", "is_baseline_action",
    ] + current_state_columns
    if not set(plan_columns).issubset(action_plan.columns):
        raise ValueError
    if action_plan.duplicated(["decision_step", "grid_id"]).any():
        raise ValueError
    period = period.merge(
        action_plan[plan_columns],
        on=keys,
        how="inner",
        validate="one_to_one")
    if len(period) != len(action_plan):
        raise RuntimeError
    planned_minutes = (
        period["decision_end_minute_step_exclusive"]
        - period["decision_start_minute_step"])
    if not np.array_equal(
            period["interval_minutes"].to_numpy(),
            planned_minutes.to_numpy()):
        raise RuntimeError
    if not np.allclose(
            period["decision_end_sim_time_exclusive"]
            - period["decision_start_sim_time"],
            period["interval_minutes"] * step_seconds[0]):
        raise RuntimeError

    period.insert(0, "run_id", run_metadata["run_id"])
    period.insert(1, "date", run_metadata["date"])
    period.insert(2, "seed", int(run_metadata["simulator_seed"]))
    return period.sort_values(keys).reset_index(drop=True)


def build_history_samples(period_metrics, history_steps, joint_action_context=False):
    """用历史指标、历史动作和当前动作构建当前周期观测标签。"""
    if (isinstance(history_steps, bool)
            or int(history_steps) != history_steps
            or history_steps <= 0):
        raise ValueError
    history_steps = int(history_steps)
    required = set(PERIOD_METRIC_COLUMNS + (
        "run_id", "date", "seed", "decision_step", "grid_id",
        "decision_start_minute_step", "decision_end_minute_step_exclusive",
        "decision_start_sim_time", "decision_end_sim_time_exclusive",
        "interval_minutes", "joint_action_id", "price_multiplier",
        "matching_radius_km", "is_baseline_action",
    ))
    if not required.issubset(period_metrics.columns):
        raise ValueError

    joint_context_columns = []
    if joint_action_context:
        period_metrics, joint_context_columns = add_joint_context(period_metrics)

    period_metrics = period_metrics.sort_values(
        ["run_id", "grid_id", "decision_step"]).reset_index(drop=True)
    groups = period_metrics.groupby(["run_id", "grid_id"], sort=False)
    gaps = groups["decision_step"].diff().dropna()
    if not gaps.eq(1).all():
        raise ValueError

    history = {}
    for lag in range(1, history_steps + 1):
        for column in PERIOD_METRIC_COLUMNS + HISTORY_ACTION_COLUMNS:
            history[f"lag_{lag}_{column}"] = groups[column].shift(lag)
    history_frame = pd.DataFrame(history, index=period_metrics.index)

    target_frame = pd.DataFrame({
        f"target_{column}": period_metrics[column]
        for column in PERIOD_METRIC_COLUMNS
    })
    metadata_columns = [
        "run_id", "date", "seed", "decision_step",
        "decision_start_minute_step", "decision_end_minute_step_exclusive",
        "decision_start_sim_time", "decision_end_sim_time_exclusive",
        "interval_minutes", "grid_id", "joint_action_id", "price_multiplier",
        "matching_radius_km", "is_baseline_action",
    ] + state_columns(period_metrics.columns) + joint_context_columns
    dataset = pd.concat([
        period_metrics[metadata_columns], history_frame, target_frame,
    ], axis=1)
    dataset = dataset[history_frame.notna().all(axis=1)]
    return dataset.reset_index(drop=True)


def resolve_split_by(dates, split_by, validation_ratio, test_ratio):
    """多日期默认整天隔离；单日期保留 run 划分并明确评估范围。"""
    if (split_by not in ("auto", "date", "run")
            or not np.isfinite(validation_ratio)
            or not np.isfinite(test_ratio)
            or validation_ratio < 0 or test_ratio < 0
            or validation_ratio + test_ratio >= 1):
        raise ValueError
    date_count = len(set(dates))
    if split_by == "auto":
        split_by = "date" if date_count > 1 else "run"
    if split_by == "date" and date_count < 1 + (validation_ratio > 0) + (test_ratio > 0):
        raise ValueError
    return split_by


def assign_run_splits(dataset, validation_ratio=0.15, test_ratio=0.15,
                      split_seed=42, split_by="run"):
    """按 run 随机划分或按日期先后划分，同一天的所有时段可整体隔离。"""
    if dataset.empty:
        raise ValueError
    dates = dataset["date"].unique() if "date" in dataset else []
    split_by = resolve_split_by(dates, split_by, validation_ratio, test_ratio)
    if split_by == "date" and not dataset.groupby("run_id")["date"].nunique().eq(1).all():
        raise ValueError
    group_column = "date" if split_by == "date" else "run_id"
    group_ids = np.asarray(sorted(dataset[group_column].unique()))
    if split_by == "run":
        np.random.default_rng(split_seed).shuffle(group_ids)
    group_count = len(group_ids)
    validation_count = int(group_count * validation_ratio)
    test_count = int(group_count * test_ratio)
    if group_count >= 3 and validation_ratio > 0:
        validation_count = max(1, validation_count)
    if group_count >= 3 and test_ratio > 0:
        test_count = max(1, test_count)
    if split_by == "date":
        validation_count = max(int(validation_ratio > 0), validation_count)
        test_count = max(int(test_ratio > 0), test_count)
    while validation_count + test_count >= group_count:
        if test_count >= validation_count and test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1

    if split_by == "date":
        train_count = group_count - validation_count - test_count
        validation_groups = group_ids[train_count:train_count + validation_count]
        test_groups = group_ids[train_count + validation_count:]
    else:
        test_groups = group_ids[:test_count]
        validation_groups = group_ids[test_count:test_count + validation_count]
    dataset = dataset.copy()
    dataset["split"] = "train"
    dataset.loc[dataset[group_column].isin(validation_groups), "split"] = "validation"
    dataset.loc[dataset[group_column].isin(test_groups), "split"] = "test"
    return dataset


def _load_runs(input_dir):
    """固定已发布 run 的清单，先只读元数据，不同时加载所有 CSV。"""
    manifests = []
    for path in sorted(Path(input_dir).rglob("manifest.json")):
        if any(part.startswith(".working_") for part in path.parts):
            continue
        with path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        files = metadata.get("files", {})
        action_path = path.parent / files.get("action_plan", "action_plan.csv")
        minute_path = path.parent / files.get("minute_metrics", "grid_minute_metrics.csv")
        if not action_path.is_file() or not minute_path.is_file():
            raise FileNotFoundError
        manifests.append((metadata, action_path, minute_path))
    if not manifests:
        raise FileNotFoundError
    run_ids = [metadata.get("run_id") for metadata, _, _ in manifests]
    if None in run_ids or len(run_ids) != len(set(run_ids)):
        raise ValueError
    return sorted(manifests, key=lambda run: run[0]["run_id"])


def _read_run(metadata, action_path, minute_path):
    """逐次验证压缩文件摘要和采集行数，再交给原有聚合逻辑。"""
    for path in (action_path, minute_path):
        expected = metadata.get("file_sha256", {}).get(path.name)
        if expected is not None:
            with path.open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                    raise ValueError
    plan, minute = pd.read_csv(action_path), pd.read_csv(minute_path)
    quality = metadata.get("quality", {})
    if (len(plan) != quality.get("action_plan_rows", len(plan))
            or len(minute) != quality.get("minute_metric_rows", len(minute))):
        raise ValueError
    if "grid_num" in metadata:
        expected_grids = set(range(metadata["grid_num"]))
        if set(plan["grid_id"]) != expected_grids or set(minute["grid_id"]) != expected_grids:
            raise ValueError
    return plan, minute


def _run_split_plan(runs, split_by, validation_ratio, test_ratio, split_seed):
    """优先沿用采集时的固定划分；旧轨迹才按原有规则生成划分。"""
    table = pd.DataFrame([{"run_id": run[0]["run_id"], "date": run[0]["date"]}
                          for run in runs])
    planned = [run[0].get("planned_split") for run in runs]
    has_plan = any(value is not None for value in planned)
    if has_plan:
        if not all(value in ("train", "validation", "test") for value in planned):
            raise ValueError
        split_by = resolve_split_by(table["date"], "run" if split_by == "auto" else split_by,
                                    validation_ratio, test_ratio)
        if split_by != "run":
            raise ValueError
        table["split"] = planned
        split_order = "collection_plan"
    else:
        split_by = resolve_split_by(table["date"], split_by, validation_ratio, test_ratio)
        table = assign_run_splits(table, validation_ratio, test_ratio, split_seed, split_by)
        split_order = "chronological" if split_by == "date" else "random"
    return table, split_by, split_order


def build_supervised_dataset(input_dir="trajectories", output_dir="datasets",
                             history_steps=4, validation_ratio=0.15,
                             test_ratio=0.15, split_seed=42,
                             overwrite=False, split_by="auto", return_dataset=True,
                             joint_action_context=False, max_runs=None):
    """逐 run 构建并落盘；CLI 不返回整张表，Python 调用保留原返回接口。"""
    if max_runs is not None and (isinstance(max_runs, bool)
                                or not isinstance(max_runs, int) or max_runs <= 0):
        raise ValueError
    output_dir = Path(output_dir)
    dataset_path = output_dir / "supervised_dataset.csv"
    manifest_path = output_dir / "dataset_manifest.json"
    if ((dataset_path.exists() or manifest_path.exists()) and not overwrite):
        raise FileExistsError
    runs = _load_runs(input_dir)
    selection_order = "run_id"
    if max_runs is not None:
        available_runs = len(runs)
        if all("run_index" in run[0] for run in runs):
            selection_order = "run_index"
            runs.sort(key=lambda run: run[0]["run_index"])
        runs = sorted(runs[:max_runs], key=lambda run: run[0]["run_id"])
        print(f"已选择前 {len(runs):,}/{available_runs:,} 个完整 run，"
              f"选择顺序：{selection_order}", flush=True)
    decision_frequencies = {run[0]["decision_freq_minutes"] for run in runs}
    if len(decision_frequencies) != 1:
        raise ValueError
    for key in ("grid_num", "grid_geometry_sha256", "collection_config_sha256",
                "decision_state_schema", "neighbor_adjacency_sha256"):
        if len({run[0].get(key) for run in runs}) > 1:
            raise ValueError
    sampling_policies = {json.dumps(run[0].get("action_sampling"), sort_keys=True) for run in runs}
    if len(sampling_policies) != 1:
        raise ValueError
    run_table, split_by, split_order = _run_split_plan(
        runs, split_by, validation_ratio, test_ratio, split_seed)
    split_map = run_table.set_index("run_id")["split"].to_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = Counter()
    source_sample_counts = {}
    row_count = 0
    # 临时文件只在全部 run 成功后发布；异常不会留下貌似完整的训练集。
    with tempfile.TemporaryDirectory(prefix=".building_", dir=output_dir) as temporary:
        staged_dataset = Path(temporary) / dataset_path.name
        columns = None
        with staged_dataset.open("w", encoding="utf-8", newline="") as file:
            for index, (metadata, action_path, minute_path) in enumerate(runs):
                plan, minute = _read_run(metadata, action_path, minute_path)
                periods = aggregate_decision_metrics(plan, minute, metadata)
                samples = build_history_samples(
                    periods, history_steps, joint_action_context=joint_action_context)
                if samples.empty:
                    raise ValueError
                split = split_map[metadata["run_id"]]
                samples["split"] = split
                if columns is None:
                    columns = samples.columns.tolist()
                elif columns != samples.columns.tolist():
                    raise ValueError
                samples.to_csv(file, index=False, header=index == 0)
                row_count += len(samples)
                split_counts[split] += len(samples)
                source_sample_counts[metadata["run_id"]] = len(samples)
                if (index + 1) % 100 == 0 or index + 1 == len(runs):
                    print(f"已构建 {index + 1:,}/{len(runs):,} 个 run，{row_count:,} 条样本", flush=True)
        staged_dataset.replace(dataset_path)
    feature_columns = [
        "grid_id", "decision_start_minute_step", "decision_start_sim_time",
        "interval_minutes",
        "price_multiplier", "matching_radius_km",
    ] + state_columns(columns) + [
        column for column in columns
        if column.startswith("global_") or column.startswith("joint_price_grid_")
        or column.startswith("joint_radius_grid_")
    ] + [column for column in columns if column.startswith("lag_")]
    target_columns = [
        column for column in columns if column.startswith("target_")]
    additional_context_columns = state_columns(columns)
    if joint_action_context:
        additional_context_columns += [
            column for column in columns
            if column.startswith("global_") or column.startswith("joint_price_grid_")
            or column.startswith("joint_radius_grid_")
        ]
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_run_count": len(runs),
        "source_run_limit": max_runs,
        "source_selection_order": selection_order,
        "source_run_ids": run_table["run_id"].tolist(),
        "source_dates": sorted(run_table["date"].unique().tolist()),
        "source_run_splits": split_map,
        "source_run_sample_counts": source_sample_counts,
        "split_by": split_by,
        "split_order": split_order,
        "row_order": "run_id,grid_id,decision_step",
        "split_dates": {name: sorted(frame["date"].unique().tolist())
                        for name, frame in run_table.groupby("split")},
        "split_run_counts": {name: int(frame["run_id"].nunique())
                             for name, frame in run_table.groupby("split")},
        "history_steps": int(history_steps),
        "decision_freq_minutes": int(next(iter(decision_frequencies))),
        "sample_definition": (
            "historical metrics and actions + available pre-action state + current joint action -> "
            "current-period observed metrics"),
        "target_period": "current decision period observations",
        "queued_order_pricing": (
            "queued orders retain the price assigned when submitted"),
        "row_count": row_count,
        "split_counts": dict(split_counts),
        "feature_columns": feature_columns,
        "additional_context_columns": additional_context_columns,
        "joint_action_context": bool(joint_action_context),
        "state_feature_columns": state_columns(columns),
        "decision_state_schema": runs[0][0].get("decision_state_schema"),
        "neighbor_adjacency": runs[0][0].get("neighbor_adjacency"),
        "neighbor_adjacency_sha256": runs[0][0].get("neighbor_adjacency_sha256"),
        "target_columns": target_columns,
        "action_sampling": runs[0][0].get("action_sampling"),
        "sampling_mode_run_counts": dict(Counter(
            run[0].get("sampling_mode", "legacy_balanced") for run in runs)),
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print(f"监督学习数据集已保存至：{dataset_path}")
    print(f"  样本数：{row_count:,}")
    print(f"  历史窗口：{history_steps} 个决策周期")
    print(f"  数据划分：{manifest['split_counts']}")
    print(f"  划分依据：{split_by}，各集合日期：{manifest['split_dates']}")
    return (pd.read_csv(dataset_path) if return_dataset else None), manifest


def main():
    parser = argparse.ArgumentParser(
        description="将联合动作仿真轨迹构建为监督学习数据集")
    parser.add_argument("--input", default="trajectories")
    parser.add_argument("--output", default="datasets")
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-by", choices=["auto", "date", "run"], default="auto")
    parser.add_argument("--joint-action-context", action="store_true",
                        help="为每个网格增加全城状态与全网格动作向量特征（BCS 专用）")
    parser.add_argument("--max-runs", type=int,
                        help="仅使用前 N 次已发布完整轨迹，优先按采集编号选择")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_supervised_dataset(
        input_dir=args.input,
        output_dir=args.output,
        history_steps=args.history_steps,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        split_by=args.split_by,
        overwrite=args.overwrite,
        joint_action_context=args.joint_action_context,
        max_runs=args.max_runs,
        return_dataset=False)


if __name__ == "__main__":
    main()
