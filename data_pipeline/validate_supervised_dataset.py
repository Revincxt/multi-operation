"""检查监督学习数据集的结构和指标质量。

使用方式:
    PYTHONPATH=. python3 -m data_pipeline.validate_supervised_dataset --input datasets
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.env.observation import (
    LOCAL_STATE_COLUMNS, STATE_SCHEMA, state_columns, validate_decision_states,
)
from src.utils.joint_context import GLOBAL_STATE_COLUMNS, joint_action_columns
from data_pipeline.collect_trajectories import ACTION_SAMPLING_VERSION


REPORT_SCHEMA_VERSION = "supervised_dataset_validation_v1"

BOUNDED_RATE_SUFFIXES = (
    "driver_accept_rate",
    "unmatched_cancel_rate",
    "pickup_cancel_rate",
    "order_completion_rate",
    "vehicle_occupancy_rate",
)


def _paths(input_path, output_path):
    """解析数据集、manifest和报告路径。"""
    input_path = Path(input_path)
    if input_path.is_dir():
        dataset_path = input_path / "supervised_dataset.csv"
        manifest_path = input_path / "dataset_manifest.json"
    else:
        dataset_path = input_path
        manifest_path = input_path.parent / "dataset_manifest.json"
    report_path = (
        Path(output_path) if output_path
        else dataset_path.parent / "validation_report.json")
    if not dataset_path.exists() or not manifest_path.exists():
        raise FileNotFoundError
    return dataset_path, manifest_path, report_path


def _range_summary(dataset, columns):
    """返回一组列的整体最小值和最大值。"""
    if not columns:
        return None
    values = dataset[columns].to_numpy(dtype=float)
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _validate_frame(dataset, manifest, dataset_path):
    """检查包含若干完整 run 的一块数据，不加载整个数据集。"""
    key_columns = ["run_id", "decision_step", "grid_id"]
    action_columns = ["price_multiplier", "matching_radius_km"]
    time_columns = [
        "decision_start_minute_step",
        "decision_end_minute_step_exclusive",
        "decision_start_sim_time",
        "decision_end_sim_time_exclusive",
        "interval_minutes",
    ]
    feature_columns = manifest.get("feature_columns", [])
    current_state_columns = state_columns(dataset.columns)
    target_columns = manifest.get("target_columns", [])
    required = set(
        key_columns + action_columns + time_columns
        + ["date", "split"] + feature_columns + target_columns)
    if dataset.empty or not required.issubset(dataset.columns):
        raise ValueError

    numeric = dataset.select_dtypes(include=[np.number])
    metric_columns = [
        column for column in dataset
        if column.startswith("lag_") or column.startswith("target_")]
    bounded_rate_columns = [
        column for column in metric_columns
        if column.endswith(BOUNDED_RATE_SUFFIXES)]
    request_rate_columns = [
        column for column in metric_columns
        if column.endswith("request_rate")]

    run_split_counts = dataset.groupby("run_id")["split"].nunique()
    steps = dataset.sort_values(key_columns).groupby(
        ["run_id", "grid_id"])["decision_step"].diff().dropna()
    decision_grid_counts = dataset.groupby(
        ["run_id", "decision_step"])["grid_id"].nunique()
    run_grid_counts = dataset.groupby("run_id")["grid_id"].nunique()
    complete_grids = all(
        count == run_grid_counts.loc[run_id]
        for (run_id, _), count in decision_grid_counts.items())

    action_coverage = dataset.groupby(action_columns).size().rename(
        "sample_count").reset_index()
    action_key = list(zip(
        dataset["price_multiplier"], dataset["matching_radius_km"]))
    action_work = dataset[["run_id", "decision_step"]].assign(_action_key=action_key)
    period_action_counts = action_work.groupby(
        ["run_id", "decision_step", "_action_key"]).size()
    actions_per_period = period_action_counts.groupby(
        ["run_id", "decision_step"]).size()
    extremes = period_action_counts.groupby(["run_id", "decision_step"]).agg(["min", "max"])
    action_spread = extremes["max"] - extremes["min"]

    expected_completion_rate = np.divide(
        dataset["target_matched_request_num"].to_numpy(dtype=float),
        (
            dataset["target_candidate_request_num"].to_numpy(dtype=float)
            * dataset["interval_minutes"].to_numpy(dtype=float)
            + dataset["target_unmatched_cancel_num"].to_numpy(dtype=float)
        ),
        out=np.zeros(len(dataset), dtype=float),
        where=(
            dataset["target_candidate_request_num"].to_numpy(dtype=float)
            * dataset["interval_minutes"].to_numpy(dtype=float)
            + dataset["target_unmatched_cancel_num"].to_numpy(dtype=float)
        ) > 0,
    )

    actual_split_counts = {
        key: int(value)
        for key, value in dataset["split"].value_counts().items()
    }
    split_by = manifest.get("split_by", "run")
    split_dates = {name: sorted(frame["date"].unique().tolist())
                   for name, frame in dataset.groupby("split")}
    dates_are_isolated = dataset.groupby("date")["split"].nunique().eq(1).all()
    checks = {
        "schema_version_supported": (
            manifest.get("schema_version") == "grid_supervised_dataset"),
        "row_count_matches_manifest": (
            len(dataset) == manifest.get("row_count")),
        "split_counts_match_manifest": (
            actual_split_counts == manifest.get("split_counts")),
        "no_duplicate_keys": not dataset.duplicated(key_columns).any(),
        "no_missing_values": not dataset.isna().any().any(),
        "numeric_values_are_finite": np.isfinite(numeric.to_numpy()).all(),
        "metrics_are_non_negative": (
            dataset[metric_columns].to_numpy(dtype=float) >= 0).all(),
        "bounded_rates_are_valid": (
            (dataset[bounded_rate_columns].to_numpy(dtype=float) >= 0).all()
            and (dataset[bounded_rate_columns].to_numpy(dtype=float) <= 1).all()),
        "request_rates_are_non_negative": (
            dataset[request_rate_columns].to_numpy(dtype=float) >= 0).all(),
        "actions_are_positive": (
            dataset[action_columns].to_numpy(dtype=float) > 0).all(),
        "run_splits_are_isolated": run_split_counts.eq(1).all(),
        "run_dates_are_consistent": dataset.groupby("run_id")["date"].nunique().eq(1).all(),
        "split_by_is_supported": split_by in ("date", "run"),
        "decision_steps_are_continuous": steps.eq(1).all(),
        "grid_coverage_is_complete": complete_grids,
        "simulation_times_are_valid": (
            (dataset["decision_end_sim_time_exclusive"]
             > dataset["decision_start_sim_time"]).all()
            and np.allclose(
                dataset["decision_end_sim_time_exclusive"]
                - dataset["decision_start_sim_time"],
                dataset["interval_minutes"] * 60)),
        "order_completion_rate_is_consistent": np.allclose(
            dataset["target_order_completion_rate"],
            expected_completion_rate),
    }
    sampling = manifest.get("action_sampling")
    if sampling is None:
        # 旧数据仍按当时的平衡采样约定校验，不把新旧采集方式混为一谈。
        checks.update({
            "joint_actions_cover_each_period": actions_per_period.eq(
                decision_grid_counts.clip(upper=len(action_coverage))).all(),
            "joint_action_product_is_complete": (
                len(action_coverage) == dataset["price_multiplier"].nunique()
                * dataset["matching_radius_km"].nunique()),
            "joint_actions_are_balanced": action_spread.le(1).all(),
        })
    else:
        valid_sampling = (isinstance(sampling, dict)
                          and sampling.get("schema_version") == ACTION_SAMPLING_VERSION
                          and sampling.get("with_replacement") is True)
        checks["action_sampling_schema_supported"] = valid_sampling
        checks["actions_match_declared_candidates"] = bool(valid_sampling and (
            dataset["price_multiplier"].isin(sampling.get("price_action_values", [])).all()
            and dataset["matching_radius_km"].isin(sampling.get("matching_radius_values_km", [])).all()))
    if "split_dates" in manifest:
        checks["split_dates_match_manifest"] = split_dates == manifest["split_dates"]
    checks["state_features_match_manifest"] = (
        current_state_columns == manifest.get("state_feature_columns", [])
        and set(current_state_columns).issubset(feature_columns))
    if manifest.get("joint_action_context", False):
        grid_ids = sorted(dataset["grid_id"].astype(int).unique())
        action_context = list(joint_action_columns(grid_ids))
        expected_extra = action_context
        if current_state_columns:
            expected_extra = list(GLOBAL_STATE_COLUMNS) + expected_extra
        context_columns = manifest.get("additional_context_columns", [])
        context_complete = (context_columns == current_state_columns + expected_extra
                            and set(context_columns).issubset(dataset.columns)
                            and set(context_columns).issubset(feature_columns))
        matches = context_complete
        if matches:
            period_groups = dataset.groupby(["run_id", "decision_step"], sort=False)
            period_index = period_groups.size().index
            period_actions = period_groups[list(action_context)].first()
            matches = bool(period_groups["grid_id"].nunique().eq(len(grid_ids)).all())
            if matches:
                for grid_id in grid_ids:
                    grid_rows = dataset.loc[
                        dataset["grid_id"].astype(int).eq(grid_id),
                        ["run_id", "decision_step", "price_multiplier", "matching_radius_km"],
                    ].set_index(["run_id", "decision_step"]).reindex(period_index)
                    if (grid_rows.isna().any().any()
                            or not np.allclose(
                                period_actions[f"joint_price_grid_{grid_id}"].to_numpy(dtype=float),
                                grid_rows["price_multiplier"].to_numpy(dtype=float))
                            or not np.allclose(
                                period_actions[f"joint_radius_grid_{grid_id}"].to_numpy(dtype=float),
                                grid_rows["matching_radius_km"].to_numpy(dtype=float))):
                        matches = False
                        break
            if matches and current_state_columns:
                expected_state = period_groups[list(LOCAL_STATE_COLUMNS)].sum()
                actual_state = period_groups[list(GLOBAL_STATE_COLUMNS)].first()
                matches = (np.allclose(
                    actual_state.to_numpy(dtype=float), expected_state.to_numpy(dtype=float))
                    and period_groups[list(GLOBAL_STATE_COLUMNS)].nunique().eq(1).all().all())
        checks["joint_action_context_matches_each_grid"] = bool(matches)
    if current_state_columns:
        valid_state = manifest.get("decision_state_schema") == STATE_SCHEMA
        try:
            for _, run in dataset.groupby("run_id"):
                validate_decision_states(run, manifest["neighbor_adjacency"])
        except (ValueError, KeyError, TypeError):
            valid_state = False
        checks["pre_action_state_and_neighbor_sums_are_valid"] = valid_state
    if split_by == "date":
        checks["date_splits_are_isolated"] = dates_are_isolated
        ordered = [split_dates[name] for name in ("train", "validation", "test")
                   if name in split_dates]
        checks["date_splits_are_chronological"] = all(
            max(earlier) < min(later) for earlier, later in zip(ordered, ordered[1:]))
    checks = {key: bool(value) for key, value in checks.items()}
    warnings = []
    if dataset["date"].nunique() == 1:
        warnings.append("只有一个日期，不能据此评估跨日期泛化。")
    elif not dates_are_isolated:
        warnings.append("不同集合共享日期，当前划分只用于同日期不同 run 的评估。")

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "dataset_schema_version": manifest.get("schema_version"),
        "passed": all(checks.values()),
        "checks": checks,
        "warnings": warnings,
        "summary": {
            "row_count": int(len(dataset)),
            "run_count": int(dataset["run_id"].nunique()),
            "date_count": int(dataset["date"].nunique()),
            "split_by": split_by,
            "split_dates": split_dates,
            "split_run_counts": {name: int(frame["run_id"].nunique())
                                 for name, frame in dataset.groupby("split")},
            "grid_count": int(dataset["grid_id"].nunique()),
            "decision_count": int(dataset["decision_step"].nunique()),
            "joint_action_count": int(len(action_coverage)),
            "decision_period_count": int(len(actions_per_period)),
            "periods_with_repeated_grid_actions": int(actions_per_period.lt(decision_grid_counts).sum()),
            "split_counts": actual_split_counts,
            "simulation_time_range": [
                float(dataset["decision_start_sim_time"].min()),
                float(dataset["decision_end_sim_time_exclusive"].max()),
            ],
            "bounded_rate_range": _range_summary(
                dataset, bounded_rate_columns),
            "request_rate_range": _range_summary(
                dataset, request_rate_columns),
        },
        "joint_action_coverage": action_coverage.to_dict(orient="records"),
    }
    period_shapes = set(zip(actions_per_period.tolist(), decision_grid_counts.tolist()))
    return report, period_shapes


def _run_chunks(dataset_path, chunk_size):
    """CSV 按 run 连续保存；保留块尾 run，避免截断历史连续性检查。"""
    tail = None
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
        if "run_id" not in chunk:
            raise ValueError
        frame = pd.concat([tail, chunk], ignore_index=True) if tail is not None else chunk
        if frame.empty:
            continue
        last_run = frame["run_id"].iloc[-1]
        mask = frame["run_id"].eq(last_run)
        ready, tail = frame.loc[~mask], frame.loc[mask]
        if not ready.empty:
            yield ready
    if tail is not None and not tail.empty:
        yield tail


def validate_supervised_dataset(input_path="datasets", output_path=None, chunk_size=50000):
    """分块执行原有指标检查，并在全数据范围检查划分和动作覆盖。"""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError
    dataset_path, manifest_path, report_path = _paths(input_path, output_path)
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    checks = {}
    row_count = decision_period_count = repeated_action_periods = 0
    split_counts, run_counts, coverage = Counter(), Counter(), Counter()
    split_dates, run_splits, run_dates = defaultdict(set), defaultdict(set), defaultdict(set)
    seen_runs, grids, decisions, period_shapes = set(), set(), set(), set()
    ranges = {}
    time_range = [float("inf"), float("-inf")]
    contiguous = True
    for frame in _run_chunks(dataset_path, chunk_size):
        local_counts = {name: int(count) for name, count in frame["split"].value_counts().items()}
        local_dates = {name: sorted(part["date"].unique().tolist())
                       for name, part in frame.groupby("split")}
        local_manifest = {**manifest, "row_count": len(frame), "split_counts": local_counts,
                          "split_dates": local_dates}
        partial, shapes = _validate_frame(frame, local_manifest, dataset_path)
        for key, passed in partial["checks"].items():
            checks[key] = checks.get(key, True) and passed
        block_runs = set(frame["run_id"])
        contiguous = contiguous and not bool(seen_runs & block_runs)
        seen_runs.update(block_runs)
        row_count += len(frame)
        split_counts.update(local_counts)
        run_counts.update({name: int(count) for name, count in frame.groupby("run_id").size().items()})
        for row in frame[["run_id", "date", "split"]].drop_duplicates().itertuples(index=False):
            split_dates[row.split].add(row.date)
            run_splits[row.run_id].add(row.split)
            run_dates[row.run_id].add(row.date)
        for row in partial["joint_action_coverage"]:
            coverage[(row["price_multiplier"], row["matching_radius_km"])]+=row["sample_count"]
        grids.update(frame["grid_id"].unique())
        decisions.update(frame["decision_step"].unique())
        period_shapes.update(shapes)
        summary = partial["summary"]
        decision_period_count += summary["decision_period_count"]
        repeated_action_periods += summary["periods_with_repeated_grid_actions"]
        time_range[0] = min(time_range[0], summary["simulation_time_range"][0])
        time_range[1] = max(time_range[1], summary["simulation_time_range"][1])
        for key in ("bounded_rate_range", "request_rate_range"):
            value = summary[key]
            if value is not None:
                previous = ranges.get(key, value)
                ranges[key] = {"minimum": min(value["minimum"], previous["minimum"]),
                               "maximum": max(value["maximum"], previous["maximum"])}
    if not row_count:
        raise ValueError
    split_dates = {name: sorted(dates) for name, dates in split_dates.items()}
    dates = set(date for values in split_dates.values() for date in values)
    dates_are_isolated = sum(len(values) for values in split_dates.values()) == len(dates)
    split_by = manifest.get("split_by", "run")
    prices = {key[0] for key in coverage}
    radii = {key[1] for key in coverage}
    checks.update({
        "row_count_matches_manifest": row_count == manifest.get("row_count"),
        "split_counts_match_manifest": dict(split_counts) == manifest.get("split_counts"),
        "run_blocks_are_contiguous": contiguous,
        "run_splits_are_isolated": all(len(values) == 1 for values in run_splits.values()),
        "run_dates_are_consistent": all(len(values) == 1 for values in run_dates.values()),
    })
    if manifest.get("action_sampling") is None:
        checks.update({
            "joint_actions_cover_each_period": all(
                actions == min(grid_count, len(coverage)) for actions, grid_count in period_shapes),
            "joint_action_product_is_complete": len(coverage) == len(prices) * len(radii),
        })
    if "source_run_ids" in manifest:
        checks["source_runs_are_complete"] = seen_runs == set(manifest["source_run_ids"])
    if "source_run_sample_counts" in manifest:
        checks["source_run_sample_counts_match"] = dict(run_counts) == manifest["source_run_sample_counts"]
    if "source_run_splits" in manifest:
        checks["source_run_splits_are_preserved"] = (
            {name: sorted(values) for name, values in run_splits.items()}
            == {name: [value] for name, value in manifest["source_run_splits"].items()})
    if "split_dates" in manifest:
        checks["split_dates_match_manifest"] = split_dates == manifest["split_dates"]
    if split_by == "date":
        checks["date_splits_are_isolated"] = dates_are_isolated
        ordered = [split_dates[name] for name in ("train", "validation", "test") if name in split_dates]
        checks["date_splits_are_chronological"] = all(
            max(first) < min(second) for first, second in zip(ordered, ordered[1:]))
    warnings = []
    if len(dates) == 1:
        warnings.append("只有一个日期，不能据此评估跨日期泛化。")
    elif not dates_are_isolated:
        warnings.append("不同集合共享日期，当前划分只用于同日期不同 run 的评估。")
    sampling = manifest.get("action_sampling")
    if isinstance(sampling, dict) and sampling.get("schema_version") == ACTION_SAMPLING_VERSION:
        candidate_count = len(sampling.get("price_action_values", [])) * len(
            sampling.get("matching_radius_values_km", []))
        if len(coverage) < candidate_count:
            warnings.append(f"有放回采样已覆盖 {len(coverage)}/{candidate_count} 种候选组合；覆盖不足作为提示，不强制逐周期平衡。")
    report = {
        "schema_version": REPORT_SCHEMA_VERSION, "dataset_path": str(dataset_path),
        "dataset_schema_version": manifest.get("schema_version"),
        "passed": all(checks.values()), "checks": checks, "warnings": warnings,
        "summary": {
            "row_count": row_count, "run_count": len(seen_runs), "date_count": len(dates),
            "split_by": split_by, "split_dates": split_dates,
            "split_run_counts": {name: sum(name in values for values in run_splits.values())
                                 for name in split_dates},
            "grid_count": len(grids), "decision_count": len(decisions),
            "joint_action_count": len(coverage), "split_counts": dict(split_counts),
            "decision_period_count": decision_period_count,
            "periods_with_repeated_grid_actions": repeated_action_periods,
            "simulation_time_range": time_range, "bounded_rate_range": ranges.get("bounded_rate_range"),
            "request_rate_range": ranges.get("request_rate_range"),
        },
        "joint_action_coverage": [{"price_multiplier": price, "matching_radius_km": radius,
                                   "sample_count": count}
                                  for (price, radius), count in sorted(coverage.items())],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"数据集校验：{'通过' if report['passed'] else '未通过'}")
    print(f"  样本数：{row_count:,}")
    print(f"  run 数：{report['summary']['run_count']}")
    print(f"  联合动作数：{len(coverage)}")
    print(f"  报告：{report_path}")
    for warning in warnings:
        print(f"  提示：{warning}")
    return report


def main():
    parser = argparse.ArgumentParser(description="检查监督学习数据集质量")
    parser.add_argument("--input", default="datasets")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = validate_supervised_dataset(args.input, args.output)
    if not report["passed"]:
        raise RuntimeError


if __name__ == "__main__":
    main()
