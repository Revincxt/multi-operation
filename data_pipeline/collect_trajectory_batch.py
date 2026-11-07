"""批量采集监督学习联合动作原始轨迹。

核心逻辑:
1. 遍历多个日期、日内时段和仿真随机种子
2. 为每个 run 生成独立的联合动作随机种子
3. 调用单次轨迹采集器并保存批次清单

使用方式:
    PYTHONPATH=. python3 -m data_pipeline.collect_trajectory_batch \
        --dates 2015-05-18 --seeds 42,43,44 --output trajectories
"""

import argparse
import json
from pathlib import Path

from data_pipeline.collect_trajectories import (
    check_collection_inputs,
    run_trajectory_collection,
    simulation_window,
    validate_sample_ratio,
)
from src.env.actions import (
    DEFAULT_MATCHING_RADII_KM,
    DEFAULT_PRICE_MULTIPLIERS,
)


BATCH_SCHEMA_VERSION = "supervised_joint_action_batch_v1"


def _parse_list(raw_value, value_type):
    """解析逗号分隔的参数。"""
    try:
        return [value_type(value.strip()) for value in raw_value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError from exc


def run_batch_collection(
    dates,
    seeds,
    grid_num=263,
    decision_freq=5,
    driver_num=1000,
    simulation_steps=300,
    price_action_values=None,
    matching_radius_values=None,
    price_response_model="utility_choice",
    enable_reposition=False,
    action_seed_offset=0,
    output_dir="trajectories",
    overwrite=False,
    start_times=("05:00",),
    data_dir="my_data",
    order_sample_ratio=1.0,
):
    """按日期、时段和随机种子的笛卡尔积批量采集轨迹。"""
    dates = [str(date).strip() for date in dates]
    seeds = list(seeds)
    order_sample_ratio = validate_sample_ratio(order_sample_ratio)
    if (not dates or any(not date for date in dates)
            or len(dates) != len(set(dates))
            or not seeds or len(seeds) != len(set(seeds))):
        raise ValueError
    if (isinstance(action_seed_offset, bool)
            or int(action_seed_offset) != action_seed_offset):
        raise ValueError
    if any(isinstance(seed, bool) or int(seed) != seed for seed in seeds):
        raise ValueError

    seeds = [int(seed) for seed in seeds]
    action_seed_offset = int(action_seed_offset)
    start_times = [simulation_window(time, simulation_steps)[0] for time in start_times]
    if not start_times or len(start_times) != len(set(start_times)):
        raise ValueError
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError
    check_collection_inputs(dates, grid_num, data_dir)
    scenarios = [(date, time) for date in dates for time in start_times]
    action_seeds = [seed + action_seed_offset + index * 100000
                    for index in range(len(scenarios)) for seed in seeds]
    if min(action_seeds) < 0 or len(action_seeds) != len(set(action_seeds)):
        raise ValueError
    output_dir = Path(output_dir)
    manifest_path = output_dir / "batch_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError
    output_dir.mkdir(parents=True, exist_ok=True)

    run_records = []
    run_count = len(scenarios) * len(seeds)
    current_run = 0
    for scenario_index, (date, start_time) in enumerate(scenarios):
        for seed in seeds:
            current_run += 1
            action_seed = seed + action_seed_offset + scenario_index * 100000
            print(
                f"\n=== Batch run {current_run}/{run_count}: "
                f"date={date}, start={start_time}, "
                f"seed={seed}, action_seed={action_seed} ===")
            result = run_trajectory_collection(
                date=date,
                grid_num=grid_num,
                decision_freq=decision_freq,
                driver_num=driver_num,
                order_sample_ratio=order_sample_ratio,
                seed=seed,
                action_seed=action_seed,
                simulation_steps=simulation_steps,
                start_time=start_time,
                data_dir=data_dir,
                price_action_values=price_action_values,
                matching_radius_values=matching_radius_values,
                price_response_model=price_response_model,
                enable_reposition=enable_reposition,
                output_dir=output_dir,
                overwrite=overwrite)
            run_manifest = result["manifest"]
            run_records.append({
                "run_id": run_manifest["run_id"],
                "run_dir": result["run_dir"],
                "date": date,
                "start_time": start_time,
                "simulator_seed": seed,
                "action_seed": action_seed,
                "quality": run_manifest["quality"],
                "summary": run_manifest["summary"],
            })

    batch_manifest = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "dates": dates,
        "start_times": start_times,
        "data_dir": str(Path(data_dir).resolve()),
        "simulator_seeds": seeds,
        "action_seed_offset": action_seed_offset,
        "run_count": run_count,
        "grid_num": int(grid_num),
        "driver_num": int(driver_num),
        "order_sample_ratio": order_sample_ratio,
        "simulation_steps": int(simulation_steps),
        "decision_freq_minutes": int(decision_freq),
        "price_response_model": price_response_model,
        "reposition_enabled": bool(enable_reposition),
        "price_action_values": list(
            DEFAULT_PRICE_MULTIPLIERS
            if price_action_values is None else price_action_values),
        "matching_radius_values_km": list(
            DEFAULT_MATCHING_RADII_KM
            if matching_radius_values is None else matching_radius_values),
        "runs": run_records,
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(batch_manifest, file, indent=2, ensure_ascii=False)

    print(f"\n批量采集完成：{run_count} 个 run")
    print(f"批次清单：{manifest_path}")
    return batch_manifest


def main():
    parser = argparse.ArgumentParser(
        description="批量采集多日期、多随机种子的联合动作轨迹")
    parser.add_argument(
        "--dates", type=lambda value: _parse_list(value, str),
        default=["2015-05-18"])
    parser.add_argument(
        "--seeds", type=lambda value: _parse_list(value, int),
        default=[42, 43, 44])
    parser.add_argument("--grid-num", type=int, default=263)
    parser.add_argument("--decision-freq", type=int, default=5)
    parser.add_argument("--simulation-steps", type=int, default=300)
    parser.add_argument(
        "--start-times", type=lambda value: _parse_list(value, str), default=["05:00"])
    parser.add_argument("--data-dir", default="my_data")
    parser.add_argument("--driver-num", type=int, default=1000)
    parser.add_argument("--order-sample-ratio", type=float, default=1.0)
    parser.add_argument("--action-seed-offset", type=int, default=0)
    parser.add_argument(
        "--price-actions", type=lambda value: _parse_list(value, float),
        default=list(DEFAULT_PRICE_MULTIPLIERS))
    parser.add_argument(
        "--matching-radii-km", type=lambda value: _parse_list(value, float),
        default=list(DEFAULT_MATCHING_RADII_KM))
    parser.add_argument(
        "--price-response-model",
        choices=[
            "utility_choice",
            "aggregate_elasticity",
            "bounded_rational_agent",
        ],
        default="utility_choice")
    parser.add_argument("--enable-reposition", action="store_true")
    parser.add_argument("--output", default="trajectories")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_batch_collection(
        dates=args.dates,
        seeds=args.seeds,
        grid_num=args.grid_num,
        decision_freq=args.decision_freq,
        driver_num=args.driver_num,
        order_sample_ratio=args.order_sample_ratio,
        simulation_steps=args.simulation_steps,
        start_times=args.start_times,
        data_dir=args.data_dir,
        price_action_values=args.price_actions,
        matching_radius_values=args.matching_radii_km,
        price_response_model=args.price_response_model,
        enable_reposition=args.enable_reposition,
        action_seed_offset=args.action_seed_offset,
        output_dir=args.output,
        overwrite=args.overwrite)


if __name__ == "__main__":
    main()
