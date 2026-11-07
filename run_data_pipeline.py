"""依次执行联合动作数据管线的三个阶段。

执行顺序:
1. 批量采集原始仿真轨迹
2. 构建监督学习数据集
3. 检查数据集质量

使用方式:
    PYTHONPATH=. python3 run_data_pipeline.py
"""

import argparse

from data_pipeline.build_supervised_dataset import build_supervised_dataset, resolve_split_by
from data_pipeline.collect_trajectory_batch import run_batch_collection
from data_pipeline.validate_supervised_dataset import validate_supervised_dataset


def run_data_pipeline(
    dates=("2015-05-18",),
    seeds=tuple(range(42, 52)),
    grid_num=6,
    driver_num=1000,
    simulation_steps=300,
    decision_freq=5,
    history_steps=4,
    price_action_values=None,
    matching_radius_values=None,
    price_response_model="utility_choice",
    enable_reposition=False,
    action_seed_offset=0,
    validation_ratio=0.15,
    test_ratio=0.15,
    split_seed=42,
    raw_output_dir="trajectories_grid6",
    dataset_output_dir="datasets_grid6",
    overwrite=False,
    start_times=("05:00",),
    data_dir="my_data/grid6",
    split_by="auto",
    order_sample_ratio=1.0,
    joint_action_context=False,
):
    """运行完整的监督学习数据管线。"""
    dates = list(dates)
    # 日期不足时先报错，不先运行耗时仿真；不会把同一天伪装成独立测试日期。
    split_by = resolve_split_by(dates, split_by, validation_ratio, test_ratio)
    if split_by == "run" and len(set(dates)) == 1:
        print("提示：只有一个日期，将按 run 划分；结果不能用于评估跨日期泛化。")
    print("\n=== 1/3 批量采集原始轨迹 ===")
    batch_manifest = run_batch_collection(
        dates=dates,
        seeds=seeds,
        grid_num=grid_num,
        driver_num=driver_num,
        order_sample_ratio=order_sample_ratio,
        simulation_steps=simulation_steps,
        start_times=start_times,
        data_dir=data_dir,
        decision_freq=decision_freq,
        price_action_values=price_action_values,
        matching_radius_values=matching_radius_values,
        price_response_model=price_response_model,
        enable_reposition=enable_reposition,
        action_seed_offset=action_seed_offset,
        output_dir=raw_output_dir,
        overwrite=overwrite)

    print("\n=== 2/3 构建监督学习数据集 ===")
    dataset, dataset_manifest = build_supervised_dataset(
        input_dir=raw_output_dir,
        output_dir=dataset_output_dir,
        history_steps=history_steps,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        split_by=split_by,
        overwrite=overwrite,
        joint_action_context=joint_action_context)

    print("\n=== 3/3 检查数据集质量 ===")
    validation_report = validate_supervised_dataset(dataset_output_dir)
    if not validation_report["passed"]:
        raise RuntimeError

    print("\n数据管线执行完成")
    return {
        "batch_manifest": batch_manifest,
        "dataset_manifest": dataset_manifest,
        "validation_report": validation_report,
        "dataset": dataset,
    }


def _parse_dates(raw_value):
    """解析日期列表。"""
    return [value.strip() for value in raw_value.split(",")]


def _parse_seeds(raw_value):
    """解析随机种子列表。"""
    try:
        return [int(value) for value in raw_value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError from exc


def _parse_float_values(raw_value):
    """解析浮点数候选列表。"""
    try:
        return [float(value) for value in raw_value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError from exc


def main():
    parser = argparse.ArgumentParser(
        description="执行轨迹采集、监督数据构建和质量检查")
    parser.add_argument(
        "--dates", type=_parse_dates, default=["2015-05-18"])
    parser.add_argument(
        "--seeds", type=_parse_seeds,
        default=list(range(42, 52)))
    parser.add_argument("--grid-num", type=int, default=6)
    parser.add_argument("--driver-num", type=int, default=1000)
    parser.add_argument("--order-sample-ratio", type=float, default=1.0)
    parser.add_argument("--simulation-steps", type=int, default=300)
    parser.add_argument("--start-times", type=_parse_dates, default=["05:00"])
    parser.add_argument("--data-dir", default="my_data/grid6")
    parser.add_argument("--decision-freq", type=int, default=5)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--price-actions", type=_parse_float_values)
    parser.add_argument("--matching-radii-km", type=_parse_float_values)
    parser.add_argument(
        "--price-response-model",
        choices=[
            "utility_choice",
            "aggregate_elasticity",
            "bounded_rational_agent",
        ],
        default="utility_choice")
    parser.add_argument("--enable-reposition", action="store_true")
    parser.add_argument("--action-seed-offset", type=int, default=0)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-by", choices=["auto", "date", "run"], default="auto")
    parser.add_argument("--joint-action-context", action="store_true",
                        help="为 BCS 数据集加入全城状态与全网格动作向量")
    parser.add_argument("--raw-output", default="trajectories_grid6")
    parser.add_argument("--dataset-output", default="datasets_grid6")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_data_pipeline(
        dates=args.dates,
        seeds=args.seeds,
        grid_num=args.grid_num,
        driver_num=args.driver_num,
        order_sample_ratio=args.order_sample_ratio,
        simulation_steps=args.simulation_steps,
        start_times=args.start_times,
        data_dir=args.data_dir,
        decision_freq=args.decision_freq,
        history_steps=args.history_steps,
        price_action_values=args.price_actions,
        matching_radius_values=args.matching_radii_km,
        price_response_model=args.price_response_model,
        enable_reposition=args.enable_reposition,
        action_seed_offset=args.action_seed_offset,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        split_by=args.split_by,
        raw_output_dir=args.raw_output,
        dataset_output_dir=args.dataset_output,
        overwrite=args.overwrite,
        joint_action_context=args.joint_action_context)


if __name__ == "__main__":
    main()
