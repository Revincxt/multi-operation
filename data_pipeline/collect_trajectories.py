"""监督学习联合动作原始数据采集脚本。

核心逻辑:
1. 每个决策周期为每个 grid 分配一个
   ``(price_multiplier, matching_radius_km)`` 联合动作
2. 动作在 ``decision_freq`` 个一分钟仿真步内保持不变
3. 保存 action plan、分钟级原始指标和 manifest

使用方式:
    PYTHONPATH=. python3 -m data_pipeline.collect_trajectories \
        --date 2015-05-18 --decision-freq 5 --output trajectories
"""

import argparse
import json
import math
from datetime import date as calendar_date, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.env.actions import (
    DEFAULT_MATCHING_RADII_KM,
    DEFAULT_PRICE_MULTIPLIERS,
    GridAction,
)
from src.env.simulator_env import Simulator
from src.env.simulator_pattern import SimulatorPattern, input_data_paths, load_grid_manifest
from src.env.observation import capture_decision_state, load_neighbor_adjacency, state_metadata


SCHEMA_VERSION = "supervised_joint_action_raw"
ACTION_SAMPLING_VERSION = "joint_action_mixture_v1"
REFERENCE_ACTIONS = ((1.6, 2.0), (1.5, 2.0), (1.6, 1.5),
                     (1.6, 3.0), (1.5, 3.0), (1.9, 2.5))


def validate_sample_ratio(value):
    """密度实验只对原始订单做无放回下采样。"""
    if isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError
    return float(value)


def collection_run_id(date, seed, action_seed, grid_num, decision_freq,
                      price_response_model, start_time, simulation_steps,
                      driver_num, order_sample_ratio):
    run_id = (
        f"{date}_simseed{seed}_actionseed{action_seed}_grid{grid_num}_"
        f"{decision_freq}min_{price_response_model}_"
        f"start{start_time.replace(':', '')}_steps{simulation_steps}")
    if order_sample_ratio != 1.0:
        run_id += f"_drivers{driver_num}_sample{order_sample_ratio}"
    return run_id


def simulation_window(start_time, simulation_steps):
    """解析日内仿真窗口；一步为一分钟，不跨日期。"""
    if (isinstance(simulation_steps, bool)
            or int(simulation_steps) != simulation_steps or simulation_steps <= 0):
        raise ValueError
    parsed = datetime.strptime(str(start_time), "%H:%M")
    start = parsed.hour * 3600 + parsed.minute * 60
    end = start + int(simulation_steps) * 60
    if end > 86400:
        raise ValueError
    return parsed.strftime("%H:%M"), start, end


def check_collection_inputs(dates, grid_num, data_dir="my_data"):
    """采集前检查所有日期的输入，避免完成部分仿真后才发现缺文件。"""
    grid_manifest = load_grid_manifest(data_dir)
    if grid_manifest is not None and grid_manifest['grid_num'] != grid_num:
        raise ValueError
    paths = []
    for date in dates:
        calendar_date.fromisoformat(date)
        paths.extend(input_data_paths(date, data_dir))
        if grid_num == 263 and grid_manifest is None:
            paths.append(Path(data_dir) / f"orders_grid35_{date}-map263.csv")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError


def _action_values(values):
    """校验动作候选值。"""
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or len(values) == 0
            or not np.all(np.isfinite(values)) or np.any(values <= 0)
            or len(np.unique(values)) != len(values)):
        raise ValueError
    return values


def joint_action_sampling(prices, radii):
    """记录采样支持集及 run 级混合比例；自定义集合缺少某模式时回退全域探索。"""
    pairs = [(float(price), float(radius)) for price in prices for radius in radii]
    preferred = [pair for pair in pairs if 1.4 <= pair[0] <= 1.8 and 1.5 <= pair[1] <= 2.5]
    references = [pair for pair in REFERENCE_ACTIONS if pair in pairs]
    probabilities = {"preferred": 0.6 if preferred else 0.0,
                     "uniform": 0.2, "reference": 0.2 if references else 0.0}
    probabilities["uniform"] = 1.0 - probabilities["preferred"] - probabilities["reference"]
    reference_probabilities = [1.0 / len(references) for _ in references]
    if len(references) > 1 and REFERENCE_ACTIONS[0] in references:
        reference_probabilities = [0.5 if pair == REFERENCE_ACTIONS[0]
                                   else 0.5 / (len(references) - 1) for pair in references]
    return {
        "schema_version": ACTION_SAMPLING_VERSION,
        "with_replacement": True, "mixture_unit": "run",
        "mode_probabilities": probabilities,
        "price_action_values": [float(value) for value in prices],
        "matching_radius_values_km": [float(value) for value in radii],
        "preferred_joint_actions": [list(pair) for pair in preferred],
        "preferred_hold_decisions": 3,
        "reference_joint_actions": [list(pair) for pair in references],
        "reference_probabilities": reference_probabilities,
        "reference_hold": "whole_run_same_action_for_all_grids",
    }


def build_joint_action_plan(
    rng,
    grid_num,
    simulation_steps,
    decision_freq=5,
    price_action_values=None,
    matching_radius_values=None,
):
    """生成允许跨网格重复的联合动作；混合优势分段、全域随机与持续参考轨迹。"""
    if (isinstance(grid_num, bool) or int(grid_num) != grid_num
            or grid_num <= 0):
        raise ValueError
    if (isinstance(simulation_steps, bool)
            or int(simulation_steps) != simulation_steps
            or simulation_steps <= 0):
        raise ValueError
    if (isinstance(decision_freq, bool)
            or int(decision_freq) != decision_freq
            or decision_freq <= 0):
        raise ValueError

    grid_num = int(grid_num)
    simulation_steps = int(simulation_steps)
    decision_freq = int(decision_freq)
    prices = _action_values(
        DEFAULT_PRICE_MULTIPLIERS if price_action_values is None
        else price_action_values)
    radii = _action_values(
        DEFAULT_MATCHING_RADII_KM if matching_radius_values is None
        else matching_radius_values)
    joint_actions = np.asarray(
        [(price, radius) for price in prices for radius in radii])

    sampling = joint_action_sampling(prices, radii)
    modes = list(sampling["mode_probabilities"])
    mode = str(rng.choice(modes, p=[sampling["mode_probabilities"][name] for name in modes]))
    indices = {tuple(pair): index for index, pair in enumerate(joint_actions)}
    preferred_ids = [indices[tuple(pair)] for pair in sampling["preferred_joint_actions"]]
    reference_pair = None
    if mode == "reference":
        reference_index = int(rng.choice(len(sampling["reference_joint_actions"]),
                                         p=sampling["reference_probabilities"]))
        reference_pair = sampling["reference_joint_actions"][reference_index]
        action_ids = np.full(grid_num, indices[tuple(reference_pair)], dtype=int)

    records = []
    for decision_step in range(math.ceil(simulation_steps / decision_freq)):
        if mode == "uniform":
            action_ids = rng.integers(len(joint_actions), size=grid_num)
        elif mode == "preferred" and decision_step % sampling["preferred_hold_decisions"] == 0:
            action_ids = rng.choice(preferred_ids, size=grid_num, replace=True)
        start_step = decision_step * decision_freq
        end_step = min(start_step + decision_freq, simulation_steps)

        for grid_id, action_id in enumerate(action_ids):
            price, radius = joint_actions[action_id]
            records.append({
                "decision_step": decision_step,
                "decision_start_minute_step": start_step,
                "decision_end_minute_step_exclusive": end_step,
                "grid_id": grid_id,
                "joint_action_id": int(action_id),
                "price_multiplier": float(price),
                "matching_radius_km": float(radius),
                "is_baseline_action": bool(
                    np.isclose(price, 1.0) and np.isclose(radius, 1.0)),
            })
    plan = pd.DataFrame(records)
    plan.attrs.update(action_sampling=sampling, sampling_mode=mode,
                      reference_joint_action=reference_pair)
    return plan


def action_mapping_for_decision(action_plan, decision_step, grid_num):
    """将一个决策周期转换为仿真器所需的完整网格动作。"""
    period = action_plan[
        action_plan["decision_step"] == decision_step
    ].sort_values("grid_id")
    if (len(period) != grid_num
            or not np.array_equal(
                period["grid_id"].to_numpy(dtype=int), np.arange(grid_num))):
        raise ValueError
    return {
        int(row.grid_id): GridAction(
            row.price_multiplier, row.matching_radius_km)
        for row in period.itertuples(index=False)
    }


def _build_minute_metrics(sim, date, seed, decision_freq, action_plan):
    """展开仿真器分钟级指标，并关联计划动作。"""
    executed_steps = int(sim.current_step)
    table = sim.evaluate_table[:executed_steps]
    minute_metrics = pd.DataFrame(
        table.reshape(-1, table.shape[-1]),
        columns=sim.evaluate_df.columns,
    )
    minute_metrics.insert(0, "date", date)
    minute_metrics.insert(1, "seed", int(seed))
    minute_metrics.insert(
        2, "minute_step",
        np.repeat(np.arange(executed_steps), sim.grid_num))
    minute_metrics.insert(
        3, "sim_time",
        np.repeat(
            sim.t_initial + np.arange(executed_steps) * sim.delta_t,
            sim.grid_num))
    minute_metrics["grid_id"] = minute_metrics["origin_grid_id"].astype(int)
    minute_metrics["decision_step"] = (
        minute_metrics["minute_step"] // decision_freq)

    planned = action_plan[[
        "decision_step", "grid_id", "joint_action_id",
        "price_multiplier", "matching_radius_km", "is_baseline_action",
    ]].rename(columns={
        "price_multiplier": "planned_price_multiplier",
        "matching_radius_km": "planned_matching_radius_km",
    })
    minute_metrics = minute_metrics.merge(
        planned,
        on=["decision_step", "grid_id"],
        how="left",
        validate="many_to_one")

    if (not np.allclose(
            minute_metrics["action_price_multiplier"],
            minute_metrics["planned_price_multiplier"])
            or not np.allclose(
                minute_metrics["matching_radius_km"],
                minute_metrics["planned_matching_radius_km"])):
        raise RuntimeError
    return minute_metrics


def _validate_output(action_plan, minute_metrics, grid_num,
                     simulation_steps, decision_freq):
    """校验行数、主键和联合动作覆盖。"""
    decision_count = math.ceil(simulation_steps / decision_freq)
    if (len(action_plan) != decision_count * grid_num
            or action_plan.duplicated(["decision_step", "grid_id"]).any()):
        raise RuntimeError
    if (len(minute_metrics) != simulation_steps * grid_num
            or minute_metrics.duplicated(["minute_step", "grid_id"]).any()):
        raise RuntimeError

    coverage = action_plan.groupby(
        ["price_multiplier", "matching_radius_km"]).size()
    distinct = action_plan.groupby("decision_step")["joint_action_id"].nunique()
    return {
        "decision_count": decision_count,
        "action_plan_rows": int(len(action_plan)),
        "minute_metric_rows": int(len(minute_metrics)),
        "observed_joint_action_count": int(len(coverage)),
        "minimum_samples_per_joint_action": int(coverage.min()),
        "maximum_samples_per_joint_action": int(coverage.max()),
        "periods_with_repeated_grid_actions": int(distinct.lt(grid_num).sum()),
    }


def _parse_action_list(raw_value):
    """解析逗号分隔的动作候选值。"""
    try:
        return [float(value) for value in raw_value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError from exc


def run_trajectory_collection(
    date,
    grid_num=263,
    decision_freq=5,
    driver_num=1000,
    seed=42,
    action_seed=None,
    simulation_steps=300,
    price_action_values=None,
    matching_radius_values=None,
    price_response_model="utility_choice",
    enable_reposition=False,
    output_dir="trajectories",
    overwrite=False,
    start_time="05:00",
    data_dir="my_data",
    order_sample_ratio=1.0,
    compress=False,
    _pattern=None,
):
    """运行一次联合动作仿真并保存原始数据。"""
    started = perf_counter()
    order_sample_ratio = validate_sample_ratio(order_sample_ratio)
    start_time, start_seconds, end_seconds = simulation_window(
        start_time, simulation_steps)
    simulation_steps = int(simulation_steps)
    check_collection_inputs([date], grid_num, data_dir)
    if action_seed is None:
        action_seed = seed

    prices = _action_values(
        DEFAULT_PRICE_MULTIPLIERS if price_action_values is None
        else price_action_values)
    radii = _action_values(
        DEFAULT_MATCHING_RADII_KM if matching_radius_values is None
        else matching_radius_values)
    run_id = collection_run_id(
        date, seed, action_seed, grid_num, decision_freq, price_response_model,
        start_time, simulation_steps, driver_num, order_sample_ratio)
    run_dir = Path(output_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError

    # === 加载数据 ===
    print(f"正在加载 {date} 的仿真数据...")
    pattern = SimulatorPattern(date, data_dir=data_dir) if _pattern is None else _pattern
    adjacency = load_neighbor_adjacency(data_dir, grid_num)
    grid_manifest = load_grid_manifest(data_dir)
    mapping = (SimulatorPattern.load_mapping(date, data_dir=data_dir)
               if grid_num == 263 and grid_manifest is None else None)
    if (not pattern.request_all
            or start_seconds < min(pattern.request_all)
            or end_seconds > max(pattern.request_all) + 1):
        raise ValueError

    # === 构建联合动作计划 ===
    print("正在生成允许重复的全网格混合联合动作计划...")
    action_plan = build_joint_action_plan(
        np.random.default_rng(action_seed),
        grid_num,
        simulation_steps,
        decision_freq,
        prices,
        radii)
    sampling_metadata = dict(action_plan.attrs)
    action_maps = {
        step: action_mapping_for_decision(action_plan, step, grid_num)
        for step in action_plan["decision_step"].unique()
    }

    def grid_action_fn(sim_time, _grid_state):
        minute_step = int((sim_time - start_seconds) // 60)
        return action_maps[minute_step // decision_freq]

    # === 创建仿真器 ===
    sim = Simulator(
        grid_num=grid_num,
        decision_freq=decision_freq,
        price_response_model=price_response_model,
        grid_action=grid_action_fn,
        allowed_price_multipliers=prices,
        allowed_matching_radii_km=radii,
        order_sample_ratio=order_sample_ratio,
        driver_num=driver_num,
        base_fare_start=3.0,
        price_per_km=0.7,
        commission_rate=0.25,
        enable_reposition=enable_reposition,
        reposition_adjacency_path=str(Path(data_dir) / (
            "grid_adjacency.csv" if grid_manifest is not None else "hexo_updated_r300_adj.csv")),
        reposition_grid_info_path=str(Path(data_dir) / (
            "grid_info.csv" if grid_manifest is not None else "hex_updated_r300_info.csv")))
    # reset 会按窗口分配记录表和初始化司机，必须先设置起止时间。
    sim.t_initial = start_seconds
    sim.t_end = end_seconds
    sim.finish_run_step = int(simulation_steps)
    sim.reset(
        seed=seed,
        given_data=True,
        request_databases=pattern.request_all,
        driver_info=pattern.driver_info)
    sim.mapping_dict = {date: mapping} if mapping is not None else None
    sim.experiment_date = date

    # === 运行仿真 ===
    print(
        f"开始运行 {simulation_steps} 个一分钟仿真步，"
        f"共 {math.ceil(simulation_steps / decision_freq)} 个决策周期...")
    decision_states = []
    while not sim.end_of_episode:
        if sim.current_step % decision_freq == 0:
            state = capture_decision_state(sim, adjacency)
            state.insert(0, "decision_step", sim.current_step // decision_freq)
            decision_states.append(state)
        sim.step()

    action_plan = action_plan.merge(
        pd.concat(decision_states, ignore_index=True), on=["decision_step", "grid_id"],
        validate="one_to_one")

    # === 整理与校验 ===
    minute_metrics = _build_minute_metrics(
        sim, date, seed, decision_freq, action_plan)
    quality = _validate_output(
        action_plan, minute_metrics, grid_num,
        simulation_steps, decision_freq)

    action_plan.insert(0, "date", date)
    action_plan.insert(1, "seed", int(seed))
    action_plan.insert(2, "action_seed", int(action_seed))
    minute_metrics.insert(0, "run_id", run_id)

    # === 保存 ===
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compress else ".csv"
    action_plan_path = run_dir / f"action_plan{suffix}"
    minute_metrics_path = run_dir / f"grid_minute_metrics{suffix}"
    action_plan.to_csv(action_plan_path, index=False)
    minute_metrics.to_csv(minute_metrics_path, index=False)
    if enable_reposition:
        pd.DataFrame(sim.reposition_events).to_csv(
            run_dir / "reposition_events.csv", index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "date": date,
        "data_dir": str(Path(data_dir).resolve()),
        "simulator_seed": int(seed),
        "action_seed": int(action_seed),
        "grid_num": int(grid_num),
        "driver_num": int(driver_num),
        "order_sample_ratio": order_sample_ratio,
        "order_sampling": "per-minute without replacement, rounded count",
        "simulation_steps": int(simulation_steps),
        "start_time": start_time,
        "start_sim_time": start_seconds,
        "end_sim_time_exclusive": end_seconds,
        "initial_state": "independent reset with supplied driver locations",
        "decision_freq_minutes": int(decision_freq),
        "price_response_model": price_response_model,
        "reposition_enabled": bool(enable_reposition),
        "price_action_values": prices.tolist(),
        "matching_radius_values_km": radii.tolist(),
        "joint_action_count": int(len(prices) * len(radii)),
        **sampling_metadata,
        "quality": quality,
        **state_metadata(adjacency),
        "files": {"action_plan": action_plan_path.name,
                  "minute_metrics": minute_metrics_path.name},
        "elapsed_seconds": perf_counter() - started,
        "summary": {
            "baseline_requests": int(sim.baseline_request_num),
            "submitted_requests": int(sim.accepted_quote_num),
            "total_gbv": float(sim.total_reward),
            "matched_requests": int(sim.matched_requests_num),
            "occupancy_rate": float(sim.occupancy_rate),
        },
    }
    if grid_manifest is not None:
        manifest['grid_geometry_sha256'] = grid_manifest['grid_geometry_sha256']
        manifest['grid_shape'] = grid_manifest['grid_shape']
        manifest['grid_radius_km'] = grid_manifest['radius_km']
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print(f"\n原始数据已保存至：{run_dir}")
    print(
        f"  action plan：{len(action_plan):,} 行\n"
        f"  minute metrics：{len(minute_metrics):,} 行\n"
        f"  联合动作：{quality['observed_joint_action_count']} 种")

    return {
        "run_dir": str(run_dir),
        "manifest": manifest,
        "action_plan": action_plan,
        "minute_metrics": minute_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="采集全网格联合动作的监督学习原始数据")
    parser.add_argument("--date", default="2015-05-18")
    parser.add_argument("--grid-num", type=int, default=263)
    parser.add_argument("--decision-freq", type=int, default=5)
    parser.add_argument("--simulation-steps", type=int, default=300)
    parser.add_argument("--start-time", default="05:00")
    parser.add_argument("--data-dir", default="my_data")
    parser.add_argument("--driver-num", type=int, default=1000)
    parser.add_argument("--order-sample-ratio", type=float, default=1.0)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-seed", type=int)
    parser.add_argument(
        "--price-actions",
        default=",".join(str(value) for value in DEFAULT_PRICE_MULTIPLIERS))
    parser.add_argument(
        "--matching-radii-km",
        default=",".join(str(value) for value in DEFAULT_MATCHING_RADII_KM))
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

    run_trajectory_collection(
        date=args.date,
        grid_num=args.grid_num,
        decision_freq=args.decision_freq,
        driver_num=args.driver_num,
        order_sample_ratio=args.order_sample_ratio,
        compress=args.compress,
        seed=args.seed,
        action_seed=args.action_seed,
        simulation_steps=args.simulation_steps,
        start_time=args.start_time,
        data_dir=args.data_dir,
        price_action_values=_parse_action_list(args.price_actions),
        matching_radius_values=_parse_action_list(args.matching_radii_km),
        price_response_model=args.price_response_model,
        enable_reposition=args.enable_reposition,
        output_dir=args.output,
        overwrite=args.overwrite)


if __name__ == "__main__":
    main()
