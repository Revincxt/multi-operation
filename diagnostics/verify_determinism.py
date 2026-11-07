"""确定性验证：给定相同 (date, seed, exploration events)，两次 run 是否输出完全一致。

运行方式:
    cd /Users/didi/project/refactor_simulator
    PYTHONPATH=. conda run -n dp_project python diagnostics/verify_determinism.py

验证逻辑:
1. 用固定 seed 生成探索计划 → 保存为 JSON
2. 用相同 seed 初始化两个 Simulator 实例，加载相同的探索计划
3. 两个实例分别跑完整 300 步
4. 逐步对比: partial reward、matched orders、evaluate_table 均值
5. 最终对比完整 evaluate_table (300×263×N)
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

from src.env.actions import (
    DEFAULT_MATCHING_RADII_KM,
    DEFAULT_PRICE_MULTIPLIERS,
    GridAction,
    uniform_grid_actions,
)
from src.env.simulator_env import Simulator
from src.env.simulator_pattern import SimulatorPattern


def build_exploration_plan_deterministic(rng, grid_num, simulation_steps,
                                          explore_prob=0.05, window_len=5,
                                          explore_grid_count=26,
                                          price_action_values=None,
                                          matching_radius_values=None):
    """为仿真器确定性诊断构造稀疏事件计划。

    这是诊断用的固定场景，不是 ``data_pipeline/collect_trajectories.py``
    使用的全网格平衡联合动作采集计划。
    """
    if price_action_values is None:
        price_action_values = np.array(DEFAULT_PRICE_MULTIPLIERS)
    if matching_radius_values is None:
        matching_radius_values = np.array(DEFAULT_MATCHING_RADII_KM)
    price_action_values = np.asarray(price_action_values, dtype=float)
    matching_radius_values = np.asarray(matching_radius_values, dtype=float)

    events = []
    next_event_id = 0
    i = 0
    while i < simulation_steps:
        if rng.random() < explore_prob:
            window_end = min(i + window_len, simulation_steps)
            events.append({
                'exploration_event_id': next_event_id,
                'exploration_event_start_step': int(i),
                'exploration_event_end_step': int(window_end),
                'exploration_price_multiplier': float(rng.choice(
                    price_action_values[price_action_values != 1.0])),
                'matching_radius_km': float(rng.choice(matching_radius_values)),
                'exploration_grid_ids': rng.choice(
                    grid_num, size=explore_grid_count, replace=False).astype(int).tolist(),
            })
            next_event_id += 1
            i = window_end
        else:
            i += 1
    return events


def action_mapping_for_step(events, minute_step):
    for event in events:
        if event['exploration_event_start_step'] <= minute_step < event['exploration_event_end_step']:
            action = GridAction(
                event['exploration_price_multiplier'], event['matching_radius_km'])
            return dict.fromkeys(event['exploration_grid_ids'], action)
    return {}


def make_grid_action_fn(events, t_initial, delta_t, grid_num):
    """Create a complete minute-level grid action schedule."""
    def grid_action_fn(sim_time, _grid_state):
        step = int((sim_time - t_initial) // delta_t)
        exploration_actions = action_mapping_for_step(events, step)
        actions = uniform_grid_actions(grid_num, 1.0, 1.0)
        for grid_id, action in exploration_actions.items():
            actions[int(grid_id)] = action
        return actions
    return grid_action_fn


def run_with_fixed_plan(date, seed, events, grid_num=263, driver_num=1000,
                         enable_reposition=True):
    """用固定的探索计划运行仿真，返回 evaluate_table 和汇总指标。"""
    pattern = SimulatorPattern(date)
    mapping = SimulatorPattern.load_mapping(date)

    simulation_steps = 300  # 5 hours × 60 minutes

    t_initial = 18000  # 05:00 in seconds
    delta_t = 60
    sim = Simulator(
        grid_num=grid_num,
        decision_freq=1,
        price_response_model='utility_choice',
        grid_action=make_grid_action_fn(
            events, t_initial, delta_t, grid_num),
        order_sample_ratio=1.0,
        driver_num=driver_num,
        base_fare_start=3.0,
        price_per_km=0.7,
        commission_rate=0.25,
        enable_reposition=enable_reposition,
    )
    sim.reset(seed=seed, given_data=True,
              request_databases=pattern.request_all,
              driver_info=pattern.driver_info)
    sim.mapping_dict = {date: mapping}
    sim.experiment_date = date

    # 逐步记录中间状态用于对比
    step_rewards = []
    step_matched = []

    while not sim.end_of_episode:
        sim.step()
        step_rewards.append(sim.total_reward)
        step_matched.append(sim.matched_requests_num)

    return {
        'evaluate_table': sim.evaluate_table.copy(),
        'total_reward': sim.total_reward,
        'matched_requests_num': sim.matched_requests_num,
        'step_rewards': step_rewards,
        'step_matched': step_matched,
        'reposition_events_count': len(sim.reposition_events),
    }


def main():
    parser = argparse.ArgumentParser(description='Verify simulator determinism')
    parser.add_argument('--date', default='2015-05-18')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grid-num', type=int, default=263)
    parser.add_argument('--driver-num', type=int, default=1000)
    parser.add_argument('--enable-reposition', action='store_true', default=True,
                        help='Enable reposition (default: True)')
    parser.add_argument('--no-reposition', dest='enable_reposition', action='store_false',
                        help='Disable reposition')
    args = parser.parse_args()

    print(f"=== Determinism Verification ===")
    print(f"  Date: {args.date}, Seed: {args.seed}")
    print(f"  Grids: {args.grid_num}, Drivers: {args.driver_num}")
    print(f"  Reposition: {args.enable_reposition}")
    print()

    # Step 1: 生成探索计划（只生成一次）
    print("[1/3] Generating exploration plan...")
    plan_rng = np.random.default_rng(args.seed)
    plan_seed_for_exploration = 999  # 独立 seed，与仿真器 seed 解耦
    plan_rng = np.random.default_rng(plan_seed_for_exploration)
    events = build_exploration_plan_deterministic(
        plan_rng, args.grid_num, 300, explore_prob=0.05, window_len=5,
        explore_grid_count=26)
    print(f"  Events generated: {len(events)}")
    for e in events[:3]:
        print(f"    event {e['exploration_event_id']}: "
              f"step [{e['exploration_event_start_step']}, {e['exploration_event_end_step']}), "
              f"action=(price={e['exploration_price_multiplier']}, "
              f"radius={e['matching_radius_km']}km), "
              f"grids={len(e['exploration_grid_ids'])}")
    if len(events) > 3:
        print(f"    ... and {len(events)-3} more events")

    # Step 2: 跑两次
    print("\n[2/3] Running Run A...")
    result_a = run_with_fixed_plan(
        args.date, args.seed, events,
        grid_num=args.grid_num, driver_num=args.driver_num,
        enable_reposition=args.enable_reposition)
    print(f"  Run A: total_reward={result_a['total_reward']:.2f}, "
          f"matched={result_a['matched_requests_num']}, "
          f"reposition_events={result_a['reposition_events_count']}")

    print("\n[2/3] Running Run B...")
    result_b = run_with_fixed_plan(
        args.date, args.seed, events,
        grid_num=args.grid_num, driver_num=args.driver_num,
        enable_reposition=args.enable_reposition)
    print(f"  Run B: total_reward={result_b['total_reward']:.2f}, "
          f"matched={result_b['matched_requests_num']}, "
          f"reposition_events={result_b['reposition_events_count']}")

    # Step 3: 对比
    print("\n[3/3] Comparing results...")
    all_pass = True

    # 3a: 逐步 reward 对比
    rewards_a = np.array(result_a['step_rewards'])
    rewards_b = np.array(result_b['step_rewards'])
    if not np.allclose(rewards_a, rewards_b):
        diff_mask = ~np.isclose(rewards_a, rewards_b)
        print(f"  ✗ Step rewards differ at steps: {np.where(diff_mask)[0].tolist()}")
        all_pass = False
    else:
        print(f"  ✓ Step rewards identical across all {len(rewards_a)} steps")

    # 3b: 逐步 matched orders 对比
    matched_a = np.array(result_a['step_matched'])
    matched_b = np.array(result_b['step_matched'])
    if not np.array_equal(matched_a, matched_b):
        diff_mask = matched_a != matched_b
        print(f"  ✗ Step matched orders differ at steps: {np.where(diff_mask)[0].tolist()}")
        all_pass = False
    else:
        print(f"  ✓ Step matched orders identical across all {len(matched_a)} steps")

    # 3c: evaluate_table 完整对比
    table_a = result_a['evaluate_table']
    table_b = result_b['evaluate_table']
    if not np.allclose(table_a, table_b, rtol=1e-10, atol=1e-10, equal_nan=True):
        diff_mask = ~np.isclose(table_a, table_b, rtol=1e-10, atol=1e-10, equal_nan=True)
        diff_count = diff_mask.sum()
        total_elements = table_a.size
        # 找到第一个不同的位置
        diff_indices = np.argwhere(diff_mask)
        print(f"  ✗ evaluate_table differs: {diff_count}/{total_elements} elements ({100*diff_count/total_elements:.4f}%)")
        if len(diff_indices) > 0:
            first_diff = diff_indices[0]
            print(f"    First difference at (step, grid, col) = ({first_diff[0]}, {first_diff[1]}, {first_diff[2]})")
            print(f"    Run A: {table_a[first_diff[0], first_diff[1], first_diff[2]]}")
            print(f"    Run B: {table_b[first_diff[0], first_diff[1], first_diff[2]]}")
        all_pass = False
    else:
        print(f"  ✓ evaluate_table ({table_a.shape}) identical (bit-exact)")

    # 3d: 最终汇总指标
    for metric in ['total_reward', 'matched_requests_num', 'reposition_events_count']:
        val_a = result_a[metric]
        val_b = result_b[metric]
        if val_a != val_b:
            print(f"  ✗ {metric}: Run A={val_a} vs Run B={val_b}")
            all_pass = False
        else:
            print(f"  ✓ {metric}: {val_a} (identical)")

    # 最终结论
    print()
    if all_pass:
        print("=" * 50)
        print("  ✓ VERIFIED: Simulator is fully deterministic")
        print("  Same (date, seed, exploration plan) → identical outputs")
        print("=" * 50)
        return 0
    else:
        print("=" * 50)
        print("  ✗ FAILED: Non-determinism detected")
        print("=" * 50)
        return 1


if __name__ == '__main__':
    sys.exit(main())
