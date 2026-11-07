"""Run parallel fixed-price and local-event price-response diagnostics.

Example:
    PYTHONPATH=. conda run -n dp_project python diagnose_price_response.py
"""

import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.env.actions import DEFAULT_PRICE_MULTIPLIERS, GridAction, uniform_grid_actions
from src.env.simulator_env import Simulator
from src.env.simulator_pattern import SimulatorPattern


ACTIONS = DEFAULT_PRICE_MULTIPLIERS
SEEDS = (42, 43)
LOCAL_START_STEP = 120
LOCAL_WINDOW_LEN = 20
LOCAL_GRID_COUNT = 26


def _metadata_for_grid(grid_id, step, scenario):
    if scenario['kind'] == 'city_fixed':
        return {
            'price_multiplier': scenario['action'],
            'is_treated': True,
            'event_phase': 'city_fixed',
        }
    treated = grid_id in scenario['treated_grid_ids']
    start = scenario['event_start_step']
    end = scenario['event_end_step']
    active = start <= step < end
    if step < start:
        phase = 'pre'
    elif active:
        phase = 'intervention'
    elif step < end + scenario['window_len']:
        phase = 'recovery_1'
    elif step < end + 2 * scenario['window_len']:
        phase = 'recovery_2'
    else:
        phase = 'other'
    return {
        'price_multiplier': scenario['action'] if active and treated else 1.0,
        'is_treated': treated,
        'event_phase': phase,
    }


def _build_scenario(kind, action, seed, grid_num):
    if kind == 'city_fixed':
        return {'kind': kind, 'action': action, 'seed': seed}
    # Separate deterministic RNG keeps the treated subset fixed across local actions.
    sampled_grids = np.random.default_rng(seed).choice(
        grid_num, size=LOCAL_GRID_COUNT, replace=False).astype(int).tolist()
    return {
        'kind': kind,
        'action': action,
        'seed': seed,
        'treated_grid_ids': sampled_grids,
        'event_start_step': LOCAL_START_STEP,
        'event_end_step': LOCAL_START_STEP + LOCAL_WINDOW_LEN,
        'window_len': LOCAL_WINDOW_LEN,
    }


def _run_scenario(scenario, date, grid_num, driver_num, output_dir):
    pattern = SimulatorPattern(date)
    mapping = SimulatorPattern.load_mapping(date)
    if scenario['kind'] == 'city_fixed':
        grid_action = uniform_grid_actions(
            grid_num, scenario['action'], 1.0)
    else:
        treated = set(scenario['treated_grid_ids'])

        def grid_action(sim_time, _grid_state):
            step = int((sim_time - 18000) // 60)
            active = scenario['event_start_step'] <= step < scenario['event_end_step']
            actions = uniform_grid_actions(grid_num, 1.0, 1.0)
            if active:
                treatment_action = GridAction(scenario['action'], 1.0)
                for grid_id in treated:
                    actions[grid_id] = treatment_action
            return actions

    sim = Simulator(
        grid_num=grid_num,
        decision_freq=1,
        price_response_model='utility_choice',
        grid_action=grid_action,
        order_sample_ratio=1.0,
        driver_num=driver_num,
        base_fare_start=3.0,
        price_per_km=0.7,
        commission_rate=0.25,
    )
    sim.reset(seed=scenario['seed'], given_data=True,
              request_databases=pattern.request_all, driver_info=pattern.driver_info)
    sim.mapping_dict = {date: mapping}
    sim.experiment_date = date

    records = []
    while not sim.end_of_episode:
        sim.step()
        step = sim.current_step - 1
        metrics = sim.evaluate_df
        for metric in metrics.itertuples(index=False):
            grid_id = int(metric.origin_grid_id)
            records.append({
                'scenario': scenario['kind'], 'action': scenario['action'],
                'seed': scenario['seed'], 'minute_step': step, 'grid_id': grid_id,
                **_metadata_for_grid(grid_id, step, scenario),
                **metric._asdict(),
            })

    df = pd.DataFrame(records)
    stem = f"{scenario['kind']}_action{scenario['action']:.1f}_seed{scenario['seed']}"
    path = Path(output_dir) / 'runs' / f'{stem}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    with open(path.with_suffix('.json'), 'w') as handle:
        json.dump(scenario, handle, indent=2)
    return str(path), sim.total_reward, sim.matched_requests_num


def _weighted_rate(frame, numerator, denominator):
    denominator_sum = frame[denominator].sum()
    return np.nan if denominator_sum == 0 else frame[numerator].sum() / denominator_sum


def _summarize_city(run_files, output_dir):
    rows = []
    for path in run_files:
        df = pd.read_csv(path)
        request = df[df['baseline_request_num'] > 0]
        matched = df[df['matched_request_num'] > 0]
        rows.append({
            'action': df['action'].iloc[0], 'seed': df['seed'].iloc[0],
            'gbv': df['gbv'].sum(),
            'platform_revenue': df['platform_revenue'].sum(),
            'matched_orders': df['matched_request_num'].sum(),
            'baseline_requests': df['baseline_request_num'].sum(),
            'submitted_requests': df['submitted_request_num'].sum(),
            'request_rate': _weighted_rate(request, 'submitted_request_num', 'baseline_request_num'),
            'driver_accept_rate': _weighted_rate(
                df, 'driver_accept_num', 'driver_offer_num'),
            'unmatched_cancels': df['unmatched_cancel_num'].sum(),
            'pickup_cancels': df['pickup_cancel_num'].sum(),
            'mean_pickup_distance_km': matched['mean_pickup_distance_km'].mean(),
            'mean_pickup_time_seconds': matched['mean_pickup_time_seconds'].mean(),
            'mean_occupancy': df['vehicle_occupancy_rate'].mean(),
        })
    by_seed = pd.DataFrame(rows).sort_values(['action', 'seed'])
    numeric = by_seed.select_dtypes(include='number').columns.drop('seed')
    by_action = by_seed.groupby('action', as_index=False)[numeric].mean()
    by_seed.to_csv(Path(output_dir) / 'city_fixed_summary.csv', index=False)
    by_action.to_csv(Path(output_dir) / 'city_fixed_by_action.csv', index=False)
    return by_seed, by_action


def _summarize_local(run_files, output_dir):
    frames = [pd.read_csv(path) for path in run_files]
    df = pd.concat(frames, ignore_index=True)
    df = df[df['event_phase'].isin(['pre', 'intervention', 'recovery_1', 'recovery_2'])]
    rows = []
    for (action, seed, phase, treated), group in df.groupby(
            ['action', 'seed', 'event_phase', 'is_treated']):
        request = group[group['baseline_request_num'] > 0]
        matched = group[group['matched_request_num'] > 0]
        rows.append({
            'action': action, 'seed': seed, 'event_phase': phase,
            'is_treated': treated, 'grid_minutes': len(group),
            'gbv_per_grid_minute': group['gbv'].mean(),
            'matched_per_grid_minute': group['matched_request_num'].mean(),
            'unmatched_cancel_per_grid_minute': group['unmatched_cancel_num'].mean(),
            'idle_drivers_per_grid_minute': group['idle_driver_num'].mean(),
            'request_rate': _weighted_rate(request, 'submitted_request_num', 'baseline_request_num'),
            'mean_pickup_distance_km': matched['mean_pickup_distance_km'].mean(),
        })
    by_seed = pd.DataFrame(rows)
    numeric = by_seed.select_dtypes(include='number').columns.drop('seed')
    by_window = by_seed.groupby(['action', 'event_phase', 'is_treated'], as_index=False)[numeric].mean()
    by_seed.to_csv(Path(output_dir) / 'local_event_summary.csv', index=False)
    by_window.to_csv(Path(output_dir) / 'local_event_by_action_window.csv', index=False)
    return by_seed, by_window


def _write_report(city, local, output_dir):
    # Avoid a runtime dependency on the optional ``tabulate`` package.
    city_text = city.round(4).to_csv(index=False)
    local_text = local.round(4).to_csv(index=False)
    report = f"""# 价格响应诊断结果

> 运行日期：2026-09-07。每个情景 2 个 seed（42、43）；结果是机制诊断，不是最终校准。

## 城市固定价格（seed 平均）

{city_text}

## 局部 20 分钟干预（seed 平均）

`is_treated=True` 表示 26 个随机处理 grid；`False` 表示同一城市内的 control grid。

{local_text}

## 读取提示

- 需求仅使用 `request_rate` 与 `request_price_multiplier` 对齐，且只在有 `baseline_request_num` 的 grid-minute 上定义。
- 接驾距离只在 `matched_request_num > 0` 时有含义。
- 当前没有 idle-driver reposition；局部干预后的供给/取消滞后因此是现有状态转移机制的结果。
"""
    Path(output_dir, 'diagnostic_report.md').write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default='2015-05-18')
    parser.add_argument('--grid-num', type=int, default=263)
    parser.add_argument('--driver-num', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', default='diagnostics/price_response_2026-09-07')
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        _build_scenario('city_fixed', action, seed, args.grid_num)
        for action in ACTIONS for seed in SEEDS
    ] + [
        _build_scenario('local_event', action, seed, args.grid_num)
        for action in ACTIONS if action != 1.0 for seed in SEEDS
    ]
    print(f'Running {len(jobs)} diagnostic episodes with {args.workers} workers...')
    run_files = {'city_fixed': [], 'local_event': []}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(
            _run_scenario, job, args.date, args.grid_num, args.driver_num, str(output_dir))
            for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            path, gbv, matched = future.result()
            kind = Path(path).name.split('_action')[0]
            run_files[kind].append(path)
            print(f'Finished {path}: GBV={gbv:.0f}, matched={matched}')

    _, city_by_action = _summarize_city(run_files['city_fixed'], output_dir)
    _, local_by_window = _summarize_local(run_files['local_event'], output_dir)
    _write_report(city_by_action, local_by_window, output_dir)
    print(f'Wrote diagnostics to {output_dir}')


if __name__ == '__main__':
    main()
