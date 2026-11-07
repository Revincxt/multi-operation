"""Compare baseline pricing with repositioning disabled versus enabled.

Example:
    PYTHONPATH=. conda run -n dp_project python diagnose_reposition.py
"""

import argparse
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd

from src.env.actions import uniform_grid_actions
from src.env.simulator_env import Simulator
from src.env.simulator_pattern import SimulatorPattern


SEEDS = (42, 43)


def run_episode(enable_reposition, seed, date, grid_num, driver_num, output_dir):
    pattern = SimulatorPattern(date)
    mapping = SimulatorPattern.load_mapping(date)
    sim = Simulator(
        grid_num=grid_num,
        decision_freq=1,
        price_response_model='utility_choice',
        grid_action=uniform_grid_actions(grid_num, 1.0, 1.0),
        order_sample_ratio=1.0,
        driver_num=driver_num,
        base_fare_start=3.0,
        price_per_km=0.7,
        commission_rate=0.25,
        enable_reposition=enable_reposition,
    )
    sim.reset(seed=seed, given_data=True,
              request_databases=pattern.request_all, driver_info=pattern.driver_info)
    sim.mapping_dict = {date: mapping}
    sim.experiment_date = date

    metrics = []
    while not sim.end_of_episode:
        sim.step()
        frame = sim.evaluate_df.copy()
        frame.insert(0, 'minute_step', sim.current_step - 1)
        frame.insert(0, 'seed', seed)
        frame.insert(0, 'reposition_enabled', enable_reposition)
        metrics.append(frame)

    label = 'on' if enable_reposition else 'off'
    run_dir = Path(output_dir) / 'runs'
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / f'baseline_reposition_{label}_seed{seed}.csv'
    pd.concat(metrics, ignore_index=True).to_csv(metrics_path, index=False)
    event_path = run_dir / f'reposition_events_seed{seed}.csv'
    if enable_reposition:
        event_frame = pd.DataFrame(sim.reposition_events)
        event_frame.insert(0, 'seed', seed)
        event_frame.to_csv(event_path, index=False)
    return {
        'enable_reposition': enable_reposition,
        'seed': seed,
        'metrics_path': str(metrics_path),
        'event_path': str(event_path) if enable_reposition else None,
        'total_gbv': sim.total_reward,
        'matched_orders': sim.matched_requests_num,
        'total_reposition_distance_km': sim.total_reposition_distance_km,
        'total_reposition_cost': sim.total_reposition_cost,
    }


def weighted_rate(frame, numerator, denominator):
    total_denominator = frame[denominator].sum()
    if total_denominator == 0:
        return np.nan
    return frame[numerator].sum() / total_denominator


def summarize_run(result):
    frame = pd.read_csv(result['metrics_path'])
    matched = frame[frame['matched_request_num'] > 0]
    return {
        'reposition_enabled': result['enable_reposition'],
        'seed': result['seed'],
        'gbv': frame['gbv'].sum(),
        'platform_revenue': frame['platform_revenue'].sum(),
        'matched_orders': frame['matched_request_num'].sum(),
        'unmatched_cancels': frame['unmatched_cancel_num'].sum(),
        'pickup_cancels': frame['pickup_cancel_num'].sum(),
        'driver_accept_rate': weighted_rate(frame, 'driver_accept_num', 'driver_offer_num'),
        'mean_idle_drivers': frame['idle_driver_num'].sum() / 300,
        'mean_busy_drivers': frame['busy_driver_num'].sum() / 300,
        'mean_repositioning_drivers': frame['repositioning_driver_num'].sum() / 300,
        'mean_pickup_distance_km': matched['mean_pickup_distance_km'].mean(),
        'mean_pickup_time_seconds': matched['mean_pickup_time_seconds'].mean(),
        'reposition_events': 0 if result['event_path'] is None else len(pd.read_csv(result['event_path'])),
        'total_reposition_distance_km': result['total_reposition_distance_km'],
        'total_reposition_cost': result['total_reposition_cost'],
    }


def summarize_events(results):
    events = []
    for result in results:
        if result['event_path']:
            event_frame = pd.read_csv(result['event_path'])
            if 'seed' not in event_frame:
                event_frame.insert(0, 'seed', result['seed'])
            events.append(event_frame)
    frame = pd.concat(events, ignore_index=True)
    per_minute = frame.groupby(['seed', 'minute_step']).size()
    destination_share = frame['destination_grid_id'].value_counts(normalize=True)
    return {
        'events': len(frame),
        'drivers_repositioned': frame['driver_id'].nunique(),
        'no_income_signal_share': (frame['choice_reason'] == 'no_income_signal').mean(),
        'mean_distance_km': frame['reposition_distance_km'].mean(),
        'p95_distance_km': frame['reposition_distance_km'].quantile(0.95),
        'mean_time_seconds': frame['reposition_time_seconds'].mean(),
        'mean_cost': frame['reposition_operating_cost'].mean(),
        'max_events_one_minute': per_minute.max(),
        'max_destination_share': destination_share.iloc[0],
        'top_destination_grid': int(destination_share.index[0]),
    }


def write_report(by_seed, by_setting, events, output_dir):
    off = by_setting.loc[by_setting['reposition_enabled'] == False].iloc[0]
    on = by_setting.loc[by_setting['reposition_enabled'] == True].iloc[0]
    comparison = pd.DataFrame({
        'metric': [
            'gbv', 'platform_revenue', 'matched_orders', 'unmatched_cancels',
            'pickup_cancels', 'mean_idle_drivers', 'mean_busy_drivers',
            'mean_pickup_distance_km', 'mean_pickup_time_seconds',
        ],
        'reposition_off': [off[m] for m in [
            'gbv', 'platform_revenue', 'matched_orders', 'unmatched_cancels',
            'pickup_cancels', 'mean_idle_drivers', 'mean_busy_drivers',
            'mean_pickup_distance_km', 'mean_pickup_time_seconds',
        ]],
        'reposition_on': [on[m] for m in [
            'gbv', 'platform_revenue', 'matched_orders', 'unmatched_cancels',
            'pickup_cancels', 'mean_idle_drivers', 'mean_busy_drivers',
            'mean_pickup_distance_km', 'mean_pickup_time_seconds',
        ]],
    })
    comparison['absolute_change'] = comparison['reposition_on'] - comparison['reposition_off']
    comparison['relative_change_pct'] = 100 * comparison['absolute_change'] / comparison['reposition_off']
    comparison.to_csv(Path(output_dir) / 'off_on_comparison.csv', index=False)
    text = f"""# Idle-driver repositioning baseline comparison

> Baseline multiplier 1.0; date 2015-05-18; two seeds (42, 43). This is a mechanism diagnostic, not a calibrated causal estimate.

## Per-seed results

```csv
{by_seed.round(4).to_csv(index=False)}
```

## Seed-average off/on comparison

```csv
{comparison.round(4).to_csv(index=False)}
```

## Reposition event diagnostics (enabled runs only)

```csv
{pd.DataFrame([events]).round(4).to_csv(index=False)}
```

## Interpretation guardrails

- Reposition operating cost is recorded but excluded from GBV and platform revenue.
- `unmatched_cancels` is a flow count, while queue effects can lag the reposition decision.
- The current model uses 10-minute idle trigger, R=2 candidates, and logit choice from lagged 10-minute confirmed driver-income flow.
"""
    Path(output_dir, 'reposition_report.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default='2015-05-18')
    parser.add_argument('--grid-num', type=int, default=263)
    parser.add_argument('--driver-num', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', default='diagnostics/reposition_2026-09-07')
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(enabled, seed) for enabled in (False, True) for seed in SEEDS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(
            run_episode, enabled, seed, args.date, args.grid_num, args.driver_num, str(output_dir))
            for enabled, seed in jobs]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    by_seed = pd.DataFrame([summarize_run(result) for result in results]).sort_values(
        ['reposition_enabled', 'seed'])
    numeric = by_seed.select_dtypes(include='number').columns.drop('seed')
    by_setting = by_seed.groupby('reposition_enabled', as_index=False)[numeric].mean()
    events = summarize_events([result for result in results if result['enable_reposition']])
    by_seed.to_csv(output_dir / 'baseline_off_on_by_seed.csv', index=False)
    by_setting.to_csv(output_dir / 'baseline_off_on_summary.csv', index=False)
    write_report(by_seed, by_setting, events, output_dir)
    print(by_setting.to_string(index=False))
    print(events)


if __name__ == '__main__':
    main()
