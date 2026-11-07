"""治疗效应分析脚本 — 按 R1/R2 分层，多指标对比。

使用方式:
    PYTHONPATH=. conda run -n dp_project python diagnostics/analyze_experiment.py \
        experiments/2015-05-18_seed42_grid263/
"""

import argparse
import json
import os
import numpy as np
import pandas as pd

ZONE_ORDER = ['treated_core', 'spillover_r1', 'spillover_r2', 'clean_control']

# 关键指标：聚合方式和显示名
METRICS = {
    'gbv':                       ('sum',   'GBV'),
    'platform_revenue':          ('sum',   'Platform Rev'),
    'matched_request_num':       ('sum',   'Matched'),
    'submitted_request_num':     ('sum',   'Submitted'),
    'candidate_request_num':     ('sum',   'Candidates'),
    'baseline_request_num':      ('sum',   'Baseline'),
    'unmatched_cancel_num':      ('sum',   'Unmat.Cancel'),
    'pickup_cancel_num':         ('sum',   'Pickup.Cancel'),
    'idle_driver_num':           ('mean',  'Idle Drv'),
    'busy_driver_num':           ('mean',  'Busy Drv'),
    'repositioning_driver_num':  ('mean',  'Repo Drv'),
    'request_rate':              ('mean',  'Req Rate'),
    'driver_accept_rate':        ('mean',  'Drv Accept%'),
    'mean_wait_to_match_seconds':('mean',  'Wait (s)'),
    'mean_pickup_time_seconds':  ('mean',  'Pickup (s)'),
    'mean_pickup_distance_km':   ('mean',  'Pickup km'),
    'vehicle_occupancy_rate':    ('mean',  'Occupancy%'),
}


def load_experiment(exp_dir):
    with open(os.path.join(exp_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    control = pd.read_csv(os.path.join(exp_dir, 'evaluation_control.csv'))
    events_data = {}
    for i in range(manifest['n_events']):
        events_data[i] = pd.read_csv(
            os.path.join(exp_dir, f'evaluation_event_{i}.csv'))
    return manifest, control, events_data


def period_label(minute_step, t_start, t_end):
    if minute_step < t_start:
        return 'pre'
    elif minute_step < t_end:
        return 'treatment'
    else:
        return 'post'


def analyze_event(event_df, control_df, event, grid_num):
    """分析单个 event 的 per-zone per-period 多指标效应。"""
    t_start = event['treatment_start_step']
    t_end = event['treatment_end_step']
    zones = event['zones']
    price = event['exploration_price_multiplier']
    matching_radius = event.get('matching_radius_km', 1.0)

    results = []
    for zone_name in ZONE_ORDER:
        zone_grids = zones.get(zone_name, [])
        if not zone_grids:
            continue
        for period, p_start, p_end in [
            ('treatment', t_start, t_end),
            ('post', t_end, event_df['minute_step'].max() + 1),
        ]:
            e_zone = event_df[(event_df['minute_step'] >= p_start) &
                             (event_df['minute_step'] < p_end) &
                             event_df['grid_id'].isin(zone_grids)]
            c_zone = control_df[(control_df['minute_step'] >= p_start) &
                               (control_df['minute_step'] < p_end) &
                               control_df['grid_id'].isin(zone_grids)]
            if len(e_zone) == 0:
                continue

            entry = {
                'zone': zone_name, 'period': period,
                'n_grids': len(zone_grids),
                'price_multiplier': price,
                'matching_radius_km': matching_radius,
            }
            for col, (agg, _label) in METRICS.items():
                if col not in e_zone.columns:
                    continue
                if agg == 'sum':
                    e_val = e_zone[col].sum()
                    c_val = c_zone[col].sum()
                else:
                    e_val = e_zone[col].mean()
                    c_val = c_zone[col].mean()
                entry[f'{col}_t'] = float(e_val)
                entry[f'{col}_c'] = float(c_val)
                denom = max(abs(c_val), 1e-9)
                entry[f'{col}_delta_pct'] = float(100 * (e_val - c_val) / denom)
            results.append(entry)

    return pd.DataFrame(results)


def print_table(period_df, period_label, key_metrics=None):
    if key_metrics is None:
        key_metrics = ['gbv', 'matched_request_num', 'submitted_request_num',
                       'request_rate', 'driver_accept_rate', 'idle_driver_num',
                       'repositioning_driver_num']
    print(f"\n  === {period_label} ===")
    print(f"  {'Zone':<18s} {'n':>4s}", end="")
    for m in key_metrics:
        label = METRICS[m][1][:12]
        print(f" {label:>12s}", end="")
    print(f"\n  {'-'*18} {'-'*4}", end="")
    for _ in key_metrics:
        print(f" {'-'*12}", end="")
    print()
    for zone in ZONE_ORDER:
        z = period_df[period_df['zone'] == zone]
        if len(z) == 0:
            continue
        n = z['n_grids'].iloc[0]
        print(f"  {zone:<18s} {int(n):4d}", end="")
        for m in key_metrics:
            col = f'{m}_delta_pct'
            if col in z.columns:
                print(f" {z[col].mean():>+11.1f}%", end="")
            else:
                print(f" {'N/A':>12s}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('exp_dir', help='Path to experiment output directory')
    args = parser.parse_args()

    print(f"Loading experiment from {args.exp_dir}...")
    manifest, control, events_data = load_experiment(args.exp_dir)

    grid_num = manifest['grid_num']
    events = manifest['events']

    print(f"\n{'='*80}")
    print(f"Experiment: {manifest['manifest_id']}")
    print(f"Events: {manifest['n_events']} × {manifest['treatment_minutes']}min treatment, "
          f"margin={manifest['margin_minutes']}min")
    print(f"{'='*80}")

    all_contrasts = []

    for i, event in enumerate(events):
        contrasts = analyze_event(events_data[i], control, event, grid_num)
        price = event['exploration_price_multiplier']
        matching_radius = event.get('matching_radius_km', 1.0)
        t_start = event['treatment_start_step']
        t_end = event['treatment_end_step']
        zones = event['zones']

        print(f"\n--- Event {i} (id={event['exploration_event_id']}): "
              f"action=(price={price}, radius={matching_radius}km), "
              f"t=[{t_start},{t_end}) ---")
        print(f"  Grid breakup: core={len(zones['treated_core'])}, "
              f"R1={len(zones['spillover_r1'])}, "
              f"R2={len(zones['spillover_r2'])}, "
              f"clean={len(zones['clean_control'])}")

        contrasts['event_idx'] = i
        all_contrasts.append(contrasts)

        # Print per-zone treatment-period summary
        tc = contrasts[contrasts['period'] == 'treatment']
        print_table(tc, "Treatment period")

    all_df = pd.concat(all_contrasts, ignore_index=True)

    # Per-zone aggregate
    print(f"\n{'='*80}")
    print("PER-ZONE AGGREGATE (treatment period, mean over events)")
    print(f"{'='*80}")
    print_table(all_df[all_df['period'] == 'treatment'], "TREATMENT")

    print(f"\n{'='*80}")
    print("PER-ZONE AGGREGATE (post-treatment period, mean over events)")
    print(f"{'='*80}")
    print_table(all_df[all_df['period'] == 'post'], "POST")

    # By joint action
    print(f"\n{'='*80}")
    print("GBV Δ% BY JOINT ACTION (treated_core, treatment period)")
    print(f"{'='*80}")
    tc = all_df[(all_df['zone'] == 'treated_core') & (all_df['period'] == 'treatment')]
    joint_actions = (
        tc[['price_multiplier', 'matching_radius_km']]
        .drop_duplicates()
        .sort_values(['price_multiplier', 'matching_radius_km'])
        .itertuples(index=False, name=None)
    )
    for price, radius in joint_actions:
        a_df = tc[
            (tc['price_multiplier'] == price)
            & (tc['matching_radius_km'] == radius)
        ]
        if len(a_df) == 0:
            continue
        gbv_d = a_df['gbv_delta_pct'].mean()
        matched_d = a_df.get('matched_request_num_delta_pct', pd.Series([0])).mean()
        drv_d = a_df.get('driver_accept_rate_delta_pct', pd.Series([0])).mean()
        print(f"  price={price:g}, radius={radius:g}km: "
              f"GBV Δ={gbv_d:+.1f}%  matched Δ={matched_d:+.1f}%  "
              f"drv_accept Δ={drv_d:+.1f}%  (n={len(a_df)})")

    all_df.to_csv(os.path.join(args.exp_dir, 'contrasts_detailed.csv'), index=False)
    print(f"\nDetailed contrasts saved to {args.exp_dir}/contrasts_detailed.csv")


if __name__ == '__main__':
    main()
