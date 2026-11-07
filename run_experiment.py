"""实验运行器 — N 个联合动作事件独立运行 + 反事实 control run。

核心理念: 每个 treatment 事件独立运行，都从同一个 reset(seed) 出发。
- 事件均匀分布在仿真时段内，可重叠（独立 run 互不污染）
- MDP 构建直接从分钟级轨迹数据（minute_metrics）中构造

使用方式:
    PYTHONPATH=. conda run -n dp_project python run_experiment.py \
        --date 2015-05-18 --seed 42 --n-events 10 --output experiments/
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

from src.env.exploration import ExplorationPlanner
from src.env.actions import GridAction, uniform_grid_actions
from src.env.simulator_env import Simulator
from src.env.simulator_pattern import SimulatorPattern

# ---- 所有用于对比的指标及其聚合方式 ----
# 'sum': per-zone 按分钟 sum 后再按 zone 求和
# 'mean': per-zone 按分钟 mean（如比率类指标）

CONTRAST_METRICS = {
    'gbv':                       ('sum',   'GBV'),
    'platform_revenue':          ('sum',   'Platform Revenue'),
    'matched_request_num':       ('sum',   'Matched Orders'),
    'submitted_request_num':     ('sum',   'Submitted Requests'),
    'candidate_request_num':     ('sum',   'Candidate Requests'),
    'baseline_request_num':      ('sum',   'Baseline Requests'),
    'unmatched_cancel_num':      ('sum',   'Unmatched Cancels'),
    'pickup_cancel_num':         ('sum',   'Pickup Cancels'),
    'idle_driver_num':           ('mean',  'Idle Drivers'),
    'busy_driver_num':           ('mean',  'Busy Drivers'),
    'repositioning_driver_num':  ('mean',  'Repositioning Drivers'),
    'request_rate':              ('mean',  'Request Rate'),
    'driver_accept_rate':        ('mean',  'Driver Accept Rate'),
    'mean_wait_to_match_seconds':('mean',  'Mean Wait-to-Match (s)'),
    'mean_pickup_time_seconds':  ('mean',  'Mean Pickup Time (s)'),
    'mean_pickup_distance_km':   ('mean',  'Mean Pickup Distance (km)'),
    'vehicle_occupancy_rate':    ('mean',  'Vehicle Occupancy Rate'),
}

ZONE_ORDER = ['treated_core', 'spillover_r1', 'spillover_r2', 'clean_control']


def build_manifest(date, seed, grid_num, plan_meta, n_events, simulation_steps):
    return {
        'manifest_id': f'exp_{date}_seed{seed}',
        'date': date,
        'seed': seed,
        'grid_num': grid_num,
        'simulation_steps': simulation_steps,
        'reposition_enabled': True,
        'treatment_minutes': plan_meta['treatment_minutes'],
        'r_exposure': plan_meta['r_exposure'],
        'margin_minutes': plan_meta['margin_minutes'],
        'price_action_weights': {
            str(k): v for k, v in plan_meta['price_action_weights'].items()},
        'matching_radius_weights': {
            str(k): v for k, v in plan_meta['matching_radius_weights'].items()},
        'joint_action_sampling': plan_meta['joint_action_sampling'],
        'n_events': n_events,
    }


def make_grid_action_fn(event, t_initial, delta_t, grid_num):
    """创建完整网格联合动作。event=None 表示价格 1.0、半径 1.0。

    价格乘子用于该 origin grid 的新订单，匹配半径用于该
    origin grid 当前排队订单的候选边筛选。
    """
    def grid_action_fn(sim_time, _grid_state):
        actions = uniform_grid_actions(grid_num, 1.0, 1.0)
        if event is None:
            return actions
        step = int((sim_time - t_initial) // delta_t)
        if event['treatment_start_step'] <= step < event['treatment_end_step']:
            treatment_action = GridAction(
                event['exploration_price_multiplier'],
                event['matching_radius_km'],
            )
            for grid_id in event['exploration_grid_ids']:
                actions[int(grid_id)] = treatment_action
        return actions
    return grid_action_fn


def grid_exposure_metadata(event, minute_step, grid_id):
    if event is None:
        return {'exposure': 'baseline', 'price_multiplier': 1.0,
                'matching_radius_km_planned': 1.0,
                'exploration_event_id': None}
    if event['treatment_start_step'] <= minute_step < event['treatment_end_step']:
        if grid_id in event['exploration_grid_ids']:
            return {
                'exposure': 'treated_core',
                'price_multiplier': event['exploration_price_multiplier'],
                'matching_radius_km_planned': event['matching_radius_km'],
                'exploration_event_id': event['exploration_event_id'],
            }
    return {'exposure': 'baseline', 'price_multiplier': 1.0,
            'matching_radius_km_planned': 1.0,
            'exploration_event_id': event['exploration_event_id']}


def run_simulation(date, seed, grid_num, driver_num, event,
                   enable_reposition=True):
    """运行仿真。event=None → control run。"""
    pattern = SimulatorPattern(date)
    mapping = SimulatorPattern.load_mapping(date)

    t_initial = 18000
    delta_t = 60

    sim = Simulator(
        grid_num=grid_num,
        decision_freq=1,
        price_response_model='utility_choice',
        grid_action=make_grid_action_fn(
            event, t_initial, delta_t, grid_num),
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

    while not sim.end_of_episode:
        sim.step()

    minute_metrics = pd.DataFrame(
        sim.evaluate_table.reshape(-1, sim.evaluate_table.shape[-1]),
        columns=sim.evaluate_df.columns,
    )
    minute_metrics.insert(0, 'sim_time', np.repeat(
        np.arange(sim.t_initial, sim.t_end, sim.delta_t), grid_num))
    minute_metrics.insert(0, 'minute_step', np.repeat(
        np.arange(sim.finish_run_step), grid_num))
    minute_metrics['grid_id'] = minute_metrics['origin_grid_id'].astype(int)

    minute_metadata = pd.DataFrame([
        {'minute_step': minute_step, 'grid_id': grid_id,
         **grid_exposure_metadata(event, minute_step, grid_id)}
        for minute_step in range(sim.finish_run_step)
        for grid_id in range(grid_num)
    ])
    minute_metrics = minute_metrics.merge(
        minute_metadata, on=['minute_step', 'grid_id'],
        how='left', validate='one_to_one')

    return {
        'minute_metrics': minute_metrics,
        'total_reward': sim.total_reward,
        'matched_requests_num': sim.matched_requests_num,
        'reposition_events': sim.reposition_events,
        'summary': {
            'total_reward': sim.total_reward,
            'matched_requests_num': sim.matched_requests_num,
            'submitted_requests': sim.accepted_quote_num,
            'baseline_requests': sim.baseline_request_num,
            'occupancy_rate': sim.occupancy_rate,
            'reposition_events_count': len(sim.reposition_events),
            'reposition_distance_km': sim.total_reposition_distance_km,
            'reposition_cost': sim.total_reposition_cost,
        },
    }


def verify_pre_period_consistency(event_results, control_df, events):
    """验证每个 event run 的 pre-period 与 control run 一致。"""
    compare_cols = ['matched_request_num', 'idle_driver_num',
                    'busy_driver_num', 'repositioning_driver_num']
    all_pass = True
    for i, (event_result, event) in enumerate(zip(event_results, events)):
        event_df = event_result['minute_metrics']
        pre_end = event['treatment_start_step']
        event_pre = event_df[event_df['minute_step'] < pre_end]
        ctrl_pre = control_df[control_df['minute_step'] < pre_end]
        consistent = True
        for col in compare_cols:
            if col in event_pre.columns and col in ctrl_pre.columns:
                if not np.allclose(event_pre[col].values, ctrl_pre[col].values,
                                   rtol=1e-10, atol=1e-10, equal_nan=True):
                    diff = np.abs(event_pre[col].values - ctrl_pre[col].values).max()
                    print(f"  ✗ Event {i} pre-period [0,{pre_end}) {col}: max_diff={diff}")
                    consistent = False
                    all_pass = False
        if consistent:
            print(f"  ✓ Event {i} pre-period [0,{pre_end}): all metrics identical to control")
    return all_pass


def compute_contrasts(event_results, control_df, events, grid_num):
    """计算 treatment vs control 的 per-zone per-period 多指标对比。

    按每个 event 预计算的 zones (treated_core / spillover_r1 / spillover_r2
    / clean_control) 分别统计 treatment 期和 post-treatment 期的效应。
    """
    contrasts = []
    for i, (event_result, event) in enumerate(zip(event_results, events)):
        event_df = event_result['minute_metrics']
        t_start = event['treatment_start_step']
        t_end = event['treatment_end_step']
        zones = event['zones']
        price = event['exploration_price_multiplier']
        matching_radius = event['matching_radius_km']

        for zone_name, zone_grids in zones.items():
            if not zone_grids:
                continue
            # 分 treatment 期和 post 期
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
                    'event_idx': i,
                    'zone': zone_name,
                    'period': period,
                    'n_grids': len(zone_grids),
                    'price_multiplier': price,
                    'matching_radius_km': matching_radius,
                    't_start': t_start,
                    't_end': t_end,
                }

                for col, (agg, _label) in CONTRAST_METRICS.items():
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
                    entry[f'{col}_delta'] = float(e_val - c_val)
                    entry[f'{col}_delta_pct'] = float(100 * (e_val - c_val) / denom)

                contrasts.append(entry)

    return pd.DataFrame(contrasts)


def print_contrast_table(contrasts_df, period='treatment'):
    """打印 per-zone 多指标汇总表。"""
    period_df = contrasts_df[contrasts_df['period'] == period]
    # 关键指标列
    key_metrics = ['gbv', 'matched_request_num', 'submitted_request_num',
                   'request_rate', 'driver_accept_rate', 'idle_driver_num',
                   'repositioning_driver_num']
    print(f"\n  {'Zone':<18s} {'n':>4s}", end="")
    for m in key_metrics:
        label = CONTRAST_METRICS[m][1][:10]
        print(f" {label:>10s}", end="")
    print(f"\n  {'-'*18} {'-'*4}", end="")
    for _ in key_metrics:
        print(f" {'-'*10}", end="")
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
                print(f" {z[col].mean():>+9.1f}%", end="")
            else:
                print(f" {'N/A':>10s}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Run joint price/radius exploration experiment')
    parser.add_argument('--date', default='2015-05-18')
    parser.add_argument('--grid-num', type=int, default=263)
    parser.add_argument('--driver-num', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--simulation-steps', type=int, default=300)
    parser.add_argument('--adjacency-path',
                        default='my_data/hexo_updated_r300_adj.csv')
    parser.add_argument('--n-events', type=int, default=10)
    parser.add_argument('--treatment-minutes', type=int, default=10)
    parser.add_argument('--margin-minutes', type=int, default=15)
    parser.add_argument('--r-exposure', type=int, default=2)
    parser.add_argument('--output', default='experiments')
    args = parser.parse_args()

    sim_steps = args.simulation_steps
    stem = f'{args.date}_seed{args.seed}_grid{args.grid_num}'
    out_dir = os.path.join(args.output, stem)
    os.makedirs(out_dir, exist_ok=True)

    total_runs = 0

    # Step 1: 探索计划
    print(f"[1/5] Loading adjacency & building exploration plan...")
    adj = ExplorationPlanner.load_adjacency(args.adjacency_path)
    planner = ExplorationPlanner(
        adj, grid_num=args.grid_num, r_exposure=args.r_exposure,
        treatment_minutes=args.treatment_minutes,
        adjacency_path=args.adjacency_path)
    plan_rng = np.random.default_rng(args.seed)
    events, plan_meta = planner.build_plan(
        plan_rng, n_events=args.n_events, simulation_steps=sim_steps,
        margin_minutes=args.margin_minutes)

    print(f"  {plan_meta['n_events']} events ({args.treatment_minutes}min), "
          f"margin={plan_meta['margin_minutes']}min, "
          f"range [{plan_meta['margin_minutes']}, "
          f"{sim_steps - args.treatment_minutes - plan_meta['margin_minutes']})")
    for ev in events:
        z = ev['zones']
        print(f"    Event {ev['exploration_event_id']}: "
              f"t=[{ev['treatment_start_step']},{ev['treatment_end_step']}) "
              f"action=(price={ev['exploration_price_multiplier']}, "
              f"radius={ev['matching_radius_km']}km) "
              f"core={len(z['treated_core'])} "
              f"R1={len(z['spillover_r1'])} "
              f"R2={len(z['spillover_r2'])} "
              f"clean={len(z['clean_control'])}")

    # Step 2: Control
    print(f"\n[2/5] Control run (all-1.0)...")
    total_runs += 1
    control_result = run_simulation(
        args.date, args.seed, args.grid_num, args.driver_num,
        event=None, enable_reposition=True)
    s = control_result['summary']
    print(f"  Control: reward={s['total_reward']:,.2f}, matched={s['matched_requests_num']}, "
          f"repos={s['reposition_events_count']}")

    control_result['minute_metrics'].to_csv(
        os.path.join(out_dir, 'evaluation_control.csv'), index=False)
    if control_result['reposition_events']:
        pd.DataFrame(control_result['reposition_events']).to_csv(
            os.path.join(out_dir, 'reposition_events_control.csv'), index=False)
    control_minute_df = control_result['minute_metrics']

    # Step 3: Events
    print(f"\n[3/5] {plan_meta['n_events']} independent event runs...")
    event_results = []
    for i, event in enumerate(events):
        total_runs += 1
        result = run_simulation(args.date, args.seed, args.grid_num, args.driver_num,
                                event=event, enable_reposition=True)
        event_results.append(result)

        t_start, t_end = event['treatment_start_step'], event['treatment_end_step']
        e_gbv = result['minute_metrics'][
            result['minute_metrics']['minute_step'] >= t_start]['gbv'].sum()
        c_gbv = control_minute_df[
            control_minute_df['minute_step'] >= t_start]['gbv'].sum()
        print(f"  Event {i}: action=(price={event['exploration_price_multiplier']}, "
              f"radius={event['matching_radius_km']}km) "
              f"t=[{t_start},{t_end}) "
              f"GBV(post)={e_gbv:,.2f} vs ctrl={c_gbv:,.2f} (Δ={e_gbv-c_gbv:+,.2f})")

    # Step 4: Pre-period 校验
    print(f"\n[4/5] Verifying pre-period consistency...")
    all_consistent = verify_pre_period_consistency(event_results, control_minute_df, events)

    # Step 5: 对比 & 保存
    print(f"\n[5/5] Computing contrasts & saving...")
    contrasts_df = compute_contrasts(event_results, control_minute_df, events, args.grid_num)

    print("\n  === Treatment period per-zone ===")
    print_contrast_table(contrasts_df, period='treatment')

    print("\n  === Post-treatment period per-zone ===")
    print_contrast_table(contrasts_df, period='post')

    # Save
    for i, result in enumerate(event_results):
        result['minute_metrics'].to_csv(
            os.path.join(out_dir, f'evaluation_event_{i}.csv'), index=False)
    contrasts_df.to_csv(os.path.join(out_dir, 'contrasts.csv'), index=False)

    manifest = build_manifest(args.date, args.seed, args.grid_num,
                              plan_meta, len(events), sim_steps)
    manifest['events'] = events
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    summary = {
        'manifest': manifest,
        'control_summary': control_result['summary'],
        'event_summaries': [r['summary'] for r in event_results],
        'pre_period_consistent': bool(all_consistent),
        'total_runs': total_runs,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== Complete ({total_runs} runs) ===")
    print(f"  Output: {out_dir}/")
    print(f"  Pre-period consistency: {'✓ PASS' if all_consistent else '✗ FAIL'}")
    return 0 if all_consistent else 1


if __name__ == '__main__':
    sys.exit(main())
