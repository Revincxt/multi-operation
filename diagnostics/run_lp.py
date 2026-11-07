"""LP (Local Projections) 分析脚本。

从实验输出目录构造 staggered treatment panel，估计 impulse response：
- H*: treatment 效应达到峰值的时间跨度
- R*: 空间溢出效应随距离衰减的特征半径

方法（Jordà 2005, Callaway & Sant'Anna 2021）:
    对每个 horizon h = 0, 1, ..., H_max，估计：
    ΔY_{i,t+h} = β_h * D_{i,t} + α_i + γ_t + ε_{i,t,h}

    其中 D_{i,t} 是 treatment 指示变量。

面板构建策略:
    - Unit = (grid, event, seed, date)
    - 每个 unit 有一个确定的 treatment_start
    - Pre-period: treatment_start 之前的所有分钟
    - Post-period: treatment_start 之后的分钟
    - Control group: never-treated grids (clean_control) + not-yet-treated grids
    - Outcome: GBV_treatment(i,t) - GBV_control(i,t)，即 treatment/control 差分消除混淆

选项:
    A. "diff-in-diff LP": outcome = Y_{treatment}(i,t) - Y_{control}(i,t)
       直接用已差分的数据，控制 grid FE 和 time FE
    B. "panel LP": outcome = Y(i,t)，加入 treatment indicator + 交互项

    推荐选项 A，因为仿真器的 control run 提供了完美的反事实，
    差分后的数据无需再控制 confounding。

使用方式:
    PYTHONPATH=. conda run -n dp_project python diagnostics/run_lp.py \
        experiments/ --max-horizon 60 --output experiments/lp_results/
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm


ZONE_ORDER = ['treated_core', 'spillover_r1', 'spillover_r2', 'clean_control']


def collect_experiments(experiments_root):
    """扫描 experiments/ 目录，收集所有已完成的实验。

    Returns
    -------
    list of dict: 每个实验的 meta 信息
        date, seed, grid_num, dir, manifest
    """
    exps = []
    if not os.path.isdir(experiments_root):
        return exps
    for name in sorted(os.listdir(experiments_root)):
        exp_dir = os.path.join(experiments_root, name)
        manifest_path = os.path.join(exp_dir, 'manifest.json')
        if not os.path.isfile(manifest_path):
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        exps.append({
            'date': manifest['date'],
            'seed': manifest['seed'],
            'grid_num': manifest['grid_num'],
            'dir': exp_dir,
            'manifest': manifest,
        })
    return exps


def build_lp_panel(exp, max_pre=None, max_post=None):
    """从单个实验构建 LP 面板。

    对每个 event，将 treatment 和 control minute_metrics 差分合并：
    - 每个 (grid, relative_minute) 计算 treated - control 的 GBV 等指标
    - relative_minute: 0 = treatment_start，负值 = pre，正值 = post
    - treatment indicator D = 1 for relative_minute in [0, treatment_minutes)

    Parameters
    ----------
    exp : dict
        实验信息，包含 manifest 和 dir。
    max_pre : int or None
        使用 treatment 前最多 max_pre 分钟（None = 全部）。
    max_post : int or None
        使用 treatment 后最多 max_post 分钟（None = 全部）。

    Returns
    -------
    pd.DataFrame with columns:
        date, seed, event_id, grid_id, zone, price_multiplier,
        matching_radius_km,
        relative_minute, D (treatment indicator),
        gbv_diff, matched_diff, ... (per-metric treatment-control diff)
    """
    manifest = exp['manifest']
    exp_dir = exp['dir']
    events = manifest['events']
    treatment_minutes = manifest['treatment_minutes']
    date = manifest['date']
    seed = manifest['seed']

    control = pd.read_csv(os.path.join(exp_dir, 'evaluation_control.csv'))

    panels = []
    for ev in events:
        event_id = ev['exploration_event_id']
        t_start = ev['treatment_start_step']
        t_end = ev['treatment_end_step']
        zones = ev['zones']
        price = ev['exploration_price_multiplier']
        matching_radius = ev.get('matching_radius_km', 1.0)

        event_df = pd.read_csv(
            os.path.join(exp_dir, f'evaluation_event_{event_id}.csv'))

        # 确定相对分钟范围
        sim_end = manifest['simulation_steps']
        pre_start = max(0, t_start - max_pre) if max_pre else 0

        # 对每个 grid，在 [pre_start, sim_end) 范围内计算差分
        for zone_name in ['treated_core', 'spillover_r1', 'spillover_r2']:
            zone_grids = zones.get(zone_name, [])
            if not zone_grids:
                continue

            for g in zone_grids:
                e_g = event_df[event_df['grid_id'] == g].copy()
                c_g = control[control['grid_id'] == g].copy()

                # 对齐到共同的时间范围
                common_minutes = sorted(set(e_g['minute_step']) &
                                       set(c_g['minute_step']))
                common_minutes = [m for m in common_minutes
                                 if m >= pre_start and m < sim_end]

                if len(common_minutes) < 2:
                    continue

                e_g = e_g.set_index('minute_step').loc[common_minutes].reset_index()
                c_g = c_g.set_index('minute_step').loc[common_minutes].reset_index()

                rows = []
                for _, (e_row, c_row) in enumerate(zip(
                        e_g.itertuples(), c_g.itertuples())):
                    minute = int(e_row.minute_step)
                    rel = minute - t_start
                    D = 1 if 0 <= rel < treatment_minutes else 0

                    row = {
                        'date': date,
                        'seed': seed,
                        'event_id': event_id,
                        'grid_id': g,
                        'zone': zone_name,
                        'price_multiplier': price,
                        'matching_radius_km': matching_radius,
                        't_start': t_start,
                        'minute_step': minute,
                        'relative_minute': rel,
                        'D': D,
                        'gbv_diff': float(e_row.gbv - c_row.gbv),
                    }

                    # 附加其他指标差分（如果列存在）
                    for col in ['matched_request_num', 'submitted_request_num',
                                'idle_driver_num', 'busy_driver_num',
                                'repositioning_driver_num',
                                'request_rate', 'driver_accept_rate',
                                'mean_wait_to_match_seconds',
                                'platform_revenue']:
                        if hasattr(e_row, col) and hasattr(c_row, col):
                            row[f'{col}_diff'] = float(
                                getattr(e_row, col) - getattr(c_row, col))

                    rows.append(row)

                if rows:
                    panels.append(pd.DataFrame(rows))

    if not panels:
        return pd.DataFrame()

    panel = pd.concat(panels, ignore_index=True)
    return panel


def run_lp(panel, outcome_col='gbv_diff', max_horizon=60,
           time_bin_minutes=10):
    """对给定面板运行 Local Projections。

    估计方程（对每个 horizon h）:
        Y_{i,t+h} = β_h * D_{i,t} + α_g + γ_t + ε_{i,t,h}

    其中:
    - Y 已是 treatment - control 差分（消除共同时间冲击）
    - α_g = grid FE
    - γ_t = absolute time FE（minute_step 分箱），用绝对时间而非 relative
      时间以避免与 D 共线（不同 event 的 D=1 在不同绝对时间）

    Parameters
    ----------
    panel : pd.DataFrame
        LP 面板，来自 build_lp_panel。
    outcome_col : str
        被解释变量列名。
    max_horizon : int
        最大 horizon（分钟）。
    time_bin_minutes : int
        时间 FE 的分箱粒度（分钟），过细会导致高维，默认 10。

    Returns
    -------
    pd.DataFrame with columns: horizon, beta, se, t_stat, p_value, n_obs
    """
    panel = panel.copy()
    # 将 minute_step 分箱为时间 FE（降低维度，同时打破共线性）
    panel['time_bin'] = panel['minute_step'] // time_bin_minutes

    irf = []

    for h in range(max_horizon + 1):
        # 构建 lead 变量 Y_{t+h}（within-unit shift）
        panel_h = panel.copy()
        panel_h['Y_lead'] = panel_h.groupby(
            ['date', 'seed', 'event_id', 'grid_id']
        )[outcome_col].shift(-h)

        valid = panel_h['Y_lead'].notna()
        panel_h = panel_h[valid].copy()
        n_h = len(panel_h)

        if n_h < 50:
            irf.append({
                'horizon': h, 'beta': np.nan, 'se': np.nan,
                't_stat': np.nan, 'p_value': np.nan, 'n_obs': n_h,
            })
            continue

        # D: treatment indicator
        X = panel_h[['D']].astype(float)

        # Grid FE (drop_first to avoid dummy trap with constant)
        grid_dummies = pd.get_dummies(
            panel_h['grid_id'], prefix='g', drop_first=True).astype(float)
        X = pd.concat([X, grid_dummies], axis=1)

        # Time FE: absolute time bins (minute_step // time_bin_minutes)
        # Different events have D=1 at different absolute times → no collinearity
        time_dummies = pd.get_dummies(
            panel_h['time_bin'], prefix='t', drop_first=True).astype(float)
        X = pd.concat([X, time_dummies], axis=1)

        y = panel_h['Y_lead'].astype(float)

        try:
            model = sm.OLS(y, sm.add_constant(X)).fit()
            beta = model.params['D']
            se = model.bse['D']
            t_stat = model.tvalues['D']
            p_value = model.pvalues['D']
        except Exception:
            beta, se, t_stat, p_value = np.nan, np.nan, np.nan, np.nan

        irf.append({
            'horizon': h,
            'beta': float(beta),
            'se': float(se),
            't_stat': float(t_stat) if not np.isnan(t_stat) else np.nan,
            'p_value': float(p_value) if not np.isnan(p_value) else np.nan,
            'n_obs': n_h,
        })

    return pd.DataFrame(irf)


def identify_h_star(irf_df, significance_level=0.05):
    """从 IRF 识别 H*：效应显著且达到最大绝对值的时间点。

    Returns
    -------
    dict with keys: h_star, beta_at_h_star, significant
    """
    significant = irf_df[irf_df['p_value'] < significance_level]
    if len(significant) == 0:
        return {'h_star': None, 'beta_at_h_star': 0.0, 'significant': False}

    # 取绝对效应最大者
    idx = significant['beta'].abs().idxmax()
    row = significant.loc[idx]
    return {
        'h_star': int(row['horizon']),
        'beta_at_h_star': float(row['beta']),
        'se_at_h_star': float(row['se']),
        'significant': True,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run LP analysis on experiment outputs')
    parser.add_argument('experiments_root',
                        help='Root directory of experiment outputs')
    parser.add_argument('--max-horizon', type=int, default=60,
                        help='Maximum horizon for LP (minutes)')
    parser.add_argument('--max-pre', type=int, default=40,
                        help='Maximum pre-treatment minutes to include')
    parser.add_argument('--output', default='experiments/lp_results',
                        help='Output directory for LP results')
    args = parser.parse_args()

    exps = collect_experiments(args.experiments_root)
    if not exps:
        print("No experiments found in", args.experiments_root)
        return 1

    print(f"Found {len(exps)} experiments:")
    all_panels = []
    for exp in exps:
        m = exp['manifest']
        print(f"  {exp['dir']}: date={m['date']}, seed={m['seed']}, "
              f"events={m['n_events']}")

        panel = build_lp_panel(exp, max_pre=args.max_pre,
                               max_post=args.max_horizon)
        if len(panel) > 0:
            all_panels.append(panel)
        else:
            print(f"    WARNING: empty panel (missing data files?)")

    if not all_panels:
        print("No valid panels constructed.")
        return 1

    combined = pd.concat(all_panels, ignore_index=True)
    print(f"\nCombined panel: {len(combined):,} rows, "
          f"grids={combined['grid_id'].nunique()}, "
          f"events={combined['event_id'].nunique()}, "
          f"seeds={combined['seed'].nunique()}, "
          f"dates={combined['date'].nunique()}")
    print(f"  Zones: {combined['zone'].value_counts().to_dict()}")

    outcome = 'gbv_diff'
    if outcome not in combined.columns:
        print(f"ERROR: outcome column '{outcome}' not found")
        return 1

    # ---- LP: per-zone impulse response ----
    os.makedirs(args.output, exist_ok=True)
    all_irfs = {}

    for zone in ZONE_ORDER:
        zone_panel = combined[combined['zone'] == zone]
        n_grids = zone_panel['grid_id'].nunique()
        n_events = zone_panel['event_id'].nunique()
        print(f"\n{'='*60}")
        print(f"LP for zone: {zone} ({n_grids} grids, {n_events} events, "
              f"{len(zone_panel):,} obs)")

        if len(zone_panel) < 100 or zone_panel['D'].sum() == 0:
            print(f"  SKIP: insufficient data or no treatment observations")
            continue

        irf = run_lp(zone_panel, outcome_col=outcome,
                     max_horizon=args.max_horizon)
        all_irfs[zone] = irf

        h_star = identify_h_star(irf)
        if h_star['significant']:
            print(f"  H* = {h_star['h_star']} min, β = {h_star['beta_at_h_star']:.4f}")
        else:
            print(f"  H* = not significant")

        # Print IRF summary
        print(f"  {'h':>4s} {'β':>10s} {'SE':>10s} {'t':>8s} {'p':>8s} {'n':>6s}")
        for _, row in irf.iterrows():
            sig = '*' if row['p_value'] < 0.05 else ' '
            print(f"  {int(row['horizon']):4d} {row['beta']:>10.4f} {row['se']:>10.4f} "
                  f"{row['t_stat']:>8.2f} {row['p_value']:>8.4f} {int(row['n_obs']):>6d}{sig}")

        irf.to_csv(os.path.join(args.output, f'irf_{zone}.csv'), index=False)

    # ---- Spatial spillover identification (R*) ----
    print(f"\n{'='*60}")
    print("Spatial spillover (R*) identification")
    print(f"{'='*60}")
    for zone in ['spillover_r1', 'spillover_r2']:
        if zone not in all_irfs:
            continue
        irf = all_irfs[zone]
        h_star = identify_h_star(irf)
        print(f"  {zone}: H*={h_star['h_star']} min, "
              f"β={h_star['beta_at_h_star']:.4f}, "
              f"significant={h_star['significant']}")

    # ---- Per-joint-action IRF (treated_core only) ----
    print(f"\n{'='*60}")
    print("Per-joint-action IRF (treated_core)")
    print(f"{'='*60}")
    tc_panel = combined[combined['zone'] == 'treated_core']
    joint_actions = (tc_panel[['price_multiplier', 'matching_radius_km']]
                     .drop_duplicates().sort_values(
                         ['price_multiplier', 'matching_radius_km']))
    for price, matching_radius in joint_actions.itertuples(index=False, name=None):
        a_panel = tc_panel[
            (tc_panel['price_multiplier'] == price)
            & (tc_panel['matching_radius_km'] == matching_radius)]
        if len(a_panel) < 100:
            continue
        irf = run_lp(a_panel, outcome_col=outcome, max_horizon=args.max_horizon)
        h_star = identify_h_star(irf)
        print(f"  action=(price={price:g}, radius={matching_radius:g}km): "
              f"H*={h_star['h_star']}, "
              f"β(h*)={h_star['beta_at_h_star']:.4f}, "
              f"significant={h_star['significant']}")
        irf.to_csv(os.path.join(
            args.output,
            f'irf_price_{price:g}_radius_{matching_radius:g}.csv'),
            index=False)

    # ---- Save combined panel for external analysis ----
    combined.to_csv(os.path.join(args.output, 'lp_panel.csv'), index=False)
    print(f"\nLP results saved to {args.output}/")
    print(f"  Files: irf_treated_core.csv, irf_spillover_r1.csv, "
          f"irf_spillover_r2.csv, lp_panel.csv")
    return 0


if __name__ == '__main__':
    sys.exit(main())
