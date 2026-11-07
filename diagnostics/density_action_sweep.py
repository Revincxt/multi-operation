"""三种密度场景下，固定价格乘子与匹配半径的筛选和独立种子复测。

使用方式:
    python3 -m diagnostics.density_action_sweep --workers 6

全城、全时段共用一个动作；只比较实际仿真收益，不训练网络。
完整扫描候选集合后冻结 shortlist，再用新种子复测，并始终保留 1.5 / 3 km 对照。
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from baseline.compare_policies import sample_common_orders, summarize_operation
from data_pipeline.collect_density import SCENARIOS, sha256, save_json
from data_pipeline.collect_trajectories import check_collection_inputs, simulation_window
from src.env.actions import DEFAULT_MATCHING_RADII_KM, DEFAULT_PRICE_MULTIPLIERS, uniform_grid_actions
from src.env.simulator_env import Simulator
from src.env.simulator_pattern import SimulatorPattern, input_data_paths


REFERENCE = (1.5, 3.0)  # price, radius
METRICS = ("platform_revenue", "order_completion_rate", "driver_accept_rate",
           "request_rate", "mean_wait_to_match_seconds", "matched_request_num")
KEYS = ["scenario", "price_multiplier", "matching_radius_km"]


@lru_cache(maxsize=1)
def load_pattern(date, data_dir):
    return SimulatorPattern(date, data_dir=data_dir)


@lru_cache(maxsize=1)
def common_inputs(date, data_dir, start, end, ratio, seed):
    """每个进程复用只读订单；候选动作不会影响外生需求抽样。"""
    pattern = load_pattern(date, data_dir)
    requests, sampled = sample_common_orders(pattern.request_all, start, end, ratio, seed)
    digest = hashlib.sha256(sampled.to_csv(index=False).encode()).hexdigest()
    return requests, len(sampled), digest


def run_action(config, phase, scenario, price, radius, seed):
    """跑完整时段，逐次保存计数、网格轨迹和源参数，避免中断丢失已完成实验。"""
    started = perf_counter()
    _, start, end = simulation_window(config["start_time"], config["simulation_steps"])
    settings = SCENARIOS[scenario]
    requests, count, order_hash = common_inputs(
        config["date"], config["data_dir"], start, end, settings["order_sample_ratio"], seed)
    pattern = load_pattern(config["date"], config["data_dir"])
    sim = Simulator(
        grid_num=6, driver_num=settings["driver_num"], decision_freq=5,
        grid_action=uniform_grid_actions(6, price, radius),
        allowed_price_multipliers=[price], allowed_matching_radii_km=[radius],
        price_response_model="utility_choice", order_sample_ratio=1.0,
        base_fare_start=3.0, price_per_km=0.7, commission_rate=0.25, enable_reposition=False)
    sim.t_initial, sim.t_end = start, end
    sim.finish_run_step = config["simulation_steps"]
    sim.reset(seed=seed, given_data=True, request_databases=requests, driver_info=pattern.driver_info)
    sim.experiment_date = config["date"]
    driver_hash = hashlib.sha256(sim.driver_table.to_csv(index=False).encode()).hexdigest()
    while not sim.end_of_episode:
        sim.step()
    minutes = pd.DataFrame(sim.evaluate_table.reshape(-1, len(sim.evaluate_df.columns)),
                           columns=sim.evaluate_df.columns)
    minutes.insert(0, "minute_step", np.repeat(np.arange(config["simulation_steps"]), 6))
    if (sim.baseline_request_num != count
            or not np.isclose(minutes.action_price_multiplier, price).all()
            or not np.isclose(minutes.matching_radius_km, radius).all()):
        raise AssertionError
    summary = {
        "phase": phase, "scenario": scenario, "seed": seed,
        "price_multiplier": price, "matching_radius_km": radius, **settings,
        "generated_baseline_requests": count, "sampled_orders_sha256": order_hash,
        "initial_drivers_sha256": driver_hash, **summarize_operation(minutes),
        "end_queue_num": len(sim.wait_requests), "elapsed_seconds": perf_counter() - started,
    }
    np.testing.assert_allclose(summary["platform_revenue"], sim.total_reward * sim.commission_rate)
    np.testing.assert_equal(summary["matched_request_num"], sim.matched_requests_num)
    root = Path(config["output"]) / "runs" / phase / scenario / f"p{price:g}_r{radius:g}_seed{seed}"
    root.mkdir(parents=True, exist_ok=False)
    minutes.to_csv(root / "grid_minute_metrics.csv.gz", index=False)
    save_json(root / "summary.json", summary)
    save_json(root / "response_parameters.json", asdict(sim.passenger_response_model.config))
    return summary


def check_paired_inputs(frame):
    """同一密度同一 seed 的每个候选必须共享订单及司机初态。"""
    if frame.duplicated(KEYS + ["seed"]).any():
        raise ValueError
    for _, group in frame.groupby(["scenario", "seed"]):
        for column in ("sampled_orders_sha256", "initial_drivers_sha256",
                       "generated_baseline_requests", "baseline_request_num"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError


def rank_actions(frame):
    """每个组合的独立 run 等权平均，不混用筛选与复测结果。"""
    if frame.empty or frame.phase.nunique() != 1:
        raise ValueError
    check_paired_inputs(frame)
    # 不允许用不同数量或不同 seed 的结果进行候选排名。
    for _, group in frame.groupby("scenario"):
        seed_sets = group.groupby(KEYS[1:])["seed"].agg(lambda values: tuple(sorted(values)))
        if len(set(seed_sets)) != 1:
            raise ValueError
    result = frame.groupby(KEYS)[list(METRICS)].agg(["mean", "std"])
    result.columns = ["_".join(column) for column in result.columns]
    result = result.reset_index()
    result["seed_count"] = frame.groupby(KEYS).size().to_numpy()
    result = result.sort_values(["scenario", "platform_revenue_mean", "price_multiplier", "matching_radius_km"],
                                ascending=[True, False, True, True]).reset_index(drop=True)
    result["rank"] = result.groupby("scenario").cumcount() + 1
    return result


def select_shortlist(screening, top_k):
    """按筛选收益冻结候选；固定对照即使落榜，也独立保留。"""
    if top_k < 1 or set(screening.phase) != {"screening"}:
        raise ValueError
    ranked = rank_actions(screening)
    result = {}
    for scenario, group in ranked.groupby("scenario", sort=False):
        pairs = list(group.head(top_k)[KEYS[1:]].itertuples(index=False, name=None))
        if REFERENCE not in pairs:
            pairs.append(REFERENCE)
        result[scenario] = pairs
    return result


def paired_comparisons(frame, bootstrap_repeats=2000):
    """按 seed 配对收益差；区间是逐组合 bootstrap，未做多重比较校正。"""
    if set(frame.phase) != {"confirmation"}:
        raise ValueError
    rows = []
    for scenario, group in frame.groupby("scenario"):
        reference = group[np.isclose(group.price_multiplier, REFERENCE[0])
                          & np.isclose(group.matching_radius_km, REFERENCE[1])].set_index("seed")
        for (price, radius), candidate in group.groupby(KEYS[1:]):
            candidate = candidate.set_index("seed").sort_index()
            if set(candidate.index) != set(reference.index):
                raise ValueError
            base = reference.loc[candidate.index, "platform_revenue"].to_numpy()
            values = candidate.platform_revenue.to_numpy()
            differences = values - base
            rng = np.random.default_rng(42)
            samples = rng.choice(differences, (bootstrap_repeats, len(differences)), replace=True).mean(1)
            lower, upper = np.quantile(samples, [0.025, 0.975])
            rows.append({"scenario": scenario, "price_multiplier": price, "matching_radius_km": radius,
                         "paired_seed_count": len(differences), "mean_revenue_difference": differences.mean(),
                         "revenue_change_percent": 100 * (values.mean() / base.mean() - 1) if base.mean() else np.nan,
                         "bootstrap_95_low": lower if len(differences) > 1 else np.nan,
                         "bootstrap_95_high": upper if len(differences) > 1 else np.nan,
                         "seeds_better_than_reference": int((differences > 0).sum())})
    return pd.DataFrame(rows)


def plot_results(output):
    """热力图只描述筛选；前几名复测图同时标注收益 ↑ 和等待时间 ↓。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output = Path(output)
    screen = pd.read_csv(output / "screening_rankings.csv")
    ranked = pd.read_csv(output / "confirmation_rankings.csv")
    labels = {"low": "Small: ~5k requests / 250 drivers",
              "medium": "Medium: ~10k requests / 500 drivers",
              "high": "Large: ~25k requests / 1000 drivers"}
    manifest = json.loads((output / "manifest.json").read_text())
    scenarios = manifest["config"]["scenarios"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 7),
                                 squeeze=False, layout="constrained")
        for ax, scenario in zip(axes.flat, scenarios):
            table = screen[screen.scenario == scenario].pivot(
                index="price_multiplier", columns="matching_radius_km", values="platform_revenue_mean")
            im = ax.imshow(table.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(table.columns)), [f"{v:g}" for v in table.columns])
            ax.set_yticks(range(len(table.index)), [f"{v:g}" for v in table.index])
            ax.set(title=labels[scenario], xlabel="Matching radius (km)", ylabel="Price multiplier")
            fig.colorbar(im, ax=ax, shrink=0.85, label="Platform revenue (USD) ↑ Higher is better")
        fig.suptitle("Screening only: complete candidate grid | Not independently confirmed for every cell")
        fig.savefig(output / "screening_revenue_heatmaps.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(len(scenarios), 4, figsize=(20, 4.5 * len(scenarios)),
                                 squeeze=False, layout="constrained")
        metrics = (("platform_revenue", "Platform revenue (USD) ↑ Higher is better"),
                   ("order_completion_rate", "Order completion rate ↑ Higher is better"),
                   ("driver_accept_rate", "Driver acceptance rate ↑ Higher is better"),
                   ("mean_wait_to_match_seconds", "Matched-passenger wait (s) ↓ Lower is better"))
        for row, scenario in enumerate(scenarios):
            data = ranked[ranked.scenario == scenario].sort_values("rank").head(5)
            for ax, (metric, title) in zip(axes[row], metrics):
                ticks = [f"p={r.price_multiplier:g}\nr={r.matching_radius_km:g} km" for r in data.itertuples()]
                means, std = data[f"{metric}_mean"], data[f"{metric}_std"].fillna(0)
                ax.bar(ticks, means, yerr=std, capsize=3, color="#2563eb")
                ax.set_title(title, fontsize=10)
                ax.set_ylim(0, max(float((means + std).max()), 1e-9) * 1.25)
                ax.grid(axis="y", alpha=0.2)
                ax.set_axisbelow(True)
                if metric.endswith("rate"):
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
            axes[row, 0].set_ylabel(labels[scenario])
        fig.suptitle(f"Shortlisted actions on {len(manifest['config']['confirmation_seeds'])} new paired seeds | "
                     "Left-to-right: mean revenue rank\nError bars: ±1 seed SD, not confidence intervals; "
                     "full 05:00–10:00 window, no warm-up exclusion")
        fig.savefig(output / "confirmed_top_actions.png", dpi=160)
        plt.close(fig)


def run_sweep(output, workers=6, scenarios=None, screening_seeds=None, confirmation_seeds=None,
              prices=None, radii=None, top_k=5, simulation_steps=300):
    """固定配置后执行两阶段实验，逐次落盘，不改训练或采集任务的状态。"""
    config = {
        "date": "2015-05-18", "start_time": "05:00", "simulation_steps": simulation_steps,
        "data_dir": str(Path("my_data/grid6").resolve()), "output": str(Path(output).resolve()),
        "scenarios": list(SCENARIOS) if scenarios is None else scenarios,
        "screening_seeds": [50001] if screening_seeds is None else screening_seeds,
        "confirmation_seeds": list(range(51001, 51006)) if confirmation_seeds is None else confirmation_seeds,
        "prices": sorted(set(DEFAULT_PRICE_MULTIPLIERS) | {REFERENCE[0]}) if prices is None else prices,
        "radii": list(DEFAULT_MATCHING_RADII_KM) if radii is None else radii,
        "top_k": top_k, "workers": workers,
    }
    if (workers < 1 or top_k < 1 or not config["scenarios"]
            or not set(config["scenarios"]).issubset(SCENARIOS)
            or len(set(config["scenarios"])) != len(config["scenarios"])):
        raise ValueError
    for key in ("prices", "radii"):
        values = config[key]
        if not values or len(set(values)) != len(values) or not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError
    for key in ("screening_seeds", "confirmation_seeds"):
        values = config[key]
        if not values or len(set(values)) != len(values) or min(values) < 0 or max(values) >= 2**32:
            raise ValueError
    if set(config["screening_seeds"]) & set(config["confirmation_seeds"]):
        raise ValueError
    check_collection_inputs([config["date"]], 6, config["data_dir"])
    _, start, end = simulation_window(config["start_time"], simulation_steps)
    pattern = load_pattern(config["date"], config["data_dir"])
    if start < min(pattern.request_all) or end > max(pattern.request_all) + 1:
        raise ValueError
    counts = [sum(len(pattern.request_all.get(t, [])) for t in range(minute, minute + 60))
              for minute in range(start, end - 60, 60)]
    inputs = list(input_data_paths(config["date"], config["data_dir"])) + [Path(config["data_dir"]) / "grid_manifest.json"]
    project = Path(__file__).resolve().parents[1]
    sources = sorted((project / "src").rglob("*.py")) + [Path(__file__),
        project / "baseline/compare_policies.py", project / "data_pipeline/collect_density.py"]
    report = {
        "schema_version": "density_fixed_action_sweep_v1", "status": "screening", "config": config,
        "created_at": datetime.now(timezone.utc).isoformat(), "scope": "citywide time-invariant actions; maximize platform revenue",
        "scenario_settings": {name: {**SCENARIOS[name], "actual_sampled_orders":
            sum(int(np.rint(SCENARIOS[name]['order_sample_ratio'] * count)) for count in counts)} for name in config["scenarios"]},
        "input_sha256": {str(p): sha256(p) for p in inputs},
        "code_sha256": {str(p.relative_to(project)): sha256(p) for p in sources},
        "metric_definitions": {
            "order_completion_rate": "sum(retained matches) / sum(candidate attempts + unmatched cancellations); retries repeat",
            "driver_accept_rate": "sum(driver accepts) / sum(driver offers)",
            "mean_wait_to_match_seconds": "matched-count weighted queue wait; excludes unserved passengers and pickup time",
            "platform_revenue": "sum(retained matched fare * 0.25), not net profit",
        },
        "limitations": ["单日期、同一时段；结论仅针对测试过的固定组合，不代表动态联合最优。",
                        "筛选仅有限 seed，shortlist 之外可能有漏选；复测排序也不等于精确全局最优。",
                        "复测种子与筛选独立；复测仍用于报告排序，并非另有最终无偏测试集。",
                        "同 seed 固定外生订单和司机初态；策略分叉后行为随机数路径不同。",
                        "期末不清空积压订单；统计完整 300 分钟，不与排除预热的旧表直接对比。",
                        "不同密度的司机订单比也不同；结果不单独识别需求密度的因果效应。"],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "manifest.json", report)
    started = perf_counter()

    def execute(pool, phase, jobs):
        rows = []
        futures = [pool.submit(run_action, config, phase, *job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(f"{phase} [{index}/{len(jobs)}] {row['scenario']} p={row['price_multiplier']:g}, "
                  f"r={row['matching_radius_km']:g}, seed={row['seed']}: "
                  f"revenue={row['platform_revenue']:.2f}, {row['elapsed_seconds']:.1f}s", flush=True)
            save_json(output / "progress.json", {"phase": phase, "completed": index, "total": len(jobs),
                "elapsed_seconds": perf_counter() - started, "updated_at": datetime.now(timezone.utc).isoformat()})
        return pd.DataFrame(rows)

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [(scenario, price, radius, seed) for seed in config["screening_seeds"]
                    for price in config["prices"] for radius in config["radii"] for scenario in config["scenarios"]]
            print(f"完整筛选 {len(jobs)} 次仿真，{workers} 个进程；随后独立种子复测。", flush=True)
            print(report["scenario_settings"], flush=True)
            screening = execute(pool, "screening", jobs)
            screening.to_csv(output / "screening_runs.csv", index=False)
            rank_actions(screening).to_csv(output / "screening_rankings.csv", index=False)
            shortlist = select_shortlist(screening, top_k)
            report.update(status="confirmation", shortlist=shortlist)
            save_json(output / "manifest.json", report)
            jobs = [(scenario, price, radius, seed) for scenario, pairs in shortlist.items()
                    for seed in config["confirmation_seeds"] for price, radius in pairs]
            confirmation = execute(pool, "confirmation", jobs)
        confirmation.to_csv(output / "confirmation_runs.csv", index=False)
        ranked = rank_actions(confirmation)
        ranked.to_csv(output / "confirmation_rankings.csv", index=False)
        paired_comparisons(confirmation).to_csv(output / "paired_vs_reference.csv", index=False)
        plot_results(output)
        report.update(status="complete", elapsed_seconds=perf_counter() - started,
                      completed_at=datetime.now(timezone.utc).isoformat())
        save_json(output / "manifest.json", report)
        save_json(output / "progress.json", {"phase": "complete", "completed": len(screening) + len(confirmation),
                                            "elapsed_seconds": perf_counter() - started})
        print(ranked.groupby("scenario", sort=False).head(3).to_string(index=False), flush=True)
        return report
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__)
        save_json(output / "manifest.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description="三种规模下固定动作组合的仿真筛选与复测")
    parser.add_argument("--output", default="experiments/density_action_sweep")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--scenarios", choices=list(SCENARIOS), nargs="+")
    parser.add_argument("--screening-seeds", type=int, nargs="+")
    parser.add_argument("--confirmation-seeds", type=int, nargs="+")
    parser.add_argument("--prices", type=float, nargs="+")
    parser.add_argument("--radii", type=float, nargs="+")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--simulation-steps", type=int, default=300)
    run_sweep(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
