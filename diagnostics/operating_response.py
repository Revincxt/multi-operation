"""测试价格、匹配半径及实际接驾距离与仿真指标的关系。

使用方式（项目根目录）:
    python3 -m diagnostics.operating_response --output experiments/operating_response

固定其他条件，分别扫描价格和半径。实际距离按派单记录分组，
包含拒单记录；它是观察性关系，不是对真实距离的随机干预。
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from data_pipeline.collect_trajectories import check_collection_inputs, simulation_window
from src.env.actions import DEFAULT_MATCHING_RADII_KM, DEFAULT_PRICE_MULTIPLIERS, uniform_grid_actions
from src.env.simulator_env import Simulator
from src.env.simulator_pattern import SimulatorPattern


COUNTS = (
    "candidate_request_num", "matched_request_num", "baseline_request_num",
    "submitted_request_num", "driver_offer_num", "driver_accept_num",
    "unmatched_cancel_num", "pickup_cancel_num", "platform_revenue", "gbv",
)
METRICS = ("order_completion_rate", "driver_accept_rate", "platform_revenue")
OFFER_COLUMNS = (
    "minute_step", "order_id", "driver_id", "origin_grid_id", "price_multiplier",
    "quoted_fare", "trip_distance", "pickup_distance_km", "driver_accepted",
    "matched", "platform_revenue",
)


class RecordedSimulator(Simulator):
    """只记录派单结果；不更改匹配、概率或随机数调用。"""

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.offer_records = []

    def update_info_after_matching_multi_process(self, pairs, distances):
        if not len(pairs):
            return super().update_info_after_matching_multi_process(pairs, distances)
        pair_table = pd.DataFrame(
            pairs, columns=["order_id", "driver_id", "weight", "pickup_distance_km"])
        if len(distances):
            pair_table["pickup_distance_km"] = np.asarray(distances, dtype=float)
        # 按 order_id 对齐，不假定匹配器与订单队列的排序相同。
        offers = pair_table.merge(self.wait_requests[[
            "order_id", "origin_grid_id", "price_multiplier", "quoted_fare",
            "trip_distance", "platform_revenue",
        ]], on="order_id", validate="one_to_one")
        event_start = len(self._step_events)
        matched, waiting = super().update_info_after_matching_multi_process(pairs, distances)
        events = self._step_events.iloc[event_start:]
        accepted_ids = events.loc[events["event"] == "driver_accept", "order_id"]
        offers["driver_accepted"] = offers["order_id"].isin(accepted_ids)
        offers["matched"] = offers["order_id"].isin(matched["order_id"])
        offers["platform_revenue"] *= offers["matched"].astype(int)
        offers["minute_step"] = self.current_step
        self.offer_records.append(offers[list(OFFER_COLUMNS)])
        return matched, waiting


def ratio(numerator, denominator):
    """无分母时保留缺失，不将无派单误画为零接单率。"""
    return float(numerator / denominator) if denominator else float("nan")


def summarize_metrics(metrics):
    """先汇总计数再相除，避免对稀疏 grid-minute 的比率直接取平均。"""
    result = {column: float(metrics[column].sum()) for column in COUNTS}
    result["order_completion_rate"] = ratio(
        result["matched_request_num"], result["candidate_request_num"])
    result["driver_accept_rate"] = ratio(
        result["driver_accept_num"], result["driver_offer_num"])
    result["request_rate"] = ratio(
        result["submitted_request_num"], result["baseline_request_num"])
    result["matched_per_submitted_in_metrics"] = ratio(
        result["matched_request_num"], result["submitted_request_num"])
    result["mean_pickup_distance_km"] = ratio(
        (metrics["mean_pickup_distance_km"] * metrics["matched_request_num"]).sum(),
        result["matched_request_num"])
    return result


def summarize_distances(offers, max_distance, bin_width):
    """实际距离分箱；成功率分母为派单次数，包含重复尝试与司机拒绝。"""
    edges = np.arange(0, max_distance + bin_width, bin_width)
    rows = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        # 首箱包含零；其余左开右闭，每条派单只进入一个分箱。
        lower = offers["pickup_distance_km"] >= left if index == 0 else (
            offers["pickup_distance_km"] > left)
        group = offers[lower & (offers["pickup_distance_km"] <= right)]
        count = len(group)
        rows.append({
            "distance_left_km": left, "distance_right_km": right,
            "distance_mid_km": (left + right) / 2,
            "driver_offer_num": count,
            "driver_accept_num": int(group["driver_accepted"].sum()),
            "matched_request_num": int(group["matched"].sum()),
            "offer_success_rate": ratio(group["matched"].sum(), count),
            "driver_accept_rate": ratio(group["driver_accepted"].sum(), count),
            "platform_revenue": float(group["platform_revenue"].sum()),
            "revenue_per_offer": ratio(group["platform_revenue"].sum(), count),
        })
    return pd.DataFrame(rows)


def scenario_pairs(prices, radii, reference_price, reference_radius):
    """单因素实验，两个扫描共用同一基准点。"""
    return sorted(set([(price, reference_radius) for price in prices]
                      + [(reference_price, radius) for radius in radii]))


def run_scenario(config, price, radius, seed):
    started = time.monotonic()
    pattern = SimulatorPattern(config["date"], data_dir=config["data_dir"])
    sim = RecordedSimulator(
        grid_num=config["grid_num"], driver_num=config["driver_num"], decision_freq=5,
        grid_action=uniform_grid_actions(config["grid_num"], price, radius),
        allowed_price_multipliers=[price], allowed_matching_radii_km=[radius],
        price_response_model="utility_choice", order_sample_ratio=1.0,
        base_fare_start=3.0, price_per_km=0.7, commission_rate=0.25,
        enable_reposition=config["enable_reposition"],
        reposition_adjacency_path=str(Path(config["data_dir"]) / "hexo_updated_r300_adj.csv"),
        reposition_grid_info_path=str(Path(config["data_dir"]) / "hex_updated_r300_info.csv"),
    )
    _, sim.t_initial, sim.t_end = simulation_window(
        config["start_time"], config["simulation_steps"])
    if (not pattern.request_all or sim.t_initial < min(pattern.request_all)
            or sim.t_end > max(pattern.request_all) + 1):
        raise ValueError
    sim.finish_run_step = config["simulation_steps"]
    sim.reset(seed=seed, given_data=True,
              request_databases=pattern.request_all, driver_info=pattern.driver_info)
    if config["grid_num"] == 263:
        sim.mapping_dict = {config["date"]: SimulatorPattern.load_mapping(
            config["date"], data_dir=config["data_dir"])}
    sim.experiment_date = config["date"]
    while not sim.end_of_episode:
        sim.step()

    metrics = pd.DataFrame(sim.evaluate_table.reshape(-1, len(sim.evaluate_df.columns)),
                           columns=sim.evaluate_df.columns)
    offers = (pd.concat(sim.offer_records, ignore_index=True) if sim.offer_records
              else pd.DataFrame(columns=OFFER_COLUMNS))
    summary = {
        "price_multiplier": price, "matching_radius_km": radius, "seed": seed,
        **summarize_metrics(metrics),
        "end_queue_num": len(sim.wait_requests),
        "generated_submitted_num": sim.accepted_quote_num,
        "elapsed_seconds": time.monotonic() - started,
    }
    # 记录器必须与仿真器原有计数及收益完全一致。
    np.testing.assert_equal(len(offers), summary["driver_offer_num"])
    np.testing.assert_equal(offers["driver_accepted"].sum(), summary["driver_accept_num"])
    np.testing.assert_equal(offers["matched"].sum(), summary["matched_request_num"])
    np.testing.assert_allclose(offers["platform_revenue"].sum(), summary["platform_revenue"])
    np.testing.assert_allclose(summary["gbv"], sim.total_reward)
    if len(offers) and (offers["pickup_distance_km"] > radius + 1e-9).any():
        raise AssertionError

    run_dir = Path(config["output"]) / "runs" / f"price{price:g}_radius{radius:g}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    offers.to_csv(run_dir / "offers.csv.gz", index=False)
    # 保留分钟级全城计数，便于复核时间边界及分母。
    metrics["minute_step"] = np.repeat(np.arange(config["simulation_steps"]), config["grid_num"])
    metrics.groupby("minute_step")[list(COUNTS)].sum().to_csv(run_dir / "minute_counts.csv")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    (run_dir / "response_parameters.json").write_text(json.dumps(
        asdict(sim.passenger_response_model.config), indent=2))
    bins = summarize_distances(offers, max(config["radii"]), config["bin_width"])
    for key in ("price_multiplier", "matching_radius_km", "seed"):
        bins[key] = summary[key]
    bins.to_csv(run_dir / "distance_bins.csv", index=False)
    return summary


def plot_results(output):
    """图中误差带为跨 seed 的 ±1 个标准差，不是置信区间。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output = Path(output)
    config = json.loads((output / "manifest.json").read_text())
    run_dirs = sorted((output / "runs").iterdir())
    frame = pd.DataFrame([json.loads((path / "summary.json").read_text()) for path in run_dirs])
    frame = frame.sort_values(["price_multiplier", "matching_radius_km", "seed"])
    frame.to_csv(output / "run_summary.csv", index=False)
    averaged = frame.groupby(["price_multiplier", "matching_radius_km"])[list(METRICS)].agg(["mean", "std"])
    averaged.columns = ["_".join(column) for column in averaged.columns]
    averaged.to_csv(output / "scenario_summary.csv")
    bins = pd.concat([pd.read_csv(path / "distance_bins.csv") for path in run_dirs], ignore_index=True)
    bins.to_csv(output / "distance_bins_by_seed.csv", index=False)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def draw(ax, data, x, y, label, color, is_rate=False):
        for _, group in data.groupby("seed"):
            group = group.sort_values(x)
            ax.plot(group[x], group[y], color=color, alpha=0.18, linewidth=1)
        grouped = data.groupby(x)[y].agg(["mean", "std"]).sort_index()
        deviation = grouped["std"].fillna(0)
        ax.fill_between(grouped.index, grouped["mean"] - deviation,
                        grouped["mean"] + deviation, color=color, alpha=0.14)
        ax.plot(grouped.index, grouped["mean"], "o-", color=color, linewidth=2)
        ax.set_title(label)
        ax.grid(alpha=0.2)
        if is_rate:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(bottom=0, top=1.02)
        else:
            ax.set_ylim(bottom=0)

    labels = ("Order completion: matches / candidate attempts",
              "Driver acceptance: accepts / offers", "Platform revenue (USD / run)")
    for axis, fixed, value, title, color, filename in (
        ("price_multiplier", "matching_radius_km", config["reference_radius"],
         "Price multiplier", "#2266aa", "price_response.png"),
        ("matching_radius_km", "price_multiplier", config["reference_price"],
         "Matching radius limit (km)", "#10867a", "radius_response.png"),
    ):
        selected = frame[np.isclose(frame[fixed], value)]
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), layout="constrained")
        for ax, metric, label in zip(axes, METRICS, labels):
            draw(ax, selected, axis, metric, label, color, metric.endswith("rate"))
            ax.set_xlabel(title)
        fig.suptitle(f"Simulator response | {fixed} = {value:g} | {len(config['seeds'])} seeds\n"
                     f"{config['date']} {config['start_time']}, {config['simulation_steps']} minutes; "
                     "lines: seed means, bands: +/- 1 SD", fontsize=13)
        fig.savefig(output / filename, dpi=170)
        plt.close(fig)

    # 实际距离固定使用最大半径、基准价格；不混合不同价格与半径造成的样本选择。
    selected = bins[np.isclose(bins["price_multiplier"], config["reference_price"])
                    & np.isclose(bins["matching_radius_km"], max(config["radii"]))].copy()
    selected.to_csv(output / "actual_distance_summary.csv", index=False)
    for metric in ("offer_success_rate", "driver_accept_rate"):
        selected.loc[selected["driver_offer_num"] < 20, metric] = np.nan
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, metric, label in zip(axes.flat,
            ("offer_success_rate", "driver_accept_rate", "platform_revenue", "driver_offer_num"),
            ("Success conditional on a dispatch offer", "Driver acceptance: accepts / offers",
             "Revenue in distance bin (USD / run)", "Dispatch offers in distance bin")):
        draw(ax, selected, "distance_mid_km", metric, label, "#9260a6", metric.endswith("rate"))
        ax.set_xlabel("Actual pickup distance (km), bin midpoint")
    fig.suptitle(f"Observed pickup-distance relationship | price = {config['reference_price']:g}, "
                 f"radius = {max(config['radii']):g} km\n"
                 "Includes rejected offers; not a causal distance intervention. "
                 "Rates hidden for seed-bins with <20 offers.", fontsize=12)
    fig.savefig(output / "actual_distance_response.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/operating_response")
    parser.add_argument("--date", default="2015-05-18")
    parser.add_argument("--data-dir", default="my_data")
    parser.add_argument("--start-time", default="05:00")
    parser.add_argument("--simulation-steps", type=int, default=300)
    parser.add_argument("--grid-num", type=int, default=263)
    parser.add_argument("--driver-num", type=int, default=1000)
    parser.add_argument("--prices", type=float, nargs="+", default=list(DEFAULT_PRICE_MULTIPLIERS))
    parser.add_argument("--radii", type=float, nargs="+", default=list(DEFAULT_MATCHING_RADII_KM))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--reference-price", type=float, default=1.0)
    parser.add_argument("--reference-radius", type=float, default=1.0)
    parser.add_argument("--bin-width", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--enable-reposition", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot_results(args.output)
        return
    config = vars(args).copy()
    for values in (args.prices, args.radii, [args.reference_price, args.reference_radius, args.bin_width]):
        if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0):
            raise ValueError
    if args.workers <= 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError
    args.start_time, _, _ = simulation_window(args.start_time, args.simulation_steps)
    check_collection_inputs([args.date], args.grid_num, args.data_dir)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError
    output.mkdir(parents=True, exist_ok=True)
    # 留存源码指纹及统计口径，避免模型修改后误用旧曲线。
    config["source_sha256"] = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (
            "src/env/simulator_env.py", "src/env/price_response/utility_choice.py",
            "src/utils/utilities.py", "src/utils/dispatch_alg.py", "diagnostics/operating_response.py")}
    config["metric_definitions"] = {
        "order_completion_rate": "sum(matches) / sum(candidate attempts); retries repeat in denominator",
        "driver_accept_rate": "sum(driver accepts) / sum(driver offers)",
        "platform_revenue": "sum(retained matched fare * 0.25); not GBV or net profit",
        "offer_success_rate": "retained matches / offers within actual-distance bin; conditional on offer",
        "distance": "Haversine pickup distance, not road distance; bins include rejected offers",
        "time_boundary": "original simulator timing; no drain period; end-of-window orders may remain queued",
        "demand": "observed_requests; low-price demand resampling may yield submitted / baseline > 1",
        "uncertainty": "mean +/- sample SD across seeds, not confidence interval",
    }
    config["scope"] = "One-factor citywide fixed-action diagnostics; no policy or core simulation changes"
    (output / "manifest.json").write_text(json.dumps(config, indent=2))
    jobs = [(price, radius, seed) for price, radius in scenario_pairs(
        args.prices, args.radii, args.reference_price, args.reference_radius) for seed in args.seeds]
    print(f"开始 {len(jobs)} 组仿真，{args.workers} 个进程", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_scenario, config, *job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            print(f"[{index}/{len(jobs)}] price={row['price_multiplier']:g}, "
                  f"radius={row['matching_radius_km']:g}, seed={row['seed']}: "
                  f"completion={row['order_completion_rate']:.3f}, "
                  f"acceptance={row['driver_accept_rate']:.3f}, "
                  f"revenue={row['platform_revenue']:.2f}, {row['elapsed_seconds']:.1f}s", flush=True)
    plot_results(output)
    print(f"曲线和明细已保存至 {output}", flush=True)


if __name__ == "__main__":
    main()
