"""在仿真器中比较固定规则与预测收益最大化的 LSTM / TimesNet 策略。

使用方式:
    python3 -m baseline.compare_policies --model-results MODEL_COMPARISON_DIR --runs 10 --workers 2

固定规则：所有网格使用 3 km 半径、1.5 价格乘子。
学习策略：共同预热后，每周期逐网格枚举训练候选动作，选择预测收益最大者。
此处是单周期贪心策略，不是多步规划，也不是全网格联合最优求解。
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from baseline.data import DEFAULT_COLLECTION
from baseline.lstm import LSTMPredictor
from baseline.timesnet import TimesNetPredictor
from baseline.sequence import CONTEXT_COLUMNS
from data_pipeline.build_supervised_dataset import aggregate_decision_metrics
from data_pipeline.collect_density import make_config, sha256, validate_execution_revision
from data_pipeline.collect_trajectories import _build_minute_metrics, simulation_window
from src.env.actions import GridAction, uniform_grid_actions
from src.env.simulator_env import Simulator
from src.env.simulator_pattern import SimulatorPattern
from src.env.observation import (
    LOCAL_STATE_COLUMNS, STATE_COLUMNS, capture_decision_state, load_neighbor_adjacency,
    state_columns, state_metadata,
)
from src.utils.joint_context import GLOBAL_STATE_COLUMNS, joint_schema


POLICIES = ("fixed", "lstm", "timesnet")
RANDOM_ACTION_SEED_NAMESPACE = 937
METRICS = {
    "baseline_request_num": ("Bubble requests (input demand)", "= Controlled input; no better direction"),
    "request_rate": ("Request rate", "↑ Higher is better (service metric)"),
    "driver_accept_rate": ("Driver acceptance rate", "↑ Higher is better"),
    "order_completion_rate": ("Order completion rate", "↑ Higher is better"),
    "mean_wait_to_match_seconds": ("Mean matched-passenger wait (s)", "↓ Lower is better"),
    "platform_revenue": ("Platform revenue (USD)", "↑ Higher is better — optimization objective"),
}


def sample_common_orders(requests, start, end, ratio, seed):
    """先固定外生订单，避免策略消耗随机数不同导致抽到不同原始订单。"""
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError
    rng = np.random.default_rng(np.random.SeedSequence([seed, 731]))
    sampled, rows = {}, []
    # 沿用原仿真窗口：末分钟生成的订单来自 [end-120, end-60)。
    for minute in range(start, end - 60, 60):
        pool = [(timestamp, order) for timestamp in range(minute, minute + 60)
                for order in requests.get(timestamp, [])]
        count = int(np.rint(ratio * len(pool)))
        indices = (np.arange(len(pool)) if count == len(pool)
                   else np.sort(rng.choice(len(pool), count, replace=False)))
        for index in indices:
            timestamp, order = pool[index]
            sampled.setdefault(timestamp, []).append(order)
            rows.append({"timestamp": timestamp, "order_id": order[0]})
    return sampled, pd.DataFrame(rows, columns=["timestamp", "order_id"])


def candidate_features(periods, preprocessor, decision_step, start_time, interval, prices, radii,
                       current_state=None):
    """只读已完成的历史周期，不读取 t 周期标签或未来订单。"""
    history_steps = preprocessor.history_steps
    if decision_step < history_steps or periods["decision_step"].max() != decision_step - 1:
        raise ValueError
    recent = periods[periods.decision_step >= decision_step - history_steps]
    if recent.duplicated(["grid_id", "decision_step"]).any():
        raise ValueError
    current_columns = state_columns(getattr(preprocessor, "context_columns", CONTEXT_COLUMNS))
    if current_columns:
        if (current_state is None or current_state.grid_id.duplicated().any()
                or not set(preprocessor.grid_ids).issubset(current_state.grid_id)
                or state_columns(current_state.columns) != current_columns):
            raise ValueError
        state_by_grid = current_state.set_index("grid_id")
    records = []
    for grid in preprocessor.grid_ids:
        group = recent[recent.grid_id == grid].set_index("decision_step")
        if set(group.index) != set(range(decision_step - history_steps, decision_step)):
            raise ValueError
        row = {"grid_id": grid, "decision_start_sim_time": start_time,
               "interval_minutes": interval}
        if current_columns:
            row.update(state_by_grid.loc[grid, current_columns].to_dict())
        for lag in range(1, history_steps + 1):
            row.update({f"lag_{lag}_{feature}": group.loc[decision_step - lag, feature]
                        for feature in preprocessor.history_features})
        for price in prices:
            for radius in radii:
                records.append({**row, "price_multiplier": price, "matching_radius_km": radius})
    return pd.DataFrame(records)


def revenue_actions(predictor, periods, decision_step, start_time, interval, prices, radii,
                    current_state=None):
    """同一份历史下比较候选动作；并列时按候选集合原顺序选择。"""
    candidates = candidate_features(periods, predictor.preprocessor, decision_step,
                                    start_time, interval, prices, radii, current_state)
    predictions = predictor.predict(candidates)
    revenue = predictions["target_platform_revenue"].to_numpy()
    if not np.isfinite(revenue).all():
        raise ValueError
    candidates["predicted_platform_revenue"] = revenue
    selected = candidates.loc[candidates.groupby("grid_id", sort=True)[
        "predicted_platform_revenue"].idxmax()]
    actions = {int(row.grid_id): GridAction(row.price_multiplier, row.matching_radius_km)
               for row in selected.itertuples()}
    return actions, selected.set_index("grid_id")["predicted_platform_revenue"].to_dict()


def _joint_candidate_frame(periods, preprocessor, decision_step, start_time, interval,
                           current_state):
    """为一组完整的全网格动作向量构造六网格预测输入。"""
    context = list(preprocessor.context_columns)
    grid_ids, global_columns, action_columns = joint_schema(context[len(CONTEXT_COLUMNS):])
    if not action_columns or list(grid_ids) != list(preprocessor.grid_ids):
        raise ValueError
    if (decision_step < preprocessor.history_steps
            or periods.empty or periods["decision_step"].max() != decision_step - 1
            or periods.duplicated(["grid_id", "decision_step"]).any()):
        raise ValueError
    recent = periods[periods.decision_step >= decision_step - preprocessor.history_steps]
    current_columns = state_columns(context)
    state_by_grid = None
    if current_columns:
        if (current_state is None or current_state.grid_id.duplicated().any()
                or set(current_state.grid_id.astype(int)) != set(grid_ids)
                or state_columns(current_state.columns) != current_columns):
            raise ValueError
        state_by_grid = current_state.set_index("grid_id")

    global_values = {}
    if global_columns:
        if current_columns != list(STATE_COLUMNS):
            raise ValueError
        global_values = dict(zip(
            GLOBAL_STATE_COLUMNS,
            current_state[list(LOCAL_STATE_COLUMNS)].sum().to_numpy(dtype=float)))

    rows = []
    for grid in grid_ids:
        history = recent[recent.grid_id == grid].set_index("decision_step")
        if set(history.index) != set(range(decision_step - preprocessor.history_steps,
                                           decision_step)):
            raise ValueError
        row = {"grid_id": grid, "decision_start_sim_time": start_time,
               "interval_minutes": interval, **global_values}
        if current_columns:
            row.update(state_by_grid.loc[grid, current_columns].to_dict())
        for lag in range(1, preprocessor.history_steps + 1):
            row.update({f"lag_{lag}_{feature}":
                        history.loc[decision_step - lag, feature]
                        for feature in preprocessor.history_features})
        rows.append(row)
    return rows, grid_ids, action_columns


def _score_joint_actions(predictor, rows, grid_ids, action_columns, action_vectors):
    """对候选联合动作预测所有网格收益，并按全城总收益打分。"""
    candidates = []
    for candidate_id, actions in enumerate(action_vectors):
        if set(actions) != set(grid_ids):
            raise ValueError
        vector = {}
        for grid in grid_ids:
            action = actions[grid]
            vector[f"joint_price_grid_{grid}"] = action.price_multiplier
            vector[f"joint_radius_grid_{grid}"] = action.matching_radius_km
        if set(vector) != set(action_columns):
            raise ValueError
        for grid, row in zip(grid_ids, rows):
            candidates.append({**row, **vector, "price_multiplier": actions[grid].price_multiplier,
                               "matching_radius_km": actions[grid].matching_radius_km,
                               "_candidate_id": candidate_id})
    frame = pd.DataFrame(candidates)
    predictions = predictor.predict(frame)
    revenue = predictions["target_platform_revenue"].to_numpy(dtype=float)
    if not np.isfinite(revenue).all():
        raise ValueError
    frame["_predicted_revenue"] = revenue
    totals = frame.groupby("_candidate_id", sort=True)["_predicted_revenue"].sum().to_numpy()
    by_grid = frame.loc[frame["_candidate_id"] == len(action_vectors) - 1].set_index(
        "grid_id")["_predicted_revenue"].to_dict()
    return totals, by_grid, len(frame)


def joint_action_search(predictor, periods, decision_step, start_time, interval, prices, radii,
                        current_state, initial_actions=None, max_sweeps=5):
    """按全城预测收益做逐网格坐标搜索；最多搜索 max_sweeps 轮。"""
    if (len(prices) == 0 or len(radii) == 0 or max_sweeps <= 0
            or not np.isfinite(np.asarray(prices, dtype=float)).all()
            or not np.isfinite(np.asarray(radii, dtype=float)).all()
            or min(prices) <= 0 or min(radii) <= 0):
        raise ValueError
    rows, grid_ids, action_columns = _joint_candidate_frame(
        periods, predictor.preprocessor, decision_step, start_time, interval,
        current_state)
    if initial_actions is None:
        first = GridAction(prices[0], radii[0])
        actions = {grid: first for grid in grid_ids}
    else:
        actions = {int(grid): action for grid, action in initial_actions.items()}
        if set(actions) != set(grid_ids):
            raise ValueError

    action_pairs = [(price, radius) for price in prices for radius in radii]
    evaluations = 0
    converged = False
    sweeps = 0
    for sweep in range(max_sweeps):
        before = {grid: (actions[grid].price_multiplier, actions[grid].matching_radius_km)
                  for grid in grid_ids}
        sweeps += 1
        for grid in grid_ids:
            pairs = list(action_pairs)
            current = (actions[grid].price_multiplier, actions[grid].matching_radius_km)
            current_index = pairs.index(current) if current in pairs else None
            vectors = []
            for price, radius in pairs:
                candidate = dict(actions)
                candidate[grid] = GridAction(price, radius)
                vectors.append(candidate)
            totals, _, evaluated = _score_joint_actions(
                predictor, rows, grid_ids, action_columns, vectors)
            evaluations += evaluated
            best = float(np.max(totals))
            tied = np.flatnonzero(np.isclose(totals, best, rtol=1e-10, atol=1e-10))
            selected = (current_index if current_index is not None and current_index in tied
                        else int(tied[0]))
            actions = vectors[selected]
        after = {grid: (actions[grid].price_multiplier, actions[grid].matching_radius_km)
                 for grid in grid_ids}
        if after == before:
            converged = True
            break

    totals, revenue_by_grid, evaluated = _score_joint_actions(
        predictor, rows, grid_ids, action_columns, [actions])
    evaluations += evaluated
    diagnostics = {"sweeps": sweeps, "converged": converged,
                   "joint_candidates_evaluated": evaluations,
                   "predicted_total_platform_revenue": float(totals[0])}
    return actions, revenue_by_grid, diagnostics


def random_grid_actions(rng, grid_num, prices, radii):
    """各网格独立、均匀、有放回地抽取联合动作，不使用仿真器的随机流。"""
    if grid_num <= 0 or not len(prices) or not len(radii):
        raise ValueError
    indices = rng.integers(len(prices) * len(radii), size=grid_num)
    return {grid: GridAction(prices[index // len(radii)], radii[index % len(radii)])
            for grid, index in enumerate(indices)}


def summarize_operation(minutes):
    """全窗口先累加分子分母；等待时间只对最终匹配订单加权。"""
    columns = ("baseline_request_num", "submitted_request_num", "driver_offer_num",
               "driver_accept_num", "matched_request_num", "candidate_request_num",
               "unmatched_cancel_num", "pickup_cancel_num", "platform_revenue")
    result = {column: float(minutes[column].sum()) for column in columns}

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator > 0 else None

    result["request_rate"] = ratio(result["submitted_request_num"], result["baseline_request_num"])
    result["driver_accept_rate"] = ratio(result["driver_accept_num"], result["driver_offer_num"])
    # 与监督学习标签保持一致：匹配前候选次数 + 超时/主动取消次数。
    result["completion_denominator"] = result["candidate_request_num"] + result["unmatched_cancel_num"]
    result["order_completion_rate"] = ratio(result["matched_request_num"], result["completion_denominator"])
    result["mean_wait_to_match_seconds"] = ratio(
        (minutes["mean_wait_to_match_seconds"] * minutes["matched_request_num"]).sum(),
        result["matched_request_num"])
    return result


def run_episode(config, policy, seed, requests, drivers, predictor, output,
                action_optimizer="greedy", bcs_max_sweeps=5, action_policy=None):
    """执行真实闭环；所有策略从相同司机及订单输入独立 reset。"""
    if (action_optimizer not in ("greedy", "bcs") or bcs_max_sweeps <= 0
            or (action_optimizer == "bcs" and predictor is None)
            or (action_policy is not None and (not callable(action_policy) or predictor is not None))):
        raise ValueError
    started = perf_counter()
    grid_num, frequency = config["grid_num"], config["decision_freq"]
    adjacency = load_neighbor_adjacency(config.get("data_dir", "my_data"), grid_num)
    _, start, end = simulation_window(config["start_time"], config["simulation_steps"])
    fixed = uniform_grid_actions(grid_num, config["fixed_price"], config["fixed_radius"])
    warmup_price = config.get("warmup_price", config["fixed_price"])
    warmup_radius = config.get("warmup_radius", config["fixed_radius"])
    warmup_actions = uniform_grid_actions(grid_num, warmup_price, warmup_radius)
    sim = Simulator(
        grid_num=grid_num, driver_num=config["driver_num"], decision_freq=frequency,
        grid_action=warmup_actions, price_response_model=config["price_response_model"],
        allowed_price_multipliers=sorted(set(config["price_action_values"] + [config["fixed_price"], warmup_price])),
        allowed_matching_radii_km=sorted(set(config["matching_radius_values"] + [config["fixed_radius"], warmup_radius])),
        # 输入已按原比例逐分钟无放回采样，此处不能再次采样。
        order_sample_ratio=1.0, base_fare_start=3.0, price_per_km=0.7,
        commission_rate=0.25, enable_reposition=False)
    sim.t_initial, sim.t_end = start, end
    sim.finish_run_step = config["simulation_steps"]
    sim.reset(seed=seed, given_data=True, request_databases=requests, driver_info=drivers)
    sim.experiment_date = config["date"]
    initial_driver_hash = hashlib.sha256(sim.driver_table.to_csv(index=False).encode()).hexdigest()
    run_id = f"{policy}_seed{seed}"
    metadata = {"run_id": run_id, "date": config["date"], "simulator_seed": seed,
                **state_metadata(adjacency)}
    rows, periods, search_rows = [], None, []
    previous_actions = dict(warmup_actions)
    warmup = config["history_steps"] * frequency
    if policy == "random" and predictor is not None:
        raise ValueError
    action_rng = (np.random.default_rng(np.random.SeedSequence([seed, RANDOM_ACTION_SEED_NAMESPACE]))
                  if policy == "random" else None)
    while not sim.end_of_episode:
        step, decision = sim.current_step, sim.current_step // frequency
        interval = min(frequency, config["simulation_steps"] - step)
        current_state = capture_decision_state(sim, adjacency)
        state_by_grid = current_state.set_index("grid_id")
        actions, predicted, search = (warmup_actions if step < warmup else fixed), {}, None
        if action_policy is not None and step >= warmup:
            actions = action_policy(periods, decision, sim.time, interval, current_state)
        if predictor is not None and step >= warmup:
            if getattr(predictor.preprocessor, "has_joint_actions", False):
                if action_optimizer not in ("greedy", "bcs"):
                    raise ValueError
                actions, predicted, search = joint_action_search(
                    predictor, periods, decision, sim.time, interval,
                    config["price_action_values"], config["matching_radius_values"],
                    current_state, initial_actions=previous_actions,
                    max_sweeps=1 if action_optimizer == "greedy" else bcs_max_sweeps)
            else:
                if action_optimizer == "bcs":
                    raise ValueError
                actions, predicted = revenue_actions(
                    predictor, periods, decision, sim.time, interval,
                    config["price_action_values"], config["matching_radius_values"], current_state)
        if search is not None:
            search_rows.append({"decision_step": decision, "simulation_time": sim.time,
                                "optimizer": action_optimizer, **search})
        elif action_rng is not None and step >= warmup:
            actions = random_grid_actions(action_rng, grid_num, config["price_action_values"],
                                          config["matching_radius_values"])
        sim.grid_action = actions
        for grid, action in actions.items():
            rows.append({"decision_step": decision, "grid_id": grid,
                         "decision_start_minute_step": step,
                         "decision_end_minute_step_exclusive": step + interval,
                         "joint_action_id": -1,  # 策略评估按实际动作记账，不依赖探索编号。
                         "price_multiplier": action.price_multiplier,
                         "matching_radius_km": action.matching_radius_km,
                         "is_baseline_action": action == GridAction(),
                         "warmup": step < warmup,
                         "action_optimizer": action_optimizer if search is not None else "none",
                         "optimizer_sweeps": search["sweeps"] if search is not None else 0,
                         **state_by_grid.loc[grid].to_dict(),
                         "predicted_platform_revenue": predicted.get(grid)})
        previous_actions = dict(actions)
        for _ in range(interval):
            sim.step()
        plan = pd.DataFrame(rows)
        minutes = _build_minute_metrics(sim, config["date"], seed, frequency, plan)
        periods = aggregate_decision_metrics(plan, minutes, metadata)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan.to_csv(output / "action_plan.csv.gz", index=False)
    minutes.to_csv(output / "grid_minute_metrics.csv.gz", index=False)
    periods.to_csv(output / "decision_metrics.csv.gz", index=False)
    if search_rows:
        pd.DataFrame(search_rows).to_csv(output / "search_diagnostics.csv", index=False)
    result = {"policy": policy, "seed": seed, "initial_driver_sha256": initial_driver_hash,
              "warmup_metrics_sha256": hashlib.sha256(
                  minutes[minutes.minute_step < warmup].to_csv(index=False).encode()).hexdigest(),
              "evaluation_start_minute": warmup, "evaluation_minutes": len(minutes) // grid_num - warmup,
              **summarize_operation(minutes[minutes.minute_step >= warmup]),
              "full_window": summarize_operation(minutes), "end_queue_num": len(sim.wait_requests),
              "generated_baseline_requests": int(sim.baseline_request_num),
              "elapsed_seconds": perf_counter() - started}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def run_paired_seed(config, seed, output, model_paths):
    """一个任务完成同一 seed 的三种策略，保留输入与初始化一致性证据。"""
    torch.set_num_threads(1)
    pattern = SimulatorPattern(config["date"], data_dir=config["data_dir"])
    _, start, end = simulation_window(config["start_time"], config["simulation_steps"])
    requests, sampled = sample_common_orders(pattern.request_all, start, end,
                                            config["order_sample_ratio"], seed)
    output = Path(output) / "runs" / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=False)
    sampled.to_csv(output / "sampled_order_ids.csv.gz", index=False)
    predictors = {"fixed": None, "lstm": LSTMPredictor.load(model_paths["lstm"]),
                  "timesnet": TimesNetPredictor.load(model_paths["timesnet"])}
    results = []
    for policy in POLICIES:
        row = run_episode(config, policy, seed, requests, pattern.driver_info,
                          predictors[policy], output / policy)
        results.append(row)
        print(f"seed={seed} {policy}: revenue={row['platform_revenue']:.2f}, "
              f"{row['elapsed_seconds']:.1f}s", flush=True)
    if (len({row["initial_driver_sha256"] for row in results}) != 1
            or len({row["baseline_request_num"] for row in results}) != 1
            or any(row["generated_baseline_requests"] != len(sampled) for row in results)):
        raise AssertionError
    return results


def plot_policy_results(output):
    """运营指标画真实仿真值，而非预测误差；误差线为跨 seed 标准差。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output = Path(output)
    frame = pd.read_csv(output / "run_summary.csv")
    report = json.loads((output / "manifest.json").read_text())
    config = report["config"]
    labels = (f"Fixed\nr={config['fixed_radius']:g} km, p={config['fixed_price']:g}",
              "LSTM\nRevenue-greedy", "TimesNet\nRevenue-greedy")
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
        for ax, (metric, (title, direction)) in zip(axes.flat, METRICS.items()):
            stats = frame.groupby("policy")[metric].agg(["mean", "std"]).reindex(POLICIES)
            bars = ax.bar(labels, stats["mean"], yerr=stats["std"].fillna(0), capsize=4,
                          color=("#94a3b8", "#2563eb", "#8b5cf6"))
            values = stats["mean"].to_numpy()
            ax.bar_label(bars, labels=[f"{v:.1%}" if metric.endswith("rate") else f"{v:,.1f}"
                                      for v in values], padding=9, fontsize=9)
            ax.set_title(f"{title}\n{direction}", fontsize=10)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            # 显式预留柱顶数值和误差线空间，避免与方向箭头标题重叠。
            upper = (stats["mean"] + stats["std"].fillna(0)).max()
            ax.set_ylim(0, max(float(upper), 1e-9) * 1.25)
            if metric.endswith("rate"):
                ax.yaxis.set_major_formatter(PercentFormatter(1))
        fig.suptitle(f"Closed-loop simulator comparison | {len(report['seeds'])} paired seeds | "
                     f"{config['grid_num']} grids, {config['driver_num']} drivers\n"
                     f"Shared {config['history_steps'] * config['decision_freq']}-min warm-up excluded; "
                     "bars: seed mean, error bars: ±1 SD (not confidence intervals)")
        fig.savefig(output / "policy_comparison.png", dpi=170)
        plt.close(fig)
    # 单独记录重绘版本；不改写实际仿真执行时的源码指纹。
    report["plot_code_sha256"] = sha256(__file__)
    report["plot_generated_at"] = datetime.now(timezone.utc).isoformat()
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


def check_collection_revision(collection_root, source):
    """沿用采集器的严格版本校验；仅接受已验证的执行兼容修订。"""
    current = make_config(
        source["scenario"], source["run_count"], source["seed_start"], source["batch_size"],
        source["data_dir"], source["date"], source["start_time"], source["simulation_steps"])
    validate_execution_revision(collection_root, source, current)


def compare_policies(model_results,
                     collection_root=DEFAULT_COLLECTION, output=None, runs=10, workers=2,
                     fixed_price=1.5, fixed_radius=3.0):
    """保留验证集选出的固定模型；测试种子抽取不参考预测或运营结果。"""
    if (runs <= 0 or workers <= 0 or not np.isfinite([fixed_price, fixed_radius]).all()
            or min(fixed_price, fixed_radius) <= 0):
        raise ValueError
    model_results, collection_root = Path(model_results), Path(collection_root)
    model_report = json.loads((model_results / "comparison_report.json").read_text())
    source_config = json.loads((collection_root / "collection_config.json").read_text())
    # 不用经过修改的仿真器/输入冒充训练时的同一实验环境。
    project = Path(__file__).resolve().parents[1]
    check_collection_revision(collection_root, source_config)
    if source_config["enable_reposition"]:
        raise ValueError
    test_runs = sorted(model_report["split_run_ids"]["test"])
    if runs > len(test_runs):
        raise ValueError
    selected = np.random.default_rng(42).choice(test_runs, runs, replace=False).tolist()
    signature = hashlib.sha256(json.dumps(source_config, sort_keys=True).encode()).hexdigest()
    seeds = []
    for run_id in selected:
        paths = list(collection_root.glob(f"batch_*/{run_id}/manifest.json"))
        if len(paths) != 1:
            raise ValueError
        manifest = json.loads(paths[0].read_text())
        if (manifest["planned_split"] != "test"
                or manifest["collection_config_sha256"] != signature):
            raise ValueError
        seeds.append(manifest["simulator_seed"])
    if len(set(seeds)) != len(seeds):
        raise ValueError
    config = {key: source_config[key] for key in (
        "date", "data_dir", "start_time", "simulation_steps", "grid_num", "driver_num",
        "order_sample_ratio", "decision_freq", "price_response_model", "price_action_values",
        "matching_radius_values")}
    config.update(fixed_price=fixed_price, fixed_radius=fixed_radius,
                  history_steps=model_report["preprocessor"]["history_steps"])
    if (model_report["preprocessor"]["grid_ids"] != list(range(config["grid_num"]))
            or config["simulation_steps"] <= config["history_steps"] * config["decision_freq"]):
        raise ValueError
    checkpoints = {}
    for name, cls in (("lstm", LSTMPredictor), ("timesnet", TimesNetPredictor)):
        snapshot = model_report["snapshots"][name]
        path = model_results / snapshot["snapshot_file"]
        if (sha256(path) != snapshot["checkpoint_sha256"]
                or cls.load(path).preprocessor.state_dict() != model_report["preprocessor"]):
            raise ValueError
        checkpoints[name] = path.read_bytes()
    output = Path(output or datetime.now().strftime("experiments/policy_comparison_%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    model_paths = {}
    for name, raw in checkpoints.items():
        destination = output / f"{name}_snapshot.pt"
        destination.write_bytes(raw)
        model_paths[name] = str(destination.resolve())
    report = {
        "schema_version": "closed_loop_policy_comparison_v1", "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
        "seeds": seeds, "source_test_run_ids": selected, "selection_seed": 42,
        "model_snapshots": model_report["snapshots"], "source_sha256": source_config["source_sha256"],
        "code_sha256": {**source_config["code_sha256"], "baseline/compare_policies.py": sha256(__file__)},
        "objective": "argmax predicted next-period platform revenue independently per grid",
        "metric_definitions": {
            "order_completion_rate": "sum(retained matches) / sum(candidate attempts + unmatched cancellations); retries repeat",
            "driver_accept_rate": "sum(driver accepts) / sum(driver offers)",
            "request_rate": "sum(submissions) / sum(baseline requests); can exceed 1 with observed-demand resampling",
            "mean_wait_to_match_seconds": "matched-count-weighted wait; excludes unserved passengers and pickup time",
            "platform_revenue": "retained matched fare * 0.25; not net profit",
        },
        "limitations": [
            "单日期低密度；配对 seed 是测试子集，不是完整 588 个测试 run。",
            "订单独立预采样以固定外生输入；不是旧随机探索轨迹的逐随机数复现。",
            "司机初态一致；策略分叉后行为随机抽样路径不强行对齐。",
            "共同固定规则预热；主结果排除预热，完整窗口结果另存。",
            "固定价格 1.5 不是训练候选值；学习策略仍只枚举原 104 个候选动作。",
            "预测模型是随机策略下学习的映射；闭环分布偏移、跨网格耦合及滞后价格会影响决策收益。",
            "单步贪心不是长期收益最优；期末未清空队列；测试不用于调参或选 checkpoint。",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    all_results = []
    if workers == 1:
        for seed in seeds:
            all_results.extend(run_paired_seed(config, seed, output, model_paths))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, runs)) as pool:
            futures = [pool.submit(run_paired_seed, config, seed, output, model_paths) for seed in seeds]
            for future in as_completed(futures):
                all_results.extend(future.result())
    frame = pd.DataFrame([{key: value for key, value in row.items() if key != "full_window"}
                          for row in all_results]).sort_values(["seed", "policy"])
    frame.to_csv(output / "run_summary.csv", index=False)
    summary = frame.groupby("policy")[list(METRICS)].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv(output / "policy_summary.csv")
    paired = frame.pivot(index="seed", columns="policy", values="platform_revenue")
    for name in ("lstm", "timesnet"):
        paired[f"{name}_minus_fixed"] = paired[name] - paired["fixed"]
        paired[f"{name}_improvement_percent"] = 100 * (paired[name] / paired["fixed"].replace(0, np.nan) - 1)
    paired.to_csv(output / "paired_revenue.csv")
    report["status"] = "complete"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    plot_policy_results(output)
    print(summary.to_string(), flush=True)
    print(f"策略对比结果已保存至：{output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="固定规则与时序模型收益贪心策略的配对仿真实验")
    parser.add_argument("--model-results", required=True)
    parser.add_argument("--collection-root", default=DEFAULT_COLLECTION)
    parser.add_argument("--output")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fixed-price", type=float, default=1.5)
    parser.add_argument("--fixed-radius", type=float, default=3.0)
    compare_policies(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
