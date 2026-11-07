"""固定完整仿真状态，检验离线Q的动作排序、后续策略依赖及不确定性。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from baseline.compare_policies import _joint_candidate_frame, sample_common_orders
from data_pipeline.build_supervised_dataset import aggregate_decision_metrics
from data_pipeline.collect_trajectories import simulation_window
from offline_rl.data import encode_state, save_json, sha256
from offline_rl.evaluate import Policy
from src.env.actions import GridAction, uniform_grid_actions
from src.env.observation import STATE_COLUMNS, capture_decision_state, load_neighbor_adjacency, state_metadata
from src.env.simulator_env import Simulator
from src.env.simulator_pattern import SimulatorPattern


SOURCE = Path("models/structured_joint_search_20261103")
CHECKPOINTS = {"stq": SOURCE / "joint_start.pt", "eq": Path("models/value_guided_full100k_20261103/eq_soft.pt")}
OUTPUT = Path("experiments/rl_research_diagnosis_20261103")
_POLICIES, _CONFIG, _ADJACENCY = {}, None, None


def initialize(config):
    global _POLICIES, _CONFIG, _ADJACENCY
    torch.set_num_threads(1)
    _CONFIG = config
    _ADJACENCY = load_neighbor_adjacency(config["data_dir"], config["grid_num"])
    _POLICIES = {name: Policy(path) for name, path in CHECKPOINTS.items()}
    assert _POLICIES["stq"].metadata["preprocessor"] == _POLICIES["eq"].metadata["preprocessor"]


def observation(periods, sim):
    policy = _POLICIES["stq"]
    current = capture_decision_state(sim, _ADJACENCY)
    rows, grids, action_columns = _joint_candidate_frame(periods, policy.processor,
        sim.current_step//sim.decision_freq, sim.time, sim.decision_freq, current)
    frame = pd.DataFrame([{**row, "price_multiplier": 0., "matching_radius_km": 0.,
                           **dict.fromkeys(action_columns, 0.)} for row in rows])
    history, context, _ = policy.processor.transform_inputs(frame)
    return torch.from_numpy(encode_state(history[None], context[None], policy.metadata["state_context_indices"]))


def action_mapping(vector):
    pairs = _POLICIES["stq"].pairs
    return {grid: GridAction(*pairs[int(index)]) for grid, index in enumerate(vector)}


@torch.no_grad()
def candidates_and_scores(state, selected_grid):
    pairs = _POLICIES["stq"].pairs
    chosen = _POLICIES["stq"].actor.act(state)[0]
    eq = _POLICIES["eq"].actor.act(state)[0]
    f3 = torch.full_like(chosen, pairs.index([1.6, 2.]))
    radius_count = len({r for p, r in pairs})
    radii = sorted({r for p, r in pairs})
    radius_two = (chosen//radius_count)*radius_count + radii.index(2.)
    actions = dict(js=chosen, f3=f3, eq=eq, same_price_radius2=radius_two)
    single = chosen.clone()
    single[selected_grid] = radius_two[selected_grid]
    actions["single_grid_radius2"] = single
    unilateral = chosen.repeat(len(chosen), 1)
    unilateral[torch.arange(len(chosen)), torch.arange(len(chosen))] = radius_two
    matrix = torch.cat((torch.stack(list(actions.values())), unilateral))
    scores = {}
    interactions = {}
    for name, policy in _POLICIES.items():
        model = policy.actor
        prepared, other = model._prepare(state)
        prior = model.behavior_log_probs(state)[0]
        grids = torch.arange(len(chosen))
        baseline = prior.argmax(-1)
        values = model.cached_values(torch.cat((baseline[None], matrix)), prepared, other)
        differences = values[:, 1:]-values[:, :1]
        regularized = differences.mean(0)-model.uncertainty_weight*differences.std(0, unbiased=False)
        logp = prior[grids, matrix].sum(-1)
        regularized += model.behavior_weight*(logp-prior[grids, baseline].sum())
        for index, candidate in enumerate(actions):
            scores.setdefault(candidate, {}).update({f"{name}_q_usd": float(values[:, index+1].mean()*100),
                f"{name}_advantage_std_usd": float(differences[:, index].std(unbiased=False)*100),
                f"{name}_regularized_score": float(regularized[index]*100),
                f"{name}_behavior_log_probability": float(logp[index])})
        q = values[:, 1:].mean(0)*100
        interactions[name] = {"all_radius2_minus_js": float(q[3]-q[0]),
            "sum_unilateral_radius2_minus_js": float((q[len(actions):]-q[0]).sum()),
            "synergy_usd": float(q[3]-q[0]-(q[len(actions):]-q[0]).sum())}
    return {name: vector.numpy() for name, vector in actions.items()}, scores, interactions


def build_snapshot(seed, steps, output):
    config = _CONFIG
    pattern = SimulatorPattern(config["date"], data_dir=config["data_dir"])
    _, start, end = simulation_window(config["start_time"], config["simulation_steps"])
    requests, sampled = sample_common_orders(pattern.request_all, start, end, config["order_sample_ratio"], seed)
    sim = Simulator(grid_num=config["grid_num"], driver_num=config["driver_num"], decision_freq=config["decision_freq"],
        grid_action=uniform_grid_actions(config["grid_num"], config["warmup_price"], config["warmup_radius"]),
        price_response_model=config["price_response_model"], allowed_price_multipliers=config["price_action_values"],
        allowed_matching_radii_km=config["matching_radius_values"], order_sample_ratio=1., base_fare_start=3.,
        price_per_km=.7, commission_rate=.25, enable_reposition=False)
    sim.t_initial, sim.t_end, sim.finish_run_step = start, end, config["simulation_steps"]
    sim.reset(seed=seed, given_data=True, request_databases=requests, driver_info=pattern.driver_info)
    sim.experiment_date = config["date"]
    directory = SOURCE / "validation_new/runs" / f"seed{seed}/joint_start"
    plan = pd.read_csv(directory / "action_plan.csv.gz")
    periods = pd.read_csv(directory / "decision_metrics.csv.gz")
    minutes = pd.read_csv(directory / "grid_minute_metrics.csv.gz")
    summary = json.loads((directory / "summary.json").read_text())
    assert hashlib.sha256(sim.driver_table.to_csv(index=False).encode()).hexdigest() == summary["initial_driver_sha256"]
    assert hashlib.sha256(sampled.to_csv(index=False).encode()).hexdigest() == summary["sampled_orders_sha256"]
    results = []
    while sim.current_step <= max(steps):
        step, decision = sim.current_step, sim.current_step//sim.decision_freq
        current_plan = plan[plan.decision_step == decision].sort_values("grid_id")
        if step in steps:
            actual = capture_decision_state(sim, _ADJACENCY)
            np.testing.assert_array_equal(actual[list(STATE_COLUMNS)].to_numpy(), current_plan[list(STATE_COLUMNS)].to_numpy())
            revenue_column = list(sim.evaluate_df.columns).index("platform_revenue")
            expected = minutes[minutes.minute_step < step].sort_values(["minute_step", "grid_id"])
            np.testing.assert_allclose(sim.evaluate_table[:step, :, revenue_column].flatten(), expected.platform_revenue.to_numpy(), atol=1e-8, rtol=1e-10)
            recent = periods[(periods.decision_step < decision) & (periods.decision_step >= decision-4)].copy()
            state = observation(recent, sim)
            selected_grid = int(np.random.default_rng(np.random.SeedSequence([7300703, seed, step])).integers(sim.grid_num))
            actions, scores, interactions = candidates_and_scores(state, selected_grid)
            metadata = {"run_id": f"diagnostic_seed{seed}", "date": config["date"], "simulator_seed": seed, **state_metadata(_ADJACENCY)}
            hidden = {"queue_age_mean": float(sim.wait_requests.wait_time.astype(float).mean()) if len(sim.wait_requests) else 0.,
                "queued_quote_mean": float(sim.wait_requests.quoted_fare.astype(float).mean()) if len(sim.wait_requests) else 0.,
                "busy_release_5min": int(((sim.driver_table.status.isin([1, 2])) & (sim.driver_table.remaining_time <= 300)).sum()),
                "busy_release_15min": int(((sim.driver_table.status.isin([1, 2])) & (sim.driver_table.remaining_time <= 900)).sum())}
            path = Path(output) / f"seed{seed}_step{step}.pickle"
            with path.open("wb") as stream:
                pickle.dump({"sim": sim, "periods": recent, "state": state.numpy(), "actions": actions, "scores": scores,
                    "metadata": metadata, "seed": seed, "minute_step": step, "hidden_summary": hidden,
                    "intervention_grid": selected_grid, "predicted_interactions": interactions}, stream, protocol=5)
            results.append({"path": str(path), "seed": seed, "minute_step": step, "sha256": sha256(path),
                            "prefix_replay_verified": True, "hidden_summary": hidden, "intervention_grid": selected_grid,
                            "predicted_interactions": interactions})
        if step == max(steps):
            break
        sim.grid_action = {int(row.grid_id): GridAction(row.price_multiplier, row.matching_radius_km) for row in current_plan.itertuples()}
        for _ in range(sim.decision_freq):
            sim.step()
    return results


def period_step(sim, actions, metadata):
    start, decision = sim.current_step, sim.current_step//sim.decision_freq
    current = capture_decision_state(sim, _ADJACENCY).set_index("grid_id")
    plan = pd.DataFrame([{"decision_step": decision, "grid_id": grid,
        "decision_start_minute_step": start, "decision_end_minute_step_exclusive": start+sim.decision_freq,
        "joint_action_id": -1, "price_multiplier": action.price_multiplier,
        "matching_radius_km": action.matching_radius_km, "is_baseline_action": False,
        **current.loc[grid].to_dict()} for grid, action in actions.items()])
    sim.grid_action = actions
    for _ in range(sim.decision_freq):
        sim.step()
    table = sim.evaluate_table[start:sim.current_step]
    minutes = pd.DataFrame(table.reshape(-1, table.shape[-1]), columns=sim.evaluate_df.columns)
    minutes["minute_step"] = np.repeat(np.arange(start, sim.current_step), sim.grid_num)
    minutes["sim_time"] = sim.t_initial+minutes.minute_step*sim.delta_t
    minutes["grid_id"] = minutes.origin_grid_id.astype(int)
    minutes["decision_step"] = decision
    return aggregate_decision_metrics(plan, minutes, metadata)


def rollout(snapshot, vector, future_seed, continuation):
    base = snapshot["sim"]
    # 外生订单与司机源表不变；所有订单、司机运行状态及RNG独立复制。
    sim = deepcopy(base, {id(base.request_databases): base.request_databases, id(base.driver_info): base.driver_info})
    sim.rng = np.random.RandomState(future_seed)
    start, periods = sim.current_step, snapshot["periods"].copy()
    first = True
    while not sim.end_of_episode:
        if first:
            actions = action_mapping(vector)
        elif continuation == "f3":
            actions = uniform_grid_actions(sim.grid_num, 1.6, 2.)
        else:
            state = observation(periods, sim)
            with torch.no_grad():
                actions = action_mapping(_POLICIES["stq"].actor.act(state)[0].numpy())
        if continuation == "js":
            recent = period_step(sim, actions, snapshot["metadata"])
            periods = pd.concat((periods, recent), ignore_index=True)
            periods = periods[periods.decision_step >= sim.current_step//sim.decision_freq-4]
        else:
            sim.grid_action = actions
            for _ in range(sim.decision_freq):
                sim.step()
        first = False
    columns = list(sim.evaluate_df.columns)
    rewards = sim.evaluate_table[start:, :, columns.index("platform_revenue")].sum(1).reshape(-1, sim.decision_freq).sum(1)
    return {"return_usd": float(rewards.sum()), "return_discount99_usd": float(rewards @ (.99**np.arange(len(rewards)))),
        "first5_usd": float(rewards[0]), "first20_usd": float(rewards[:4].sum()),
        "pickup_cancellations": float(sim.evaluate_table[start:, :, columns.index("pickup_cancel_num")].sum()),
        "matched_requests": float(sim.evaluate_table[start:, :, columns.index("matched_request_num")].sum())}


def branch_group(path, repetition):
    with Path(path).open("rb") as stream:
        snapshot = pickle.load(stream)
    future_seed = int(np.random.SeedSequence([7300702, snapshot["seed"], snapshot["minute_step"], repetition]).generate_state(1)[0])
    records = []
    for continuation in ("f3", "js"):
        completed = {}
        for candidate, vector in snapshot["actions"].items():
            key = tuple(vector.tolist())
            duplicate = key in completed
            if not duplicate:
                completed[key] = rollout(snapshot, vector, future_seed, continuation)
            records.append({"seed": snapshot["seed"], "minute_step": snapshot["minute_step"], "repetition": repetition,
                "future_seed": future_seed, "continuation": continuation, "candidate": candidate,
                "intervention_grid": snapshot["intervention_grid"],
                "duplicate_action_reused": duplicate, "action_vector": json.dumps(vector.tolist()),
                **snapshot["scores"][candidate], **completed[key]})
    return records


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((SOURCE / "test/manifest.json").read_text())["config"]
    pool = json.loads((SOURCE / "protocol.json").read_text())["validation"]["seeds"]
    seeds = np.random.default_rng(7300701).choice(pool, args.seeds, replace=False).tolist()
    steps = [20, 120, 220]
    plan = {"scope": "diagnostic simulator interventions on development validation states; no parameter learning or policy selection",
        "seeds": seeds, "minute_steps": steps, "future_streams_per_state": args.repetitions,
        "candidates": ["js", "f3", "eq", "same_price_radius2", "single_grid_radius2"], "continuations": ["f3", "js"],
        "config": config, "checkpoint_sha256": {name: sha256(path) for name, path in CHECKPOINTS.items()},
        "controlled": "identical complete state and history, same exogenous orders, same future RNG starting seed per paired group; random calls may diverge afterwards",
        "source_sha256": {str(path): sha256(path) for path in Path("offline_rl").glob("*.py")},
        "simulator_sources_sha256": {path: sha256(path) for path in ("src/env/simulator_env.py", "src/utils/utilities.py",
            "src/utils/dispatch_alg.py", "src/env/price_response/utility_choice.py", "baseline/compare_policies.py",
            "data_pipeline/build_supervised_dataset.py")}, "diagnostic_sha256": sha256(__file__)}
    save_json(output / "protocol.json", plan)
    snapshots = output / "snapshots"
    snapshots.mkdir()
    timer = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(config,)) as executor:
        prepared = [executor.submit(build_snapshot, seed, steps, snapshots) for seed in seeds]
        records = []
        for result in as_completed(prepared):
            records.extend(result.result())
            print(f"verified snapshots: {len(records)}/{len(seeds)*len(steps)}", flush=True)
        save_json(output / "snapshots.json", records)
        jobs = [executor.submit(branch_group, row["path"], repetition) for row in records for repetition in range(args.repetitions)]
        rows = []
        for count, result in enumerate(as_completed(jobs), 1):
            rows.extend(result.result())
            pd.DataFrame(rows).to_csv(output / "branch_returns.csv", index=False)
            save_json(output / "status.json", {"status": "branching", "completed_groups": count, "target_groups": len(jobs),
                "elapsed_seconds": perf_counter()-timer})
            print(f"branch groups: {count}/{len(jobs)}, elapsed={perf_counter()-timer:.1f}s", flush=True)
    save_json(output / "status.json", {"status": "complete", "rows": len(rows), "states": len(records),
        "elapsed_seconds": perf_counter()-timer, "checkpoints_unchanged": all(sha256(CHECKPOINTS[name]) == value for name, value in plan["checkpoint_sha256"].items())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
