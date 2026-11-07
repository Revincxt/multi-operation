"""复用仿真物理过程和指标口径，独立维护资源状态与整轮预热序列。"""

import numpy as np
import pandas as pd
from types import SimpleNamespace

from baseline.compare_policies import sample_common_orders, summarize_operation
from data_pipeline.build_supervised_dataset import aggregate_decision_metrics
from data_pipeline.collect_trajectories import _build_minute_metrics, simulation_window
from resource_rl.observation import ResourceObservation
from src.env.actions import GridAction, uniform_grid_actions
from src.env.observation import capture_decision_state, load_neighbor_adjacency, state_metadata
from src.env.simulator_env import Simulator
from src.env.simulator_pattern import SimulatorPattern


def frame_hash(frame):
    import hashlib
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


class ResourceEnvironment:
    def __init__(self, config, pattern=None):
        self.config = config
        self.pattern = pattern or SimulatorPattern(config["date"], data_dir=config["data_dir"])
        self.adjacency = load_neighbor_adjacency(config["data_dir"], config["grid_num"])
        self.encoder = ResourceObservation(config, self.adjacency)
        self.pairs = [(p, r) for p in config["price_action_values"] for r in config["matching_radius_values"]]

    def reset(self, seed):
        c = self.config
        self.seed = int(seed)
        _, start, end = simulation_window(c["start_time"], c["simulation_steps"])
        requests, sampled = sample_common_orders(self.pattern.request_all, start, end, c["order_sample_ratio"], seed)
        self.warm_action = self.pairs.index((c["warmup_price"], c["warmup_radius"]))
        warm = uniform_grid_actions(c["grid_num"], c["warmup_price"], c["warmup_radius"])
        self.sim = Simulator(grid_num=c["grid_num"], driver_num=c["driver_num"], decision_freq=c["decision_freq"],
            grid_action=warm, price_response_model=c["price_response_model"],
            allowed_price_multipliers=sorted(set(c["price_action_values"]+[c["warmup_price"], c["fixed_price"]])),
            allowed_matching_radii_km=sorted(set(c["matching_radius_values"]+[c["warmup_radius"], c["fixed_radius"]])),
            order_sample_ratio=1., base_fare_start=3., price_per_km=.7, commission_rate=.25, enable_reposition=False)
        self.sim.t_initial, self.sim.t_end, self.sim.finish_run_step = start, end, c["simulation_steps"]
        self.sim.reset(seed=seed, given_data=True, request_databases=requests, driver_info=self.pattern.driver_info)
        self.sim.experiment_date = c["date"]
        self.rows, self.periods, self.warmup, self.minutes = [], None, [], None
        self.previous_actions = np.full(c["grid_num"], len(self.pairs), dtype="int64")
        self.run_metadata = {"run_id": f"resource_seed{seed}", "date": c["date"], "simulator_seed": seed,
                             **state_metadata(self.adjacency)}
        self.initial = {"seed": self.seed, "sampled_orders_sha256": frame_hash(sampled),
                        "initial_driver_sha256": frame_hash(self.sim.driver_table)}
        for _ in range(c["history_steps"]):
            observation = self.observation()
            reward = self._advance(warm)
            self.previous_actions[:] = self.warm_action
            self.warmup.append((observation, reward, self.transition_target()))
        self.initial["warmup_metrics_sha256"] = frame_hash(self.minutes)
        return self.observation()

    def observation(self):
        return self.encoder.capture(self.sim, self.periods, self.previous_actions)

    def _advance(self, actions):
        sim, c = self.sim, self.config
        step, decision = sim.current_step, sim.current_step//c["decision_freq"]
        interval = min(c["decision_freq"], c["simulation_steps"]-step)
        state = capture_decision_state(sim, self.adjacency).set_index("grid_id")
        sim.grid_action = actions
        self.rows.extend({"decision_step": decision, "grid_id": grid,
            "decision_start_minute_step": step, "decision_end_minute_step_exclusive": step+interval,
            "joint_action_id": -1, "price_multiplier": action.price_multiplier,
            "matching_radius_km": action.matching_radius_km, "is_baseline_action": action == GridAction(),
            **state.loc[grid].to_dict()} for grid, action in actions.items())
        self.completed_flows = np.zeros((c["grid_num"], c["grid_num"]), dtype="float32")
        for _ in range(interval):
            before = sim.driver_table[["grid_id", "status"]].to_numpy(int).copy()
            sim.step()
            after = sim.driver_table[["grid_id", "status"]].to_numpy(int)
            finished = np.isin(before[:, 1], [1, 2]) & (after[:, 1] == 0)
            np.add.at(self.completed_flows, (before[finished, 0], after[finished, 0]), 1.)
        plan = pd.DataFrame(self.rows[-c["grid_num"]:])
        # 只聚合刚执行的周期，避免每五分钟重新聚合全部历史。物理状态与共享指标公式不变。
        view = SimpleNamespace(current_step=interval, evaluate_table=sim.evaluate_table[step:step+interval],
            evaluate_df=sim.evaluate_df, grid_num=c["grid_num"], t_initial=sim.t_initial+step*sim.delta_t, delta_t=sim.delta_t)
        current = _build_minute_metrics(view, c["date"], self.seed, c["decision_freq"], plan.assign(decision_step=0))
        current["minute_step"] += step
        current["decision_step"] += decision
        period = aggregate_decision_metrics(plan, current, self.run_metadata)
        self.minutes = current if self.minutes is None else pd.concat((self.minutes, current), ignore_index=True)
        self.periods = period if self.periods is None else pd.concat((self.periods, period), ignore_index=True)
        return float(current.platform_revenue.sum())

    def transition_target(self):
        local = capture_decision_state(self.sim, self.adjacency)
        return np.r_[local[["start_idle_driver_num", "start_waiting_request_num"]].to_numpy(float).ravel(),
                     self.completed_flows.ravel()].astype("float32")/self.encoder.unit

    def step(self, actions):
        actions = np.asarray(actions)
        if (self.sim.end_of_episode or actions.shape != (self.config["grid_num"],)
                or not np.issubdtype(actions.dtype, np.integer) or (actions < 0).any() or (actions >= len(self.pairs)).any()):
            raise ValueError
        reward = self._advance({grid: GridAction(*self.pairs[int(index)]) for grid, index in enumerate(actions)})
        self.previous_actions = actions.copy()
        return self.observation(), reward, bool(self.sim.end_of_episode)

    def summary(self):
        if not self.sim.end_of_episode:
            raise ValueError
        warmup = self.config["history_steps"]*self.config["decision_freq"]
        return {**self.initial, **summarize_operation(self.minutes[self.minutes.minute_step >= warmup])}
