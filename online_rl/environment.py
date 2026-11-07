"""一个step对应一次全城决策；观察在动作前，收益覆盖随后五个仿真分钟。"""

import hashlib

import numpy as np
import pandas as pd

from baseline.compare_policies import _joint_candidate_frame, sample_common_orders, summarize_operation
from baseline.sequence import SequencePreprocessor
from data_pipeline.build_supervised_dataset import aggregate_decision_metrics
from data_pipeline.collect_trajectories import _build_minute_metrics, simulation_window
from offline_rl.data import encode_state
from src.env.actions import GridAction, uniform_grid_actions
from src.env.observation import capture_decision_state, load_neighbor_adjacency, state_metadata
from src.env.simulator_env import Simulator
from src.env.simulator_pattern import SimulatorPattern


def frame_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


class ObservationEncoder:
    """只编码过去四周期和当前动作前状态，剔除当期候选动作占位列。"""

    def __init__(self, metadata):
        self.processor = SequencePreprocessor.from_state_dict(metadata["preprocessor"])
        self.indices = metadata["state_context_indices"]

    def __call__(self, periods, decision, time, interval, current_state):
        rows, _, action_columns = _joint_candidate_frame(periods, self.processor, decision, time, interval, current_state)
        rows = pd.DataFrame([{**row, "price_multiplier": 0., "matching_radius_km": 0.,
                              **dict.fromkeys(action_columns, 0.)} for row in rows])
        history, context, _ = self.processor.transform_inputs(rows)
        return encode_state(history[None], context[None], self.indices)[0]


class DecisionEnvironment:
    """复用现有仿真业务逻辑和指标聚合，预热后有56条联合转移。"""

    def __init__(self, config, metadata, pattern=None):
        self.config, self.metadata = config, metadata
        self.encoder = ObservationEncoder(metadata)
        self.pattern = pattern or SimulatorPattern(config["date"], data_dir=config["data_dir"])
        self.adjacency = load_neighbor_adjacency(config["data_dir"], config["grid_num"])
        self.pairs = metadata["action_pairs"]

    def reset(self, seed):
        c = self.config
        self.seed = int(seed)
        _, start, end = simulation_window(c["start_time"], c["simulation_steps"])
        requests, sampled = sample_common_orders(self.pattern.request_all, start, end, c["order_sample_ratio"], seed)
        warm = uniform_grid_actions(c["grid_num"], c["warmup_price"], c["warmup_radius"])
        self.sim = Simulator(grid_num=c["grid_num"], driver_num=c["driver_num"], decision_freq=c["decision_freq"],
            grid_action=warm, price_response_model=c["price_response_model"],
            allowed_price_multipliers=sorted(set(c["price_action_values"]+[c["warmup_price"], c["fixed_price"]])),
            allowed_matching_radii_km=sorted(set(c["matching_radius_values"]+[c["warmup_radius"], c["fixed_radius"]])),
            order_sample_ratio=1., base_fare_start=3., price_per_km=.7, commission_rate=.25, enable_reposition=False)
        self.sim.t_initial, self.sim.t_end, self.sim.finish_run_step = start, end, c["simulation_steps"]
        self.sim.reset(seed=seed, given_data=True, request_databases=requests, driver_info=self.pattern.driver_info)
        self.sim.experiment_date = c["date"]
        self.rows, self.periods = [], None
        self.run_metadata = {"run_id": f"online_seed{seed}", "date": c["date"], "simulator_seed": seed,
                             **state_metadata(self.adjacency)}
        self.initial = {"seed": int(seed), "sampled_orders_sha256": frame_hash(sampled),
                        "initial_driver_sha256": frame_hash(self.sim.driver_table)}
        for _ in range(c["history_steps"]):
            self._advance(warm)
        self.initial["warmup_metrics_sha256"] = frame_hash(self.minutes)
        return self.observation()

    def observation(self):
        if self.sim.end_of_episode:
            return np.zeros(self.metadata["state_size"], dtype="float32")
        return self.encoder(self.periods, self.sim.current_step//self.config["decision_freq"], self.sim.time,
            self.config["decision_freq"], capture_decision_state(self.sim, self.adjacency))

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
        for _ in range(interval):
            sim.step()
        plan = pd.DataFrame(self.rows)
        self.minutes = _build_minute_metrics(sim, c["date"], self.seed, c["decision_freq"], plan)
        self.periods = aggregate_decision_metrics(plan, self.minutes, self.run_metadata)
        return float(self.minutes.loc[self.minutes.minute_step >= step, "platform_revenue"].sum())

    def step(self, actions):
        actions = np.asarray(actions)
        if (self.sim.end_of_episode or actions.shape != (self.config["grid_num"],)
                or not np.issubdtype(actions.dtype, np.integer) or (actions < 0).any() or (actions >= len(self.pairs)).any()):
            raise ValueError
        reward = self._advance({grid: GridAction(*self.pairs[int(index)]) for grid, index in enumerate(actions)})
        terminal = bool(self.sim.end_of_episode)
        return self.observation(), reward, terminal

    def summary(self):
        if not self.sim.end_of_episode:
            raise ValueError
        warmup = self.config["history_steps"]*self.config["decision_freq"]
        return {**self.initial, **summarize_operation(self.minutes[self.minutes.minute_step >= warmup])}
