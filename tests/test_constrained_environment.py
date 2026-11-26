"""需求族不重复计分、全队列等待、预热和末端边界以及原仿真等价性。"""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from constrained_rl.environment import CohortLedger, ConstrainedEnvironment, PARENT_COLUMN, SIGNALS
from online_rl.environment import DecisionEnvironment


def table(ids):
    return pd.DataFrame({"order_id": ids})


class CohortLedgerTests(unittest.TestCase):
    def test_duplicates_receive_one_baseline_credit_but_all_wait_and_revenue_remain(self):
        ledger = CohortLedger(table([8]))
        ledger.register_lineage(pd.DataFrame({"order_id": [20, -1, -2], PARENT_COLUMN: [20, 20, 20]}))
        ledger.observe_minute(table([20, 21]), table([8]), table([20, -1, -2]), 60)
        ledger.observe_minute(table([]), table([-1]), table([20, -2]), 60)
        ledger.observe_minute(table([]), table([20]), table([]), 60)
        result = ledger.summary()
        self.assertEqual(result["cohort_demand"], 3)
        self.assertEqual(result["physical_matched_orders"], 3)
        self.assertEqual(result["unique_matched"], 2)
        self.assertEqual(result["queue_wait_seconds"], 300)
        self.assertEqual(result["baseline_demand_coverage"], 2/3)

    def test_cancelled_wait_is_preserved_and_terminal_new_demand_is_unserved(self):
        ledger = CohortLedger(table([10]))
        ledger.observe_minute(table([20]), table([]), table([10, 20]), 60)
        # 10在第二分钟取消；末分钟21发单，22拒绝报价，二者均未被服务。
        ledger.observe_minute(table([21, 22]), table([20]), table([21]), 60)
        result = ledger.summary()
        self.assertEqual(result["queue_wait_seconds"], 180)
        self.assertEqual(result["unique_matched"], 1)
        self.assertEqual(result["end_unserved"], 3)
        self.assertEqual(result["end_queue_orders"], 1)

    def test_warmup_elapsed_wait_is_not_recounted(self):
        backlog = table([4, -1]); backlog["wait_time"] = [600., 180.]
        ledger = CohortLedger(backlog)
        ledger.observe_minute(table([]), table([4]), table([-1]), 60)
        self.assertEqual(ledger.summary()["queue_wait_seconds"], 60.)
        self.assertEqual(ledger.summary()["carry_in_demand"], 2)

    def test_zero_arrivals_has_finite_defined_diagnostics(self):
        ledger = CohortLedger(table([]))
        ledger.observe_minute(table([]), table([]), table([]), 60)
        self.assertEqual(ledger.summary()["demand_satisfaction"], 0.)
        self.assertEqual(ledger.summary()["wait_per_demand_seconds"], 0.)

    def test_double_match_and_missing_parent_are_rejected(self):
        ledger = CohortLedger(table([4]))
        ledger.observe_minute(table([]), table([4]), table([]), 60)
        with self.assertRaises(ValueError):
            ledger.observe_minute(table([]), table([4]), table([]), 60)
        with self.assertRaises(ValueError):
            ledger.observe_minute(table([]), table([500]), table([]), 60)


@unittest.skipUnless(Path("my_data/grid6/orders_grid6_2015-05-18.pkl").exists(), "requires local grid6 inputs")
class ConstrainedSimulatorTests(unittest.TestCase):
    def test_instrumentation_keeps_original_transitions_observations_and_revenue(self):
        from src.env.simulator_pattern import SimulatorPattern
        torch.set_num_threads(1)
        payload = json.loads(Path("checkpoints/online_observation.json").read_text())
        config = dict(payload["config"], simulation_steps=35)
        metadata = payload["observation_metadata"]
        pattern = SimulatorPattern(config["date"], data_dir=config["data_dir"])
        seed = 3_419_000_081
        expected = DecisionEnvironment(config, metadata, pattern)
        actual = ConstrainedEnvironment(config, metadata, pattern)
        first = expected.reset(seed)
        second = actual.reset(seed)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(actual.ledger.totals()["queue_wait_seconds"], 0.)
        prewarm_baseline = expected.sim.baseline_request_num
        carry = len(expected.sim.wait_requests)
        per_step = []
        for pair in ([.4, 4.], [1.6, 2.], [1.5, 1.5]):
            action = np.full(6, metadata["action_pairs"].index(pair), dtype="int64")
            expected_state, expected_reward, expected_done = expected.step(action)
            actual_state, actual_reward, actual_done = actual.step(action)
            np.testing.assert_array_equal(expected_state, actual_state)
            self.assertEqual(expected_reward, actual_reward)
            self.assertEqual(expected_done, actual_done)
            pd.testing.assert_frame_equal(expected.sim.wait_requests, actual.sim.wait_requests)
            pd.testing.assert_frame_equal(expected.sim.driver_table, actual.sim.driver_table)
            np.testing.assert_array_equal(expected.sim.rng.get_state()[1], actual.sim.rng.get_state()[1])
            self.assertEqual(expected.sim.rng.get_state()[2:], actual.sim.rng.get_state()[2:])
            per_step.append(dict(actual.last_signals))
        result = actual.summary()
        for name, value in expected.summary().items():
            self.assertEqual(result[name], value)
        for name in SIGNALS:
            self.assertAlmostEqual(sum(step[name] for step in per_step), result[name])
        self.assertEqual(result["end_unserved"], result["cohort_demand"] - result["unique_matched"])
        self.assertEqual(result["end_queue_orders"], len(actual.sim.wait_requests))
        self.assertEqual(result["cohort_demand"], carry + expected.sim.baseline_request_num - prewarm_baseline)
        self.assertEqual(actual.sim.current_step, 35)
        self.assertEqual(actual.sim.end_of_episode, 1)


if __name__ == "__main__":
    unittest.main()
