"""原始轨迹直接编码应与监督缓存一致，且不读取当前结果或动作。"""

import unittest

import numpy as np
import pandas as pd

from baseline.sequence import CONTEXT_COLUMNS, SequencePreprocessor
from data_pipeline.build_supervised_dataset import (
    HISTORY_ACTION_COLUMNS, PERIOD_METRIC_COLUMNS, build_history_samples,
)
from offline_rl.data import decode_actions, encode_state, state_context_indices
from offline_rl.prepare_collection import encode_periods
from src.env.observation import STATE_COLUMNS
from src.utils.joint_context import joint_context_columns


class RawReplayTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for step in range(8):
            for grid in range(6):
                row = {name: float(step + grid / 10) for name in PERIOD_METRIC_COLUMNS}
                row.update({name: float(step + grid) for name in STATE_COLUMNS})
                row.update(run_id="fixture", date="2015-05-18", seed=42, decision_step=step,
                    grid_id=grid, decision_start_minute_step=step * 5,
                    decision_end_minute_step_exclusive=(step + 1) * 5,
                    decision_start_sim_time=18000 + step * 300,
                    decision_end_sim_time_exclusive=18000 + (step + 1) * 300,
                    interval_minutes=5, joint_action_id=step % 4,
                    price_multiplier=[1.5, 1.6][step % 2], matching_radius_km=[2., 3.][grid % 2],
                    is_baseline_action=False)
                rows.append(row)
        self.periods = pd.DataFrame(rows)
        self.samples = build_history_samples(self.periods, 4, joint_action_context=True)
        context = list(CONTEXT_COLUMNS) + list(STATE_COLUMNS) + list(joint_context_columns(range(6)))
        self.processor = SequencePreprocessor(4, PERIOD_METRIC_COLUMNS + HISTORY_ACTION_COLUMNS, context).fit(self.samples)
        self.metadata = dict(grid_count=6, state_context_indices=state_context_indices(self.processor),
                             periods_per_run=4, action_pairs=[[p, r] for p in (1.5, 1.6) for r in (2., 3.)])

    def test_same_as_supervised_preprocessor_with_chronological_history(self):
        encoded = encode_periods(self.periods, self.processor, self.metadata)
        frame = self.samples.sort_values(["decision_step", "grid_id"])
        history, context, _ = self.processor.transform_inputs(frame)
        history = history.reshape(4, 6, 4, -1)
        context = context.reshape(4, 6, -1)
        np.testing.assert_array_equal(encoded["states"], encode_state(history, context, self.metadata["state_context_indices"]))
        np.testing.assert_array_equal(encoded["actions"], decode_actions(
            context, self.processor, np.array([1.5, 1.6]), np.array([2., 3.])))
        expected = self.samples.sort_values(["grid_id", "decision_step"]).target_platform_revenue.to_numpy(dtype="float64").reshape(6, 4).sum(0).astype("float32")
        np.testing.assert_array_equal(encoded["rewards"], expected)

    def test_current_outcomes_and_candidate_actions_do_not_leak_into_state(self):
        before = encode_periods(self.periods, self.processor, self.metadata)
        changed = self.periods.copy()
        current = changed.decision_step >= 4
        changed.loc[current, "price_multiplier"] = 1.6
        changed.loc[current, "matching_radius_km"] = 2.
        changed.loc[current, list(PERIOD_METRIC_COLUMNS)] += 100
        after = encode_periods(changed, self.processor, self.metadata)
        np.testing.assert_array_equal(before["states"][0], after["states"][0])
        self.assertNotEqual(before["rewards"][0], after["rewards"][0])
        self.assertFalse(np.array_equal(before["actions"][0], after["actions"][0]))

    def test_missing_region_and_period_are_rejected(self):
        with self.assertRaises(ValueError):
            encode_periods(self.periods.iloc[:-1], self.processor, self.metadata)
        with self.assertRaises(ValueError):
            encode_periods(self.periods[self.periods.decision_step != 2], self.processor, self.metadata)


if __name__ == "__main__":
    unittest.main()
