"""逐网格候选动作保持联合上下文一致，综合评分保留方向与符号。"""

import unittest

import numpy as np
import pandas as pd

from baseline.dlmrd.data import MinMaxPreprocessor
from baseline.dlmrd.decision import select_actions
from baseline.sequence import CONTEXT_COLUMNS, TARGET_COLUMNS
from src.utils.joint_context import joint_action_columns


class FakePredictor:
    def __init__(self, processor, function):
        self.preprocessor, self.function, self.calls = processor, function, []

    def predict(self, frame):
        if any(column.startswith("target_") for column in frame.columns):
            raise AssertionError
        self.calls.append(frame.copy())
        return pd.DataFrame(self.function(frame), columns=TARGET_COLUMNS)


class DLMRDDecisionTests(unittest.TestCase):
    def setUp(self):
        context = list(CONTEXT_COLUMNS) + list(joint_action_columns([0, 1]))
        self.processor = MinMaxPreprocessor(1, ["history"], context)
        self.processor.grid_ids, self.processor.fit_row_count = [0, 1], 4
        self.processor.scalers = {name: {"mean": np.zeros(size), "scale": np.ones(size)}
                                 for name, size in (("history", 1), ("context", len(context)), ("targets", 6))}
        self.frame = pd.DataFrame({"grid_id": [0, 1], "decision_start_sim_time": [600, 600],
            "interval_minutes": [5, 5], "price_multiplier": [1., 1.5],
            "matching_radius_km": [2., 3.], "lag_1_history": [.3, .4],
            "joint_price_grid_0": [1., 1.], "joint_radius_grid_0": [2., 2.],
            "joint_price_grid_1": [1.5, 1.5], "joint_radius_grid_1": [3., 3.]})

    def test_revenue_candidates_joint_consistency_and_no_context_mutation(self):
        def predict(frame):
            values = np.zeros((len(frame), 6))
            values[:, -1] = (frame.grid_id + 1) * 100 + 10 * frame.price_multiplier - frame.matching_radius_km
            return values
        predictor = FakePredictor(self.processor, predict)
        original = self.frame.copy(deep=True)
        selected = select_actions(predictor, self.frame, [1., 2.], [1., 4.])
        self.assertEqual(selected.grid_id.tolist(), [0, 1])
        self.assertEqual(selected.price_multiplier.tolist(), [2., 2.])
        self.assertEqual(selected.matching_radius_km.tolist(), [1., 1.])
        self.assertEqual(selected.score.tolist(), [119., 219.])
        self.assertTrue(set(TARGET_COLUMNS).issubset(selected.columns))
        self.assertEqual(len(predictor.calls), 2)
        for grid, candidates in enumerate(predictor.calls):
            self.assertEqual(len(candidates), 4)
            self.assertEqual(set(zip(candidates.price_multiplier, candidates.matching_radius_km)),
                             {(1., 1.), (1., 4.), (2., 1.), (2., 4.)})
            np.testing.assert_array_equal(candidates[f"joint_price_grid_{grid}"], candidates.price_multiplier)
            np.testing.assert_array_equal(candidates[f"joint_radius_grid_{grid}"], candidates.matching_radius_km)
            other = 1 - grid
            np.testing.assert_array_equal(candidates[f"joint_price_grid_{other}"],
                                          np.full(4, original.loc[other, "price_multiplier"]))
            np.testing.assert_array_equal(candidates[f"joint_radius_grid_{other}"],
                                          np.full(4, original.loc[other, "matching_radius_km"]))
        pd.testing.assert_frame_equal(self.frame, original)

    def test_paper_score_wait_negative_and_negative_scores_not_absolute(self):
        def predict(frame):
            values = np.zeros((len(frame), 6))
            values[:, 4] = 10 * frame.matching_radius_km
            return values
        predictor = FakePredictor(self.processor, predict)
        selected = select_actions(predictor, self.frame, [1.], [4., 1.], "paper_score")
        self.assertEqual(selected.matching_radius_km.tolist(), [1., 1.])
        np.testing.assert_allclose(selected.score, -10. / 6)

    def test_paper_score_uses_train_minmax_with_no_lower_or_upper_clipping(self):
        self.processor.scalers["targets"]["mean"] = np.full(6, 10.)
        self.processor.scalers["targets"]["scale"] = np.full(6, 10.)

        def predict(frame):
            values = np.full((len(frame), 6), 10.)
            values[:, -1] = frame.price_multiplier * 10.  # p=3 得2，p=4 得3。
            values[:, 4] = 5.  # 等待低于训练最小值，规范化为 -0.5。
            return values
        predictor = FakePredictor(self.processor, predict)
        selected = select_actions(predictor, self.frame, [3., 4.], [1.], "paper_score")
        self.assertEqual(selected.price_multiplier.tolist(), [4., 4.])
        np.testing.assert_allclose(selected.score, (3. + .5) / 6)
        self.assertEqual(selected.target_platform_revenue.tolist(), [40., 40.])

    def test_explicit_weights_and_tie_order(self):
        predictor = FakePredictor(self.processor, lambda frame: np.ones((len(frame), 6)))
        weights = {name: float(name == "target_platform_revenue") for name in TARGET_COLUMNS}
        selected = select_actions(predictor, self.frame, [2., 1.], [4., 1.], "paper_score", weights)
        self.assertEqual(selected.price_multiplier.tolist(), [2., 2.])
        self.assertEqual(selected.matching_radius_km.tolist(), [4., 4.])
        self.assertEqual(selected.score.tolist(), [1., 1.])

    def test_invalid_inputs_reject_current_labels_grids_actions_and_context(self):
        predictor = FakePredictor(self.processor, lambda frame: np.ones((len(frame), 6)))
        invalid = [self.frame.assign(target_platform_revenue=100.),
                   pd.concat([self.frame, self.frame.iloc[:1]], ignore_index=True),
                   self.frame.iloc[:1], self.frame.assign(grid_id=[0, 99]),
                   self.frame.assign(joint_price_grid_1=[1.5, 2.]),
                   self.frame.drop(columns="joint_radius_grid_0")]
        for frame in invalid:
            with self.subTest(columns=frame.columns), self.assertRaises(ValueError):
                select_actions(predictor, frame, [1.], [1.])
        for prices, radii in (([], [1.]), ([1., 1.], [1.]), ([np.nan], [1.]), ([1.], [0.])):
            with self.subTest(prices=prices, radii=radii), self.assertRaises(ValueError):
                select_actions(predictor, self.frame, prices, radii)
        with self.assertRaises(ValueError):
            select_actions(predictor, self.frame, [1.], [1.], "paper_score", [-1., 1., 1., 1., 1., 1.])
        self.assertEqual(predictor.calls, [])


if __name__ == "__main__":
    unittest.main()
