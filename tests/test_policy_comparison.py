"""检查闭环对比的历史边界、共同输入和运营指标口径。"""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from baseline.compare_policies import (
    candidate_features, revenue_actions, run_episode, sample_common_orders,
    summarize_operation, plot_policy_results, METRICS, POLICIES,
)
from data_pipeline.build_supervised_dataset import (
    HISTORY_ACTION_COLUMNS, PERIOD_METRIC_COLUMNS, build_history_samples,
)


class FixedPredictor:
    def __init__(self):
        self.preprocessor = SimpleNamespace(
            grid_ids=[0], history_steps=2,
            history_features=list(PERIOD_METRIC_COLUMNS + HISTORY_ACTION_COLUMNS))
        self.inputs = []

    def predict(self, frame):
        self.inputs.append(frame.copy())
        return pd.DataFrame({"target_platform_revenue":
                             -abs(frame.price_multiplier - 1.0) - abs(frame.matching_radius_km - 1.0)})


class PolicyComparisonTests(unittest.TestCase):
    def test_common_sampling_is_repeatable_and_keeps_minute_counts(self):
        requests = {second: [[second * 10 + j] for j in range(10)]
                    for second in (18001, 18060, 18121, 18181)}
        first, ids = sample_common_orders(requests, 18000, 18240, 0.3, 42)
        second, repeated = sample_common_orders(requests, 18000, 18240, 0.3, 42)
        self.assertEqual(first, second)
        pd.testing.assert_frame_equal(ids, repeated)
        self.assertEqual(len(ids), 9)
        self.assertEqual(ids.timestamp.tolist(), [18001] * 3 + [18060] * 3 + [18121] * 3)
        self.assertTrue(ids.order_id.is_unique)
        for timestamp, orders in first.items():
            self.assertTrue(all(order in requests[timestamp] for order in orders))
        _, different = sample_common_orders(requests, 18000, 18240, 0.3, 43)
        self.assertFalse(ids.equals(different))

    def test_operation_uses_counts_including_cancellations(self):
        frame = pd.DataFrame({
            "baseline_request_num": [1, 9], "submitted_request_num": [1, 4],
            "driver_offer_num": [1, 9], "driver_accept_num": [1, 4],
            "matched_request_num": [1, 3], "candidate_request_num": [1, 7],
            "unmatched_cancel_num": [0, 2], "pickup_cancel_num": [0, 1],
            "platform_revenue": [1, 9], "mean_wait_to_match_seconds": [0, 20],
        })
        result = summarize_operation(frame)
        self.assertEqual(result["request_rate"], 0.5)
        self.assertEqual(result["driver_accept_rate"], 0.5)
        self.assertEqual(result["order_completion_rate"], 0.4)
        self.assertEqual(result["mean_wait_to_match_seconds"], 15.0)
        self.assertEqual(result["platform_revenue"], 10.0)
        frame.loc[:] = 0
        self.assertIsNone(summarize_operation(frame)["order_completion_rate"])

    def test_same_actions_reproduce_fixed_simulation_and_offline_features(self):
        config = {"grid_num": 1, "driver_num": 2, "decision_freq": 2,
                  "history_steps": 2, "start_time": "05:00", "simulation_steps": 10,
                  "date": "2015-05-18", "fixed_price": 1.0, "fixed_radius": 1.0,
                  "price_response_model": "utility_choice", "price_action_values": [1.0, 1.5],
                  "matching_radius_values": [0.5, 1.0]}
        drivers = pd.DataFrame({"driver_id": [1, 2], "start_time": [18000, 18000],
                                "end_time": [36000, 36000], "lng": [-73.99, -73.98],
                                "lat": [40.75, 40.75], "grid_id": [0, 0]})
        requests = {18000 + minute * 60: [
            [minute * 10 + j, 0, 40.75, -73.99 + j * 0.007, 0, 40.75,
             -73.98, 1.0, 18000 + minute * 60, 0, 0, [], [], 160, 3.7, 0]
            for j in range(3)] for minute in range(9)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed = run_episode(config, "fixed", 42, requests, drivers, None, root / "fixed")
            predictor = FixedPredictor()
            model = run_episode(config, "lstm", 42, requests, drivers, predictor, root / "lstm")
            calls = []

            def policy(periods, decision, start, interval, state):
                from src.env.actions import GridAction
                calls.append(decision)
                self.assertEqual(periods.decision_step.max(), decision - 1)
                return {0: GridAction(1.0, 1.0)}

            custom = run_episode(config, "custom", 42, requests, drivers, None, root / "custom",
                                 action_policy=policy)
            self.assertEqual(calls, [2, 3, 4])
            self.assertEqual(custom["platform_revenue"], fixed["platform_revenue"])
            self.assertEqual(custom["warmup_metrics_sha256"], fixed["warmup_metrics_sha256"])
            self.assertEqual(len(predictor.inputs), 3)
            for key in ("platform_revenue", "baseline_request_num", "initial_driver_sha256",
                        "driver_accept_rate", "order_completion_rate", "full_window"):
                self.assertEqual(fixed[key], model[key])
            fixed_minutes = pd.read_csv(root / "fixed" / "grid_minute_metrics.csv.gz")
            model_minutes = pd.read_csv(root / "lstm" / "grid_minute_metrics.csv.gz")
            pd.testing.assert_frame_equal(fixed_minutes, model_minutes)
            periods = pd.read_csv(root / "lstm" / "decision_metrics.csv.gz")
            offline = build_history_samples(periods, 2)
            for index, candidates in enumerate(predictor.inputs):
                self.assertFalse(any(column.startswith("target_") for column in candidates))
                history_columns = [column for column in candidates if column.startswith("lag_")]
                np.testing.assert_allclose(candidates.iloc[0][history_columns].to_numpy(dtype=float),
                                           offline.iloc[index][history_columns].to_numpy(dtype=float))
                self.assertEqual(candidates.decision_start_sim_time.iloc[0], 18000 + (index + 2) * 120)
            with self.assertRaises(ValueError):
                candidate_features(periods, predictor.preprocessor, 2, 18240, 2, [1.0], [1.0])
            actions, predicted = revenue_actions(
                predictor, periods[periods.decision_step < 2], 2, 18240, 2, [1.0, 1.5], [0.5, 1.0])
            self.assertEqual(actions[0].price_multiplier, 1.0)
            self.assertEqual(actions[0].matching_radius_km, 1.0)
            self.assertEqual(predicted[0], 0.0)

    def test_charts_distinguish_error_and_operational_directions(self):
        try:
            import matplotlib.figure
        except ModuleNotFoundError:
            self.skipTest("matplotlib not installed")
        from baseline.compare_models import MODELS, plot_comparison
        from baseline.plot_results import plot_results
        from tests.test_plot_results import PlotResultsTests

        figures = []

        def capture(figure, *args, **kwargs):
            figures.append([(axis.get_title(), axis.get_ylabel()) for axis in figure.axes])

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(matplotlib.figure.Figure, "savefig", capture):
            root = Path(temporary)
            scores = pd.DataFrame([{"model": model, "metric": metric, "mae": 1.0}
                                   for model in MODELS for metric in METRICS])
            plot_comparison(scores, root / "errors.png")
            self.assertTrue(all("↓" in label for _, label in figures[-1]))
            PlotResultsTests()._write_run(root)
            plot_results(root)
            self.assertTrue(all("↓" in label for _, label in figures[-1]))
            config = {"fixed_radius": 3.0, "fixed_price": 1.5, "grid_num": 6,
                      "driver_num": 250, "history_steps": 4, "decision_freq": 5}
            (root / "manifest.json").write_text(json.dumps({"config": config, "seeds": [1, 2]}))
            pd.DataFrame([{"policy": policy, "seed": seed, **dict.fromkeys(METRICS, 0.5)}
                          for policy in POLICIES for seed in (1, 2)]).to_csv(
                              root / "run_summary.csv", index=False)
            plot_policy_results(root)
            titles = [title for title, _ in figures[-1]]
            self.assertIn("no better direction", titles[0])
            self.assertTrue(all("↑" in titles[index] for index in (1, 2, 3, 5)))
            self.assertIn("↓", titles[4])


if __name__ == "__main__":
    unittest.main()
