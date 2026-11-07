"""六指标热力图的完整网格、口径和颜色方向测试。"""

import unittest

import numpy as np
import pandas as pd

from diagnostics.plot_density_heatmaps import METRICS, color_limits, draw_heatmap, prepare_metrics


def example_data():
    config = {"scenarios": ["low", "high"], "prices": [1.5, 1.6],
              "radii": [2.0, 3.0], "screening_seeds": [51, 52]}
    rows = []
    for scenario in config["scenarios"]:
        for seed in config["screening_seeds"]:
            for price in config["prices"]:
                for radius in config["radii"]:
                    rows.append({
                        "phase": "screening", "scenario": scenario, "seed": seed,
                        "price_multiplier": price, "matching_radius_km": radius,
                        "sampled_orders_sha256": f"orders-{scenario}-{seed}",
                        "initial_drivers_sha256": f"drivers-{scenario}-{seed}",
                        "generated_baseline_requests": 101, "baseline_request_num": 100,
                        "submitted_request_num": 120, "request_rate": 1.2,
                        "driver_offer_num": 100, "driver_accept_num": 80, "driver_accept_rate": 0.8,
                        "matched_request_num": 60, "candidate_request_num": 195,
                        "unmatched_cancel_num": 5, "completion_denominator": 200,
                        "order_completion_rate": 0.3, "mean_wait_to_match_seconds": 70 + seed,
                        "platform_revenue": 200 + seed,
                    })
    return pd.DataFrame(rows), config


class DensityHeatmapTests(unittest.TestCase):
    def test_six_requested_metrics_and_seed_means(self):
        frame, config = example_data()
        data = prepare_metrics(frame, config)
        self.assertEqual(len(METRICS), 6)
        self.assertIn("baseline_request_num", METRICS)
        self.assertNotIn("matched_request_num", METRICS)
        self.assertEqual(len(data), 2 * 2 * 2 * 6)
        self.assertTrue(data.seed_count.eq(2).all())
        self.assertTrue(data.valid_seed_count.eq(2).all())
        self.assertTrue(data.loc[data.metric == "request_rate", "value"].eq(1.2).all())
        self.assertTrue(data.loc[data.metric == "platform_revenue", "value"].eq(251.5).all())

    def test_rejects_mixed_phases_incomplete_grid_and_unpaired_inputs(self):
        frame, config = example_data()
        for bad in (frame.iloc[1:], pd.concat([frame, frame.iloc[:1]]),
                    frame.assign(phase="confirmation")):
            with self.assertRaises(ValueError):
                prepare_metrics(bad, config)
        for column in ("sampled_orders_sha256", "initial_drivers_sha256"):
            bad = frame.copy()
            bad.loc[0, column] = "different"
            with self.assertRaises(ValueError):
                prepare_metrics(bad, config)
        bad = frame.copy()
        bad.loc[0, "request_rate"] = 1.0
        with self.assertRaises(AssertionError):
            prepare_metrics(bad, config)

    def test_no_matches_wait_is_missing_not_zero(self):
        frame, config = example_data()
        frame["matched_request_num"] = 0
        frame["order_completion_rate"] = 0.0
        frame["mean_wait_to_match_seconds"] = np.nan
        data = prepare_metrics(frame, config)
        wait = data[data.metric == "mean_wait_to_match_seconds"]
        self.assertTrue(wait.value.isna().all())
        self.assertTrue(wait.valid_seed_count.eq(0).all())
        frame["mean_wait_to_match_seconds"] = 0.0
        with self.assertRaises(ValueError):
            prepare_metrics(frame, config)

    def test_shared_limits_do_not_clip_expanded_demand(self):
        self.assertEqual(color_limits([0.5, 1.17], "request_rate"), (0.0, 1.2))
        self.assertEqual(color_limits([0.2, 0.9], "driver_accept_rate"), (0.0, 1.0))
        self.assertEqual(color_limits([5000, 25000], "baseline_request_num"), (0.0, 25000.0))
        self.assertEqual(color_limits([np.nan], "mean_wait_to_match_seconds"), (0.0, 1.0))
        lower, upper = color_limits([60, 60], "mean_wait_to_match_seconds")
        self.assertLess(lower, upper)

    def test_heatmap_values_axes_reference_and_wait_color_direction(self):
        try:
            import matplotlib
        except ModuleNotFoundError:
            self.skipTest("matplotlib not installed")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frame, config = example_data()
        data = prepare_metrics(frame, config)
        data = data[data.scenario == "low"]
        fig, axes = plt.subplots(1, 2)
        try:
            request = draw_heatmap(fig, axes[0], data, "request_rate", config["prices"],
                                   config["radii"], (0, 1.2), "↑")
            wait = draw_heatmap(fig, axes[1], data, "mean_wait_to_match_seconds", config["prices"],
                                config["radii"], (60, 180), "↓")
            np.testing.assert_allclose(request.get_array(), 1.2)
            self.assertTrue(all(text.get_text() == "120.0%" for text in axes[0].texts))
            self.assertEqual(len(axes[0].patches), 1)
            self.assertEqual([tick.get_text() for tick in axes[0].get_xticklabels()], ["2", "3"])
            self.assertEqual([tick.get_text() for tick in axes[0].get_yticklabels()], ["1.5", "1.6"])
            self.assertEqual(wait.cmap.name, "viridis_r")
            self.assertEqual(axes[0].get_title(), "↑")
            self.assertEqual(axes[1].get_title(), "↓")
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
