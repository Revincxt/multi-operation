"""运营参数诊断的统计口径与无侵入记录测试。"""

import unittest

import numpy as np
import pandas as pd

from diagnostics.operating_response import (
    COUNTS, RecordedSimulator, ratio, scenario_pairs,
    summarize_distances, summarize_metrics,
)
from src.env.actions import uniform_grid_actions
from src.env.simulator_env import Simulator


class OperatingResponseTests(unittest.TestCase):
    def test_ratios_use_total_counts_and_weighted_distance(self):
        frame = pd.DataFrame(0.0, index=range(2), columns=COUNTS)
        frame["candidate_request_num"] = [1, 99]
        frame["matched_request_num"] = [1, 9]
        frame["driver_offer_num"] = [1, 19]
        frame["driver_accept_num"] = [1, 9]
        frame["mean_pickup_distance_km"] = [0, 2]
        result = summarize_metrics(frame)
        self.assertAlmostEqual(result["order_completion_rate"], 0.1)
        self.assertAlmostEqual(result["driver_accept_rate"], 0.5)
        self.assertAlmostEqual(result["mean_pickup_distance_km"], 1.8)
        self.assertTrue(np.isnan(ratio(0, 0)))

    def test_distance_bins_include_rejections_and_exact_edges_once(self):
        offers = pd.DataFrame({
            "pickup_distance_km": [0, 0.25, 0.26, 0.5],
            "driver_accepted": [True, False, True, False],
            "matched": [True, False, False, False],
            "platform_revenue": [2, 0, 0, 0],
        })
        result = summarize_distances(offers, 1.0, 0.25)
        self.assertEqual(result["driver_offer_num"].sum(), 4)
        self.assertEqual(result["platform_revenue"].sum(), 2)
        self.assertAlmostEqual(result.iloc[0]["offer_success_rate"], 0.5)
        self.assertAlmostEqual(result.iloc[1]["driver_accept_rate"], 0.5)
        self.assertEqual(result.iloc[1]["offer_success_rate"], 0)
        self.assertTrue(np.isnan(result.iloc[2]["driver_accept_rate"]))

    def test_common_reference_scenario_is_not_repeated(self):
        pairs = scenario_pairs([0.4, 1.0, 4.0], [0.5, 1.0, 4.0], 1.0, 1.0)
        self.assertEqual(len(pairs), 5)
        self.assertEqual(pairs.count((1.0, 1.0)), 1)

    @staticmethod
    def run_small_simulator(sim_class):
        drivers = pd.DataFrame({
            "driver_id": [1, 2], "start_time": [18000, 18000],
            "end_time": [36000, 36000], "lng": [-73.99, -73.98],
            "lat": [40.75, 40.75], "grid_id": [0, 0],
        })
        requests = {18000 + minute * 60: [
            [minute * 10 + j, 0, 40.75, -73.99 + j * 0.007, 0, 40.75,
             -73.98, 1.0, 18000 + minute * 60, 0, 0, [], [], 160, 3.7, 0]
            for j in range(3)] for minute in range(8)}
        sim = sim_class(grid_num=1, driver_num=2,
                        price_response_model="utility_choice",
                        grid_action=uniform_grid_actions(1, 0.4, 4.0))
        sim.finish_run_step = 12
        sim.t_end = sim.t_initial + 12 * 60
        sim.reset(seed=42, given_data=True, request_databases=requests, driver_info=drivers)
        while not sim.end_of_episode:
            sim.step()
        return sim

    def test_recording_does_not_change_simulator_or_random_state(self):
        original = self.run_small_simulator(Simulator)
        recorded = self.run_small_simulator(RecordedSimulator)
        np.testing.assert_array_equal(original.evaluate_table, recorded.evaluate_table)
        pd.testing.assert_frame_equal(original.matched_requests, recorded.matched_requests)
        pd.testing.assert_frame_equal(original.wait_requests, recorded.wait_requests)
        pd.testing.assert_frame_equal(original.driver_table, recorded.driver_table)
        np.testing.assert_array_equal(original.rng.rand(20), recorded.rng.rand(20))
        offers = pd.concat(recorded.offer_records)
        self.assertGreater((~offers["driver_accepted"]).sum(), 0)
        self.assertGreater(offers["matched"].sum(), 0)
        self.assertTrue((offers.loc[~offers["matched"], "platform_revenue"] == 0).all())
        self.assertEqual(offers["matched"].sum(), recorded.matched_requests_num)
        self.assertEqual(len(offers), offers["driver_accepted"].sum()
                         + recorded.driver_rejected_requests_num)


if __name__ == "__main__":
    unittest.main()
