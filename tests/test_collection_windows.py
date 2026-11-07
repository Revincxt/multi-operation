"""日内采集窗口、输出隔离和分钟时间戳检查。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data_pipeline.collect_trajectories import run_trajectory_collection, simulation_window


class CollectionWindowTests(unittest.TestCase):
    def test_window_supports_more_than_five_hours_and_rejects_midnight_overflow(self):
        self.assertEqual(simulation_window("12:00", 360), ("12:00", 43200, 64800))
        self.assertEqual(simulation_window("23:00", 60), ("23:00", 82800, 86400))
        for start, steps in (("23:00", 61), ("24:00", 1), ("05:00", 0),
                             ("05:00", 1.5), ("05:00", True)):
            with self.subTest(start=start, steps=steps), self.assertRaises(ValueError):
                simulation_window(start, steps)

    @patch("data_pipeline.collect_trajectories.check_collection_inputs")
    @patch("data_pipeline.collect_trajectories.SimulatorPattern")
    def test_collector_uses_window_before_reset_and_keeps_outputs_distinct(self, pattern, check):
        pattern.return_value.request_all = {0: [], 86399: []}
        pattern.return_value.driver_info = pd.DataFrame({
            "driver_id": [1], "start_time": [18000], "end_time": [36000],
            "lng": [-73.99], "lat": [40.75], "grid_id": [0]})
        with tempfile.TemporaryDirectory() as directory:
            results = []
            for start, steps in (("05:00", 2), ("12:00", 2), ("12:00", 301)):
                result = run_trajectory_collection(
                    "2015-05-18", grid_num=1, driver_num=1, seed=42,
                    start_time=start, simulation_steps=steps, decision_freq=2,
                    price_action_values=[1.0], matching_radius_values=[1.0],
                    output_dir=directory, data_dir="custom_data")
                results.append(result)
                expected = simulation_window(start, steps)[1]
                minute = result["minute_metrics"]
                self.assertEqual(len(minute), steps)
                self.assertEqual(minute["sim_time"].min(), expected)
                self.assertEqual(minute["sim_time"].max(), expected + (steps - 1) * 60)
                self.assertTrue(minute["idle_driver_num"].eq(1).all())
                self.assertEqual(result["manifest"]["start_sim_time"], expected)
                self.assertEqual(result["manifest"]["end_sim_time_exclusive"],
                                 expected + steps * 60)
            self.assertEqual(len({result["run_dir"] for result in results}), 3)
            pattern.load_mapping.assert_not_called()
            with self.assertRaises(FileExistsError):
                run_trajectory_collection(
                    "2015-05-18", grid_num=1, driver_num=1, start_time="05:00",
                    simulation_steps=2, decision_freq=2, output_dir=directory)

    @patch("data_pipeline.collect_trajectories.check_collection_inputs")
    @patch("data_pipeline.collect_trajectories.SimulatorPattern")
    def test_rejects_window_outside_order_index_without_creating_run(self, pattern, check):
        pattern.return_value.request_all = {18000: [], 35999: []}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_trajectory_collection(
                    "2015-05-18", grid_num=1, start_time="12:00",
                    simulation_steps=2, output_dir=directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
