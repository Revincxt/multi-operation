import unittest
from unittest.mock import patch

import pandas as pd

from run_data_pipeline import run_data_pipeline


class DataPipelineTests(unittest.TestCase):
    @patch("run_data_pipeline.validate_supervised_dataset")
    @patch("run_data_pipeline.build_supervised_dataset")
    @patch("run_data_pipeline.run_batch_collection")
    def test_runs_all_three_stages(self, collect, build, validate):
        collect.return_value = {"run_count": 3}
        build.return_value = (
            pd.DataFrame({"sample": [1]}),
            {"row_count": 1},
        )
        validate.return_value = {"passed": True}

        result = run_data_pipeline(
            dates=["2015-05-18"],
            seeds=[42, 43, 44],
            grid_num=35,
            driver_num=500,
            order_sample_ratio=0.077,
            simulation_steps=4,
            start_times=["05:00", "12:00"],
            data_dir="custom_data",
            decision_freq=2,
            history_steps=1,
            price_action_values=[0.7, 1.0],
            matching_radius_values=[0.5, 1.0],
            price_response_model="aggregate_elasticity",
            action_seed_offset=9,
            validation_ratio=0.2,
            test_ratio=0.1,
            split_seed=7,
            raw_output_dir="raw",
            dataset_output_dir="dataset",
            joint_action_context=True)

        collect.assert_called_once_with(
            dates=["2015-05-18"],
            seeds=[42, 43, 44],
            grid_num=35,
            driver_num=500,
            order_sample_ratio=0.077,
            simulation_steps=4,
            start_times=["05:00", "12:00"],
            data_dir="custom_data",
            decision_freq=2,
            price_action_values=[0.7, 1.0],
            matching_radius_values=[0.5, 1.0],
            price_response_model="aggregate_elasticity",
            enable_reposition=False,
            action_seed_offset=9,
            output_dir="raw",
            overwrite=False)
        build.assert_called_once_with(
            input_dir="raw",
            output_dir="dataset",
            history_steps=1,
            validation_ratio=0.2,
            test_ratio=0.1,
            split_seed=7,
            split_by="run",
            overwrite=False,
            joint_action_context=True)
        validate.assert_called_once_with("dataset")
        self.assertEqual(result["batch_manifest"]["run_count"], 3)
        self.assertTrue(result["validation_report"]["passed"])

    @patch("run_data_pipeline.run_batch_collection")
    def test_date_split_requires_enough_dates_before_collection(self, collect):
        for dates in (["2015-05-18"], ["2015-05-18", "2015-05-19"]):
            with self.subTest(dates=dates), self.assertRaises(ValueError):
                run_data_pipeline(dates=dates, split_by="date")
        collect.assert_not_called()

    @patch("run_data_pipeline.validate_supervised_dataset")
    @patch("run_data_pipeline.build_supervised_dataset")
    @patch("run_data_pipeline.run_batch_collection")
    def test_stops_when_validation_fails(self, collect, build, validate):
        collect.return_value = {}
        build.return_value = (pd.DataFrame(), {})
        validate.return_value = {"passed": False}

        with self.assertRaises(RuntimeError):
            run_data_pipeline(
                dates=["2015-05-18"],
                seeds=[42],
                raw_output_dir="raw",
                dataset_output_dir="dataset")


if __name__ == "__main__":
    unittest.main()
