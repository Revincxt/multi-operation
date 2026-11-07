import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_pipeline.collect_trajectory_batch import run_batch_collection


class BatchCollectionTests(unittest.TestCase):
    @patch("data_pipeline.collect_trajectory_batch.check_collection_inputs")
    @patch("data_pipeline.collect_trajectory_batch.run_trajectory_collection")
    def test_runs_date_seed_product_and_writes_manifest(self, collect, check_inputs):
        def fake_collection(**kwargs):
            run_id = (
                f"{kwargs['date']}_simseed{kwargs['seed']}_"
                f"actionseed{kwargs['action_seed']}")
            return {
                "run_dir": str(Path(kwargs["output_dir"]) / run_id),
                "manifest": {
                    "run_id": run_id,
                    "quality": {"decision_count": 2},
                    "summary": {"total_gbv": 1.0},
                },
            }

        collect.side_effect = fake_collection
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = run_batch_collection(
                dates=["2015-05-18", "2015-05-19"],
                seeds=[42, 43],
                simulation_steps=4,
                action_seed_offset=7,
                output_dir=temporary_dir)

            self.assertEqual(collect.call_count, 4)
            self.assertEqual(manifest["run_count"], 4)
            self.assertEqual(
                [run["action_seed"] for run in manifest["runs"]],
                [49, 50, 100049, 100050])
            manifest_path = Path(temporary_dir) / "batch_manifest.json"
            self.assertTrue(manifest_path.exists())
            with manifest_path.open(encoding="utf-8") as file:
                saved = json.load(file)
            self.assertEqual(saved["runs"], manifest["runs"])
            check_inputs.assert_called_once_with(
                ["2015-05-18", "2015-05-19"], 263, "my_data")

    @patch("data_pipeline.collect_trajectory_batch.check_collection_inputs")
    @patch("data_pipeline.collect_trajectory_batch.run_trajectory_collection")
    def test_runs_date_time_seed_product(self, collect, check_inputs):
        def fake_collection(**kwargs):
            run_id = f"{kwargs['date']}_{kwargs['start_time']}_{kwargs['seed']}"
            return {"run_dir": run_id, "manifest": {
                "run_id": run_id, "quality": {}, "summary": {}}}

        collect.side_effect = fake_collection
        with tempfile.TemporaryDirectory() as directory:
            manifest = run_batch_collection(
                dates=["2015-05-18", "2015-05-19"], seeds=[42, 43],
                start_times=["05:00", "12:00", "18:00"], simulation_steps=360,
                data_dir="custom_data", output_dir=directory)
        self.assertEqual(collect.call_count, 12)
        self.assertEqual(manifest["run_count"], 12)
        self.assertEqual(len({r["run_id"] for r in manifest["runs"]}), 12)
        self.assertEqual(len({r["action_seed"] for r in manifest["runs"]}), 12)
        self.assertEqual(manifest["start_times"], ["05:00", "12:00", "18:00"])
        self.assertTrue(all(call.kwargs["simulation_steps"] == 360
                            for call in collect.call_args_list))

    @patch("data_pipeline.collect_trajectory_batch.run_trajectory_collection")
    def test_missing_later_date_fails_before_any_collection(self, collect):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("drivers_grid35_1000.pickle", "orders_grid35_2015-05-18.pkl",
                         "orders_grid35_2015-05-18-map263.csv"):
                (root / name).touch()
            with self.assertRaises(FileNotFoundError):
                run_batch_collection(
                    dates=["2015-05-18", "2015-05-19"], seeds=[42],
                    data_dir=root, output_dir=root / "output")
            collect.assert_not_called()
            self.assertFalse((root / "output").exists())

    def test_rejects_duplicate_times_and_cross_midnight_windows(self):
        for times in (["05:00", "5:00"], ["23:00"], []):
            with self.subTest(times=times), self.assertRaises(ValueError):
                run_batch_collection(
                    dates=["2015-05-18"], seeds=[42], start_times=times)

    def test_rejects_duplicate_dates_or_seeds(self):
        with self.assertRaises(ValueError):
            run_batch_collection(
                dates=["2015-05-18", "2015-05-18"],
                seeds=[42],
                output_dir="unused")
        with self.assertRaises(ValueError):
            run_batch_collection(
                dates=["2015-05-18"],
                seeds=[42, 42],
                output_dir="unused")


if __name__ == "__main__":
    unittest.main()
