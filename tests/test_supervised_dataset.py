import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_pipeline.build_supervised_dataset import (
    aggregate_decision_metrics,
    assign_run_splits,
    build_history_samples,
    build_supervised_dataset,
    resolve_split_by,
)
from data_pipeline.validate_supervised_dataset import validate_supervised_dataset
from data_pipeline.collect_trajectories import joint_action_sampling


def _raw_tables(run_id="run_0"):
    """构造三个决策周期的一网格原始数据。"""
    action_records = []
    minute_records = []
    for decision_step in range(3):
        start = decision_step * 2
        action_records.append({
            "decision_step": decision_step,
            "decision_start_minute_step": start,
            "decision_end_minute_step_exclusive": start + 2,
            "grid_id": 0,
            "joint_action_id": decision_step,
            "price_multiplier": 1.0 + 0.3 * decision_step,
            "matching_radius_km": 1.0 + decision_step,
            "is_baseline_action": decision_step == 0,
        })
        for offset in range(2):
            matched = decision_step + offset + 1
            minute_records.append({
                "minute_step": start + offset,
                "sim_time": 18000 + 60 * (start + offset),
                "decision_step": decision_step,
                "grid_id": 0,
                "candidate_request_num": 3 + 2 * offset,
                "matched_request_num": matched,
                "gbv": 10 * matched,
                "platform_revenue": 2.5 * matched,
                "baseline_request_num": 2 + offset,
                "submitted_request_num": 1 + offset,
                "driver_offer_num": 2 * matched,
                "driver_accept_num": matched,
                "unmatched_cancel_num": offset,
                "pickup_cancel_num": offset,
                "mean_wait_to_match_seconds": 10 + 10 * offset,
                "mean_pickup_time_seconds": 60 + 30 * offset,
                "mean_pickup_distance_km": 1.0 + 0.5 * offset,
                "idle_driver_num": 10 - offset,
                "busy_driver_num": 2 + offset,
                "repositioning_driver_num": offset,
                "vehicle_occupancy_rate": 0.2 + 0.1 * offset,
            })
    metadata = {
        "run_id": run_id,
        "date": "2015-05-18",
        "simulator_seed": 42,
        "decision_freq_minutes": 2,
    }
    return pd.DataFrame(action_records), pd.DataFrame(minute_records), metadata


class SupervisedDatasetTests(unittest.TestCase):
    @staticmethod
    def _planned_runs(root, splits=("test", "train", "validation", "train")):
        for index, split in enumerate(splits):
            plan, minute, metadata = _raw_tables(f"run_{index}")
            plan["price_multiplier"] = 1.0
            plan["matching_radius_km"] = 1.0
            plan["joint_action_id"] = 0
            if split is not None:
                metadata["planned_split"] = split
            raw = root / metadata["run_id"]
            raw.mkdir(parents=True)
            plan.to_csv(raw / "action_plan.csv.gz", index=False)
            minute.to_csv(raw / "grid_minute_metrics.csv.gz", index=False)
            metadata["files"] = {"action_plan": "action_plan.csv.gz",
                                 "minute_metrics": "grid_minute_metrics.csv.gz"}
            metadata["file_sha256"] = {
                name: hashlib.sha256((raw / name).read_bytes()).hexdigest()
                for name in metadata["files"].values()}
            (raw / "manifest.json").write_text(json.dumps(metadata))

    def test_stream_build_uses_all_runs_and_preserves_planned_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._planned_runs(root / "raw")
            # 未发布的工作目录不能混入训练数据。
            self._planned_runs(root / "raw" / ".working_unfinished", ("train",))
            dataset, manifest = build_supervised_dataset(
                root / "raw", root / "dataset", history_steps=1,
                split_seed=999, return_dataset=False)
            self.assertIsNone(dataset)
            dataset = pd.read_csv(root / "dataset" / "supervised_dataset.csv")
            self.assertEqual(len(dataset), 8)
            self.assertEqual(manifest["source_run_count"], 4)
            self.assertEqual(manifest["split_order"], "collection_plan")
            expected = {"run_0": "test", "run_1": "train",
                        "run_2": "validation", "run_3": "train"}
            self.assertEqual(dataset.groupby("run_id")["split"].first().to_dict(), expected)
            self.assertEqual(manifest["source_run_splits"], expected)
            self.assertEqual(manifest["source_run_sample_counts"], dict.fromkeys(expected, 2))
            normal = validate_supervised_dataset(root / "dataset")
            for size in (1, 2, 3, 7):
                with self.subTest(chunk_size=size):
                    streamed = validate_supervised_dataset(root / "dataset", chunk_size=size)
                    self.assertTrue(streamed["passed"])
                    self.assertEqual(normal["summary"], streamed["summary"])
                    self.assertEqual(normal["joint_action_coverage"], streamed["joint_action_coverage"])

    def test_planned_split_missing_or_conflicting_date_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._planned_runs(root / "mixed", ("train", None))
            with self.assertRaises(ValueError):
                build_supervised_dataset(root / "mixed", root / "out", history_steps=1)
            self._planned_runs(root / "complete")
            with self.assertRaises(ValueError):
                build_supervised_dataset(root / "complete", root / "out", history_steps=1, split_by="date")
            self.assertFalse((root / "out" / "supervised_dataset.csv").exists())

    def test_run_limit_uses_collection_index_and_preserves_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._planned_runs(root / "raw")
            # 采集编号与文件名字典序不同，不能按文件名误选轨迹。
            for run_id, run_index in enumerate((3, 0, 2, 1)):
                path = root / "raw" / f"run_{run_id}" / "manifest.json"
                metadata = json.loads(path.read_text())
                metadata["run_index"] = run_index
                path.write_text(json.dumps(metadata))
            dataset, manifest = build_supervised_dataset(
                root / "raw", root / "dataset", history_steps=1, max_runs=3)
            self.assertEqual(set(dataset.run_id), {"run_1", "run_2", "run_3"})
            self.assertEqual(manifest["source_run_limit"], 3)
            self.assertEqual(manifest["source_selection_order"], "run_index")
            self.assertEqual(manifest["source_run_count"], 3)
            self.assertEqual(manifest["row_count"], 6)
            self.assertEqual(manifest["source_run_splits"], {
                "run_1": "train", "run_2": "validation", "run_3": "train"})
            self.assertEqual(dataset.run_id.drop_duplicates().tolist(),
                             ["run_1", "run_2", "run_3"])
            for limit in (0, -1, True, 1.5):
                with self.subTest(max_runs=limit), self.assertRaises(ValueError):
                    build_supervised_dataset(root / "raw", root / "invalid", max_runs=limit)

    def test_corrupt_raw_file_fails_before_publishing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._planned_runs(root / "raw")
            (root / "raw" / "run_3" / "action_plan.csv.gz").write_bytes(b"broken")
            with self.assertRaises(ValueError):
                build_supervised_dataset(root / "raw", root / "dataset", history_steps=1)
            self.assertFalse((root / "dataset" / "supervised_dataset.csv").exists())
            self.assertFalse((root / "dataset" / "dataset_manifest.json").exists())

    def test_stream_validation_detects_split_changes_and_late_repeated_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._planned_runs(root / "raw")
            dataset, _ = build_supervised_dataset(root / "raw", root / "dataset", history_steps=1)
            modified = dataset.copy()
            modified.loc[modified["run_id"] == "run_0", "split"] = "train"
            modified.loc[modified["run_id"] == "run_1", "split"] = "test"
            modified.to_csv(root / "dataset" / "supervised_dataset.csv", index=False)
            report = validate_supervised_dataset(root / "dataset", chunk_size=2)
            self.assertFalse(report["passed"])
            self.assertTrue(report["checks"]["split_counts_match_manifest"])
            self.assertFalse(report["checks"]["source_run_splits_are_preserved"])
            pd.concat([dataset, dataset.iloc[[0]]]).to_csv(
                root / "dataset" / "supervised_dataset.csv", index=False)
            report = validate_supervised_dataset(root / "dataset", chunk_size=2)
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["run_blocks_are_contiguous"])

    def test_action_coverage_with_fewer_grids_than_joint_actions(self):
        """6 网格不能同时执行 8 种动作，但跨周期仍须覆盖完整组合。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            plans, minutes = [], []
            actions = [(price, radius) for price in (1.0, 1.3)
                       for radius in (0.5, 1.0, 1.5, 2.0)]
            for grid_id in range(6):
                plan, minute, metadata = _raw_tables()
                plan["grid_id"] = grid_id
                minute["grid_id"] = grid_id
                for index in plan.index:
                    action_id = (int(plan.loc[index, "decision_step"]) * 6 + grid_id) % 8
                    plan.loc[index, "joint_action_id"] = action_id
                    plan.loc[index, "price_multiplier"] = actions[action_id][0]
                    plan.loc[index, "matching_radius_km"] = actions[action_id][1]
                plans.append(plan)
                minutes.append(minute)
            pd.concat(plans).to_csv(raw / "action_plan.csv", index=False)
            pd.concat(minutes).to_csv(raw / "grid_minute_metrics.csv", index=False)
            (raw / "manifest.json").write_text(json.dumps(metadata))
            dataset, _ = build_supervised_dataset(raw, root / "dataset", history_steps=1)
            report = validate_supervised_dataset(root / "dataset")
            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["joint_action_count"], 8)

            # 同周期出现重复、遗漏应有的第 6 种动作，仍然必须报错。
            first_period = dataset.index[dataset["decision_step"] == 1]
            columns = ["price_multiplier", "matching_radius_km", "joint_action_id"]
            dataset.loc[first_period[1], columns] = dataset.loc[first_period[0], columns]
            dataset.to_csv(root / "dataset" / "supervised_dataset.csv", index=False)
            report = validate_supervised_dataset(root / "dataset")
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["joint_actions_cover_each_period"])

    def test_mixture_sampling_allows_repeated_and_partially_covered_actions(self):
        """新采样允许网格间重复；候选外动作和重复主键仍须失败。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            plans, minutes = [], []
            for grid in range(6):
                plan, minute, metadata = _raw_tables()
                plan["grid_id"], minute["grid_id"] = grid, grid
                plans.append(plan)
                minutes.append(minute)
            pd.concat(plans).to_csv(raw / "action_plan.csv", index=False)
            pd.concat(minutes).to_csv(raw / "grid_minute_metrics.csv", index=False)
            metadata.update(action_sampling=joint_action_sampling([1.0, 1.3, 1.6], [1.0, 2.0, 3.0]),
                            sampling_mode="uniform")
            (raw / "manifest.json").write_text(json.dumps(metadata))
            dataset, manifest = build_supervised_dataset(raw, root / "dataset", history_steps=1)
            self.assertEqual(manifest["sampling_mode_run_counts"], {"uniform": 1})
            normal = validate_supervised_dataset(root / "dataset")
            streamed = validate_supervised_dataset(root / "dataset", chunk_size=2)
            self.assertTrue(normal["passed"])
            self.assertEqual(normal["checks"], streamed["checks"])
            self.assertEqual(normal["summary"]["periods_with_repeated_grid_actions"], 2)
            self.assertNotIn("joint_actions_are_balanced", normal["checks"])
            dataset.loc[dataset.index[0], "price_multiplier"] = 9.0
            dataset.to_csv(root / "dataset" / "supervised_dataset.csv", index=False)
            invalid = validate_supervised_dataset(root / "dataset")
            self.assertFalse(invalid["passed"])
            self.assertFalse(invalid["checks"]["actions_match_declared_candidates"])

    def test_aggregate_and_build_history_samples(self):
        action_plan, minute_metrics, metadata = _raw_tables()

        periods = aggregate_decision_metrics(
            action_plan, minute_metrics, metadata)
        dataset = build_history_samples(periods, history_steps=2)

        first_period = periods.iloc[0]
        self.assertEqual(first_period["interval_minutes"], 2)
        self.assertEqual(first_period["decision_start_sim_time"], 18000)
        self.assertEqual(
            first_period["decision_end_sim_time_exclusive"], 18120)
        self.assertEqual(first_period["candidate_request_num"], 4)
        self.assertEqual(first_period["matched_request_num"], 3)
        self.assertEqual(first_period["gbv"], 30)
        self.assertAlmostEqual(first_period["request_rate"], 3 / 5)
        self.assertAlmostEqual(first_period["driver_accept_rate"], 0.5)
        self.assertAlmostEqual(first_period["order_completion_rate"], 1 / 3)
        self.assertAlmostEqual(first_period["mean_wait_to_match_seconds"], 50 / 3)

        self.assertEqual(len(dataset), 1)
        sample = dataset.iloc[0]
        self.assertEqual(sample["decision_step"], 2)
        self.assertEqual(sample["lag_1_matched_request_num"], 5)
        self.assertEqual(sample["lag_2_matched_request_num"], 3)
        self.assertAlmostEqual(sample["lag_1_price_multiplier"], 1.3)
        self.assertAlmostEqual(sample["lag_2_price_multiplier"], 1.0)
        self.assertAlmostEqual(sample["lag_1_matching_radius_km"], 2.0)
        self.assertAlmostEqual(sample["lag_2_matching_radius_km"], 1.0)
        self.assertEqual(sample["target_matched_request_num"], 7)
        self.assertAlmostEqual(sample["target_order_completion_rate"], 7 / 9)
        self.assertEqual(sample["decision_start_sim_time"], 18240)
        self.assertEqual(sample["decision_end_sim_time_exclusive"], 18360)
        self.assertAlmostEqual(sample["price_multiplier"], 1.6)
        self.assertAlmostEqual(sample["matching_radius_km"], 3.0)

    def test_joint_action_context_contains_complete_vector_and_global_state(self):
        from src.env.observation import LOCAL_STATE_COLUMNS, NEIGHBOR_STATE_COLUMNS

        plan, minute, metadata = _raw_tables()
        for index, column in enumerate(LOCAL_STATE_COLUMNS):
            plan[column] = plan.decision_step + index
        for column in NEIGHBOR_STATE_COLUMNS:
            plan[column] = 0
        periods = aggregate_decision_metrics(plan, minute, metadata)
        samples = build_history_samples(periods, 1, joint_action_context=True)
        self.assertIn("joint_price_grid_0", samples)
        self.assertIn("joint_radius_grid_0", samples)
        self.assertTrue((samples["global_start_idle_driver_num"]
                         == samples["start_idle_driver_num"]).all())
        self.assertTrue((samples["joint_price_grid_0"] == samples["price_multiplier"]).all())

    def test_history_rejects_missing_decision_step(self):
        action_plan, minute_metrics, metadata = _raw_tables()
        periods = aggregate_decision_metrics(
            action_plan, minute_metrics, metadata)

        with self.assertRaises(ValueError):
            build_history_samples(
                periods[periods["decision_step"] != 1], history_steps=1)

    def test_split_keeps_each_run_together(self):
        dataset = pd.DataFrame({
            "run_id": [f"run_{index}" for index in range(6) for _ in range(2)]
        })

        result = assign_run_splits(
            dataset, validation_ratio=0.2, test_ratio=0.2, split_seed=7)

        self.assertTrue(
            (result.groupby("run_id")["split"].nunique() == 1).all())
        self.assertEqual(set(result["split"]), {"train", "validation", "test"})

    def test_date_split_keeps_all_seeds_and_times_of_day_together(self):
        dataset = pd.DataFrame([
            {"date": date, "run_id": f"{date}_{seed}_{time}"}
            for date in ("2015-05-20", "2015-05-18", "2015-05-19")
            for seed in (42, 43) for time in ("05:00", "12:00", "18:00")])
        first = assign_run_splits(dataset, split_by="auto")
        second = assign_run_splits(dataset, split_by="date", split_seed=99)
        self.assertTrue(first.equals(second))
        self.assertTrue(first.groupby("date")["split"].nunique().eq(1).all())
        self.assertEqual(first.groupby("date")["split"].first().to_dict(), {
            "2015-05-18": "train", "2015-05-19": "validation", "2015-05-20": "test"})

    def test_date_split_does_not_fall_back_to_runs(self):
        with self.assertRaises(ValueError):
            resolve_split_by(["2015-05-18", "2015-05-19"], "auto", 0.15, 0.15)
        with self.assertRaises(ValueError):
            resolve_split_by(["2015-05-18"], "date", 0.15, 0.15)
        self.assertEqual(resolve_split_by(["2015-05-18"], "auto", 0.15, 0.15), "run")
        two_days = pd.DataFrame({"date": ["2015-05-18", "2015-05-19"],
                                 "run_id": ["first", "second"]})
        result = assign_run_splits(two_days, validation_ratio=0, test_ratio=0.15,
                                   split_by="date")
        self.assertEqual(result["split"].tolist(), ["train", "test"])

    def test_date_split_manifest_and_validation_detect_date_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for day in (18, 19, 20):
                for seed in (42, 43):
                    run_id = f"day{day}_seed{seed}"
                    plan, minute, metadata = _raw_tables(run_id)
                    metadata["date"] = f"2015-05-{day}"
                    metadata["simulator_seed"] = seed
                    raw = root / "raw" / run_id
                    raw.mkdir(parents=True)
                    plan.to_csv(raw / "action_plan.csv", index=False)
                    minute.to_csv(raw / "grid_minute_metrics.csv", index=False)
                    with (raw / "manifest.json").open("w", encoding="utf-8") as file:
                        json.dump(metadata, file)
            dataset, manifest = build_supervised_dataset(
                root / "raw", root / "dataset", history_steps=2)
            self.assertEqual(manifest["split_by"], "date")
            self.assertEqual(manifest["split_dates"], {
                "train": ["2015-05-18"], "validation": ["2015-05-19"],
                "test": ["2015-05-20"]})
            self.assertEqual(manifest["split_run_counts"], {
                "train": 2, "validation": 2, "test": 2})
            report = validate_supervised_dataset(root / "dataset")
            self.assertTrue(report["passed"])
            self.assertEqual(report["warnings"], [])
            dataset.loc[dataset["split"] == "test", "date"] = "2015-05-18"
            dataset.to_csv(root / "dataset" / "supervised_dataset.csv", index=False)
            report = validate_supervised_dataset(root / "dataset")
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["date_splits_are_isolated"])

    def test_build_and_save_dataset(self):
        action_plan, minute_metrics, metadata = _raw_tables()
        with tempfile.TemporaryDirectory() as temporary_dir:
            raw_dir = Path(temporary_dir) / "raw" / metadata["run_id"]
            output_dir = Path(temporary_dir) / "dataset"
            raw_dir.mkdir(parents=True)
            action_plan.to_csv(raw_dir / "action_plan.csv", index=False)
            minute_metrics.to_csv(
                raw_dir / "grid_minute_metrics.csv", index=False)
            with (raw_dir / "manifest.json").open("w", encoding="utf-8") as file:
                json.dump(metadata, file)

            dataset, manifest = build_supervised_dataset(
                input_dir=raw_dir.parent,
                output_dir=output_dir,
                history_steps=2)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(manifest["row_count"], 1)
            self.assertEqual(manifest["split_counts"], {"train": 1})
            self.assertIn(
                "lag_1_price_multiplier", manifest["feature_columns"])
            self.assertIn(
                "lag_1_matching_radius_km", manifest["feature_columns"])
            self.assertEqual(
                manifest["target_period"],
                "current decision period observations")
            self.assertEqual(
                manifest["queued_order_pricing"],
                "queued orders retain the price assigned when submitted")
            self.assertTrue((output_dir / "supervised_dataset.csv").exists())
            self.assertTrue((output_dir / "dataset_manifest.json").exists())
            report = validate_supervised_dataset(output_dir)
            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["date_count"], 1)
            self.assertTrue(report["warnings"])
            self.assertTrue((output_dir / "validation_report.json").exists())

            duplicated = pd.concat([dataset, dataset.iloc[[0]]], ignore_index=True)
            duplicated.to_csv(
                output_dir / "supervised_dataset.csv", index=False)
            report = validate_supervised_dataset(output_dir)
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["no_duplicate_keys"])


if __name__ == "__main__":
    unittest.main()
