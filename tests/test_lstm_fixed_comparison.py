"""多固定动作对比：共同预热、留出种子、配对收益及可重绘输出。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from baseline.compare_lstm_fixed import (
    METRICS, plot_results, policy_definitions, select_test_runs, summarize_results,
)
from baseline.compare_policies import RANDOM_ACTION_SEED_NAMESPACE, random_grid_actions, run_episode
from data_pipeline.collect_density import config_signature, sha256


class LSTMFixedComparisonTests(unittest.TestCase):
    def test_fixed_combinations_are_validated(self):
        definitions = policy_definitions([(1, 2), (1.5, 3)])
        self.assertEqual([d["policy"] for d in definitions], ["fixed_1", "fixed_2", "lstm"])
        self.assertEqual([d["policy"] for d in policy_definitions([(1, 2)], include_random=True)],
                         ["fixed_1", "random", "lstm"])
        for actions in ([], [(0, 2)], [(1, -1)], [(1, float("nan"))], [(1, 2), (1, 2)], [(1, 2, 3)]):
            with self.assertRaises(ValueError):
                policy_definitions(actions)

    def test_random_actions_are_reproducible_per_grid_and_use_the_same_candidates(self):
        prices, radii = [1.0, 1.5, 1.6], [1.0, 2.0, 3.0]
        seed = [42, RANDOM_ACTION_SEED_NAMESPACE]
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        state = np.random.get_state()
        first = random_grid_actions(rng, 6, prices, radii)
        second = random_grid_actions(rng, 6, prices, radii)
        repeated = random_grid_actions(np.random.default_rng(np.random.SeedSequence(seed)), 6, prices, radii)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)
        self.assertEqual(set(first), set(range(6)))
        self.assertGreater(len(set(first.values())), 1)
        self.assertTrue(all(action.price_multiplier in prices and action.matching_radius_km in radii
                            for action in list(first.values()) + list(second.values())))
        expected = np.random.default_rng(np.random.SeedSequence(seed)).integers(9, size=6)
        self.assertEqual([(action.price_multiplier, action.matching_radius_km) for action in first.values()],
                         [(prices[index // 3], radii[index % 3]) for index in expected])
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_held_out_run_selection_and_source_validation(self):
        training = {"split_run_ids": {"train": ["train_run"], "validation": [], "test": ["a", "b"]}}
        source = {"simulation_steps": 4, "grid_num": 1}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "collection_config.json").write_text(json.dumps(source))
            for seed, run_id in enumerate(("a", "b")):
                path = root / "batch_00000" / run_id
                path.mkdir(parents=True)
                data = path / "sample.csv"
                data.write_text("example")
                (path / "manifest.json").write_text(json.dumps({
                    "planned_split": "test", "simulator_seed": seed,
                    "collection_config_sha256": config_signature(source),
                    "quality": {"minute_metric_rows": 4}, "file_sha256": {"sample.csv": sha256(data)}}))
            first = select_test_runs(training, root, 2)
            self.assertEqual(first, select_test_runs(training, root, 2))
            self.assertEqual(set(first[1]), {0, 1})
            with self.assertRaises(ValueError):
                select_test_runs(training, root, 3)
            training["split_run_ids"]["train"].append("a")
            with self.assertRaises(ValueError):
                select_test_runs(training, root, 2)
            training["split_run_ids"]["train"].remove("a")
            data.write_text("changed")
            with self.assertRaises(ValueError):
                select_test_runs(training, root, 2)

    def test_multiple_fixed_actions_share_the_same_warmup(self):
        config = {"grid_num": 1, "driver_num": 2, "decision_freq": 2,
                  "history_steps": 2, "start_time": "05:00", "simulation_steps": 10,
                  "date": "2015-05-18", "fixed_price": 1.0, "fixed_radius": 1.0,
                  "warmup_price": 1.5, "warmup_radius": 2.0,
                  "price_response_model": "utility_choice", "price_action_values": [1.0, 1.5],
                  "matching_radius_values": [0.5, 1.0, 2.0]}
        drivers = pd.DataFrame({"driver_id": [1, 2], "start_time": [18000, 18000],
                                "end_time": [36000, 36000], "lng": [-73.99, -73.98],
                                "lat": [40.75, 40.75], "grid_id": [0, 0]})
        requests = {18000 + minute * 60: [
            [minute * 10 + j, 0, 40.75, -73.99 + j * 0.007, 0, 40.75,
             -73.98, 1.0, 18000 + minute * 60, 0, 0, [], [], 160, 3.7, 0]
            for j in range(3)] for minute in range(9)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = run_episode(config, "fixed_1", 42, requests, drivers, None, root / "first")
            second = run_episode({**config, "fixed_price": 1.5, "fixed_radius": 0.5},
                                 "fixed_2", 42, requests, drivers, None, root / "second")
            for key in ("warmup_metrics_sha256", "initial_driver_sha256", "baseline_request_num"):
                self.assertEqual(first[key], second[key])
            self.assertEqual(first["evaluation_minutes"], 6)
            for directory, price, radius in (("first", 1, 1), ("second", 1.5, 0.5)):
                plan = pd.read_csv(root / directory / "action_plan.csv.gz")
                self.assertTrue(plan.loc[plan.warmup, "price_multiplier"].eq(1.5).all())
                self.assertTrue(plan.loc[plan.warmup, "matching_radius_km"].eq(2).all())
                self.assertTrue(plan.loc[~plan.warmup, "price_multiplier"].eq(price).all())
                self.assertTrue(plan.loc[~plan.warmup, "matching_radius_km"].eq(radius).all())
            random = run_episode(config, "random", 42, requests, drivers, None, root / "random")
            repeated = run_episode(config, "random", 42, requests, drivers, None, root / "repeated")
            self.assertEqual(random["warmup_metrics_sha256"], first["warmup_metrics_sha256"])
            self.assertEqual(random["initial_driver_sha256"], first["initial_driver_sha256"])
            self.assertEqual(random["platform_revenue"], repeated["platform_revenue"])
            pd.testing.assert_frame_equal(pd.read_csv(root / "random" / "action_plan.csv.gz"),
                                          pd.read_csv(root / "repeated" / "action_plan.csv.gz"))
            plan = pd.read_csv(root / "random" / "action_plan.csv.gz")
            self.assertTrue(plan.loc[plan.warmup, "price_multiplier"].eq(1.5).all())
            self.assertTrue(plan.loc[plan.warmup, "matching_radius_km"].eq(2).all())
            self.assertTrue(plan.predicted_platform_revenue.isna().all())
            # 只有一个候选时，随机策略必须与相同固定策略逐分钟一致，不能干扰仿真 RNG。
            single_config = {**config, "price_action_values": [1.0], "matching_radius_values": [1.0]}
            single = run_episode(single_config, "random", 42, requests, drivers, None, root / "single")
            self.assertEqual(single["full_window"], first["full_window"])
            pd.testing.assert_frame_equal(pd.read_csv(root / "single" / "grid_minute_metrics.csv.gz"),
                                          pd.read_csv(root / "first" / "grid_minute_metrics.csv.gz"))

    def test_paired_statistics_and_vector_plots(self):
        definitions = policy_definitions([(1, 2), (1.5, 3)], include_random=True)
        frame = pd.DataFrame([
            {"seed": seed, "policy": policy, **dict.fromkeys(METRICS, 0.5),
             "platform_revenue": seed * 10 + revenue}
            for seed in (1, 2, 3) for policy, revenue in
            (("fixed_1", 100), ("fixed_2", 200), ("random", 80), ("lstm", 150))])
        summary, paired, comparisons = summarize_results(frame, definitions)
        np.testing.assert_allclose(comparisons.mean_revenue_difference, [50, -50, 70])
        np.testing.assert_allclose(comparisons.bootstrap_95_lower, [50, -50, 70])
        self.assertEqual(comparisons.lstm_win_count.tolist(), [3, 0, 3])
        self.assertTrue(paired.lstm_minus_fixed_1.eq(50).all())
        self.assertTrue(paired.lstm_minus_random.eq(70).all())
        self.assertEqual(comparisons.set_index("policy").loc["random", "baseline_revenue_mean"], 100)
        with self.assertRaises(ValueError):
            summarize_results(frame.iloc[1:], definitions)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary.to_csv(output / "policy_summary.csv")
            (output / "manifest.json").write_text(json.dumps({
                "policies": definitions, "seeds": [1, 2, 3], "code_sha256": {"simulation": "unchanged"}, "config": {
                    "grid_num": 6, "driver_num": 250, "history_steps": 4, "decision_freq": 5}}))
            report = plot_results(output)
            self.assertEqual(report["code_sha256"], {"simulation": "unchanged"})
            self.assertIn("plot_code_sha256", report)
            pdf = (output / "policy_comparison.pdf").read_bytes()
            self.assertTrue(pdf.startswith(b"%PDF"))
            self.assertNotIn(b"/Subtype /Image", pdf)
            self.assertTrue((output / "policy_comparison.png").is_file())
            self.assertTrue((output / "policy_comparison.svg").is_file())
            svg = (output / "policy_comparison.svg").read_text()
            self.assertIn("Rand.", svg)
            self.assertIn("fixed and random", svg)


if __name__ == "__main__":
    unittest.main()
