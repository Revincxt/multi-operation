"""LSTM 动作统计：预热排除、网格边界、候选频率和矢量导出。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from diagnostics.analyze_lstm_actions import analyze, action_statistics, load_actions, time_frequency


def write_example(root):
    report = {"status": "complete", "seeds": [10, 20], "config": {
        "grid_num": 2, "simulation_steps": 15, "decision_freq": 5, "history_steps": 1,
        "start_time": "05:00", "price_action_values": [1.0, 1.5, 2.0],
        "matching_radius_values": [1.0, 2.0, 3.0], "warmup_price": 1.5, "warmup_radius": 3.0}}
    (root / "manifest.json").write_text(json.dumps(report))
    for seed in report["seeds"]:
        rows = []
        for step in range(3):
            for grid in range(2):
                price = 1.5 if step == 0 else 1.0 if step == 1 and grid == 0 else 2.0
                if seed == 20 and step == 2 and grid == 1:
                    price = 1.0
                rows.append({"decision_step": step, "grid_id": grid,
                             "decision_start_minute_step": step * 5,
                             "decision_end_minute_step_exclusive": (step + 1) * 5,
                             "price_multiplier": price,
                             "matching_radius_km": 3.0 if step == 0 or grid == 1 else 1.0,
                             "warmup": step == 0})
        path = root / "runs" / f"seed{seed}" / "lstm"
        path.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(path / "action_plan.csv.gz", index=False)


class LSTMActionAnalysisTests(unittest.TestCase):
    def test_switches_exclude_warmup_and_cross_grid_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_example(root)
            history, _ = load_actions(root)
            active, tables = action_statistics(history)
            self.assertEqual(len(active), 8)
            self.assertEqual(active.joint_changed.count(), 4)
            self.assertEqual(active.joint_changed.sum(), 3)
            self.assertAlmostEqual(active.joint_changed.mean(), 0.75)
            self.assertEqual(active.matching_radius_km_changed.sum(), 0)
            self.assertTrue(active.loc[active.decision_step == 1, "joint_changed"].isna().all())
            self.assertEqual(tables["time_summary"].clock_time.tolist(), ["05:05", "05:10"])
            self.assertEqual(tables["hourly_summary"].time_interval.tolist(), ["05:05–05:15"])
            self.assertAlmostEqual(tables["joint_action_frequency"].share.sum(), 1.0)

    def test_frequency_keeps_unused_candidates_and_normalizes_each_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_example(root)
            history, _ = load_actions(root)
            active, _ = action_statistics(history)
            frequency = time_frequency(active, "matching_radius_km", [1.0, 2.0, 3.0])
            self.assertEqual(len(frequency), 6)
            self.assertTrue(frequency.loc[frequency.matching_radius_km == 2.0, "share"].eq(0).all())
            np.testing.assert_allclose(frequency.groupby("clock_minute").share.sum(), 1)
            np.testing.assert_allclose(frequency.groupby("clock_minute").decisions.sum(), 4)

    def test_rejects_missing_or_invalid_actions_and_wrong_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_example(root)
            path = root / "runs/seed10/lstm/action_plan.csv.gz"
            frame = pd.read_csv(path)
            for bad in (frame.iloc[1:], pd.concat([frame, frame.iloc[:1]]),
                        frame.assign(price_multiplier=7.0), frame.assign(warmup=False)):
                bad.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_actions(root)

    def test_exports_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_example(root)
            original = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            result = analyze(root)
            self.assertEqual(result["example_seed"], 10)
            self.assertEqual(result["learned_decision_count"], 8)
            self.assertEqual(result["joint_switch_rate"], 0.75)
            output = root / "lstm_action_analysis"
            pdf = (output / "lstm_action_analysis.pdf").read_bytes()
            self.assertIn(b"/Count 3", pdf)
            self.assertNotIn(b"/Subtype /Image", pdf)
            for name in ("lstm_action_trends", "lstm_action_frequencies", "lstm_actions_example"):
                self.assertTrue((output / f"{name}.png").is_file())
                self.assertTrue((output / f"{name}.svg").is_file())
            for path, content in original.items():
                self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
