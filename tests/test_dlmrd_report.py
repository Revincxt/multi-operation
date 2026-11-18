"""验证DL-MRD报告的训练/终测阶段、归一化标注和输出单位。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from baseline.dlmrd import report


class DLMRDReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.rows = []
        for epoch in (1, 2):
            row = {"epoch": epoch, "train_mse": .1 / epoch, "validation_mse": .2 / epoch}
            for metric in report.METRICS:
                row.update({f"train_mse_{metric}": .1 / epoch,
                            f"validation_mse_{metric}": .2 / epoch,
                            f"mean_weight_{metric}": 1 / 6})
            self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.output / "training_history.csv", index=False)

    def metrics(self):
        values = {f"target_{metric}": {"mae": .02, "rmse": .03, "r2": .5}
                  for metric in report.METRICS}
        return {"model_name": "dlmrd", "epochs_run": 2, "best_epoch": 2,
                "loss_weighting": "wesm", "splits": {"test": {
                    "dlmrd": values, "persistence": values}}}

    def test_in_progress_reporting_does_not_require_test_metrics(self):
        with patch.object(report, "plot_training", return_value=2):
            result = report.run(self.output)
        self.assertEqual(result["status"], "training")
        self.assertFalse((self.output / "test_results.csv").exists())
        text = (self.output / "paper_figures" / "captions.txt").read_text()
        self.assertIn("min-max", text)
        self.assertNotIn("standardized", text)

    def test_final_csv_keeps_raw_rate_units_and_all_six_targets(self):
        metrics = self.metrics()
        metrics["splits"]["test"]["dlmrd"]["target_baseline_request_num"]["r2"] = None
        (self.output / "metrics.json").write_text(json.dumps(metrics))
        with patch.object(report, "plot_training", return_value=2), patch.object(report, "plot_test"):
            result = report.run(self.output)
        self.assertEqual(result["status"], "complete")
        frame = pd.read_csv(self.output / "test_results.csv")
        self.assertEqual(len(frame), 12)
        self.assertEqual(set(frame.metric), set(report.METRICS))
        self.assertTrue((frame.rmse == .03).all())
        self.assertTrue(frame[frame.metric == "baseline_request_num"].r2.isna().all())

    def test_mismatched_checkpoint_and_invalid_history_are_rejected(self):
        metrics = self.metrics()
        metrics["best_epoch"] = 1
        (self.output / "metrics.json").write_text(json.dumps(metrics))
        with patch.object(report, "plot_training", return_value=2), self.assertRaises(ValueError):
            report.run(self.output)
        self.rows[1]["epoch"] = 3
        pd.DataFrame(self.rows).to_csv(self.output / "training_history.csv", index=False)
        with self.assertRaises(ValueError):
            report.load_history(self.output / "training_history.csv")

    def test_test_plot_converts_rate_errors_to_percentage_points(self):
        figures = []
        with patch.object(report, "save_figure", side_effect=lambda fig, path: figures.append(fig)):
            report.plot_test(report.test_results(self.metrics()), self.output)
        self.assertEqual(figures[0].axes[1].patches[0].get_height(), 3.)
        self.assertIn("pp", figures[0].axes[1].get_ylabel())
        self.assertAlmostEqual(figures[0].axes[0].patches[0].get_height(), .03)


if __name__ == "__main__":
    unittest.main()
