"""训练曲线与测试结果导出的轻量检查。"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    from baseline.plot_results import METRICS, plot_results
except ModuleNotFoundError as error:
    if error.name != "matplotlib":
        raise
    plot_results = None


@unittest.skipIf(plot_results is None, "matplotlib not installed")
class PlotResultsTests(unittest.TestCase):
    def _write_run(self, directory):
        history = {"epoch": [1, 2, 3], "train_mse": [0.6, 0.4, 0.3],
                   "validation_mse": [0.6, 0.5, 0.7]}
        scores = {"lstm": {}, "persistence": {}}
        for index, metric in enumerate(METRICS):
            history[f"train_mse_{metric}"] = history["train_mse"]
            history[f"validation_mse_{metric}"] = history["validation_mse"]
            scores["lstm"][f"target_{metric}"] = {"mae": 1.0, "rmse": 1.5, "r2": 0.5}
            scores["persistence"][f"target_{metric}"] = {
                "mae": 2.0 if index else 0.0, "rmse": 2.5, "r2": None}
        pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
        report = {"epochs_run": 3, "best_epoch": 2, "loss_weighting": "wesm",
                  "splits": {"test": scores}}
        with (directory / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(report, file)

    def test_exports_curves_and_six_original_unit_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_run(directory)
            original = (directory / "metrics.json").read_bytes()
            output = directory / "figures"
            results = plot_results(directory, output)
            self.assertEqual(len(results), 6)
            self.assertEqual(results["metric"].tolist(), list(METRICS))
            self.assertTrue(pd.isna(results.loc[0, "mae_improvement_percent"]))
            self.assertTrue(results.loc[1:, "mae_improvement_percent"].eq(50).all())
            self.assertTrue(results["lstm_mae"].eq(1).all())
            self.assertTrue(results["persistence_r2"].isna().all())
            self.assertEqual(len(pd.read_csv(output / "test_results.csv")), 6)
            for name in ("training_curves.png", "task_curves.png"):
                with (output / name).open("rb") as file:
                    self.assertEqual(file.read(8), b"\x89PNG\r\n\x1a\n")
            self.assertEqual((directory / "metrics.json").read_bytes(), original)

    def test_rejects_incomplete_training_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_run(directory)
            path = directory / "training_history.csv"
            history = pd.read_csv(path)
            history.iloc[:2].to_csv(path, index=False)
            with self.assertRaises(ValueError):
                plot_results(directory)

    def test_timesnet_results_are_not_labeled_as_lstm(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_run(directory)
            path = directory / "metrics.json"
            report = json.loads(path.read_text())
            report["model_name"] = "timesnet"
            scores = report["splits"]["test"]
            scores["timesnet"] = scores.pop("lstm")
            path.write_text(json.dumps(report))
            results = plot_results(directory)
            self.assertIn("timesnet_mae", results)
            self.assertNotIn("lstm_mae", results)
            self.assertTrue(results["timesnet_mae"].eq(1).all())


if __name__ == "__main__":
    unittest.main()
