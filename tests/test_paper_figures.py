"""论文曲线使用原始记录，选点和导出格式与训练结果一致。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from tests import test_plot_results as plot_tests

if plot_tests.plot_results is not None:
    from baseline.plot_paper_figures import _load_history, plot_paper_figures


@unittest.skipIf(plot_tests.plot_results is None, "matplotlib not installed")
class PaperFigureTests(unittest.TestCase):
    def test_exports_vector_and_raster_figures_without_changing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plot_tests.PlotResultsTests()._write_run(root)
            sources = {path.name: path.read_bytes() for path in root.iterdir()}
            result = plot_paper_figures(root)
            output = root / "paper_figures"
            self.assertEqual(result["selected_epoch"], 2)
            self.assertEqual(result["zoom_start_epoch"], 1)
            self.assertEqual(result["zoom_end_epoch"], 3)
            self.assertFalse(result["smoothing"])
            for stem in ("fig01_training_convergence", "fig02_task_convergence"):
                self.assertTrue((output / f"{stem}.pdf").read_bytes().startswith(b"%PDF-"))
                self.assertIn("<svg", (output / f"{stem}.svg").read_text())
                self.assertTrue((output / f"{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            for name, content in sources.items():
                self.assertEqual((root / name).read_bytes(), content)
            self.assertIn("unsmoothed", (output / "captions.txt").read_text())

    def test_right_panel_contains_both_curves_for_first_100_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plot_tests.PlotResultsTests()._write_run(root)
            path = root / "training_history.csv"
            template = pd.read_csv(path).iloc[[0]]
            history = pd.concat([template] * 120, ignore_index=True)
            history["epoch"] = range(1, 121)
            history["validation_mse"] = [1 / epoch for epoch in history["epoch"]]
            history.to_csv(path, index=False)
            report_path = root / "metrics.json"
            report = json.loads(report_path.read_text())
            report.update(epochs_run=120, best_epoch=120)
            report_path.write_text(json.dumps(report))

            def inspect_figure(fig, output, name):
                if name == "fig01_training_convergence":
                    left, right = fig.axes
                    self.assertEqual(right.get_title(loc="left"), "(b) First 100 epochs")
                    self.assertEqual(left.get_ylim(), right.get_ylim())
                    self.assertEqual(len(right.lines), 2)
                    for line in right.lines:
                        self.assertEqual(list(line.get_xdata()), list(range(1, 101)))
                    self.assertEqual(len(left.lines[0].get_xdata()), 120)
                from matplotlib import pyplot as plt
                plt.close(fig)
                return []

            with patch("baseline.plot_paper_figures._save", side_effect=inspect_figure):
                result = plot_paper_figures(root)
            self.assertEqual(result["zoom_end_epoch"], 100)

    def test_rejects_wrong_checkpoint_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plot_tests.PlotResultsTests()._write_run(root)
            path = root / "metrics.json"
            report = json.loads(path.read_text())
            report["best_epoch"] = 3
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                _load_history(root)
            report["best_epoch"] = 2
            path.write_text(json.dumps(report))
            history = pd.read_csv(root / "training_history.csv")
            history.loc[0, "train_mse"] = float("inf")
            history.to_csv(root / "training_history.csv", index=False)
            with self.assertRaises(ValueError):
                _load_history(root)


if __name__ == "__main__":
    unittest.main()
