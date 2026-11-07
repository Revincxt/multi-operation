"""检查阶段模型对比的训练/验证隔离、流式指标和 checkpoint 快照。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from tests.test_lstm import _dataset, _write_dataset
from baseline.compare_models import MODELS, compare_models, fit_ridge, linear_features
from baseline.data import load_dataset
from baseline.sequence import TARGET_COLUMNS
from baseline.train_lstm import train_lstm
from baseline.train_timesnet import train_timesnet
from baseline.training import regression_metrics


class ModelComparisonTests(unittest.TestCase):
    def test_streamed_ridge_matches_direct_train_fit_and_validation_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            _write_dataset(root, _dataset())
            splits, preprocessor, _ = load_dataset(root, chunk_size=3)
            alphas = (0.1, 10.0, 100.0)
            coef, mean, info = fit_ridge(splits["train"], splits["validation"], 2,
                                         chunk_size=3, alphas=alphas)
            x = linear_features(splits["train"], 0, len(splits["train"]), 2)
            vx = linear_features(splits["validation"], 0, len(splits["validation"]), 2)
            y = np.asarray(splits["train"].arrays["targets"], dtype=float)
            vy = np.asarray(splits["validation"].arrays["targets"], dtype=float)
            penalty = np.eye(x.shape[1])
            penalty[-1, -1] = 0
            candidates, scores = [], []
            for alpha in alphas:
                direct = np.linalg.solve(x.T @ x + alpha * penalty, x.T @ y)
                candidates.append(direct)
                scores.append(np.square(vx @ direct - vy).mean())
            best = int(np.argmin(scores))
            np.testing.assert_allclose(coef, candidates[best], atol=1e-8)
            np.testing.assert_allclose([item["validation_standardized_mse"] for item in info["candidates"]], scores)
            np.testing.assert_allclose(mean, splits["train"].arrays["actual"].mean(0))
            self.assertEqual(info["selected_alpha"], alphas[best])
            self.assertEqual(preprocessor.fit_row_count, len(splits["train"]))

    def test_comparison_preserves_models_and_scores_every_test_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_dataset(root / "data", _dataset())
            train_lstm(root / "data", root / "lstm", epochs=1, hidden_size=4, threads=1)
            train_timesnet(root / "data", root / "timesnet", epochs=1, hidden_size=4,
                           num_layers=1, d_ff=4, num_kernels=1, threads=1)
            original = {name: (root / name / "best_model.pt").read_bytes()
                        for name in ("lstm", "timesnet")}
            original_reports = {name: (root / name / "metrics.json").read_bytes()
                                for name in ("lstm", "timesnet")}
            report = compare_models(root / "data", root / "lstm", root / "timesnet",
                                    root / "comparison", batch_size=3, threads=1, make_plots=False)
            self.assertFalse(report["test_selection"])
            self.assertEqual(report["split_counts"]["test"], 8)
            self.assertEqual(report["preprocessor"]["fit_row_count"], 8)
            predictions = pd.read_csv(root / "comparison" / "test_predictions.csv.gz")
            scores = pd.read_csv(root / "comparison" / "comparison_metrics.csv")
            self.assertEqual(len(predictions), 8)
            self.assertEqual(set(predictions["split"]), {"test"})
            self.assertEqual(set(scores["model"]), set(MODELS))
            self.assertEqual(len(scores), len(MODELS) * len(TARGET_COLUMNS))
            actual = predictions[list(TARGET_COLUMNS)].to_numpy()
            for name in MODELS:
                predicted = predictions[[f"{name}_{target.removeprefix('target_')}"
                                         for target in TARGET_COLUMNS]].to_numpy()
                direct = regression_metrics(actual, predicted)
                for target in TARGET_COLUMNS:
                    row = scores[(scores.model == name) & (scores.metric == target.removeprefix("target_"))].iloc[0]
                    for metric in ("mae", "rmse", "r2"):
                        self.assertAlmostEqual(row[metric], direct[target][metric], places=6)
            for name in ("lstm", "timesnet"):
                self.assertEqual((root / name / "best_model.pt").read_bytes(), original[name])
                self.assertEqual((root / name / "metrics.json").read_bytes(), original_reports[name])
                self.assertEqual((root / "comparison" / f"{name}_best_snapshot.pt").read_bytes(), original[name])
            with self.assertRaises(FileExistsError):
                compare_models(root / "data", root / "lstm", root / "timesnet", root / "comparison")
            # checkpoint 来自另一数据版本时，不允许混用数据或模型。
            path = root / "timesnet" / "training_config.json"
            config = json.loads(path.read_text())
            config["source_sha256"]["csv"] = "other-dataset"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                compare_models(root / "data", root / "lstm", root / "timesnet",
                               root / "invalid", make_plots=False)


if __name__ == "__main__":
    unittest.main()
