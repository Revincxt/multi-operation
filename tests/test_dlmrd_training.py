"""DL-MRD完整样本训练、WESM隔离以及按epoch暂停恢复的检查。"""

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from baseline.dlmrd import train
from baseline.dlmrd.data import MinMaxPreprocessor
from baseline.dlmrd.model import GridDLMRD
from baseline.losses import WESMLoss
from baseline.training import _make_loader


class SyntheticDataset(Dataset):
    """支持批量索引的小型确定性数据，保留实际量纲评价字段。"""

    def __init__(self, count=17, seed=11):
        rng = np.random.default_rng(seed)
        history = rng.random((count, 2, 2), dtype=np.float32)
        context = rng.random((count, 4), dtype=np.float32)
        grids = np.arange(count, dtype=np.int64) % 2
        targets = np.stack([.1 + .2 * context[:, task % 4] + .1 * history.mean((1, 2))
                            + .05 * grids for task in range(6)], axis=1).astype("float32")
        self.arrays = dict(history=history, context=context, grid_indices=grids, targets=targets,
                           actual=targets.astype("float64"), persistence=targets.astype("float64") + .1)
        self.sample_count = count
        self.visited = []

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        positions = np.arange(self.sample_count)[index]
        self.visited.extend(np.asarray(positions).reshape(-1).tolist())
        return tuple(torch.from_numpy(np.asarray(self.arrays[name][index]))
                     for name in ("history", "context", "grid_indices", "targets"))


class ZeroPredictor(nn.Module):
    def forward(self, history, context, grids):
        return torch.zeros((len(history), 6), dtype=history.dtype, device=history.device)


class DLMRDTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        self.addCleanup(setattr, train, "STOP", False)

    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_epoch_mse_uses_every_sample_and_all_six_targets(self):
        dataset = SyntheticDataset(count=7)
        dataset.arrays["targets"] = np.arange(42, dtype=np.float32).reshape(7, 6) / 10
        expected = (dataset.arrays["targets"] ** 2).mean(0)
        result = train.epoch(ZeroPredictor(), _make_loader(dataset, 4, shuffle=True, seed=59),
                             "cpu", criterion=WESMLoss(6))
        self.assertEqual(sorted(dataset.visited), list(range(7)))
        np.testing.assert_allclose(result["task_mse"], expected, rtol=1e-6)
        self.assertAlmostEqual(result["mse"], float(expected.mean()), places=6)
        self.assertAlmostEqual(result["weighted_loss"], float(expected.mean()), places=6)
        np.testing.assert_allclose(result["mean_weights"], np.full(6, 1 / 6))
        dropped = DataLoader(SyntheticDataset(count=7), batch_size=4, drop_last=True)
        with self.assertRaises(ValueError):
            train.epoch(ZeroPredictor(), dropped, "cpu", criterion=WESMLoss(6))

    def test_wesm_cpu_weighted_loss_keeps_gradient_and_validation_is_read_only(self):
        criterion = WESMLoss(6, update_interval=1)
        losses = torch.tensor([1., 2., 3., 4., 5., 6.], requires_grad=True)
        objective = criterion(losses.to("cpu"))
        used_weights = criterion.weights.clone()
        objective.backward()
        torch.testing.assert_close(losses.grad, used_weights, rtol=0, atol=0)
        before = deepcopy(criterion.state_dict())
        result = train.epoch(ZeroPredictor(), _make_loader(SyntheticDataset(7), 4),
                             "cpu", criterion=criterion)
        self.assert_nested_equal(before, criterion.state_dict())
        self.assertFalse(criterion.training)
        self.assertAlmostEqual(result["weighted_loss"], result["mse"], places=7)
        np.testing.assert_allclose(result["mean_weights"], np.full(6, 1 / 6))

    def test_training_reduces_synthetic_regression_error_without_dropping_tail(self):
        dataset = SyntheticDataset(count=33)
        model = GridDLMRD(2, 2, history_steps=2, hidden_size=16, num_heads=4,
                          num_layers=1, d_ff=24, context_size=4)
        criterion = WESMLoss(6, update_interval=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        before = train.epoch(model, _make_loader(dataset, 8), "cpu", criterion=criterion)["mse"]
        for number in range(20):
            dataset.visited.clear()
            train.epoch(model, _make_loader(dataset, 8, shuffle=True, seed=number),
                        "cpu", optimizer, criterion)
            self.assertEqual(sorted(dataset.visited), list(range(33)))
        after = train.epoch(model, _make_loader(dataset, 8), "cpu", criterion=criterion)["mse"]
        self.assertLess(after, before * .1)
        self.assertEqual(int(criterion.step_count), 20 * 5)

    def test_pause_resume_restores_optimizer_wesm_rng_and_reproduces_full_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, cache = root / "source", root / "cache"
            source.mkdir()
            cache.mkdir()
            (source / "dataset_manifest.json").write_text("{}")
            (cache / "cache_manifest.json").write_text("{}")

            def dataset_loader(input_dir, cache_dir, output_dir):
                processor = MinMaxPreprocessor(2, ["history_a", "history_b"])
                processor.grid_ids, processor.fit_row_count = [0, 1], 17
                processor.scalers = {name: {"mean": np.zeros(width, dtype=np.float32),
                                            "scale": np.ones(width, dtype=np.float32)}
                                     for name, width in (("history", 2), ("context", 4), ("targets", 6))}
                (Path(output_dir) / "normalization.json").write_text(json.dumps(processor.state_dict()))
                return ({"train": SyntheticDataset(17), "validation": SyntheticDataset(9, 13),
                         "test": SyntheticDataset(7, 15)}, processor, {})

            def args(output):
                return Namespace(input=str(source), prepared_cache=str(cache), output=str(output),
                                 epochs=2, batch_size=8, seed=42, device="cpu", threads=1, workers=0, smoke=True)

            uninterrupted, resumed = root / "full", root / "resumed"
            with patch.object(train, "load_dataset", side_effect=dataset_loader), \
                    patch.object(train, "source_hashes", return_value={"fixture": "constant"}), \
                    patch.object(train.signal, "signal"), patch("baseline.dlmrd.report.run"):
                train.run(args(uninterrupted))
                with patch("baseline.dlmrd.report.run", side_effect=lambda output: train.stop()):
                    train.run(args(resumed))
                paused = json.loads((resumed / "progress.json").read_text())
                self.assertEqual(paused["status"], "paused")
                self.assertEqual(paused["epochs_completed"], 1)
                checkpoint = torch.load(resumed / "latest.pt", map_location="cpu", weights_only=True)
                self.assertTrue(checkpoint["optimizer_state_dict"]["state"])
                self.assertEqual(int(checkpoint["loss_state_dict"]["step_count"]), 3)
                self.assertEqual(len(checkpoint["history"]), 1)
                self.assertFalse((resumed / "metrics.json").exists())
                (resumed / "training_history.csv").unlink()
                torch.manual_seed(999)
                torch.randn(27)
                train.run(args(resumed))

            full = torch.load(uninterrupted / "latest.pt", map_location="cpu", weights_only=True)
            restored = torch.load(resumed / "latest.pt", map_location="cpu", weights_only=True)
            for key in ("model_state_dict", "optimizer_state_dict", "loss_state_dict", "torch_rng_state",
                        "best_loss", "best_epoch"):
                self.assert_nested_equal(full[key], restored[key])
            for left, right in zip(full["history"], restored["history"]):
                for key in left:
                    if not key.endswith("seconds"):
                        self.assertEqual(left[key], right[key])
            self.assertEqual(int(restored["loss_state_dict"]["step_count"]), 6)
            history = pd.read_csv(resumed / "training_history.csv")
            self.assertEqual(history.epoch.tolist(), [1, 2])
            progress = json.loads((resumed / "progress.json").read_text())
            self.assertEqual(progress["status"], "complete")
            metrics = json.loads((resumed / "metrics.json").read_text())
            self.assertEqual(metrics["epochs_run"], 2)
            self.assertEqual(set(metrics["splits"]), {"validation", "test"})
            self.assertEqual(len(metrics["splits"]["test"]["dlmrd"]), 6)


if __name__ == "__main__":
    unittest.main()
