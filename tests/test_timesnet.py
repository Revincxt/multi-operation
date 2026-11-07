"""TimesNet 的周期提取、联合动作输入与完整训练流程检查。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("需要 baseline/requirements.txt") from error

from baseline.lstm import LSTMPredictor
from baseline.sequence import CONTEXT_COLUMNS, TARGET_COLUMNS
from baseline.timesnet import GridTimesNet, TimesBlock, TimesNetPredictor, fft_for_period
from baseline.train_timesnet import train_timesnet
from tests.test_lstm import _dataset, _write_dataset


class TimesNetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_fft_detects_individual_periods_and_excludes_dc(self):
        time = torch.arange(24, dtype=torch.float32)
        signals = torch.stack([torch.sin(2 * torch.pi * time / period) + 20
                               for period in (4, 6)]).unsqueeze(-1).requires_grad_()
        periods, amplitudes = fft_for_period(signals, 1)
        self.assertEqual(periods.tolist(), [[4], [6]])
        self.assertEqual(tuple(amplitudes.shape), (2, 1))
        amplitudes.sum().backward()
        self.assertTrue(torch.isfinite(signals.grad).all())
        for steps in (2, 3, 4, 7):
            for value in (0.0, 10.0):
                periods, amplitudes = fft_for_period(torch.full((3, steps, 2), value), 99)
                self.assertEqual(tuple(periods.shape), (3, steps // 2))
                self.assertTrue((periods >= 2).all())
                self.assertTrue(torch.isfinite(amplitudes).all())

    def test_padding_aggregation_and_residual_restore_sequence(self):
        block = TimesBlock(3, 4, top_k=2, num_kernels=2)
        block.conv = torch.nn.Identity()
        values = torch.randn(2, 7, 3, requires_grad=True)
        # 包含不能整除的周期，以及同一样本中重复的周期。
        with patch("baseline.timesnet.fft_for_period", return_value=(
                torch.tensor([[3, 3], [2, 3]]), torch.tensor([[1., 2.], [3., 4.]]))):
            output = block(values)
        torch.testing.assert_close(output, values * 2)
        output.sum().backward()
        torch.testing.assert_close(values.grad, torch.full_like(values, 2))

    def test_six_outputs_and_gradients_for_history_actions_and_grid(self):
        model = GridTimesNet(8, 3, history_steps=4, hidden_size=8,
                            embedding_size=3, num_layers=2, d_ff=12, num_kernels=2,
                            dropout=0.0)
        history = torch.randn(4, 4, 8, requires_grad=True)
        context = torch.randn(4, len(CONTEXT_COLUMNS), requires_grad=True)
        output = model(history, context, torch.tensor([0, 1, 2, 0]))
        self.assertEqual(tuple(output.shape), (4, 6))
        output.square().mean().backward()
        for values in (history.grad, context.grad[:, -2], context.grad[:, -1],
                       model.value_embedding.weight.grad, model.temporal_projection.weight.grad,
                       model.grid_embedding.weight.grad,
                       model.blocks[0].conv[0].kernels[1].weight.grad):
            self.assertTrue(torch.isfinite(values).all())
            self.assertGreater(values.abs().sum().item(), 0)

    def test_prediction_is_independent_of_batch_companions(self):
        # 较长窗口且 top_k 小于频率数，确实触发不同样本选择不同周期。
        model = GridTimesNet(3, 2, history_steps=12, hidden_size=7,
                            embedding_size=3, num_layers=2, top_k=2,
                            d_ff=8, num_kernels=2, dropout=0.0).eval()
        history = torch.randn(5, 12, 3)
        context = torch.randn(5, len(CONTEXT_COLUMNS))
        grids = torch.tensor([0, 1, 0, 1, 0])
        with torch.inference_mode():
            together = model(history, context, grids)
            alone = torch.cat([model(history[i:i+1], context[i:i+1], grids[i:i+1])
                               for i in range(5)])
            order = torch.tensor([4, 0, 2, 3, 1])
            shuffled = model(history[order], context[order], grids[order])
        torch.testing.assert_close(together, alone, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(together[order], shuffled, rtol=1e-5, atol=1e-6)

    def test_short_constant_history_and_invalid_arguments(self):
        model = GridTimesNet(3, 1, history_steps=2, hidden_size=4,
                            num_layers=1, d_ff=4, num_kernels=1, top_k=99, dropout=0.0)
        output = model(torch.zeros(2, 2, 3), torch.zeros(2, 4), torch.zeros(2, dtype=torch.long))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                            if p.grad is not None))
        for kwargs in ({"history_steps": 1}, {"top_k": 0}, {"d_ff": 0},
                       {"num_kernels": 0}, {"num_layers": 0}, {"dropout": float("nan")}):
            with self.assertRaises(ValueError):
                GridTimesNet(3, 1, **kwargs)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3, 3), torch.zeros(2, 4), torch.zeros(2, dtype=torch.long))

    def test_train_wesm_checkpoint_and_label_free_prediction(self):
        frame = _dataset()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "data", root / "model"
            _write_dataset(source, frame)
            report = train_timesnet(source, output, epochs=2, batch_size=4, hidden_size=8,
                                   embedding_size=3, num_layers=1, d_ff=8, num_kernels=2,
                                   threads=1, dropout=0.0, wesm_update_interval=1)
            self.assertEqual(report["epochs_run"], 2)
            self.assertEqual(report["model_name"], "timesnet")
            self.assertEqual(len(report["splits"]["test"]["timesnet"]), 6)
            history = pd.read_csv(output / "training_history.csv")
            self.assertTrue(np.isfinite(history.to_numpy()).all())
            np.testing.assert_allclose(history.filter(like="mean_weight_").sum(axis=1), 1, atol=1e-6)
            checkpoint = torch.load(output / "best_model.pt", weights_only=True)
            self.assertEqual(checkpoint["schema_version"], "grid_timesnet_v1")
            self.assertAlmostEqual(checkpoint["validation_mse"], history.validation_mse.min())
            self.assertEqual(int(checkpoint["loss_state_dict"]["step_count"]), report["best_epoch"] * 2)
            config = json.loads((output / "training_config.json").read_text())
            self.assertEqual(config["model"]["history_steps"], 2)
            for record in (config, checkpoint, report):
                self.assertEqual(record["optimizer"], "adamw")
                self.assertEqual(record["weight_decay"], 1e-3)
            predictor = TimesNetPredictor.load(output / "best_model.pt")
            test = frame[frame.split == "test"].drop(columns=list(TARGET_COLUMNS))
            predictions = predictor.predict(test, batch_size=3)
            self.assertEqual(predictions.index.tolist(), test.index.tolist())
            saved = pd.read_csv(output / "test_predictions.csv")
            for target in TARGET_COLUMNS:
                np.testing.assert_allclose(predictions[target],
                                           saved[target.replace("target_", "predicted_", 1)],
                                           atol=1e-5)
            with self.assertRaises(ValueError):
                LSTMPredictor.load(output / "best_model.pt")
            with self.assertRaises(FileExistsError):
                train_timesnet(source, output, epochs=1)

    def test_equal_loss_and_no_early_stopping(self):
        losses = iter([1., 2., 0.8, 3., 0.6, 4.])

        def fake_epoch(*args, **kwargs):
            value = next(losses)
            return {"mse": value, "weighted_loss": value,
                    "task_mse": [value] * 6, "mean_weights": [1 / 6] * 6}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_dataset(root / "data", _dataset())
            with patch("baseline.training._run_epoch", side_effect=fake_epoch):
                report = train_timesnet(
                    root / "data", root / "model", epochs=3, hidden_size=4,
                    num_layers=1, d_ff=4, num_kernels=1, loss_weighting="equal", threads=1)
            self.assertEqual(report["epochs_run"], 3)
            self.assertEqual(report["best_epoch"], 1)
            checkpoint = torch.load(root / "model" / "best_model.pt", weights_only=True)
            self.assertIsNone(checkpoint["loss_state_dict"])
            np.testing.assert_allclose(list(report["best_task_weights"].values()), 1 / 6)


if __name__ == "__main__":
    unittest.main()
