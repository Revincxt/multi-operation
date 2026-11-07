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

from baseline.lstm import (
    CONTEXT_COLUMNS,
    TARGET_COLUMNS,
    GridLSTM,
    LSTMPredictor,
    SequencePreprocessor,
)
from baseline.train_lstm import load_dataset, regression_metrics, train_lstm


HISTORY_FEATURES = [column.removeprefix("target_") for column in TARGET_COLUMNS]
HISTORY_FEATURES += ["price_multiplier", "matching_radius_km"]


def _dataset():
    """三个独立 run、两个不连续编号的网格，构造两步历史。"""
    rows = []
    for run, split in enumerate(("train", "validation", "test")):
        for step in range(2, 6):
            for grid_id in (7, 31):
                row = {
                    "run_id": f"run_{run}", "split": split,
                    "decision_step": step, "grid_id": grid_id,
                    "decision_start_sim_time": 18000 + step * 300,
                    "interval_minutes": 5, "price_multiplier": 1.3,
                    "matching_radius_km": 2.0,
                }
                for index, target in enumerate(TARGET_COLUMNS):
                    # 验证/测试值与 train 明显不同，用于检查标准化泄漏。
                    row[target] = run * 100 + step + grid_id / 10 + index
                for lag in (1, 2):
                    for index, feature in enumerate(HISTORY_FEATURES):
                        row[f"lag_{lag}_{feature}"] = run * 100 + step - lag + index
                rows.append(row)
    return pd.DataFrame(rows)


def _write_dataset(directory, frame, schema="grid_supervised_dataset_v3"):
    directory.mkdir(parents=True)
    frame.to_csv(directory / "supervised_dataset.csv", index=False)
    manifest = {
        "schema_version": schema, "history_steps": 2,
        "row_count": len(frame),
        "split_counts": {key: int(value) for key, value in frame["split"].value_counts().items()},
        "feature_columns": list(CONTEXT_COLUMNS) + [
            f"lag_{lag}_{feature}" for lag in (1, 2) for feature in HISTORY_FEATURES],
    }
    with (directory / "dataset_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file)


class LSTMBaselineTests(unittest.TestCase):
    def setUp(self):
        self.frame = _dataset()
        self.train = self.frame[self.frame["split"] == "train"]
        self.preprocessor = SequencePreprocessor(2, HISTORY_FEATURES).fit(self.train)

    def test_history_is_oldest_first_and_uses_train_statistics(self):
        history, context, grids = self.preprocessor.transform_inputs(self.frame)
        stats = self.preprocessor.scalers["history"]
        raw = history * stats["scale"] + stats["mean"]
        np.testing.assert_allclose(raw[:, 0, 0], self.frame["lag_2_baseline_request_num"], atol=1e-5)
        np.testing.assert_allclose(raw[:, 1, 0], self.frame["lag_1_baseline_request_num"], atol=1e-5)
        expected_mean = self.train[list(TARGET_COLUMNS)].mean().to_numpy()
        np.testing.assert_allclose(self.preprocessor.scalers["targets"]["mean"], expected_mean)
        self.assertEqual(history.shape, (24, 2, 8))
        self.assertEqual(context.shape, (24, len(CONTEXT_COLUMNS)))
        self.assertEqual(set(grids), {0, 1})
        self.assertEqual(self.preprocessor.grid_ids, [7, 31])
        self.assertTrue(np.isfinite(context).all())

    def test_input_transform_does_not_read_current_targets(self):
        without_labels = self.frame.drop(columns=list(TARGET_COLUMNS))
        before = self.preprocessor.transform_inputs(self.frame)
        after = self.preprocessor.transform_inputs(without_labels)
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)
        restored = SequencePreprocessor.from_state_dict(self.preprocessor.state_dict())
        np.testing.assert_allclose(
            restored.inverse_targets(restored.transform_targets(self.frame)),
            self.frame[list(TARGET_COLUMNS)].to_numpy(), atol=1e-5)

    def test_rejects_unknown_grids_and_non_finite_features(self):
        unknown = self.frame.iloc[[0]].copy()
        unknown["grid_id"] = 99
        with self.assertRaises(ValueError):
            self.preprocessor.transform_inputs(unknown)
        invalid = self.frame.copy()
        invalid.loc[0, "lag_1_request_rate"] = np.nan
        with self.assertRaises(ValueError):
            self.preprocessor.transform_inputs(invalid)

    def test_load_supports_both_schemas_and_rejects_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, schema in enumerate(("grid_supervised_dataset_v3", "grid_supervised_dataset")):
                path = Path(directory) / str(index)
                _write_dataset(path, self.frame, schema)
                splits, preprocessor, _ = load_dataset(path)
                self.assertEqual([len(value) for value in splits.values()], [8, 8, 8])
                np.testing.assert_array_equal(
                    preprocessor.scalers["targets"]["mean"],
                    self.preprocessor.scalers["targets"]["mean"])
            leaked = self.frame.copy()
            leaked.loc[leaked["split"] == "validation", "run_id"] = "run_0"
            _write_dataset(Path(directory) / "leaked", leaked)
            with self.assertRaises(ValueError):
                load_dataset(Path(directory) / "leaked")
            missing = self.frame[self.frame["split"] != "validation"]
            _write_dataset(Path(directory) / "missing", missing)
            with self.assertRaises(ValueError):
                load_dataset(Path(directory) / "missing")

    def test_model_predicts_six_outputs_and_all_branches_receive_gradients(self):
        torch.manual_seed(7)
        model = GridLSTM(8, 2, hidden_size=8, embedding_size=3, dropout=0.0)
        inputs = [torch.from_numpy(values) for values in self.preprocessor.transform_inputs(self.train)]
        inputs[1].requires_grad_()
        outputs = model(*inputs)
        self.assertEqual(tuple(outputs.shape), (8, 6))
        outputs.square().mean().backward()
        self.assertGreater(model.encoder.weight_ih_l0.grad.abs().sum().item(), 0)
        self.assertGreater(model.grid_embedding.weight.grad.abs().sum().item(), 0)
        self.assertGreater(inputs[1].grad[:, -2:].abs().sum().item(), 0)

    def test_date_split_loader_rejects_shared_dates_and_reversed_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            frame = self.frame.copy()
            frame["date"] = frame["split"].map({
                "train": "2015-05-18", "validation": "2015-05-19", "test": "2015-05-20"})
            _write_dataset(root, frame)
            path = root / "dataset_manifest.json"
            with path.open(encoding="utf-8") as file:
                manifest = json.load(file)
            manifest["split_by"] = "date"
            manifest["split_dates"] = {name: sorted(group["date"].unique().tolist())
                                       for name, group in frame.groupby("split")}
            with path.open("w", encoding="utf-8") as file:
                json.dump(manifest, file)
            splits, _, _ = load_dataset(root)
            self.assertEqual(splits["test"].metadata["split_dates"]["test"], ["2015-05-20"])
            for test_date in ("2015-05-18", "2015-05-17"):
                frame.loc[frame["split"] == "test", "date"] = test_date
                frame.to_csv(root / "supervised_dataset.csv", index=False)
                manifest["split_dates"]["test"] = [test_date]
                with path.open("w", encoding="utf-8") as file:
                    json.dump(manifest, file)
                with self.assertRaises(ValueError):
                    load_dataset(root)

    def test_metrics_handle_constant_targets(self):
        actual = np.ones((3, 6))
        metrics = regression_metrics(actual, actual * 2)
        for values in metrics.values():
            self.assertEqual(values["mae"], 1.0)
            self.assertEqual(values["rmse"], 1.0)
            self.assertIsNone(values["r2"])

    def test_runs_all_epochs_and_keeps_best_validation_checkpoint(self):
        epochs = 12
        losses = iter(value for epoch in range(epochs)
                      for value in (1.0 / (epoch + 1), 2.0 + epoch))

        def fake_epoch(model, loader, device, optimizer=None, criterion=None):
            if optimizer is not None:
                with torch.no_grad():
                    next(model.parameters()).add_(1.0)
            loss = next(losses)
            return {"mse": loss, "weighted_loss": loss,
                    "task_mse": [loss] * 6, "mean_weights": [1 / 6] * 6}

        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "dataset"
            output_dir = Path(directory) / "model"
            _write_dataset(input_dir, self.frame)
            with patch("baseline.training._run_epoch", side_effect=fake_epoch) as run_epoch:
                report = train_lstm(
                    input_dir, output_dir, epochs=epochs,
                    hidden_size=8, embedding_size=3, dropout=0.0, threads=1)
            self.assertEqual(run_epoch.call_count, epochs * 2)
            self.assertEqual(report["epochs_run"], epochs)
            self.assertEqual(report["best_epoch"], 1)
            history = pd.read_csv(output_dir / "training_history.csv")
            self.assertEqual(history["epoch"].tolist(), list(range(1, epochs + 1)))
            with (output_dir / "training_config.json").open(encoding="utf-8") as file:
                self.assertNotIn("patience", json.load(file))
            saved = torch.load(output_dir / "best_model.pt", weights_only=True)
            self.assertEqual(saved["best_epoch"], 1)
            torch.manual_seed(42)
            original = GridLSTM(**saved["model_config"])
            np.testing.assert_allclose(
                saved["model_state_dict"]["encoder.weight_ih_l0"].numpy(),
                original.encoder.weight_ih_l0.detach().numpy() + 1.0)

    def test_training_checkpoint_and_label_free_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "dataset"
            output_dir = Path(directory) / "model"
            _write_dataset(input_dir, self.frame)
            report = train_lstm(
                input_dir, output_dir, epochs=2, batch_size=4,
                hidden_size=8, embedding_size=3, dropout=0.0, threads=1,
                workers=2, wesm_update_interval=1)
            self.assertEqual(report["epochs_run"], 2)
            self.assertGreaterEqual(report["best_epoch"], 1)
            self.assertEqual(set(report["splits"]), {"validation", "test"})
            self.assertEqual(len(report["splits"]["test"]["lstm"]), 6)
            predictor = LSTMPredictor.load(output_dir / "best_model.pt")
            test_frame = self.frame[self.frame["split"] == "test"]
            predictions = predictor.predict(test_frame.drop(columns=list(TARGET_COLUMNS)), batch_size=3)
            self.assertEqual(predictions.index.tolist(), test_frame.index.tolist())
            saved = pd.read_csv(output_dir / "test_predictions.csv")
            for target in TARGET_COLUMNS:
                column = target.replace("target_", "predicted_", 1)
                np.testing.assert_allclose(predictions[target], saved[column], atol=1e-5)
            checkpoint = torch.load(output_dir / "best_model.pt", weights_only=True)
            history = pd.read_csv(output_dir / "training_history.csv")
            self.assertAlmostEqual(checkpoint["validation_mse"], history["validation_mse"].min())
            self.assertEqual(checkpoint["loss_weighting"], "wesm")
            config = json.loads((output_dir / "training_config.json").read_text())
            self.assertEqual(config["data_loader_workers"], 2)
            for record in (checkpoint, report, config):
                self.assertEqual(record["optimizer"], "adamw")
                self.assertEqual(record["weight_decay"], 1e-3)
            self.assertEqual(int(checkpoint["loss_state_dict"]["step_count"]), report["best_epoch"] * 2)
            weights = history.filter(like="mean_weight_")
            self.assertEqual(weights.shape[1], 6)
            np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-6)
            self.assertEqual(history.filter(like="validation_mse_").shape[1], 6)
            self.assertTrue((output_dir / "training_config.json").exists())
            self.assertTrue((output_dir / "metrics.json").exists())
            with self.assertRaises(FileExistsError):
                train_lstm(input_dir, output_dir, epochs=1)

    def test_equal_weight_training_remains_available(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "dataset"
            output_dir = Path(directory) / "model"
            _write_dataset(input_dir, self.frame)
            report = train_lstm(
                input_dir, output_dir, epochs=1, batch_size=4,
                hidden_size=8, embedding_size=3, dropout=0.0, threads=1,
                loss_weighting="equal")
            self.assertEqual(report["loss_weighting"], "equal")
            np.testing.assert_allclose(list(report["best_task_weights"].values()), 1 / 6)
            history = pd.read_csv(output_dir / "training_history.csv")
            np.testing.assert_array_equal(history["train_weighted_loss"], history["train_mse"])
            checkpoint = torch.load(output_dir / "best_model.pt", weights_only=True)
            self.assertIsNone(checkpoint["loss_state_dict"])

    def test_adamw_applies_decay_and_adam_zero_decay_preserves_original_control(self):
        """用零梯度隔离权重衰减，确认不是只记录配置却没有实际生效。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_dataset(root / "data", self.frame)
            for name, decay in (("adamw", 1e-3), ("adamw", 0.0), ("adam", 0.0)):
                seen = []

                def zero_gradient_epoch(model, loader, device, optimizer=None, criterion=None):
                    if optimizer is not None:
                        seen.append(optimizer)
                        for parameter in model.parameters():
                            parameter.grad = torch.zeros_like(parameter)
                        optimizer.step()
                    return {"mse": 1.0, "weighted_loss": 1.0,
                            "task_mse": [1.0] * 6, "mean_weights": [1 / 6] * 6}

                output = root / f"{name}_{decay}"
                with patch("baseline.training._run_epoch", side_effect=zero_gradient_epoch):
                    report = train_lstm(
                        root / "data", output, epochs=1, hidden_size=8, embedding_size=3,
                        dropout=0.0, learning_rate=0.01, threads=1,
                        optimizer_name=name, weight_decay=decay)
                expected_class = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
                self.assertIsInstance(seen[0], expected_class)
                self.assertEqual(seen[0].param_groups[0]["weight_decay"], decay)
                checkpoint = torch.load(output / "best_model.pt", weights_only=True)
                torch.manual_seed(42)
                original = GridLSTM(**checkpoint["model_config"])
                for key, initial in original.state_dict().items():
                    torch.testing.assert_close(
                        checkpoint["model_state_dict"][key], initial * (1 - 0.01 * decay))
                config = json.loads((output / "training_config.json").read_text())
                for record in (checkpoint, report, config):
                    self.assertEqual(record["optimizer"], name)
                    self.assertEqual(record["weight_decay"], decay)


if __name__ == "__main__":
    unittest.main()
