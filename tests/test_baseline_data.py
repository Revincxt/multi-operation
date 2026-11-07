"""全量基线数据：分块拟合、磁盘缓存、独立划分与训练前检查。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests.test_lstm import HISTORY_FEATURES, _dataset, _write_dataset
from baseline.data import load_dataset, load_prepared_dataset
from baseline.sequence import SequencePreprocessor, TARGET_COLUMNS
from baseline.training import RegressionAccumulator, _make_loader, training_parser
from baseline.train_lstm import train_lstm
from baseline.train_timesnet import train_timesnet
from src.utils.joint_context import joint_action_columns


class BaselineDataTests(unittest.TestCase):
    def test_prepared_cache_skips_csv_scan_and_rejects_another_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "data"
            _write_dataset(source, _dataset())
            original, processor, _ = load_dataset(source, chunk_size=3)
            cache = original["train"].cache_dir
            with patch("baseline.data._csv_chunks", side_effect=AssertionError), \
                    patch("baseline.data.pd.read_csv", side_effect=AssertionError):
                reused, restored, _ = load_prepared_dataset(source, cache)
            self.assertEqual(restored.state_dict(), processor.state_dict())
            np.testing.assert_array_equal(reused["train"].arrays["targets"],
                                          original["train"].arrays["targets"])
            other = Path(temporary) / "other"
            _write_dataset(other, _dataset())
            with self.assertRaises(ValueError):
                load_prepared_dataset(other, cache)

    def test_chunked_cache_matches_in_memory_and_reuses_without_csv_reload(self):
        frame = _dataset()
        expected = SequencePreprocessor(2, HISTORY_FEATURES).fit(frame[frame.split == "train"])
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "data"
            _write_dataset(source, frame)
            original_read = pd.read_csv

            def bounded_read(*args, **kwargs):
                self.assertTrue(kwargs.get("chunksize") or kwargs.get("nrows") == 0)
                return original_read(*args, **kwargs)

            with patch("baseline.data.pd.read_csv", side_effect=bounded_read):
                splits, processor, _ = load_dataset(source, chunk_size=3)
            self.assertEqual(processor.fit_row_count, 8)
            for name, dataset in splits.items():
                part = frame[frame.split == name]
                reference = (*expected.transform_inputs(part), expected.transform_targets(part))
                for actual, wanted in zip(dataset[slice(None)], reference):
                    np.testing.assert_allclose(actual.numpy(), wanted, rtol=1e-6, atol=1e-6)
                self.assertTrue(all(isinstance(value, np.memmap) for value in dataset.arrays.values()))
                self.assertTrue(all(not value.flags.writeable for value in dataset.arrays.values()))
                np.testing.assert_array_equal(dataset.arrays["actual"], part[list(TARGET_COLUMNS)])
                self.assertEqual(pd.read_csv(dataset.keys_path).run_id.tolist(), part.run_id.tolist())
            with patch("baseline.data._csv_chunks", side_effect=AssertionError):
                reused, restored, _ = load_dataset(source, chunk_size=1)
            self.assertEqual(restored.state_dict(), processor.state_dict())
            self.assertEqual(reused["train"].cache_dir, splits["train"].cache_dir)

    def test_joint_action_context_is_loaded_and_scaled(self):
        frame = _dataset()
        extras = list(joint_action_columns([7, 31]))
        for column, source in zip(extras, (
                "price_multiplier", "matching_radius_km",
                "price_multiplier", "matching_radius_km")):
            frame[column] = frame.groupby(["run_id", "decision_step"])[source].transform("first")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "data"
            _write_dataset(source, frame)
            path = source / "dataset_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["feature_columns"] += extras
            manifest["additional_context_columns"] = extras
            manifest["joint_action_context"] = True
            path.write_text(json.dumps(manifest))
            splits, processor, _ = load_dataset(source, chunk_size=3)
            self.assertEqual(processor.context_columns[-len(extras):], extras)
            self.assertEqual(processor.joint_grid_ids, [7, 31])
            frame_context = frame[frame.split == "test"]
            _, context, _ = processor.transform_inputs(frame_context)
            restored = (context[:, -len(extras):]
                        * processor.scalers["context"]["scale"][-len(extras):]
                        + processor.scalers["context"]["mean"][-len(extras):])
            np.testing.assert_allclose(restored, frame_context[extras].to_numpy())

    def test_each_epoch_contains_every_training_row_once_including_tail_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "data"
            _write_dataset(source, _dataset())
            splits, _, _ = load_dataset(source, chunk_size=1)
            dataset = splits["train"]
            loader = _make_loader(dataset, batch_size=3, shuffle=True, seed=8)
            for _ in range(2):
                indices = [index for batch in loader.sampler for index in batch]
                self.assertEqual(sorted(indices), list(range(len(dataset))))
            first = _make_loader(dataset, 3, True, 42)
            second = _make_loader(dataset, 3, True, 42)
            self.assertEqual(list(first.sampler), list(second.sampler))
            sequential = list(_make_loader(dataset, 3))
            self.assertEqual([len(batch[0]) for batch in sequential], [3, 3, 2])
            np.testing.assert_array_equal(
                np.concatenate([batch[3].numpy() for batch in sequential]), dataset.arrays["targets"])

    def test_parallel_loader_reopens_memmaps_and_keeps_all_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "data"
            _write_dataset(source, _dataset())
            splits, _, _ = load_dataset(source, chunk_size=3)
            dataset = splits["train"]
            worker_state = dataset.__getstate__()
            self.assertEqual(worker_state["arrays"], {})
            single = _make_loader(dataset, 3, True, seed=42, shuffle_block_size=3)
            parallel = _make_loader(dataset, 3, True, seed=42, workers=2,
                                    shuffle_block_size=3)
            for _ in range(2):
                reference, batches = list(single), list(parallel)
                self.assertEqual(sum(len(batch[0]) for batch in batches), len(dataset))
                np.testing.assert_array_equal(
                    np.concatenate([np.asarray(batch[3]) for batch in batches]),
                    np.concatenate([np.asarray(batch[3]) for batch in reference]))

    def test_block_shuffle_covers_all_rows_and_preserves_partial_last_block(self):
        for block_size in (0, 1, 7, 65536):
            loader = _make_loader(range(103), 8, True, seed=42,
                                  shuffle_block_size=block_size)
            first = list(loader.sampler)
            second = list(loader.sampler)
            for batches in (first, second):
                self.assertEqual(sorted(row for batch in batches for row in batch), list(range(103)))
                self.assertEqual(len(batches[-1]), 7)
            self.assertNotEqual(first, second)
            same_seed = _make_loader(range(103), 8, True, seed=42,
                                     shuffle_block_size=block_size)
            self.assertEqual(first, list(same_seed.sampler))

    def test_streamed_statistics_match_population_and_reject_non_training_rows(self):
        train = _dataset().query("split == 'train'")
        processor = SequencePreprocessor(2, HISTORY_FEATURES).fit_chunks(
            train.iloc[start:start + 3] for start in range(0, len(train), 3))
        history, context = processor._raw_inputs(train)
        for name, values in (("history", history.reshape(-1, len(HISTORY_FEATURES))),
                             ("context", context),
                             ("targets", train[list(TARGET_COLUMNS)].to_numpy(dtype=np.float32))):
            np.testing.assert_allclose(processor.scalers[name]["mean"], values.mean(0, dtype=np.float64))
            scale = values.std(0, dtype=np.float64)
            scale[scale < 1e-6] = 1
            np.testing.assert_allclose(processor.scalers[name]["scale"], scale)
        with self.assertRaises(ValueError):
            processor.fit_chunks([_dataset()])

    def test_cross_chunk_errors_cannot_publish_a_cache(self):
        frame = _dataset()
        duplicate = pd.concat([frame.iloc[:1], frame], ignore_index=True)
        leaked = frame.copy()
        leaked.loc[leaked.split == "validation", "run_id"] = "run_0"
        unknown = frame.copy()
        unknown.loc[unknown.split == "test", "grid_id"] = 99
        non_finite = frame.copy()
        non_finite.loc[23, TARGET_COLUMNS[0]] = np.inf
        for invalid in (duplicate, leaked, unknown, non_finite):
            with self.subTest(), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "data"
                _write_dataset(source, invalid)
                with self.assertRaises(ValueError):
                    load_dataset(source, chunk_size=1)
                self.assertEqual(list(source.rglob("cache_manifest.json")), [])

    def test_source_changes_invalidate_cache_and_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "data"
            frame = _dataset()
            _write_dataset(source, frame)
            original, _, _ = load_dataset(source, chunk_size=3)
            frame.loc[0, TARGET_COLUMNS[0]] += 10
            frame.to_csv(source / "supervised_dataset.csv", index=False)
            updated, _, _ = load_dataset(source, chunk_size=3)
            self.assertNotEqual(updated["train"].cache_dir, original["train"].cache_dir)
            path = updated["train"].cache_dir / "train" / "targets.npy"
            with path.open("ab") as file:
                file.write(b"corrupt")
            with self.assertRaises(ValueError):
                load_dataset(source)

    def test_frozen_source_splits_must_not_be_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "data"
            _write_dataset(source, _dataset())
            path = source / "dataset_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["source_run_splits"] = {"run_0": "test", "run_1": "validation", "run_2": "train"}
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_dataset(source, chunk_size=3)

    def test_preparation_checks_six_grids_and_models_without_training(self):
        base = _dataset().query("grid_id == 7")
        frame = pd.concat([base.assign(grid_id=grid) for grid in range(6)], ignore_index=True)
        frame = frame.sort_values(["run_id", "decision_step", "grid_id"])
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "data"
            output = Path(temporary) / "unused_model"
            _write_dataset(source, frame)
            configs = []
            for trainer in (train_lstm, train_timesnet):
                with patch("baseline.training._run_epoch", side_effect=AssertionError), \
                        patch("baseline.training.torch.optim.Adam", side_effect=AssertionError), \
                        patch("baseline.training.torch.optim.AdamW", side_effect=AssertionError):
                    config = trainer(source, output, epochs=1000, prepare_only=True,
                                     hidden_size=8, batch_size=5, chunk_size=3, threads=1)
                configs.append(config)
                self.assertFalse(output.exists())
                self.assertEqual(config["grid_ids"], list(range(6)))
                self.assertEqual(config["model"]["grid_count"], 6)
                self.assertEqual(config["normalization_fit_rows"], 24)
                self.assertEqual(config["target_columns"], list(TARGET_COLUMNS))
                self.assertEqual(config["forward_check"]["output_shape"], [5, 6])
                self.assertEqual(config["epochs"], 1000)
                self.assertEqual(config["initialization"], "from_scratch")
                self.assertFalse(config["early_stopping"])
                self.assertEqual(config["loss_weighting"], "wesm")
                self.assertEqual(config["optimizer"], "adamw")
                self.assertEqual(config["weight_decay"], 1e-3)
            self.assertEqual(configs[0]["data_cache"], configs[1]["data_cache"])
            self.assertEqual(configs[0]["source_sha256"], configs[1]["source_sha256"])
            self.assertEqual(configs[0]["split_run_ids"], configs[1]["split_run_ids"])
            for model in ("lstm", "timesnet"):
                args = training_parser(model).parse_args(["--prepare-only"])
                self.assertEqual(args.input_dir, "datasets_grid6_low_mixed47k_20261029")
                self.assertIsNone(args.output_dir)
                self.assertEqual(args.optimizer_name, "adamw")
                self.assertEqual(args.weight_decay, 1e-3)
                control = training_parser(model).parse_args([
                    "--optimizer", "adam", "--weight-decay", "0"])
                self.assertEqual(control.optimizer_name, "adam")
                self.assertEqual(control.weight_decay, 0)

    def test_optimizer_validation_precedes_data_loading(self):
        for trainer in (train_lstm, train_timesnet):
            for options in ({"optimizer_name": "invalid"}, {"weight_decay": -1},
                            {"weight_decay": float("nan")}, {"weight_decay": float("inf")}):
                with self.subTest(trainer=trainer.__name__, options=options), \
                        patch("baseline.training.load_dataset", side_effect=AssertionError):
                    with self.assertRaises(ValueError):
                        trainer(prepare_only=True, **options)

    def test_streamed_metrics_equal_direct_metrics(self):
        actual = np.arange(60, dtype=float).reshape(10, 6)
        actual[:, 0] = 9
        predicted = actual + np.cos(actual)
        accumulator = RegressionAccumulator()
        for start in range(0, 10, 3):
            accumulator.update(actual[start:start + 3], predicted[start:start + 3])
        result = accumulator.result()
        for index, target in enumerate(TARGET_COLUMNS):
            errors = predicted[:, index] - actual[:, index]
            self.assertAlmostEqual(result[target]["mae"], np.abs(errors).mean())
            self.assertAlmostEqual(result[target]["rmse"], np.sqrt((errors ** 2).mean()))
            if index == 0:
                self.assertIsNone(result[target]["r2"])
            else:
                self.assertAlmostEqual(result[target]["r2"], 1 - (errors ** 2).sum()
                                       / ((actual[:, index] - actual[:, index].mean()) ** 2).sum())


if __name__ == "__main__":
    unittest.main()
