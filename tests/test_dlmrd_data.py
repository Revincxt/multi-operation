"""DL-MRD 的 train-only Min-Max 与现有只读缓存保持一致。"""

import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from baseline.data import CACHE_VERSION, SPLITS, _sha256, load_prepared_dataset
from baseline.dlmrd.data import MinMaxPreprocessor, load_dataset
from baseline.sequence import CONTEXT_COLUMNS, TARGET_COLUMNS, SequencePreprocessor


class DLMRDDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.source = root / "source"
        self.cache = self.source / "sequence_cache/fake"
        self.output = root / "model"
        self.cache.mkdir(parents=True)
        rows = []
        for i, split in enumerate(("train", "train", "train", "validation", "validation", "test")):
            values = (i * 4, i * 4 + 2) if split == "train" else (-10, 20)
            row = {"run_id": f"run_{split}", "split": split, "decision_step": i + 2,
                   "grid_id": 3 if i % 2 == 0 else 7, "decision_start_sim_time": i * 5.,
                   "interval_minutes": 5. if split == "train" else 7.,
                   "price_multiplier": 1. + i / 2, "matching_radius_km": 1. + i,
                   "lag_2_requests": values[0], "lag_1_requests": values[1],
                   "lag_2_price": 2., "lag_1_price": 2.}
            for j, name in enumerate(TARGET_COLUMNS):
                row[name] = 42. if j == len(TARGET_COLUMNS) - 1 else (i + 1) * (j + 1)
            rows.append(row)
        self.frame = pd.DataFrame(rows)
        train = self.frame[self.frame.split == "train"]
        self.original = SequencePreprocessor(2, ["requests", "price"]).fit(train)
        manifest = {"row_count": len(self.frame), "history_steps": 2,
                    "split_counts": self.frame.split.value_counts().to_dict()}
        manifest_path = self.source / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        metadata = {**manifest, "schema_version": CACHE_VERSION,
                    "source_sha256": {"manifest": _sha256(manifest_path)},
                    "normalization_fit_split": "train", "preprocessor": self.original.state_dict(),
                    "split_run_ids": {split: [f"run_{split}"] for split in SPLITS}}
        for split in SPLITS:
            part = self.frame[self.frame.split == split]
            history, context, grids = self.original.transform_inputs(part)
            values = dict(history=history, context=context, grid_indices=grids,
                          targets=self.original.transform_targets(part),
                          actual=part[list(TARGET_COLUMNS)].to_numpy(dtype=np.float64),
                          persistence=part[list(TARGET_COLUMNS)].to_numpy(dtype=np.float64) - 1)
            target = self.cache / split
            target.mkdir()
            for name, array in values.items():
                np.save(target / f"{name}.npy", array)
            part[["run_id", "decision_step", "grid_id", "split"]].to_csv(target / "keys.csv", index=False)
        (self.cache / "cache_manifest.json").write_text(json.dumps(metadata))

    def test_train_only_extrema_conversion_and_validation_not_clipped(self):
        with patch("baseline.data._csv_chunks", side_effect=AssertionError), \
                patch("baseline.data.pd.read_csv", side_effect=AssertionError):
            splits, processor, manifest = load_dataset(self.source, self.cache, self.output)
        self.assertEqual(processor.fit_row_count, 3)
        self.assertEqual(manifest["row_count"], 6)
        self.assertEqual(processor.grid_ids, [3, 7])
        np.testing.assert_allclose(processor.scalers["history"]["mean"], [0, 2], atol=1e-6)
        np.testing.assert_allclose(processor.scalers["history"]["scale"], [10, 1], atol=1e-6)
        expected = MinMaxPreprocessor(2, ["requests", "price"]).fit(self.frame[self.frame.split == "train"])
        for name, dataset in splits.items():
            frame = self.frame[self.frame.split == name]
            reference = (*expected.transform_inputs(frame), expected.transform_targets(frame))
            actual = dataset[slice(None)]
            self.assertEqual([value.dtype for value in actual],
                             [torch.float32, torch.float32, torch.int64, torch.float32])
            for tensor, wanted in zip(actual, reference):
                np.testing.assert_allclose(tensor.numpy(), wanted, atol=1e-6, rtol=1e-6)
        validation = splits["validation"][:]
        self.assertLess(float(validation[0].min()), 0)
        self.assertGreater(float(validation[0].max()), 1)
        self.assertGreater(float(validation[3].max()), 1)
        np.testing.assert_allclose(validation[1][:, 1], 2.)  # 训练集常量 5，验证集 7，不裁剪。

    def test_chunked_cache_fit_constant_features_and_raw_fit_agree(self):
        splits, base, _ = load_prepared_dataset(self.source, self.cache)
        cached = MinMaxPreprocessor(2, ["requests", "price"]).fit_cache(splits["train"], base, chunk_size=1)
        train = self.frame[self.frame.split == "train"]
        direct = MinMaxPreprocessor(2, ["requests", "price"]).fit_chunks([train.iloc[:1], train.iloc[1:]])
        for name in cached.scalers:
            for key in ("mean", "scale"):
                np.testing.assert_allclose(cached.scalers[name][key], direct.scalers[name][key], atol=1e-6)
        self.assertEqual(cached.scalers["history"]["scale"][1], 1.)
        self.assertEqual(cached.scalers["context"]["scale"][1], 1.)
        self.assertEqual(cached.scalers["targets"]["scale"][-1], 1.)
        with self.assertRaises(ValueError):
            cached.fit_cache(splits["validation"], base)
        with self.assertRaises(ValueError):
            direct.fit(self.frame)

    def test_saved_normalization_reused_without_rescan_and_rejects_changed_cache(self):
        splits, processor, _ = load_dataset(self.source, self.cache, self.output)
        state = json.loads((self.output / "normalization.json").read_text())
        self.assertEqual(state["source_cache_manifest_sha256"], _sha256(self.cache / "cache_manifest.json"))
        self.assertEqual(state["normalization_fit_split"], "train")
        with patch.object(MinMaxPreprocessor, "fit_cache", side_effect=AssertionError):
            reused, restored, _ = load_dataset(self.source, self.cache, self.output)
        self.assertEqual(restored.state_dict(), processor.state_dict())
        np.testing.assert_array_equal(reused["train"][:][0], splits["train"][:][0])
        manifest = self.cache / "cache_manifest.json"
        manifest.write_text(manifest.read_text() + "\n")
        with self.assertRaises(ValueError):
            load_dataset(self.source, self.cache, self.output)

    def test_roundtrip_inverse_targets_and_unknown_grids(self):
        processor = MinMaxPreprocessor(2, ["requests", "price"]).fit(self.frame[self.frame.split == "train"])
        state = json.loads(json.dumps(processor.state_dict()))
        self.assertEqual(state["normalization"], "train_minmax")
        restored = MinMaxPreprocessor.from_state_dict(state)
        self.assertEqual(state, restored.state_dict())
        normalized = restored.transform_targets(self.frame)
        np.testing.assert_allclose(restored.inverse_targets(normalized), self.frame[list(TARGET_COLUMNS)], atol=1e-6)
        with self.assertRaises(ValueError):
            restored.transform_inputs(self.frame.assign(grid_id=99))
        state.pop("normalization")
        with self.assertRaises(ValueError):
            MinMaxPreprocessor.from_state_dict(state)

    def test_evaluation_fields_preserved_cache_immutable_and_spawn_reopens(self):
        before = {str(path): _sha256(path) for path in self.cache.rglob("*") if path.is_file()}
        splits, _, _ = load_dataset(self.source, self.cache, self.output)
        dataset = splits["test"]
        actual = self.frame[self.frame.split == "test"][list(TARGET_COLUMNS)].to_numpy()
        np.testing.assert_array_equal(dataset.arrays["actual"], actual)
        np.testing.assert_array_equal(dataset.arrays["persistence"], actual - 1)
        self.assertEqual(dataset.run_ids, ["run_test"])
        self.assertEqual(dataset.keys_path, (self.cache / "test/keys.csv").resolve())
        self.assertTrue(all(isinstance(array, np.memmap) and not array.flags.writeable
                            for array in dataset.arrays.values()))
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertEqual(restored.arrays, {})
        for wanted, tensor in zip(dataset[0], restored[0]):
            np.testing.assert_array_equal(tensor.numpy(), wanted.numpy())
        after = {str(path): _sha256(path) for path in self.cache.rglob("*") if path.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
