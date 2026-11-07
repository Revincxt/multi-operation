"""时序基线的分块预处理与磁盘数组读取。

首次读取 CSV 时生成共享缓存；之后 LSTM / TimesNet 按 batch 读取同一份
数组，不常驻完整 DataFrame，也不重复保存整份训练张量。
"""

from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from baseline.sequence import CONTEXT_COLUMNS, TARGET_COLUMNS, SequencePreprocessor
from src.env.observation import state_columns


DEFAULT_INPUT = "datasets_grid6_low_mixed47k_20261029"
DEFAULT_COLLECTION = "trajectories_grid6_low_mixed_20261027"
DEFAULT_MODEL = "models/lstm_grid6_low_mixed47k_mps100_20261029"
DEFAULT_CHUNK_SIZE = 20000
CACHE_VERSION = "sequence_memmap_v1"
SPLITS = ("train", "validation", "test")
KEY_COLUMNS = ("run_id", "decision_step", "grid_id", "split")
PERSISTENCE_COLUMNS = tuple(column.replace("target_", "lag_1_", 1)
                            for column in TARGET_COLUMNS)


def _sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _csv_chunks(path, columns, chunk_size):
    with pd.read_csv(path, usecols=columns, chunksize=chunk_size,
                     dtype={key: str for key in ("run_id", "split", "date")
                            if key in columns}) as reader:
        yield from reader


def _fit_training_data(csv_path, columns, chunk_size, manifest, preprocessor):
    """检查全部来源与划分，仅把 train 行交给标准化器。"""
    counts = Counter()
    run_counts = Counter()
    run_splits = {}
    split_dates = defaultdict(set)
    run_dates = defaultdict(set)
    current_run = None
    current_keys = set()

    def training_frames():
        nonlocal current_run, current_keys
        for frame in _csv_chunks(csv_path, columns, chunk_size):
            if (frame.isna().any().any() or not set(frame["split"]).issubset(SPLITS)
                    or frame.duplicated(["run_id", "decision_step", "grid_id"]).any()):
                raise ValueError
            numeric_columns = [c for c in columns if c not in ("run_id", "split", "date")]
            numeric = frame[numeric_columns].to_numpy(dtype=float)
            steps = frame["decision_step"].to_numpy(dtype=float)
            if (not np.isfinite(numeric).all() or (steps < preprocessor.history_steps).any()
                    or not np.equal(steps, np.floor(steps)).all()):
                raise ValueError
            preprocessor._grid_values(frame)
            counts.update(frame["split"].value_counts().to_dict())
            # 管线按 run 连续输出。只保留当前 run 的键，仍可检查跨块重复行。
            blocks = frame["run_id"].ne(frame["run_id"].shift()).cumsum()
            for _, block in frame.groupby(blocks, sort=False):
                run_id = block["run_id"].iloc[0]
                split = block["split"].iloc[0]
                if run_id != current_run:
                    if run_id in run_splits:
                        raise ValueError
                    current_run, current_keys = run_id, set()
                if (not block["split"].eq(split).all()
                        or run_splits.get(run_id, split) != split):
                    raise ValueError
                keys = set(zip(block["decision_step"], block["grid_id"]))
                if current_keys.intersection(keys):
                    raise ValueError
                current_keys.update(keys)
                run_splits[run_id] = split
                run_counts[run_id] += len(block)
                if "date" in block:
                    dates = set(block["date"])
                    run_dates[run_id].update(dates)
                    split_dates[split].update(dates)
            train = frame[frame["split"] == "train"]
            if not train.empty:
                yield train

    preprocessor.fit_chunks(training_frames())
    if (set(counts) != set(SPLITS) or dict(counts) != manifest["split_counts"]
            or sum(counts.values()) != manifest["row_count"]
            or any(len(dates) != 1 for dates in run_dates.values())):
        raise ValueError
    for field, actual in (
        ("source_run_splits", run_splits), ("source_run_sample_counts", dict(run_counts)),
        ("source_run_count", len(run_splits)),
    ):
        if field in manifest and manifest[field] != actual:
            raise ValueError
    if "source_run_ids" in manifest and set(manifest["source_run_ids"]) != set(run_splits):
        raise ValueError
    split_dates = {name: sorted(values) for name, values in split_dates.items()}
    if "split_dates" in manifest and split_dates != manifest["split_dates"]:
        raise ValueError
    if manifest.get("split_by") == "date":
        if (set(split_dates) != set(SPLITS)
                or max(split_dates["train"]) >= min(split_dates["validation"])
                or max(split_dates["validation"]) >= min(split_dates["test"])):
            raise ValueError
    return {
        "row_count": sum(counts.values()), "split_counts": dict(counts),
        "split_run_ids": {name: sorted(run for run, split in run_splits.items() if split == name)
                          for name in SPLITS},
        "split_dates": split_dates,
    }


def _array_specs(count, preprocessor):
    return {
        "history": ((count, preprocessor.history_steps, len(preprocessor.history_features)), "float32"),
        "context": ((count, len(preprocessor.context_columns)), "float32"),
        "grid_indices": ((count,), "int64"),
        "targets": ((count, len(TARGET_COLUMNS)), "float32"),
        "actual": ((count, len(TARGET_COLUMNS)), "float64"),
        "persistence": ((count, len(TARGET_COLUMNS)), "float64"),
    }


def _write_arrays(directory, csv_path, columns, chunk_size, summary, preprocessor):
    """第二遍读取写入标准化数组；原量纲标签用于最终评估。"""
    arrays = {}
    offsets = Counter()
    with ExitStack() as stack:
        key_files = {}
        for split in SPLITS:
            target = directory / split
            target.mkdir()
            arrays[split] = {
                name: np.lib.format.open_memmap(target / f"{name}.npy", mode="w+",
                                               dtype=dtype, shape=shape)
                for name, (shape, dtype) in _array_specs(
                    summary["split_counts"][split], preprocessor).items()
            }
            key_files[split] = stack.enter_context(
                (target / "keys.csv").open("w", encoding="utf-8", newline=""))
        for frame in _csv_chunks(csv_path, columns, chunk_size):
            for split, group in frame.groupby("split", sort=False):
                start, end = offsets[split], offsets[split] + len(group)
                if end > summary["split_counts"][split]:
                    raise ValueError
                history, context, grids = preprocessor.transform_inputs(group)
                values = {
                    "history": history, "context": context, "grid_indices": grids,
                    "targets": preprocessor.transform_targets(group),
                    "actual": group[list(TARGET_COLUMNS)].to_numpy(dtype=float),
                    "persistence": group[list(PERSISTENCE_COLUMNS)].to_numpy(dtype=float),
                }
                for name, value in values.items():
                    if not np.isfinite(value).all():
                        raise ValueError
                    arrays[split][name][start:end] = value
                key_columns = list(KEY_COLUMNS) + (["date"] if "date" in group else [])
                group[key_columns].to_csv(key_files[split], index=False, header=start == 0)
                offsets[split] = end
        if dict(offsets) != summary["split_counts"]:
            raise ValueError
        for split_arrays in arrays.values():
            for array in split_arrays.values():
                array.flush()


class SequenceDataset(Dataset):
    """只读内存映射；一次索引一个 batch，不复制全体训练样本。"""

    def __init__(self, cache_dir, split, metadata, preprocessor):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.metadata = metadata
        self.run_ids = metadata["split_run_ids"][split]
        self.sample_count = metadata["split_counts"][split]
        self.keys_path = self.cache_dir / split / "keys.csv"
        self.array_specs = _array_specs(self.sample_count, preprocessor)
        self.arrays = self._open_arrays()

    def _open_arrays(self):
        arrays = {}
        for name, (shape, dtype) in self.array_specs.items():
            array = np.load(self.cache_dir / self.split / f"{name}.npy",
                            mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != np.dtype(dtype):
                raise ValueError
            arrays[name] = array
        return arrays

    def __getstate__(self):
        # macOS 使用 spawn 创建 DataLoader worker；不要把父进程的 memmap
        # 序列化给每个 worker，worker 收到轻量状态后自行打开只读映射。
        state = self.__dict__.copy()
        state["arrays"] = {}
        return state

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        # 仅复制当前 batch，避免 PyTorch 写入只读 memmap 或读取整表。
        if not self.arrays:
            self.arrays = self._open_arrays()
        values = []
        for name in ("history", "context", "grid_indices", "targets"):
            array = np.asarray(self.arrays[name][index])
            # 列表索引已经生成可写副本，仅 slice 等只读视图需要额外复制。
            if not array.flags.writeable:
                array = array.copy()
            values.append(torch.from_numpy(array))
        return tuple(values)


def load_prepared_dataset(input_dir, cache_dir):
    """直接打开已准备的缓存，只核对来源元数据，不重读 CSV 或扫描数组摘要。"""
    input_dir, cache_dir = Path(input_dir).resolve(), Path(cache_dir).resolve()
    manifest_path = input_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    if (cache_dir.parent != input_dir / "sequence_cache"
            or metadata["schema_version"] != CACHE_VERSION
            or metadata["source_sha256"]["manifest"] != _sha256(manifest_path)
            or metadata["row_count"] != manifest["row_count"]
            or metadata["split_counts"] != manifest["split_counts"]
            or metadata["normalization_fit_split"] != "train"):
        raise ValueError
    processor = SequencePreprocessor.from_state_dict(metadata["preprocessor"])
    if (processor.fit_row_count != manifest["split_counts"]["train"]
            or processor.history_steps != manifest["history_steps"]):
        raise ValueError
    splits = {name: SequenceDataset(cache_dir, name, metadata, processor) for name in SPLITS}
    print(f"直接复用已准备缓存：{cache_dir}，共 {manifest['row_count']:,} 条样本。", flush=True)
    return splits, processor, manifest


def load_dataset(input_dir=DEFAULT_INPUT, chunk_size=DEFAULT_CHUNK_SIZE):
    """生成或复用共享缓存，返回三份磁盘 Dataset、标准化器和原 manifest。"""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError
    input_dir = Path(input_dir)
    csv_path = input_dir / "supervised_dataset.csv"
    manifest_path = input_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") not in (
            "grid_supervised_dataset", "grid_supervised_dataset_v3")
            or manifest.get("split_by", "run") not in ("run", "date")):
        raise ValueError
    history_features = [column.removeprefix("lag_1_") for column in manifest["feature_columns"]
                        if column.startswith("lag_1_")]
    additional_context = manifest.get(
        "additional_context_columns", state_columns(manifest["feature_columns"]))
    if (len(set(additional_context)) != len(additional_context)
            or not set(additional_context).issubset(manifest["feature_columns"])):
        raise ValueError
    context_columns = list(CONTEXT_COLUMNS) + list(additional_context)
    preprocessor = SequencePreprocessor(manifest["history_steps"], history_features, context_columns)
    columns = list(KEY_COLUMNS) + context_columns + list(TARGET_COLUMNS) + [
        f"lag_{lag}_{feature}" for lag in range(1, preprocessor.history_steps + 1)
        for feature in history_features]
    header = pd.read_csv(csv_path, nrows=0).columns
    if "date" in header:
        columns.append("date")
    if (not set(columns).issubset(header) or not set(PERSISTENCE_COLUMNS).issubset(columns)
            or (manifest.get("split_by") == "date" and "date" not in columns)):
        raise ValueError
    source_hashes = {"csv": _sha256(csv_path), "manifest": _sha256(manifest_path)}
    fingerprint = hashlib.sha256(json.dumps(
        [CACHE_VERSION, source_hashes, list(TARGET_COLUMNS), context_columns],
        sort_keys=True).encode()).hexdigest()
    cache_root = input_dir / "sequence_cache"
    cache_dir = cache_root / fingerprint
    if not cache_dir.exists():
        print("分块检查数据并拟合训练集标准化参数……", flush=True)
        summary = _fit_training_data(csv_path, columns, chunk_size, manifest, preprocessor)
        cache_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".building_", dir=cache_root) as temporary:
            staged = Path(temporary) / "cache"
            staged.mkdir()
            print(f"写入磁盘数组：{summary['row_count']:,} 条样本……", flush=True)
            _write_arrays(staged, csv_path, columns, chunk_size, summary, preprocessor)
            # 防止构建过程中源数据被替换，留下与指纹不一致的缓存。
            if source_hashes != {"csv": _sha256(csv_path), "manifest": _sha256(manifest_path)}:
                raise ValueError
            metadata = {
                "schema_version": CACHE_VERSION, "source_sha256": source_hashes,
                **summary, "preprocessor": preprocessor.state_dict(),
                "normalization_fit_split": "train",
                "files": {str(path.relative_to(staged)): _sha256(path)
                          for path in sorted(staged.glob("*/*")) if path.is_file()},
            }
            (staged / "cache_manifest.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            staged.rename(cache_dir)
    metadata = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    expected_files = {f"{split}/{name}.npy" for split in SPLITS
                      for name in _array_specs(manifest["split_counts"][split], preprocessor)}
    expected_files.update(f"{split}/keys.csv" for split in SPLITS)
    if (metadata["schema_version"] != CACHE_VERSION or metadata["source_sha256"] != source_hashes
            or metadata["split_counts"] != manifest["split_counts"]
            or metadata["row_count"] != manifest["row_count"]
            or metadata["normalization_fit_split"] != "train"
            or set(metadata["files"]) != expected_files
            or any(_sha256(cache_dir / path) != digest for path, digest in metadata["files"].items())):
        raise ValueError
    preprocessor = SequencePreprocessor.from_state_dict(metadata["preprocessor"])
    if (preprocessor.fit_row_count != manifest["split_counts"]["train"]
            or preprocessor.history_steps != manifest["history_steps"]
            or preprocessor.history_features != history_features
            or preprocessor.context_columns != context_columns):
        raise ValueError
    splits = {name: SequenceDataset(cache_dir, name, metadata, preprocessor) for name in SPLITS}
    print(f"磁盘数据已就绪：{sum(map(len, splits.values())):,} 条，"
          f"网格 {preprocessor.grid_ids}，仅用 {preprocessor.fit_row_count:,} 条训练样本拟合标准化。",
          flush=True)
    return splits, preprocessor, manifest
