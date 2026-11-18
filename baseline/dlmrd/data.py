"""复用时序缓存，仅用训练样本拟合 DL-MRD 的 Min-Max 归一化。"""

import json
from pathlib import Path

import numpy as np
import torch

from baseline.data import SequenceDataset, _sha256, load_prepared_dataset
from baseline.sequence import SequencePreprocessor, TARGET_COLUMNS, _numeric_values


CHUNK_SIZE = 131072
NORMALIZATION_VERSION = "dlmrd_train_minmax_v1"


class MinMaxPreprocessor(SequencePreprocessor):
    """沿用输入与网格映射接口；mean/scale 字段分别保存最小值与极差。"""

    def _set_scalers(self, bounds):
        self.scalers = {}
        for name, (minimum, maximum) in bounds.items():
            scale = (maximum - minimum).astype(np.float32)
            scale[scale == 0] = 1.
            self.scalers[name] = {"mean": minimum.astype(np.float32), "scale": scale}

    def fit_chunks(self, frames):
        bounds, grids, count = {}, set(), 0
        for frame in frames:
            if frame.empty:
                continue
            if "split" in frame and not frame["split"].eq("train").all():
                raise ValueError
            history, context = self._raw_inputs(frame)
            grids.update(self._grid_values(frame).tolist())
            count += len(frame)
            for name, values in (("history", history.reshape(-1, history.shape[-1])),
                                 ("context", context),
                                 ("targets", _numeric_values(frame, TARGET_COLUMNS))):
                low, high = values.min(axis=0), values.max(axis=0)
                if name in bounds:
                    low, high = np.minimum(bounds[name][0], low), np.maximum(bounds[name][1], high)
                bounds[name] = (low, high)
        if not grids or (self.has_joint_actions and self.joint_grid_ids != sorted(grids)):
            raise ValueError
        self.grid_ids, self.fit_row_count = sorted(grids), count
        self._set_scalers(bounds)
        return self

    def fit_cache(self, dataset, original, chunk_size=CHUNK_SIZE):
        """分块扫描 train 的 z-score 数组，在原量纲下拟合全局特征极值。"""
        if (dataset.split != "train" or len(dataset) == 0 or chunk_size <= 0
                or len(dataset) != original.fit_row_count
                or self.history_steps != original.history_steps
                or self.history_features != original.history_features
                or self.context_columns != original.context_columns):
            raise ValueError
        if not dataset.arrays:
            dataset.arrays = dataset._open_arrays()
        bounds = {}
        for name in ("history", "context", "targets"):
            stats = original.scalers[name]
            mean, scale = stats["mean"].astype(np.float64), stats["scale"].astype(np.float64)
            if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError
            array = dataset.arrays[name]
            low, high = np.full_like(mean, np.inf), np.full_like(mean, -np.inf)
            for start in range(0, len(dataset), chunk_size):
                values = array[start:start + chunk_size]
                if not np.isfinite(values).all():
                    raise ValueError
                axes = tuple(range(values.ndim - 1))
                # 标准差为正，先缩减极值再还原，可避免整块创建 float64 历史副本。
                low = np.minimum(low, values.min(axis=axes).astype(np.float64) * scale + mean)
                high = np.maximum(high, values.max(axis=axes).astype(np.float64) * scale + mean)
            bounds[name] = (low, high)
        self.grid_ids = list(original.grid_ids)
        self.fit_row_count = len(dataset)
        self._set_scalers(bounds)
        return self

    def state_dict(self):
        return {**super().state_dict(), "normalization": "train_minmax"}

    @classmethod
    def from_state_dict(cls, state):
        if state.get("normalization") != "train_minmax":
            raise ValueError
        instance = super().from_state_dict(state)
        dimensions = {"history": len(instance.history_features),
                      "context": len(instance.context_columns), "targets": len(TARGET_COLUMNS)}
        if (instance.grid_ids != sorted(set(instance.grid_ids)) or not instance.grid_ids
                or instance.fit_row_count is None or instance.fit_row_count <= 0
                or set(instance.scalers) != set(dimensions)
                or (instance.has_joint_actions and instance.joint_grid_ids != instance.grid_ids)):
            raise ValueError
        for name, width in dimensions.items():
            stats = instance.scalers[name]
            if (set(stats) != {"mean", "scale"}
                    or any(value.shape != (width,) or not np.isfinite(value).all()
                           for value in stats.values())
                    or (stats["scale"] <= 0).any()):
                raise ValueError
        return instance


class MinMaxDataset(SequenceDataset):
    """保留只读缓存与评价字段；索引时把当前 batch 转成 Min-Max 张量。"""

    def __init__(self, dataset, original, preprocessor):
        super().__init__(dataset.cache_dir, dataset.split, dataset.metadata, original)
        if original.grid_ids != preprocessor.grid_ids:
            raise ValueError
        self.grid_count = len(preprocessor.grid_ids)
        self.original_scalers = original.scalers
        self.minmax_scalers = preprocessor.scalers

    def __getitem__(self, index):
        if not self.arrays:
            self.arrays = self._open_arrays()
        values = []
        for name in ("history", "context", "grid_indices", "targets"):
            array = np.asarray(self.arrays[name][index])
            if name == "grid_indices":
                if (array < 0).any() or (array >= self.grid_count).any():
                    raise ValueError
                if not array.flags.writeable:
                    array = array.copy()
            else:
                original, normalized = self.original_scalers[name], self.minmax_scalers[name]
                array = array * original["scale"] + original["mean"]
                array -= normalized["mean"]
                array /= normalized["scale"]
            values.append(torch.from_numpy(array))
        return tuple(values)


def load_dataset(input_dir, cache_dir, output_dir):
    """首次只扫描训练数组；重启读取已保存的归一化参数，不重建共享缓存。"""
    splits, original, manifest = load_prepared_dataset(input_dir, cache_dir)
    cache_dir, output_dir = Path(cache_dir).resolve(), Path(output_dir)
    source_sha256 = _sha256(cache_dir / "cache_manifest.json")
    normalization_path = output_dir / "normalization.json"
    if normalization_path.exists():
        saved = json.loads(normalization_path.read_text(encoding="utf-8"))
        if (saved["schema_version"] != NORMALIZATION_VERSION
                or saved["source_cache_manifest_sha256"] != source_sha256
                or saved["normalization_fit_split"] != "train"):
            raise ValueError
        preprocessor = MinMaxPreprocessor.from_state_dict(saved["preprocessor"])
        if (preprocessor.fit_row_count != original.fit_row_count
                or preprocessor.grid_ids != original.grid_ids
                or preprocessor.history_steps != original.history_steps
                or preprocessor.history_features != original.history_features
                or preprocessor.context_columns != original.context_columns):
            raise ValueError
    else:
        preprocessor = MinMaxPreprocessor(original.history_steps, original.history_features,
                                          original.context_columns)
        print(f"仅扫描 {len(splits['train']):,} 条训练缓存，拟合 Min-Max 归一化……", flush=True)
        preprocessor.fit_cache(splits["train"], original)
        if _sha256(cache_dir / "cache_manifest.json") != source_sha256:
            raise ValueError
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = {"schema_version": NORMALIZATION_VERSION,
                 "source_cache": str(cache_dir), "source_cache_manifest_sha256": source_sha256,
                 "normalization_fit_split": "train", "preprocessor": preprocessor.state_dict()}
        temporary = normalization_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False),
                             encoding="utf-8")
        temporary.replace(normalization_path)
    return ({name: MinMaxDataset(dataset, original, preprocessor) for name, dataset in splits.items()},
            preprocessor, manifest)
