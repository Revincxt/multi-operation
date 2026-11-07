"""复用监督缓存构造全城转移；当前动作与当期结果不进入状态。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from baseline.data import DEFAULT_COLLECTION, DEFAULT_INPUT, load_prepared_dataset
from baseline.sequence import SequencePreprocessor


DEFAULT_REPLAY = "datasets_grid6_low_mixed47k_20261029/offline_rl"
VERSION = "city_offline_replay_v1"


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def state_context_indices(processor):
    """历史动作可观测；当前自身动作和其他网格当前动作必须排除。"""
    return [i for i, name in enumerate(processor.context_columns)
            if name not in ("price_multiplier", "matching_radius_km")
            and not name.startswith(("joint_price_grid_", "joint_radius_grid_"))]


def encode_state(history, context, indices):
    """输入为 batch × grid × history/context，网格顺序固定表示区域身份。"""
    if history.shape[:2] != context.shape[:2]:
        raise ValueError
    return np.concatenate((history.reshape(len(history), -1),
                           context[..., indices].reshape(len(context), -1)), axis=1).astype("float32")


def local_state_indices(processor):
    """每个智能体只读自身历史、时间和本区/邻区状态，不读全城状态。"""
    context = [processor.context_columns[i] for i in state_context_indices(processor)]
    local = [i for i, name in enumerate(context) if not name.startswith("global_")]
    history_size = processor.history_steps * len(processor.history_features)
    history_total = len(processor.grid_ids) * history_size
    return [list(range(grid * history_size, (grid + 1) * history_size))
            + [history_total + grid * len(context) + i for i in local]
            for grid in range(len(processor.grid_ids))]


def decode_actions(context, processor, prices, radii):
    raw = context * processor.scalers["context"]["scale"] + processor.scalers["context"]["mean"]
    price = raw[..., processor.context_columns.index("price_multiplier")]
    radius = raw[..., processor.context_columns.index("matching_radius_km")]
    p = np.abs(price[..., None] - prices).argmin(-1)
    r = np.abs(radius[..., None] - radii).argmin(-1)
    if (not np.allclose(prices[p], price, atol=2e-5, rtol=0)
            or not np.allclose(radii[r], radius, atol=2e-5, rtol=0)):
        raise ValueError
    # 验证每一行记录的全城动作确实对应这六个网格的实际动作。
    for grid, grid_id in enumerate(processor.grid_ids):
        for prefix, own in (("joint_price_grid_", price), ("joint_radius_grid_", radius)):
            index = processor.context_columns.index(f"{prefix}{grid_id}")
            if not np.allclose(raw[..., index], own[..., grid, None], atol=2e-5, rtol=0):
                raise ValueError
    return (p * len(radii) + r).astype("int16")


def prepare(input_dir=DEFAULT_INPUT, collection_root=DEFAULT_COLLECTION, output=DEFAULT_REPLAY,
            chunk_runs=64):
    """保留原划分，把每个决策周期的六行组合为一条城市级转移。"""
    input_dir, collection_root, output = map(Path, (input_dir, collection_root, output))
    caches = list((input_dir / "sequence_cache").glob("*/cache_manifest.json"))
    if len(caches) != 1 or chunk_runs <= 0 or output.exists():
        raise ValueError
    splits, processor, source = load_prepared_dataset(input_dir, caches[0].parent)
    collection = json.loads((collection_root / "collection_config.json").read_text())
    if (not processor.has_joint_actions or source["row_order"] != "run_id,grid_id,decision_step"
            or source["history_steps"] != processor.history_steps):
        raise ValueError
    grids = len(processor.grid_ids)
    periods = collection["simulation_steps"] // collection["decision_freq"] - processor.history_steps
    if collection["simulation_steps"] % collection["decision_freq"] or periods <= 1:
        raise ValueError
    indices = state_context_indices(processor)
    state_size = grids * (processor.history_steps * len(processor.history_features) + len(indices))
    prices = np.asarray(collection["price_action_values"], dtype="float32")
    radii = np.asarray(collection["matching_radius_values"], dtype="float32")
    metadata = {
        "schema_version": VERSION, "status": "building", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(input_dir.resolve()), "source_manifest_sha256": sha256(input_dir / "dataset_manifest.json"),
        "source_cache": str(caches[0].parent.resolve()), "source_cache_manifest_sha256": sha256(caches[0]),
        "collection_root": str(collection_root.resolve()), "collection_config_sha256": sha256(collection_root / "collection_config.json"),
        "grid_ids": processor.grid_ids, "grid_count": grids, "state_size": state_size,
        "state_context_indices": indices, "state_context_columns": [processor.context_columns[i] for i in indices],
        "local_state_indices": local_state_indices(processor),
        "preprocessor": processor.state_dict(), "periods_per_run": periods,
        "decision_steps": list(range(processor.history_steps, periods + processor.history_steps)),
        "action_pairs": [[p, r] for p in collection["price_action_values"]
                         for r in collection["matching_radius_values"]],
        "reward": "sum of six grids' current-period platform commission revenue in USD",
        "terminal": "10:00 is the finite evaluation horizon; final reward retained, no bootstrap",
        "normalization": "existing train-only feature scalers; no current actions or outcomes in state",
        "observation": "four-step histories and pre-action aggregate states; partial observation of simulator",
        "splits": {},
    }
    output.mkdir(parents=True)
    save_json(output / "manifest.json", metadata)
    revenue_index = processor.state_dict()["target_columns"].index("target_platform_revenue")
    time_index = processor.context_columns.index("decision_start_sim_time")
    start_seconds = int(collection["start_time"][:2]) * 3600 + int(collection["start_time"][3:]) * 60
    expected_times = start_seconds + np.asarray(metadata["decision_steps"]) * collection["decision_freq"] * 60
    try:
        for split, dataset in splits.items():
            run_ids = dataset.metadata["split_run_ids"][split]
            runs, count = len(run_ids), len(dataset) // grids
            if len(dataset) != runs * grids * periods or any(
                    source["source_run_sample_counts"][run_id] != grids * periods for run_id in run_ids):
                raise ValueError
            target = output / split
            target.mkdir()
            arrays = {
                "states": np.lib.format.open_memmap(target / "states.npy", mode="w+", dtype="float32", shape=(count, state_size)),
                "actions": np.lib.format.open_memmap(target / "actions.npy", mode="w+", dtype="int16", shape=(count, grids)),
                "rewards": np.lib.format.open_memmap(target / "rewards.npy", mode="w+", dtype="float32", shape=(count,)),
            }
            for first in range(0, runs, chunk_runs):
                last = min(runs, first + chunk_runs)
                n = last - first
                a, b = first * grids * periods, last * grids * periods
                history = np.asarray(dataset.arrays["history"][a:b]).reshape(n, grids, periods, processor.history_steps, -1).transpose(0, 2, 1, 3, 4).reshape(n * periods, grids, processor.history_steps, -1)
                context = np.asarray(dataset.arrays["context"][a:b]).reshape(n, grids, periods, -1).transpose(0, 2, 1, 3).reshape(n * periods, grids, -1)
                grid_indices = np.asarray(dataset.arrays["grid_indices"][a:b]).reshape(n, grids, periods)
                times = context[..., time_index] * processor.scalers["context"]["scale"][time_index] + processor.scalers["context"]["mean"][time_index]
                if (not np.array_equal(grid_indices, np.broadcast_to(np.arange(grids)[None, :, None], grid_indices.shape))
                        or not np.allclose(times.reshape(n, periods, grids), expected_times[None, :, None], atol=0.01)):
                    raise ValueError
                states = encode_state(history, context, indices)
                actions = decode_actions(context, processor, prices, radii)
                rewards = np.asarray(dataset.arrays["actual"][a:b, revenue_index]).reshape(n, grids, periods).sum(axis=1).reshape(-1)
                if not np.isfinite(states).all() or not np.isfinite(rewards).all() or (rewards < 0).any():
                    raise ValueError
                lo, hi = first * periods, last * periods
                arrays["states"][lo:hi], arrays["actions"][lo:hi], arrays["rewards"][lo:hi] = states, actions, rewards
                if first % (chunk_runs * 40) == 0 or last == runs:
                    print(f"{split}: 已构建 {last:,}/{runs:,} 条轨迹，{hi:,} 条城市级转移", flush=True)
            for array in arrays.values():
                array.flush()
            (target / "run_ids.json").write_text(json.dumps(run_ids), encoding="utf-8")
            metadata["splits"][split] = {"runs": runs, "transitions": count, "terminal_transitions": runs}
            save_json(output / "manifest.json", metadata)
        metadata.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
        save_json(output / "manifest.json", metadata)
    except BaseException:
        metadata["status"] = "failed"
        save_json(output / "manifest.json", metadata)
        raise
    return metadata


class Replay:
    """下一状态用连续索引读取，绝不跨轨迹；终止状态不复制一份大数组。"""

    def __init__(self, root, split):
        root = Path(root)
        self.metadata = json.loads((root / "manifest.json").read_text())
        if self.metadata["schema_version"] != VERSION or self.metadata["status"] != "complete":
            raise ValueError
        self.periods = self.metadata["periods_per_run"]
        self.arrays = {name: np.load(root / split / f"{name}.npy", mmap_mode="r")
                       for name in ("states", "actions", "rewards")}
        self.count = len(self.arrays["states"])
        if self.count != self.metadata["splits"][split]["transitions"]:
            raise ValueError

    def batch(self, indices):
        indices = np.asarray(indices)
        terminal = (indices + 1) % self.periods == 0
        next_indices = np.where(terminal, indices, indices + 1)
        next_states = np.array(self.arrays["states"][next_indices])
        next_states[terminal] = 0
        return {"states": np.array(self.arrays["states"][indices]),
                "actions": np.array(self.arrays["actions"][indices], dtype="int64"),
                "rewards": np.array(self.arrays["rewards"][indices]),
                "next_states": next_states, "masks": (~terminal).astype("float32")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从现有 47k 数据缓存构建全城离线 RL 转移")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--collection-root", default=DEFAULT_COLLECTION)
    parser.add_argument("--output", default=DEFAULT_REPLAY)
    prepare(**vars(parser.parse_args()))
