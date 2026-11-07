"""直接从完整轨迹构建城市级离线转移，复用已验证缓存和训练集标准化器。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from baseline.sequence import SequencePreprocessor
from data_pipeline.build_supervised_dataset import _read_run, aggregate_decision_metrics
from data_pipeline.collect_density import config_signature, planned_splits, run_path
from offline_rl.data import DEFAULT_REPLAY, Replay, decode_actions, encode_state, save_json, sha256
from src.utils.joint_context import add_joint_context


DEFAULT_OUTPUT = "datasets_grid6_low_mixed100k_20261103/offline_rl"
_WORKER = None


def encode_periods(periods, processor, metadata):
    """与监督缓存相同的过去四期顺序；当期动作和结果均不进入状态。"""
    periods, _ = add_joint_context(periods)
    periods = periods.sort_values(["decision_step", "grid_id"])
    grids, history_steps = metadata["grid_count"], processor.history_steps
    steps = periods.decision_step.nunique()
    if (len(periods) != grids * steps
            or not np.array_equal(periods.decision_step.to_numpy().reshape(steps, grids),
                                  np.broadcast_to(np.arange(steps)[:, None], (steps, grids)))
            or not np.array_equal(periods.grid_id.to_numpy().reshape(steps, grids),
                                  np.broadcast_to(np.arange(grids), (steps, grids)))):
        raise ValueError
    values = periods[processor.history_features].to_numpy(dtype="float32").reshape(steps, grids, -1)
    times = np.arange(history_steps, steps)[:, None] + np.arange(-history_steps, 0)
    history = values[times].transpose(0, 2, 1, 3)
    context = periods[processor.context_columns].to_numpy(dtype="float32").reshape(steps, grids, -1)[history_steps:]
    history = processor._standardize("history", history)
    context = processor._standardize("context", context)
    states = encode_state(history, context, metadata["state_context_indices"])
    pairs = np.asarray(metadata["action_pairs"], dtype="float32")
    actions = decode_actions(context, processor, np.unique(pairs[:, 0]), np.unique(pairs[:, 1]))
    revenue = periods.platform_revenue.to_numpy(dtype="float64").reshape(steps, grids)[history_steps:]
    # 原缓存的actual是float64；六区域求和之后才写入float32奖励数组。
    rewards = revenue.sum(1).astype("float32")
    if (len(states) != metadata["periods_per_run"] or not np.isfinite(states).all()
            or not np.isfinite(rewards).all() or (rewards < 0).any()):
        raise ValueError
    return {"states": states, "actions": actions, "rewards": rewards}


def read_transition(path, processor, metadata, signature):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["collection_config_sha256"] != signature:
        raise ValueError
    plan, minute = _read_run(manifest, *(path / manifest["files"][key]
                                       for key in ("action_plan", "minute_metrics")))
    periods = aggregate_decision_metrics(plan, minute, manifest)
    return encode_periods(periods, processor, metadata)


def initialize_worker(root, config, metadata):
    global _WORKER
    _WORKER = Path(root), config, metadata, SequencePreprocessor.from_state_dict(metadata["preprocessor"])


def build_chunk(jobs):
    root, config, metadata, processor = _WORKER
    signature = config_signature(config)
    return [(index, split, position,
             read_transition(run_path(root, config, index), processor, metadata, signature))
            for index, split, position in jobs]


def prepare(collection_root, reference_replay=DEFAULT_REPLAY, output=DEFAULT_OUTPUT,
            workers=8, chunk_runs=24, resume=False):
    root, reference, output = map(Path, (collection_root, reference_replay, output))
    if workers < 1 or chunk_runs < 1 or (output.exists() and not resume):
        raise ValueError
    began = perf_counter()
    config = json.loads((root / "collection_config.json").read_text())
    original = json.loads((reference / "manifest.json").read_text())
    if sha256(root / "collection_config.json") != original["collection_config_sha256"]:
        raise ValueError
    output.mkdir(parents=True, exist_ok=resume)
    runs, jobs, split_order = {s: [] for s in ("train", "validation", "test")}, [], planned_splits(config)
    signature = config_signature(config)
    for index, split in enumerate(split_order):
        path = run_path(root, config, index)
        manifest = json.loads((path / "manifest.json").read_text())
        if (manifest["run_index"] != index or manifest["planned_split"] != split
                or manifest["simulator_seed"] != config["seed_start"] + index
                or manifest["collection_config_sha256"] != signature
                or manifest["run_id"] != path.name
                or any(not (path / name).is_file() for name in manifest["files"].values())):
            raise ValueError
        jobs.append((index, str(split), len(runs[split])))
        runs[split].append(path.name)
        if (index + 1) % 10000 == 0:
            print(f"inventory {index+1:,}/{config['run_count']:,}", flush=True)
    metadata = deepcopy(original)
    for name in ("source_manifest_sha256", "source_cache", "source_cache_manifest_sha256"):
        if name in metadata:
            metadata[f"reference_{name}"] = metadata.pop(name)
    metadata.update(status="building", created_at=datetime.now(timezone.utc).isoformat(),
        source_dataset=str(root.resolve()), source_raw_run_count=config["run_count"],
        reference_replay_root=str(reference.resolve()), reference_replay_manifest_sha256=sha256(reference / "manifest.json"),
        build_method="raw collection plus unchanged 47k replay reuse; no supervised CSV intermediate",
        normalization="frozen original 47k training-only scalers; reference train runs remain in full train split",
        normalization_fit_training_runs=original["splits"]["train"]["runs"],
        builder_sha256=sha256(Path(__file__)), splits={})
    arrays, reused = {}, set()
    for split, run_ids in runs.items():
        old_ids = json.loads((reference / split / "run_ids.json").read_text())
        if run_ids[:len(old_ids)] != old_ids:
            raise ValueError
        reused.update(old_ids)
        directory = output / split
        directory.mkdir(exist_ok=resume)
        save_json(directory / "run_ids.json", run_ids)
        count = len(run_ids) * original["periods_per_run"]
        metadata["splits"][split] = {"runs": len(run_ids), "transitions": count, "terminal_transitions": len(run_ids)}
        arrays[split] = {}
        for name, shape, dtype in (("states", (count, original["state_size"]), "float32"),
                                    ("actions", (count, original["grid_count"]), "int16"),
                                    ("rewards", (count,), "float32")):
            path = directory / f"{name}.npy"
            arrays[split][name] = (np.load(path, mmap_mode="r+") if resume and path.exists() else
                np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape))
            if arrays[split][name].shape != shape or arrays[split][name].dtype != np.dtype(dtype):
                raise ValueError
    manifest_path, done_path = output / "manifest.json", output / "completed_runs.npy"
    if resume:
        previous = json.loads(manifest_path.read_text())
        if any(previous[key] != metadata[key] for key in (
                "builder_sha256", "reference_replay_manifest_sha256", "collection_config_sha256", "splits")):
            raise ValueError
        if previous["status"] == "complete":
            return previous
    save_json(manifest_path, metadata)
    done = (np.load(done_path, mmap_mode="r+") if resume and done_path.exists() else
            np.lib.format.open_memmap(done_path, mode="w+", dtype="uint8", shape=(config["run_count"],)))
    periods = original["periods_per_run"]

    def progress():
        save_json(output / "build_progress.json", {"status": metadata["status"],
            "completed_runs": int(done.sum()), "target_runs": config["run_count"],
            "elapsed_seconds": perf_counter() - began, "workers": workers,
            "updated_at": datetime.now(timezone.utc).isoformat()})

    try:
        for split in runs:
            old = Replay(reference, split)
            split_jobs = [job for job in jobs if job[1] == split and runs[split][job[2]] in reused]
            for first in range(0, len(split_jobs), 512):
                group = split_jobs[first:first + 512]
                if all(done[item[0]] for item in group):
                    continue
                a, b = group[0][2] * periods, (group[-1][2] + 1) * periods
                for name in arrays[split]:
                    arrays[split][name][a:b] = old.arrays[name][a:b]
                    arrays[split][name].flush()
                done[[item[0] for item in group]] = 1
                done.flush()
            print(f"reused {split}: {len(split_jobs):,} runs", flush=True)
            progress()
        missing = [job for job in jobs if not done[job[0]]]
        chunks = [missing[i:i + chunk_runs] for i in range(0, len(missing), chunk_runs)]
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                                 initargs=(str(root), config, metadata)) as pool:
            for batch in pool.map(build_chunk, chunks, buffersize=workers * 2):
                for index, split, position, values in batch:
                    a, b = position * periods, (position + 1) * periods
                    for name, value in values.items():
                        arrays[split][name][a:b] = value
                for split in {item[1] for item in batch}:
                    for value in arrays[split].values():
                        value.flush()
                done[[item[0] for item in batch]] = 1
                done.flush()
                progress()
                if int(done.sum()) % (chunk_runs * 20) < chunk_runs:
                    print(f"built {int(done.sum()):,}/{config['run_count']:,} runs", flush=True)
        if not np.all(done == 1):
            raise ValueError
        hashes = {}
        for split, values in arrays.items():
            for start in range(0, len(values["states"]), 8192):
                if (not np.isfinite(values["states"][start:start+8192]).all()
                        or not np.isfinite(values["rewards"][start:start+8192]).all()
                        or (values["rewards"][start:start+8192] < 0).any()
                        or (values["actions"][start:start+8192] < 0).any()
                        or (values["actions"][start:start+8192] >= len(metadata["action_pairs"])).any()):
                    raise ValueError
            for name in values:
                hashes[f"{split}/{name}.npy"] = sha256(output / split / f"{name}.npy")
        metadata.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                        array_sha256=hashes, reused_reference_runs=len(reused),
                        fresh_raw_runs=config["run_count"] - len(reused))
        save_json(manifest_path, metadata)
        progress()
        return metadata
    except BaseException:
        metadata["status"] = "interrupted"
        save_json(manifest_path, metadata)
        progress()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", default="trajectories_grid6_low_mixed_20261027")
    parser.add_argument("--reference-replay", default=DEFAULT_REPLAY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-runs", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    prepare(**vars(parser.parse_args()))
