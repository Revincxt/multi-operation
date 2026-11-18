"""DL-MRD六指标适配版：Min-Max、三层Transformer和WESM完整训练。"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from baseline.data import DEFAULT_INPUT
from baseline.dlmrd.data import load_dataset
from baseline.dlmrd.model import GridDLMRD, DLMRDPredictor
from baseline.losses import WESMLoss
from baseline.sequence import TARGET_COLUMNS
from baseline.training import _make_loader, RegressionAccumulator
from offline_rl.data import save_json, sha256
from online_rl.train import atomic_save, cpu_weights


STOP = False
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = (Path(DEFAULT_INPUT) / "sequence_cache" /
                 "8041947369ab73f93ab7a9e3e42a9f64a1672b4326f3aa3edde3b223cadd4814")


def stop(*_):
    global STOP
    STOP = True


def save_csv(rows, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def source_hashes():
    paths = list(Path("baseline/dlmrd").glob("*.py"))
    paths += [Path("baseline") / name for name in ("data.py", "sequence.py", "training.py", "losses.py")]
    return {str(path): sha256(path) for path in sorted(paths)}


def epoch(model, loader, device, optimizer=None, criterion=None, callback=None):
    """训练更新WESM，验证只计算固定等权误差，每轮保留全部样本。"""
    model.train(optimizer is not None)
    criterion.train(optimizer is not None)
    sums = np.zeros((2, len(TARGET_COLUMNS)), dtype=float)
    count, weighted = 0, 0.
    begin = perf_counter()
    with torch.set_grad_enabled(optimizer is not None):
        for batch_index, batch in enumerate(loader, 1):
            inputs = [(torch.from_numpy(value) if isinstance(value, np.ndarray) else value).to(device) for value in batch]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            predicted = model(*inputs[:3])
            losses = (predicted - inputs[3]).square().mean(0)
            if optimizer is not None:
                loss = criterion(losses.to("cpu"))
                weights = criterion.weights.detach().numpy().copy()
            else:
                loss = losses.mean()
                weights = np.full(6, 1 / 6)
            if not torch.isfinite(loss):
                raise FloatingPointError
            if optimizer is not None:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError
                optimizer.step()
            size = len(inputs[3])
            count += size
            weighted += float(loss.detach().cpu()) * size
            sums[0] += losses.detach().cpu().numpy() * size
            sums[1] += weights * size
            if callback and (batch_index % 100 == 0 or count == len(loader.dataset)):
                callback(count, perf_counter() - begin)
    if count != len(loader.dataset):
        raise ValueError
    return {"mse": float(sums[0].mean() / count), "weighted_loss": weighted / count,
            "task_mse": (sums[0] / count).tolist(), "mean_weights": (sums[1] / count).tolist(),
            "seconds": perf_counter() - begin}


def evaluate(model, dataset, preprocessor, device, batch_size):
    """全量评估原单位六指标，不收集全部预测到内存，也不更新WESM。"""
    metrics, persistence = RegressionAccumulator(), RegressionAccumulator()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            batch = dataset[slice(start, end)]
            outputs = model(*(value.to(device) for value in batch[:3])).cpu().numpy()
            predictions = preprocessor.inverse_targets(outputs)
            actual = dataset.arrays["actual"][start:end]
            metrics.update(actual, predictions)
            persistence.update(actual, dataset.arrays["persistence"][start:end])
    return {"dlmrd": metrics.result(), "persistence": persistence.result()}


def run(args):
    global STOP
    STOP = False
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".train.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if min(args.epochs, args.batch_size, args.threads) <= 0 or args.workers < 0:
        raise ValueError
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    torch.set_num_threads(args.threads)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    save_json(root / "progress.json", {"status": "preparing", "pid": os.getpid()})
    splits, processor, manifest = load_dataset(args.input, args.prepared_cache, root)
    if args.smoke:
        for dataset in splits.values():
            dataset.sample_count = min(len(dataset), 4096)
    model_config = dict(history_size=len(processor.history_features), history_steps=processor.history_steps,
        context_size=len(processor.context_columns), grid_count=len(processor.grid_ids),
        hidden_size=64, num_heads=8, num_layers=3, d_ff=128)
    config = {"schema_version": "dlmrd_training_v1", "model_name": "dlmrd", "model": model_config,
        "input_dir": str(Path(args.input).resolve()), "cache_dir": str(Path(args.prepared_cache).resolve()),
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": .001,
        "seed": args.seed, "device": args.device, "threads": args.threads, "workers": args.workers,
        "optimizer": "adam", "weight_decay": 0., "loss_weighting": "wesm",
        "wesm": {"gamma": .1, "history_steps": 10, "update_interval": 10},
        "split_counts": {name: len(value) for name, value in splits.items()},
        "target_columns": list(TARGET_COLUMNS), "early_stopping": False,
        "initialization": "random", "smoke": args.smoke, "normalization": "train_minmax",
        "model_selection": "validation_equal_weight_minmax_mse",
        "dataset_manifest_sha256": sha256(Path(args.input) / "dataset_manifest.json"),
        "cache_manifest_sha256": sha256(Path(args.prepared_cache) / "cache_manifest.json"),
        "normalization_sha256": sha256(root / "normalization.json"), "source_sha256": source_hashes(),
        "paper": "10.1016/j.tre.2024.103822",
        "upstream_reference_commit": "1e627ad895f92a06382d7ec01571aeeb5fa5f48c",
        "adaptation": "Six targets and joint price/radius context; 6 grids, four historical windows; retained run-level splits; independent implementation of paper equations 5-16 and WESM."}
    config_path = root / "training_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if config != old:
            raise ValueError("Training configuration differs from saved run")
    else:
        save_json(config_path, config)
    torch.manual_seed(args.seed)
    model = GridDLMRD(**model_config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    criterion = WESMLoss(6, **config["wesm"])
    history, best, best_epoch = [], float("inf"), 0
    latest = root / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["config_sha256"] != sha256(config_path):
            raise ValueError
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        criterion.load_state_dict(state["loss_state_dict"])
        history, best, best_epoch = state["history"], state["best_loss"], state["best_epoch"]
        torch.set_rng_state(state["torch_rng_state"])
    if history:
        save_csv(history, root / "training_history.csv")
    validation_loader = _make_loader(splits["validation"], args.batch_size, workers=args.workers)

    def progress(status, **extra):
        save_json(root / "progress.json", {"status": status, "pid": os.getpid(),
            "epochs_completed": len(history), "epochs_target": args.epochs,
            "best_epoch": best_epoch, "best_validation_mse": best if np.isfinite(best) else None, **extra})

    try:
        for number in range(len(history) + 1, args.epochs + 1):
            if STOP:
                break
            loader = _make_loader(splits["train"], args.batch_size, shuffle=True, seed=args.seed + number,
                                  workers=args.workers)
            train_stats = epoch(model, loader, args.device, optimizer, criterion,
                callback=lambda count, seconds: progress("training", epoch=number,
                    samples_in_epoch=count, samples_per_epoch=len(splits["train"]), seconds_in_epoch=seconds))
            del loader
            progress("validating", epoch=number)
            valid_stats = epoch(model, validation_loader, args.device, criterion=criterion)
            row = {"epoch": number, "train_mse": train_stats["mse"],
                   "train_weighted_loss": train_stats["weighted_loss"], "validation_mse": valid_stats["mse"],
                   "train_seconds": train_stats["seconds"], "validation_seconds": valid_stats["seconds"]}
            for i, target in enumerate(TARGET_COLUMNS):
                key = target.removeprefix("target_")
                row["train_mse_" + key] = train_stats["task_mse"][i]
                row["validation_mse_" + key] = valid_stats["task_mse"][i]
                row["mean_weight_" + key] = train_stats["mean_weights"][i]
            history.append(row)
            if row["validation_mse"] < best:
                best, best_epoch = row["validation_mse"], number
                atomic_save(root / "best_model.pt", {"schema_version": DLMRDPredictor.schema_version,
                    "model_config": model_config, "model_state_dict": cpu_weights(model),
                    "preprocessor": processor.state_dict(), "best_epoch": best_epoch,
                    "validation_mse": best, "loss_weighting": "wesm", "wesm_config": config["wesm"],
                    "loss_state_dict": cpu_weights(criterion)})
            atomic_save(latest, {"config_sha256": sha256(config_path), "model_state_dict": cpu_weights(model),
                "optimizer_state_dict": optimizer.state_dict(), "loss_state_dict": cpu_weights(criterion),
                "history": history, "best_loss": best, "best_epoch": best_epoch,
                "torch_rng_state": torch.get_rng_state()})
            save_csv(history, root / "training_history.csv")
            print(f"epoch={number}/{args.epochs} train_mse={row['train_mse']:.8f} "
                  f"validation_mse={row['validation_mse']:.8f} best_epoch={best_epoch} "
                  f"train_seconds={row['train_seconds']:.1f} validation_seconds={row['validation_seconds']:.1f}", flush=True)
            from baseline.dlmrd.report import run as report
            report(root)
            progress("paused" if STOP else "training", epoch=number)
        if STOP:
            progress("paused")
            return
        progress("testing")
        predictor = DLMRDPredictor.load(root / "best_model.pt", args.device)
        metrics = {"model_name": "dlmrd", "best_epoch": best_epoch, "epochs_run": len(history),
            "best_validation_minmax_mse": best, "loss_weighting": "wesm", "splits": {}}
        for name in ("validation", "test"):
            metrics["splits"][name] = evaluate(predictor.model, splits[name], processor, args.device, args.batch_size)
        save_json(root / "metrics.json", metrics)
        from baseline.dlmrd.report import run as report
        report(root)
        progress("complete")
    except BaseException as error:
        progress("failed", error=type(error).__name__)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--prepared-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default="models/dlmrd_grid6_low47k_mps1000_20261117")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
