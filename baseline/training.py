"""LSTM / TimesNet 共用的数据读取、训练和六指标评估。

核心逻辑:
1. 读取历史序列、当前联合动作和已有 train/validation/test 划分
2. 用 WESM 平衡六个任务，按等权验证集误差保存最佳参数
3. 在原始量纲下评估六个指标，与上一周期预测基线比较

两个训练入口共用同一流程，保持数据划分、标准化和模型选择口径一致。
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler, SequentialSampler

from baseline.data import DEFAULT_CHUNK_SIZE, DEFAULT_INPUT, KEY_COLUMNS, load_dataset, load_prepared_dataset
from baseline.losses import WESMLoss
from baseline.lstm import GridLSTM, LSTMPredictor
from baseline.timesnet import GridTimesNet, TimesNetPredictor
from baseline.sequence import (
    TARGET_COLUMNS,
    RunningMoments,
)


class _BlockShuffleSampler(Sampler):
    """随机块顺序、随机块内顺序；每轮覆盖全部样本，限制随机读的工作集。"""

    def __init__(self, dataset, block_size, generator):
        self.dataset = dataset
        self.block_size = block_size
        self.generator = generator

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        count = len(self.dataset)
        blocks = (count + self.block_size - 1) // self.block_size
        order = (torch.randperm(blocks, generator=self.generator).tolist()
                 if blocks > 1 else range(blocks))
        for block in order:
            start = block * self.block_size
            yield from (torch.randperm(min(self.block_size, count - start),
                                      generator=self.generator) + start).tolist()


def _make_loader(dataset, batch_size, shuffle=False, seed=42, workers=0,
                 shuffle_block_size=65536):
    """按块打乱限制随机读范围；worker 仅预取，每轮保留全部样本。"""
    if batch_size <= 0 or workers < 0 or shuffle_block_size < 0:
        raise ValueError
    generator = torch.Generator().manual_seed(seed)
    if shuffle:
        sampler = (_BlockShuffleSampler(dataset, shuffle_block_size, generator)
                   if shuffle_block_size else RandomSampler(dataset, generator=generator))
    else:
        sampler = SequentialSampler(dataset)
    options = ({"num_workers": workers, "persistent_workers": True,
                "prefetch_factor": 2} if workers else {})
    return DataLoader(
        dataset, batch_size=None,
        sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False),
        # worker 的随机种子单独管理，worker 数量不改变各轮的样本顺序。
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_numpy_batch if workers else None,
        **options)


def _numpy_batch(batch):
    """用普通 NumPy 缓冲区跨进程传 batch，避免依赖受限的 torch_shm_manager。"""
    return tuple(value.numpy() for value in batch)


def _run_epoch(model, loader, device, optimizer=None, criterion=None):
    """训练可动态加权；验证始终等权且不更新 WESM 历史。"""
    model.train(optimizer is not None)
    if criterion is not None:
        criterion.train(optimizer is not None)
    total_loss = 0.0
    total_mse = 0.0
    task_totals = np.zeros(len(TARGET_COLUMNS))
    weight_totals = np.zeros(len(TARGET_COLUMNS))
    data_wait_seconds = 0.0
    compute_seconds = 0.0
    batches = iter(loader)
    with torch.set_grad_enabled(optimizer is not None):
        while True:
            wait_started = time.perf_counter()
            try:
                history, context, grid_indices, targets = next(batches)
            except StopIteration:
                break
            data_wait_seconds += time.perf_counter() - wait_started
            compute_started = time.perf_counter()
            history, context, grid_indices, targets = (
                (torch.from_numpy(value) if isinstance(value, np.ndarray) else value).to(device)
                for value in (history, context, grid_indices, targets))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(history, context, grid_indices)
            task_losses = (predictions - targets).square().mean(dim=0)
            mse = nn.functional.mse_loss(predictions, targets)
            if optimizer is not None and criterion is not None:
                loss = criterion(task_losses.to(criterion.weights.device))
                weights = criterion.weights.detach().cpu().numpy()
            else:
                loss = mse
                weights = np.full(len(TARGET_COLUMNS), 1.0 / len(TARGET_COLUMNS))
            if not torch.isfinite(loss):
                raise FloatingPointError
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * len(targets)
            total_mse += mse.item() * len(targets)
            task_totals += task_losses.detach().cpu().numpy() * len(targets)
            weight_totals += weights * len(targets)
            compute_seconds += time.perf_counter() - compute_started
    count = len(loader.dataset)
    return {"mse": total_mse / count, "weighted_loss": total_loss / count,
            "task_mse": (task_totals / count).tolist(),
            "mean_weights": (weight_totals / count).tolist(),
            "data_wait_seconds": data_wait_seconds,
            "compute_seconds": compute_seconds}


def regression_metrics(actual, predicted):
    """分别计算各指标误差；常量标签的 R² 记为 null。"""
    metrics = RegressionAccumulator()
    metrics.update(actual, predicted)
    return metrics.result()


class RegressionAccumulator:
    """逐批累计原量纲误差，不收集整份验证/测试预测。"""

    def __init__(self):
        self.actual = RunningMoments()
        self.absolute_error = np.zeros(len(TARGET_COLUMNS))
        self.squared_error = np.zeros(len(TARGET_COLUMNS))

    def update(self, actual, predicted):
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        if (actual.shape != predicted.shape or actual.ndim != 2 or not len(actual)
                or actual.shape[1] != len(TARGET_COLUMNS)
                or not np.isfinite(actual).all() or not np.isfinite(predicted).all()):
            raise ValueError
        error = predicted - actual
        self.actual.update(actual)
        self.absolute_error += np.abs(error).sum(axis=0)
        self.squared_error += (error ** 2).sum(axis=0)

    def result(self):
        if self.actual.count == 0:
            raise ValueError
        return {
            column: {
                "mae": float(self.absolute_error[index] / self.actual.count),
                "rmse": float(np.sqrt(self.squared_error[index] / self.actual.count)),
                "r2": (float(1 - self.squared_error[index] / self.actual.m2[index])
                       if self.actual.m2[index] > 0 else None),
            }
            for index, column in enumerate(TARGET_COLUMNS)
        }


def _evaluate_split(predictor, dataset, batch_size, output_path, model_name):
    """顺序读取磁盘数组，边评估边输出预测 CSV。"""
    model_metrics, persistence_metrics = RegressionAccumulator(), RegressionAccumulator()
    offset = 0
    with pd.read_csv(dataset.keys_path, chunksize=batch_size, dtype={"run_id": str}) as reader, \
            Path(output_path).open("w", encoding="utf-8", newline="") as file, \
            torch.inference_mode():
        for keys in reader:
            end = offset + len(keys)
            history, context, grids, _ = dataset[slice(offset, end)]
            outputs = predictor.model(*(value.to(predictor.device)
                                        for value in (history, context, grids)))
            predicted = predictor.preprocessor.inverse_targets(outputs.cpu().numpy())
            actual = dataset.arrays["actual"][offset:end]
            persistence = dataset.arrays["persistence"][offset:end]
            model_metrics.update(actual, predicted)
            persistence_metrics.update(actual, persistence)
            records = keys[list(KEY_COLUMNS)].copy()
            for index, column in enumerate(TARGET_COLUMNS):
                metric = column.removeprefix("target_")
                records[column] = actual[:, index]
                records[f"predicted_{metric}"] = predicted[:, index]
                records[f"persistence_{metric}"] = persistence[:, index]
            records.to_csv(file, index=False, header=offset == 0)
            offset = end
    if offset != len(dataset):
        raise ValueError
    return {model_name: model_metrics.result(), "persistence": persistence_metrics.result()}


def train_baseline(model_name, model_options, input_dir, output_dir,
                   epochs=50, batch_size=512, learning_rate=0.001,
                   seed=42, device="cpu", threads=4, workers=0, loss_weighting="wesm",
                   wesm_gamma=0.1, wesm_history_steps=10, wesm_update_interval=10,
                   chunk_size=DEFAULT_CHUNK_SIZE, prepare_only=False,
                   optimizer_name="adamw", weight_decay=1e-3, shuffle_block_size=65536,
                   prepared_cache=None):
    """训练指定轮数、保存最佳模型并评估；已有结果目录不覆盖。"""
    if (min(epochs, batch_size, threads, model_options["hidden_size"],
            model_options["embedding_size"], model_options["num_layers"]) <= 0
            or workers < 0 or shuffle_block_size < 0
            or not 0 <= model_options["dropout"] < 1 or not np.isfinite(learning_rate)
            or learning_rate <= 0 or seed < 0 or loss_weighting not in ("equal", "wesm")
            or optimizer_name not in ("adam", "adamw")
            or not np.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError
    models = {"lstm": (GridLSTM, LSTMPredictor), "timesnet": (GridTimesNet, TimesNetPredictor)}
    if model_name not in models:
        raise ValueError
    model_class, predictor_class = models[model_name]
    # 正则化版与原始对照分开保存；显式指定目录时仍不覆盖已有结果。
    output_dir = (Path(output_dir) if output_dir is not None
                  else Path(f"models/{model_name}_grid6_low_full_{optimizer_name}"))
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    device = torch.device(device)
    # MPS 只训练网络；六维损失的历史平滑及权重更新放到 CPU，保持原公式。
    loss_device = torch.device("cpu") if device.type == "mps" else device
    criterion = (WESMLoss(
        len(TARGET_COLUMNS), gamma=wesm_gamma, history_steps=wesm_history_steps,
        update_interval=wesm_update_interval).to(loss_device)
        if loss_weighting == "wesm" else None)
    splits, preprocessor, manifest = (
        load_prepared_dataset(input_dir, prepared_cache) if prepared_cache is not None
        else load_dataset(input_dir, chunk_size))
    model_config = {
        "context_size": len(preprocessor.context_columns),
        "history_size": len(preprocessor.history_features),
        "grid_count": len(preprocessor.grid_ids),
        **model_options,
    }
    if model_name == "timesnet":
        model_config["history_steps"] = preprocessor.history_steps
        if preprocessor.history_steps < 8:
            print("提示：当前历史窗口较短，TimesNet 的周期识别能力有限。", flush=True)
    model = model_class(**model_config).to(device)
    train_loader = _make_loader(splits["train"], batch_size, True, seed, workers,
                               shuffle_block_size)
    validation_loader = _make_loader(splits["validation"], batch_size,
                                     workers=workers)
    checkpoint_path = output_dir / "best_model.pt"
    config = {
        "model_name": model_name,
        "input_dir": str(Path(input_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "seed": seed, "device": str(device),
        "optimizer": optimizer_name, "weight_decay": weight_decay,
        "threads": threads, "data_loader_workers": workers, "model": model_config,
        "shuffle_block_size": shuffle_block_size,
        "dataset_schema_version": manifest["schema_version"],
        "history_steps": preprocessor.history_steps,
        "target_columns": list(TARGET_COLUMNS),
        "split_counts": {name: len(frame) for name, frame in splits.items()},
        "split_by": manifest.get("split_by", "run"),
        "split_dates": manifest.get("split_dates", {}),
        "split_run_ids": {name: dataset.run_ids for name, dataset in splits.items()},
        "torch_version": str(torch.__version__),
        "loss_weighting": loss_weighting,
        "wesm": ({"gamma": wesm_gamma, "history_steps": wesm_history_steps,
                  "update_interval": wesm_update_interval} if criterion is not None else None),
        "normalization": "train-only z-score (unchanged from equal-weight baseline)",
        "normalization_fit_rows": preprocessor.fit_row_count,
        "grid_ids": preprocessor.grid_ids,
        "data_loading": "read-only numpy memmap, batch indexing, no dropped samples",
        "preprocessing_chunk_size": chunk_size,
        "data_cache": str(splits["train"].cache_dir.resolve()),
        "data_validation": "prepared_cache_metadata_only" if prepared_cache is not None else "full_checksums",
        "loss_device": str(loss_device) if criterion is not None else str(device),
        "source_sha256": splits["train"].metadata["source_sha256"],
        "initialization": "from_scratch",
        "early_stopping": False,
        "model_selection": "validation_equal_weight_standardized_mse",
        "weight_history": "sample-weighted mean of applied batch weights within each epoch",
    }
    if prepare_only:
        # 仅检查两个模型的输入输出尺寸，不反向传播，也不创建训练结果目录。
        model.eval()
        with torch.inference_mode():
            batch = splits["train"][slice(0, min(batch_size, len(splits["train"])))]
            outputs = model(*(values.to(device) for values in batch[:3]))
        if outputs.shape != batch[3].shape or not torch.isfinite(outputs).all():
            raise ValueError
        config["forward_check"] = {"input_history_shape": list(batch[0].shape),
                                   "output_shape": list(outputs.shape), "passed": True}
        config_path = splits["train"].cache_dir / f"{model_name}_training_config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"训练配置检查通过，未开始训练：{config_path}", flush=True)
        return config
    # 只改变优化器及权重衰减，网络、学习率、WESM 与数据划分保持不变。
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer_name]
    optimizer = optimizer_class(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    best_loss = float("inf")
    best_epoch = 0
    best_weights = [1.0 / len(TARGET_COLUMNS)] * len(TARGET_COLUMNS)
    history = []
    print(f"训练样本：{len(splits['train']):,}，历史窗口：{preprocessor.history_steps}，"
          f"网格数：{len(preprocessor.grid_ids)}，预测指标：{len(TARGET_COLUMNS)}，"
          f"损失权重：{loss_weighting}，优化器：{optimizer_name}，"
          f"权重衰减：{weight_decay:g}", flush=True)
    for epoch in range(1, epochs + 1):
        train_stats = _run_epoch(model, train_loader, device, optimizer, criterion)
        validation_stats = _run_epoch(model, validation_loader, device)
        validation_loss = validation_stats["mse"]
        row = {"epoch": epoch, "train_mse": train_stats["mse"],
               "train_weighted_loss": train_stats["weighted_loss"], "validation_mse": validation_loss,
               "train_data_wait_seconds": train_stats.get("data_wait_seconds", 0.0),
               "train_compute_seconds": train_stats.get("compute_seconds", 0.0),
               "validation_data_wait_seconds": validation_stats.get("data_wait_seconds", 0.0),
               "validation_compute_seconds": validation_stats.get("compute_seconds", 0.0)}
        for index, target in enumerate(TARGET_COLUMNS):
            metric = target.removeprefix("target_")
            row[f"train_mse_{metric}"] = train_stats["task_mse"][index]
            row[f"validation_mse_{metric}"] = validation_stats["task_mse"][index]
            row[f"mean_weight_{metric}"] = train_stats["mean_weights"][index]
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        print(f"Epoch {epoch:03d} | train MSE={train_stats['mse']:.6f} | "
              f"weighted loss={train_stats['weighted_loss']:.6f} | "
              f"validation MSE={validation_loss:.6f} | "
              f"train wait/compute={train_stats.get('data_wait_seconds', 0.0):.1f}/"
              f"{train_stats.get('compute_seconds', 0.0):.1f}s", flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_weights = (criterion.weights.detach().cpu().tolist()
                            if criterion is not None else best_weights)
            torch.save({
                "schema_version": predictor_class.schema_version,
                "model_config": model_config,
                "model_state_dict": {key: value.detach().cpu()
                                     for key, value in model.state_dict().items()},
                "preprocessor": preprocessor.state_dict(),
                "best_epoch": best_epoch,
                "validation_mse": best_loss,
                "loss_weighting": loss_weighting,
                "optimizer": optimizer_name,
                "weight_decay": weight_decay,
                "wesm_config": config["wesm"],
                "loss_state_dict": ({key: value.detach().cpu()
                                     for key, value in criterion.state_dict().items()}
                                    if criterion is not None else None),
            }, checkpoint_path)

    # 测试集只在模型选择完成后评估，不用于模型选择或标准化。
    predictor = predictor_class.load(checkpoint_path, device)
    report = {"model_name": model_name, "best_epoch": best_epoch, "epochs_run": len(history),
              "best_validation_standardized_mse": best_loss,
              "loss_weighting": loss_weighting,
              "optimizer": optimizer_name, "weight_decay": weight_decay,
              "best_task_weights": dict(zip(TARGET_COLUMNS, best_weights)), "splits": {}}
    for name in ("validation", "test"):
        report["splits"][name] = _evaluate_split(
            predictor, splits[name], batch_size, output_dir / f"{name}_predictions.csv", model_name)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"最佳 epoch：{best_epoch}，结果保存至：{output_dir}", flush=True)
    return report


def training_parser(model_name):
    """共用训练参数，模型结构参数由各自入口补充。"""
    label = {"lstm": "LSTM", "timesnet": "TimesNet"}[model_name]
    parser = argparse.ArgumentParser(description=f"训练网格级多输出 {label} 基线")
    parser.add_argument("--input", dest="input_dir", default=DEFAULT_INPUT)
    parser.add_argument("--output", dest="output_dir",
                        help="默认按模型和优化器命名目录，不覆盖已有结果")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--prepare-only", action="store_true",
                        help="仅准备全量缓存、检查配置和前向计算，不训练")
    parser.add_argument("--prepared-cache", help="直接使用指定的已准备缓存，跳过全量数据校验")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--embedding-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1 if model_name == "lstm" else 2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--optimizer", dest="optimizer_name", choices=("adamw", "adam"),
                        default="adamw", help="默认 AdamW；原始对照使用 adam 和 weight-decay 0")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0,
                        help="并行预取数据的进程数；建议先测试 2 或 4")
    parser.add_argument("--shuffle-block-size", type=int, default=65536,
                        help="训练数据按块打乱，减少随机磁盘读取；0 表示全局随机打乱")
    parser.add_argument("--loss-weighting", choices=("wesm", "equal"), default="wesm")
    parser.add_argument("--wesm-gamma", type=float, default=0.1)
    parser.add_argument("--wesm-history-steps", type=int, default=10)
    parser.add_argument("--wesm-update-interval", type=int, default=10)
    return parser
