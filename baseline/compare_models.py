"""比较当前最佳 LSTM / TimesNet 与均值、上一周期和 Ridge 基线。

训练与参数选择只使用 train / validation；测试使用完整 test 划分。
运行中的模型先保存快照，不改写原训练目录，也不暂停原训练任务。

使用方式:
    python3 -m baseline.compare_models --input DATASET --lstm-dir LSTM_DIR \
        --timesnet-dir TIMESNET_DIR --output models/comparison_grid6_low
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import torch

from baseline.data import KEY_COLUMNS, load_dataset
from baseline.lstm import LSTMPredictor
from baseline.sequence import TARGET_COLUMNS
from baseline.timesnet import TimesNetPredictor
from baseline.training import RegressionAccumulator


MODELS = ("train_mean", "persistence", "ridge", "lstm", "timesnet")
LABELS = ("Train mean", "Previous period", "Ridge", "LSTM", "TimesNet")
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def linear_features(dataset, start, end, grid_count):
    """展平相同历史与当前动作，网格用 one-hot，最后一列为截距。"""
    history = dataset.arrays["history"][start:end]
    context = dataset.arrays["context"][start:end]
    grids = dataset.arrays["grid_indices"][start:end]
    return np.column_stack((history.reshape(len(history), -1), context,
                            np.eye(grid_count)[grids], np.ones(len(history))))


def _linear_statistics(dataset, grid_count, chunk_size):
    """分块累计最小二乘统计量，不加载百万行特征矩阵。"""
    xtx = xty = None
    yty = 0.0
    actual_sum = np.zeros(len(TARGET_COLUMNS))
    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        x = linear_features(dataset, start, end, grid_count)
        y = np.asarray(dataset.arrays["targets"][start:end], dtype=float)
        if xtx is None:
            xtx = np.zeros((x.shape[1], x.shape[1]))
            xty = np.zeros((x.shape[1], y.shape[1]))
        xtx += x.T @ x
        xty += x.T @ y
        yty += float(np.square(y).sum())
        actual_sum += dataset.arrays["actual"][start:end].sum(axis=0)
    return xtx, xty, yty, actual_sum / len(dataset)


def fit_ridge(train, validation, grid_count, chunk_size=10000, alphas=ALPHAS):
    """训练集拟合，验证集选择一个共享六输出的正则化强度。"""
    if (chunk_size <= 0 or not alphas
            or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas)):
        raise ValueError
    xtx, xty, _, train_mean = _linear_statistics(train, grid_count, chunk_size)
    vx, vy, yy, _ = _linear_statistics(validation, grid_count, chunk_size)
    penalty = np.eye(len(xtx))
    penalty[-1, -1] = 0  # 不惩罚截距；其余特征均使用训练集预处理结果。
    best_score, best_coef, best_alpha = float("inf"), None, None
    scores = []
    for alpha in alphas:
        coef = np.linalg.solve(xtx + alpha * penalty, xty)
        squared_error = yy - 2 * np.sum(coef * vy) + np.sum(coef * (vx @ coef))
        score = max(float(squared_error), 0.0) / (len(validation) * len(TARGET_COLUMNS))
        if not np.isfinite(score) or not np.isfinite(coef).all():
            raise ValueError
        scores.append({"alpha": float(alpha), "validation_standardized_mse": score})
        if score < best_score:
            best_score, best_coef, best_alpha = score, coef, float(alpha)
    return best_coef, train_mean, {"selected_alpha": best_alpha, "candidates": scores,
                                  "selection": "validation_equal_weight_standardized_mse"}


def _snapshot(model_name, source_dir, output_dir, dataset, preprocessor, device):
    """读取完整 checkpoint 后保存固定快照，规避原训练继续更新最佳权重。"""
    source_dir = Path(source_dir)
    config = json.loads((source_dir / "training_config.json").read_text(encoding="utf-8"))
    if (config["source_sha256"] != dataset.metadata["source_sha256"]
            or config["split_run_ids"] != dataset.metadata["split_run_ids"]):
        raise ValueError
    for attempt in range(3):
        try:
            raw = (source_dir / "best_model.pt").read_bytes()
            checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            break
        except (EOFError, RuntimeError, pickle.UnpicklingError):
            if attempt == 2:
                raise
            time.sleep(0.1)
    if (checkpoint["preprocessor"] != preprocessor.state_dict()
            or checkpoint["model_config"] != config["model"]):
        raise ValueError
    snapshot_path = output_dir / f"{model_name}_best_snapshot.pt"
    snapshot_path.write_bytes(raw)
    predictor_class = {"lstm": LSTMPredictor, "timesnet": TimesNetPredictor}[model_name]
    predictor = predictor_class.load(snapshot_path, device)
    return predictor, {
        "source_dir": str(source_dir.resolve()),
        "snapshot_file": snapshot_path.name,
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "best_epoch": checkpoint["best_epoch"],
        "validation_standardized_mse": checkpoint["validation_mse"],
        "training_completed_at_snapshot": (source_dir / "metrics.json").exists(),
        "optimizer": checkpoint.get("optimizer", "adam"),
        "weight_decay": checkpoint.get("weight_decay", 0.0),
        "evaluation_device": str(device),
    }


def evaluate_models(dataset, preprocessor, predictors, coef, train_mean, output_path, batch_size):
    """同一批测试样本同时评估五种方法，预测和指标逐批写出。"""
    if batch_size <= 0 or set(predictors) != {"lstm", "timesnet"}:
        raise ValueError
    accumulators = {name: RegressionAccumulator() for name in MODELS}
    squared_errors = {name: np.zeros(len(TARGET_COLUMNS)) for name in MODELS}
    scale = preprocessor.scalers["targets"]["scale"]
    offset = 0
    with pd.read_csv(dataset.keys_path, chunksize=batch_size, dtype={"run_id": str}) as reader, \
            gzip.open(output_path, "wt", encoding="utf-8", newline="") as file, \
            torch.inference_mode():
        for keys in reader:
            if not keys["split"].eq("test").all():
                raise ValueError
            end = offset + len(keys)
            actual = dataset.arrays["actual"][offset:end]
            predictions = {
                "train_mean": np.broadcast_to(train_mean, actual.shape),
                "persistence": dataset.arrays["persistence"][offset:end],
                "ridge": preprocessor.inverse_targets(
                    linear_features(dataset, offset, end, len(preprocessor.grid_ids)) @ coef),
            }
            history, context, grids, _ = dataset[slice(offset, end)]
            for name, predictor in predictors.items():
                outputs = predictor.model(*(values.to(predictor.device)
                                            for values in (history, context, grids)))
                predictions[name] = preprocessor.inverse_targets(outputs.cpu().numpy())
            columns = {key: keys[key].to_numpy() for key in KEY_COLUMNS}
            columns.update({target: actual[:, index] for index, target in enumerate(TARGET_COLUMNS)})
            for name, values in predictions.items():
                accumulators[name].update(actual, values)
                squared_errors[name] += np.square((np.asarray(values, dtype=float) - actual) / scale).sum(0)
                columns.update({f"{name}_{target.removeprefix('target_')}": values[:, index]
                                for index, target in enumerate(TARGET_COLUMNS)})
            pd.DataFrame(columns).to_csv(file, index=False, header=offset == 0, float_format="%.17g")
            offset = end
            if offset % (batch_size * 100) == 0:
                print(f"已评估 {offset:,}/{len(dataset):,} 条测试样本", flush=True)
    if offset != len(dataset):
        raise ValueError
    rows, summary = [], []
    persistence = accumulators["persistence"].result()
    for name in MODELS:
        metrics = accumulators[name].result()
        normalized = squared_errors[name] / offset
        for index, target in enumerate(TARGET_COLUMNS):
            reference = persistence[target]["mae"]
            rows.append({"model": name, "metric": target.removeprefix("target_"),
                         **metrics[target], "standardized_mse": float(normalized[index]),
                         "mae_improvement_vs_persistence_percent": (
                             100 * (1 - metrics[target]["mae"] / reference) if reference > 0 else None)})
        summary.append({"model": name, "test_rows": offset,
                        "test_standardized_mse": float(normalized.mean())})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def plot_comparison(metrics, output_path):
    """每个指标独立量纲，避免把不同单位的 MAE 直接相加。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = ("Bubble requests", "Request rate", "Driver acceptance rate",
              "Order completion rate", "Mean waiting time (s)", "Platform revenue")
    colors = ("#94a3b8", "#f59e0b", "#10b981", "#2563eb", "#8b5cf6")
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        for ax, target, title in zip(axes.flat, TARGET_COLUMNS, titles):
            frame = metrics[metrics.metric == target.removeprefix("target_")].set_index("model")
            ax.bar(LABELS, frame.loc[list(MODELS), "mae"], color=colors)
            ax.set(title=title, ylabel="Test MAE ↓ (lower is better)")
            ax.tick_params(axis="x", labelrotation=25, labelsize=8)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        fig.suptitle("Same held-out test set | Validation-selected model snapshots")
        fig.savefig(output_path, dpi=160)
        plt.close(fig)


def compare_models(input_dir, lstm_dir, timesnet_dir, output_dir=None,
                   batch_size=512, threads=2, device="cpu", make_plots=True):
    """一次性阶段对比；测试结果不写回正在训练的模型目录。"""
    if min(batch_size, threads) <= 0:
        raise ValueError
    output_dir = (Path(output_dir) if output_dir is not None else
                  Path("models") / datetime.now().strftime("comparison_grid6_low_%Y%m%d_%H%M%S"))
    if output_dir.exists():
        raise FileExistsError
    # 先检查显式选择的两个模型，避免缺少权重时还去构建大规模数据缓存。
    for directory in (lstm_dir, timesnet_dir):
        for name in ("training_config.json", "best_model.pt"):
            if not (Path(directory) / name).is_file():
                raise FileNotFoundError
    torch.set_num_threads(threads)
    splits, preprocessor, manifest = load_dataset(input_dir)
    output_dir.mkdir(parents=True)
    predictors, snapshots = {}, {}
    for name, directory in (("lstm", lstm_dir), ("timesnet", timesnet_dir)):
        predictors[name], snapshots[name] = _snapshot(
            name, directory, output_dir, splits["train"], preprocessor, device)
        print(f"{name} 快照：验证集最佳第 {snapshots[name]['best_epoch']} 轮", flush=True)
    print("拟合均值与 Ridge 基线；只使用训练集和验证集……", flush=True)
    coef, train_mean, ridge = fit_ridge(splits["train"], splits["validation"], len(preprocessor.grid_ids))
    np.savez(output_dir / "ridge_model.npz", coefficients=coef, train_mean=train_mean,
             alpha=ridge["selected_alpha"])
    print(f"Ridge 选定 alpha={ridge['selected_alpha']:g}，开始完整测试集评估……", flush=True)
    metrics, summary = evaluate_models(
        splits["test"], preprocessor, predictors, coef, train_mean,
        output_dir / "test_predictions.csv.gz", batch_size)
    metrics.to_csv(output_dir / "comparison_metrics.csv", index=False)
    summary = summary.sort_values("test_standardized_mse")
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(Path(input_dir).resolve()),
        "source_sha256": splits["train"].metadata["source_sha256"],
        "source_run_count": manifest.get("source_run_count"),
        "split_counts": {name: len(dataset) for name, dataset in splits.items()},
        "split_run_ids": splits["train"].metadata["split_run_ids"],
        "snapshots": snapshots, "ridge": ridge,
        "preprocessor": preprocessor.state_dict(),
        "ridge_feature_order": "flatten oldest-to-newest history, context, grid one-hot, intercept",
        "test_selection": False, "prediction_clipping": False,
        "overall_score": "mean across six task MSEs normalized by training target variance",
        "summary": summary.to_dict(orient="records"),
        "limitations": ["当前单日期、低密度、同一时段的独立 run 测试，不代表跨日期或跨密度泛化。",
                        "模型快照只按验证集选择；本次测试结果不反馈训练或用于选择后续 checkpoint。"],
    }
    if make_plots:
        plot_comparison(metrics, output_dir / "comparison_mae.png")
    (output_dir / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"对比结果已保存至：{output_dir}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="比较当前最佳时序模型与简单监督学习基线")
    parser.add_argument("--input", dest="input_dir", required=True)
    parser.add_argument("--lstm-dir", required=True)
    parser.add_argument("--timesnet-dir", required=True)
    parser.add_argument("--output", dest="output_dir")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--no-plots", dest="make_plots", action="store_false")
    compare_models(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
