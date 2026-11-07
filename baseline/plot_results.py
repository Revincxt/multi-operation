"""绘制已完成训练的损失曲线，并导出六指标测试结果。

使用方式:
    python3 -m baseline.plot_results --input models/lstm_wesm_1000
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRICS = {
    "baseline_request_num": ("Bubble requests", "冒泡请求数量"),
    "request_rate": ("Request rate", "订单呼出率"),
    "driver_accept_rate": ("Driver acceptance rate", "司机接单率"),
    "order_completion_rate": ("Order completion rate", "订单完成率"),
    "mean_wait_to_match_seconds": ("Mean waiting time (s)", "乘客平均等待时间（秒）"),
    "platform_revenue": ("Platform revenue", "平台收益"),
}


def plot_results(input_dir, output_dir=None):
    """只读取已有结果，不重新训练或修改模型选择。"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    history = pd.read_csv(input_dir / "training_history.csv")
    with (input_dir / "metrics.json").open(encoding="utf-8") as file:
        report = json.load(file)
    # 旧的 LSTM 结果没有 model_name，仍按原格式读取。
    model_name = report.get("model_name", "lstm")
    labels = {"lstm": "LSTM", "timesnet": "TimesNet"}
    if model_name not in labels:
        raise ValueError
    best_epoch = report["best_epoch"]
    if (history.empty or len(history) != report["epochs_run"]
            or history["epoch"].tolist() != list(range(1, len(history) + 1))
            or best_epoch not in history["epoch"].values):
        raise ValueError
    output_dir.mkdir(parents=True, exist_ok=True)

    def draw_curve(ax, frame, train_column, validation_column, title):
        ax.plot(frame["epoch"], frame[train_column], color="#2563eb",
                linewidth=1.1, label="Train MSE")
        ax.plot(frame["epoch"], frame[validation_column], color="#ea580c",
                linewidth=1.1, label="Validation MSE")
        if frame["epoch"].min() <= best_epoch <= frame["epoch"].max():
            ax.axvline(best_epoch, color="#64748b", linestyle="--", linewidth=1,
                       label=f"Selected epoch: {best_epoch}")
        ax.set(title=title, xlabel="Epoch", ylabel="Standardized MSE ↓ (lower is better)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 160}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
        draw_curve(axes[0], history, "train_mse", "validation_mse", "Full training")
        early = history.head(100)
        draw_curve(axes[1], early, "train_mse", "validation_mse",
                   f"First {len(early)} epochs")
        fig.suptitle(f"{labels[model_name]} | {report['loss_weighting'].upper()} | {len(history)} epochs")
        fig.savefig(output_dir / "training_curves.png")
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        for ax, (metric, (english, _)) in zip(axes.flat, METRICS.items()):
            draw_curve(ax, history, f"train_mse_{metric}",
                       f"validation_mse_{metric}", english)
        fig.suptitle("Six prediction tasks | Train and validation MSE")
        fig.savefig(output_dir / "task_curves.png")
        plt.close(fig)

    rows = []
    for metric, (_, chinese) in METRICS.items():
        key = f"target_{metric}"
        model = report["splits"]["test"][model_name][key]
        persistence = report["splits"]["test"]["persistence"][key]
        row = {"metric": metric, "label": chinese}
        row.update({f"{model_name}_{name}": value for name, value in model.items()})
        row.update({f"persistence_{name}": value for name, value in persistence.items()})
        row["mae_improvement_percent"] = (
            100 * (1 - model["mae"] / persistence["mae"])
            if persistence["mae"] > 0 else None)
        rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "test_results.csv", index=False, encoding="utf-8-sig")
    print(f"训练曲线与测试结果已保存至：{output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="绘制时序基线训练曲线与导出测试结果")
    parser.add_argument("--input", dest="input_dir", required=True)
    parser.add_argument("--output", dest="output_dir")
    plot_results(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
