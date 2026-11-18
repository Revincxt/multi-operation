"""从DL-MRD训练CSV生成论文曲线，并在终测完成后导出六指标对比。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "baseline_request_num", "request_rate", "driver_accept_rate",
    "order_completion_rate", "mean_wait_to_match_seconds", "platform_revenue",
)
TITLES = ("Bubble requests", "Order request rate", "Driver acceptance rate",
          "Order completion rate", "Mean waiting time", "Platform revenue")
TASK_COLORS = ("#347798", "#D58B48", "#559A88", "#A66B87", "#8D82B5", "#9F9B4C")
TRAIN_COLOR, VALIDATION_COLOR = "#347798", "#D58B48"


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt, {
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
        "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
    }


def save_figure(fig, path):
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + suffix), dpi=600, facecolor="white")


def load_history(path):
    frame = pd.read_csv(path)
    required = ["epoch", "train_mse", "validation_mse"]
    required += [f"{split}_mse_{metric}" for metric in METRICS for split in ("train", "validation")]
    required += [f"mean_weight_{metric}" for metric in METRICS]
    if (frame.empty or not set(required).issubset(frame.columns)
            or frame.epoch.tolist() != list(range(1, len(frame) + 1))
            or not np.isfinite(frame[required].to_numpy(dtype=float)).all()
            or (frame[required].to_numpy(dtype=float) < 0).any()):
        raise ValueError
    if not np.allclose(frame[[f"mean_weight_{metric}" for metric in METRICS]].sum(axis=1), 1.):
        raise ValueError
    return frame


def curves(ax, frame, train, validation, best_epoch):
    marker = "o" if len(frame) == 1 else None
    ax.plot(frame.epoch, frame[train], color=TRAIN_COLOR, lw=1.05,
            marker=marker, markersize=3, label="Training")
    ax.plot(frame.epoch, frame[validation], color=VALIDATION_COLOR, lw=1.,
            ls="--", marker=marker, markersize=3, label="Validation")
    selected = frame[frame.epoch == best_epoch]
    if not selected.empty:
        ax.axvline(best_epoch, color="#7B8085", lw=.65, ls=":")
        ax.plot(best_epoch, selected.iloc[0][validation], "o", color=VALIDATION_COLOR,
                ms=3, markeredgecolor="white", markeredgewidth=.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Min-max MSE ↓")
    ax.grid(axis="y", alpha=.2, lw=.5)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set_xlim(0, max(2, int(frame.epoch.max())))


def plot_training(frame, output):
    plt, settings = plotting()
    best = int(frame.loc[frame.validation_mse.idxmin(), "epoch"])
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8), layout="constrained")
        zoom = frame[frame.epoch <= 100]
        for ax, group, title in zip(axes, (frame, zoom), ("Full training", "First 100 epochs")):
            curves(ax, group, "train_mse", "validation_mse", best)
            ax.set_title(title, loc="left")
        axes[1].set_ylim(axes[0].get_ylim())
        axes[0].legend(frameon=False)
        save_figure(fig, output / "training_convergence")
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.6), layout="constrained")
        for ax, metric, title in zip(axes.flat, METRICS, TITLES):
            curves(ax, frame, f"train_mse_{metric}", f"validation_mse_{metric}", best)
            ax.set_title(title, loc="left")
        axes[0, 0].legend(frameon=False)
        save_figure(fig, output / "task_convergence")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.16, 2.65), layout="constrained")
        for metric, title, color in zip(METRICS, TITLES, TASK_COLORS):
            ax.plot(frame.epoch, frame[f"mean_weight_{metric}"], label=title,
                    color=color, lw=1.05, marker="o" if len(frame) == 1 else None, markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean task weight")
        ax.set_xlim(0, max(2, int(frame.epoch.max())))
        ax.set_ylim(bottom=0.)
        ax.grid(axis="y", alpha=.2, lw=.5)
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.25))
        save_figure(fig, output / "task_weights")
        plt.close(fig)
    return best


def test_results(metrics):
    if metrics.get("model_name") != "dlmrd":
        raise ValueError
    rows = []
    results = metrics["splits"]["test"]
    for model in ("dlmrd", "persistence"):
        for metric in METRICS:
            values = results[model]["target_" + metric]
            mae, rmse, r2 = values["mae"], values["rmse"], values["r2"]
            if (not np.isfinite([mae, rmse]).all() or min(mae, rmse) < 0
                    or (r2 is not None and not np.isfinite(r2))):
                raise ValueError
            rows.append({"model": model, "metric": metric, "mae": mae, "rmse": rmse, "r2": r2})
    return pd.DataFrame(rows)


def plot_test(frame, output):
    plt, settings = plotting()
    units = ("orders", "pp", "pp", "pp", "s", "USD")
    with plt.rc_context(settings):
        fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.6), layout="constrained")
        for ax, metric, title, unit in zip(axes.flat, METRICS, TITLES, units):
            selected = frame[frame.metric == metric].set_index("model")
            values = selected.loc[["dlmrd", "persistence"], "rmse"].to_numpy()
            if unit == "pp":
                values = values * 100
            ax.bar([0, 1], values, color=[TRAIN_COLOR, "#A6ABB0"], width=.58)
            ax.set_xticks([0, 1], ["DL-MRD", "Persistence"])
            ax.set_title(title, loc="left")
            ax.set_ylabel(f"RMSE ({unit}) ↓")
            ax.grid(axis="y", alpha=.2, lw=.5)
            ax.set_axisbelow(True)
        save_figure(fig, output / "test_metrics")
        plt.close(fig)


def run(output):
    output = Path(output)
    source = output / "training_history.csv"
    if not source.exists():
        return {"status": "waiting_for_history"}
    frame = load_history(source)
    destination = output / "paper_figures"
    destination.mkdir(parents=True, exist_ok=True)
    best_epoch = plot_training(frame, destination)
    result = {"status": "training", "epochs_run": len(frame), "best_epoch": best_epoch,
              "normalization": "Training-only min-max; train/validation curves use equal task mean MSE.",
              "smoothing": False, "zoom_epochs": [1, min(100, len(frame))],
              "png_dpi": 600, "width_inches": 7.16}
    source = output / "metrics.json"
    if source.exists():
        metrics = json.loads(source.read_text())
        if (metrics["epochs_run"] != len(frame) or metrics["best_epoch"] != best_epoch):
            raise ValueError
        test = test_results(metrics)
        test.to_csv(output / "test_results.csv", index=False)
        plot_test(test, destination)
        result["status"] = "complete"
        result["loss_weighting"] = metrics["loss_weighting"]
    caption = (
        "Training convergence: full unsmoothed training/validation curves and the first 100 epochs "
        "(or all available epochs), with the same vertical scale. MSE is measured after training-only "
        "min-max normalization and averaged equally across six targets; it is not the dynamically "
        "weighted WESM training objective. Dotted lines mark the common checkpoint with minimum "
        "mean validation MSE; test metrics are not used to select it.\n\n"
        "Task convergence: the six targets use separate vertical scales and the same selected epoch. "
        "Task weights are sample-weighted averages of the weights used in each epoch.\n\n"
        "Test metrics, when present: RMSE of the selected DL-MRD checkpoint and last-observation "
        "persistence on the held-out test split. Rate errors are plotted in percentage points (pp); "
        "test_results.csv retains the original units and additionally contains MAE and R². "
        "These are predictive errors, not simulator policy performance. No across-seed uncertainty "
        "is estimated from this single training run.\n"
    )
    (destination / "captions.txt").write_text(caption)
    (destination / "figure_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(run(parser.parse_args().output), indent=2))
