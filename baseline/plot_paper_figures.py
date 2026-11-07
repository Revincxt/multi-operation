"""从已有训练记录导出论文曲线，不平滑数据、不重新评估模型。

使用方式:
    python3 -m baseline.plot_paper_figures --input models/lstm_grid6_low_mixed47k_mps100_20261029

默认输出到输入目录下的 paper_figures：矢量 PDF/SVG、600 dpi PNG 和图注。
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator, ScalarFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline.plot_results import METRICS


TRAIN_COLOR = "#0072B2"
VALIDATION_COLOR = "#D55E00"
SELECTION_COLOR = "#727272"
VALIDATION_STYLE = (0, (3.5, 1.6))
PANEL_NAMES = (
    "Bubble requests", "Order request rate", "Driver acceptance rate",
    "Order completion rate", "Mean waiting time", "Platform revenue",
)


def _font_family():
    """优先使用 Times，其他系统没有该字体时采用衬线字体。"""
    for family in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family),
                                  fallback_to_default=False)
            return family
        except ValueError:
            continue
    raise ValueError


def _load_history(input_dir):
    history = pd.read_csv(input_dir / "training_history.csv")
    report = json.loads((input_dir / "metrics.json").read_text(encoding="utf-8"))
    columns = ["train_mse", "validation_mse"] + [
        f"{split}_mse_{metric}" for metric in METRICS for split in ("train", "validation")]
    if (history.empty or not set(columns + ["epoch"]).issubset(history.columns)
            or len(history) != report["epochs_run"]
            or history["epoch"].tolist() != list(range(1, len(history) + 1))
            or not np.isfinite(history[columns].to_numpy()).all()
            or (history[columns].to_numpy() < 0).any()
            or report.get("model_name", "lstm") not in ("lstm", "timesnet")):
        raise ValueError
    # 检查图中选中的轮次确实来自固定等权验证误差，而非某个单项的最优值。
    best_epoch = int(history.loc[history["validation_mse"].idxmin(), "epoch"])
    best_mse = float(history.loc[history["epoch"] == best_epoch, "validation_mse"].iloc[0])
    if (best_epoch != report["best_epoch"]
            or not np.isclose(best_mse, report.get("best_validation_standardized_mse", best_mse),
                              rtol=1e-9, atol=1e-12)):
        raise ValueError
    return history, report


def _axes(ax, title, first_epoch, last_epoch):
    ax.set_title(title, loc="left", pad=7, fontsize=9)
    ax.set_xlim(first_epoch, max(first_epoch + 1, last_epoch))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.4, zorder=0)
    ax.tick_params(direction="out", width=0.55, length=3, pad=3)
    ax.margins(y=0.09)


def _curves(ax, frame, train_column, validation_column, best_epoch):
    ax.plot(frame["epoch"], frame[train_column], color=TRAIN_COLOR,
            linewidth=1.05, zorder=3)
    ax.plot(frame["epoch"], frame[validation_column], color=VALIDATION_COLOR,
            linestyle=VALIDATION_STYLE, linewidth=0.85, zorder=4)
    selected = frame.loc[frame["epoch"] == best_epoch, validation_column]
    # 最佳轮次不在前 100 轮时，不在放大图里标出范围外的点。
    if selected.empty:
        return
    ax.axvline(best_epoch, color=SELECTION_COLOR, linewidth=0.75,
               linestyle=(0, (4, 2)), zorder=2)
    ax.plot(best_epoch, selected.iloc[0], "o", color=VALIDATION_COLOR, markersize=3.2,
            markeredgecolor="white", markeredgewidth=0.45, zorder=5)


def _legend(fig, best_epoch):
    handles = [
        Line2D([], [], color=TRAIN_COLOR, linewidth=1.2, label="Training"),
        Line2D([], [], color=VALIDATION_COLOR, linewidth=1.1,
               linestyle=VALIDATION_STYLE, label="Validation"),
        Line2D([], [], color=SELECTION_COLOR, linewidth=0.8,
               linestyle=(0, (4, 2)), label=f"Selected epoch ({best_epoch})"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.995),
               ncol=3, frameon=False, handlelength=2.7, columnspacing=2.0)


def _save(fig, output_dir, name):
    paths = []
    for extension in ("pdf", "svg", "png"):
        path = output_dir / f"{name}.{extension}"
        # 不使用 tight 裁切，保持论文中的双栏物理宽度。
        fig.savefig(path, dpi=600, facecolor="white")
        paths.append(path.name)
    plt.close(fig)
    return paths


def plot_paper_figures(input_dir, output_dir=None):
    """保留原训练文件，在独立子目录输出可复现的论文版曲线。"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir / "paper_figures"
    history, report = _load_history(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_epoch, epochs = report["best_epoch"], len(history)
    best_mse = float(history.loc[history["epoch"] == best_epoch, "validation_mse"].iloc[0])
    zoom_end = min(100, epochs)
    zoom = history.head(zoom_end)
    family = _font_family()
    style = {
        "font.family": family, "font.size": 8.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "xtick.color": "#333333", "ytick.color": "#333333",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "mathtext.fontset": "stix", "path.simplify": False,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))
        fig.subplots_adjust(left=0.093, right=0.97, bottom=0.19, top=0.81, wspace=0.36)
        _curves(axes[0], history, "train_mse", "validation_mse", best_epoch)
        _axes(axes[0], f"(a) Full training ({epochs} epochs)", 0, epochs)
        _curves(axes[1], zoom, "train_mse", "validation_mse", best_epoch)
        _axes(axes[1], f"(b) First {zoom_end} epochs", 0, zoom_end)
        axes[1].set_ylim(axes[0].get_ylim())
        axes[0].set_ylabel(r"Mean standardized MSE $\downarrow$")
        for ax in axes:
            ax.set_xlabel("Epoch")
        _legend(fig, best_epoch)
        files = _save(fig, output_dir, "fig01_training_convergence")

        fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.75))
        fig.subplots_adjust(left=0.087, right=0.97, bottom=0.105, top=0.878,
                            wspace=0.40, hspace=0.38)
        for index, (ax, metric, name) in enumerate(zip(axes.flat, METRICS, PANEL_NAMES)):
            _curves(ax, history, f"train_mse_{metric}", f"validation_mse_{metric}", best_epoch)
            _axes(ax, f"({chr(97 + index)}) {name}", 0, epochs)
            ax.xaxis.set_major_locator(FixedLocator(sorted({0, epochs // 2, epochs})))
            if index >= 3:
                ax.set_xlabel("Epoch")
            if index % 3 == 0:
                ax.set_ylabel(r"Standardized MSE $\downarrow$")
        _legend(fig, best_epoch)
        files += _save(fig, output_dir, "fig02_task_convergence")

    model_name = {"lstm": "LSTM", "timesnet": "TimesNet"}[report.get("model_name", "lstm")]
    captions = (
        f"Figure 1. Convergence of the {model_name} model over {epochs} training epochs. "
        f"(a) Training and validation mean standardized MSE across six prediction tasks. "
        f"(b) The same training and validation curves for the first {zoom_end} epochs, "
        "using the same vertical scale as panel (a). "
        f"The checkpoint at epoch {best_epoch} was selected by the minimum validation MSE "
        f"({best_mse:.6f}), not by the test set.\n\n"
        f"Figure 2. Task-wise training and validation standardized MSE: "
        "(a) bubble requests, (b) order request rate, (c) driver acceptance rate, "
        "(d) order completion rate, (e) mean passenger waiting time to matching, "
        "and (f) platform revenue. Dashed vertical lines mark the same checkpoint selected "
        "by the six-task mean validation MSE; they do not denote separate task-wise minima.\n\n"
        "Notes. Curves show the original, unsmoothed per-epoch observations from one training run; "
        "no across-seed confidence intervals are available. Lower MSE is better. "
        "Figure 2 uses different vertical scales across tasks. Plotted overall MSE uses equal task "
        "weights, rather than the dynamically weighted WESM objective. Training MSE is recorded "
        "during parameter updates with dropout enabled; validation MSE is measured after each "
        "epoch with dropout disabled. Their absolute gap is therefore not a like-for-like "
        "train-versus-validation evaluation.\n"
    )
    (output_dir / "captions.txt").write_text(captions, encoding="utf-8")
    manifest = {
        "source_directory": str(input_dir.resolve()),
        "source_sha256": {name: hashlib.sha256((input_dir / name).read_bytes()).hexdigest()
                          for name in ("training_history.csv", "metrics.json")},
        "model": model_name, "epochs": epochs, "selected_epoch": best_epoch,
        "best_validation_mse": best_mse, "zoom_start_epoch": 1, "zoom_end_epoch": zoom_end,
        "smoothing": False, "font_family": family, "width_inches": 7.16,
        "png_dpi": 600, "pdf_fonttype": 42, "files": files,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"论文曲线已保存至：{output_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="导出论文用的时序模型训练曲线")
    parser.add_argument("--input", dest="input_dir", required=True)
    parser.add_argument("--output", dest="output_dir")
    plot_paper_figures(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
