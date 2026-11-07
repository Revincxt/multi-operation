"""从在线PPO记录导出完整训练曲线及前100次更新放大图，不平滑数据。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"refactor_simulator_matplotlib"))

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAIN_COLOR = "#268F85"
VALIDATION_COLOR = "#C78860"
DEFAULT = Path("models/online_ppo_100000_20261104/seed81")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font_family():
    for name in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            pass
    raise ValueError


def axes_style(ax):
    ax.grid(axis="y", color="#E1E6E9", lw=.55, alpha=.85)
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.margins(y=.08)


def save(fig, output, name):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output/f"{name}.{extension}", dpi=600, facecolor="white")
    plt.close(fig)


def returns_panel(ax, training, validation, end, title):
    train = training[training.iteration <= end]
    valid = validation[validation.iteration <= end]
    ax.plot(train.iteration, train.revenue_mean_usd, color=TRAIN_COLOR, lw=.7, alpha=.65)
    ax.plot(valid.iteration, valid.revenue_mean_usd, color=VALIDATION_COLOR,
            lw=1.25, ls=(0, (4, 1.8)), marker="o" if end <= 100 else None, ms=3.)
    ax.set_xlim(0, end)
    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel("Policy updates")
    ax.set_ylabel("Mean episode return (USD) ↑")
    axes_style(ax)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    if end <= 100:
        ax.set_xticks([0, 25, 50, 75, 100])


def run(input_dir, output=None):
    input_dir = Path(input_dir)
    output = Path(output) if output else input_dir/"paper_figures"
    output.mkdir(parents=True, exist_ok=True)
    sources = {name: digest(input_dir/name) for name in ("training_history.csv", "validation_history.csv")}
    training = pd.read_csv(input_dir/"training_history.csv")
    validation = pd.read_csv(input_dir/"validation_history.csv")
    assert training.iteration.tolist() == list(range(1, len(training)+1))
    assert validation.iteration.is_monotonic_increasing and validation.iteration.is_unique
    assert validation.iteration.isin(training.iteration).all()
    assert np.isfinite(training.select_dtypes(include=np.number).to_numpy()).all()
    assert np.isfinite(validation.select_dtypes(include=np.number).to_numpy()).all()
    assert training.episodes.eq(training.iteration*16).all()
    assert training.environment_steps.eq(training.episodes*56).all()
    end = int(training.iteration.iloc[-1])
    style = {"font.family": "serif", "font.serif": [font_family()], "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "axes.linewidth": .75,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "text.color": "#26323F",
        "axes.labelcolor": "#26323F", "axes.edgecolor": "#55606D"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.65), gridspec_kw={"width_ratios": [1.55, 1]})
        returns_panel(axes[0], training, validation, end, "(a) Full training")
        returns_panel(axes[1], training, validation, min(100, end), "(b) First 100 updates")
        axes[0].axvspan(0, min(100, end), color="#808B96", alpha=.08, lw=0)
        handles = [Line2D([], [], color=TRAIN_COLOR, lw=1.2, label="Training"),
                   Line2D([], [], color=VALIDATION_COLOR, lw=1.3, ls=(0, (4, 1.8)), label="Validation")]
        fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(.52, 1.), columnspacing=3.)
        fig.tight_layout(rect=(0, 0, 1, .91), w_pad=2.5, pad=.9)
        save(fig, output, "training_curve_overall_first100")

        fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.9))
        columns = ("value_loss", "policy_loss", "entropy", "joint_kl", "clip_fraction", "effective_epochs")
        titles = ("Normalized value loss", "Policy loss", "Conditional entropy",
                  "Final joint KL", "Clip fraction", "Effective PPO epochs")
        for index, (ax, column, title) in enumerate(zip(axes.flat, columns, titles)):
            ax.plot(training.iteration, training[column], color=TRAIN_COLOR, lw=.65, alpha=.8)
            ax.set_title(f"({chr(97+index)}) {title}", loc="left", pad=8)
            ax.set_xlabel("Policy updates")
            ax.set_xlim(0, end)
            axes_style(ax)
            if column == "value_loss":
                assert training[column].gt(0).all()
                ax.set_yscale("log")
            elif column == "entropy":
                ax.axhline(6*np.log(110), color="#8A939B", lw=.8, ls="--")
            elif column == "joint_kl":
                ax.axhline(.02, color=VALIDATION_COLOR, lw=.8, ls="--", label="Target")
                ax.legend(fontsize=8)
            elif column == "effective_epochs":
                ax.axhline(5, color="#8A939B", lw=.8, ls="--")
                ax.set_ylim(0, 5.4)
        fig.tight_layout(pad=1., w_pad=2., h_pad=2.)
        save(fig, output, "training_diagnostics")
    # 原始记录随图留存，保证图中每个点均可追溯。
    training.to_csv(output/"training_curve_data.csv", index=False)
    validation.to_csv(output/"validation_curve_data.csv", index=False)
    assert all(digest(input_dir/name) == value for name, value in sources.items())
    captions = [
        "图1：AR-PPO完整训练收益曲线，右侧放大前100次策略更新。训练曲线每点为16条新采样完整仿真的平均收益；验证曲线每点为固定30个场景的平均收益，每16次更新记录。两条曲线均为原始值，未平滑；连线仅连接已记录点。左图浅灰区域对应右图范围。",
        "图2：价值损失、策略损失、条件熵、最终联合KL、裁剪比例及有效PPO更新遍数。价值损失采用对数纵轴；熵虚线为6×ln(110)，有效遍数虚线为配置上限5。最终联合KL虚线为目标0.02，不是硬约束；训练停止本批更新的判断使用小批次估计，故最终整批KL仍可能超出目标。",
        "条件：6网格、250司机、2015-05-18；共同预热20分钟，收益统计05:20—10:00的平台佣金。",
        f"训练初始化81，共{end}次策略更新、{int(training.episodes.iloc[-1])}条完整仿真、{int(training.environment_steps.iloc[-1])}条城市级计分转移。每次更新对应896条新转移，实际优化遍数由KL条件决定。",
        "这是单个训练初始化的曲线，不是跨训练种子的均值或置信区间。当前仍保持暂停；未重新训练或评估模型，也没有按验证高点重新选择checkpoint。",
    ]
    (output/"captions.txt").write_text("\n\n".join(captions)+"\n", encoding="utf-8")
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "input": str(input_dir),
        "training_updates": end, "training_episodes": int(training.episodes.iloc[-1]),
        "validation_points": len(validation), "source_sha256": sources,
        "plot_source_sha256": digest(__file__), "smoothing": None, "training_resumed": False,
        "files_sha256": {p.name: digest(p) for p in output.iterdir() if p.suffix in ("png", "pdf", "svg", ".png", ".pdf", ".svg")}}
    (output/"figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"output": str(output), "training_updates": end,
                      "training_episodes": int(training.episodes.iloc[-1]), "validation_points": len(validation)},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.input, args.output)
