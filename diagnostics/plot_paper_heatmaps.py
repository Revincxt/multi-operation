"""把既有动作扫描结果绘制为三种规模的论文版六指标总览。

使用方式:
    python3 -m diagnostics.plot_paper_heatmaps --input experiments/density_action_sweep_20261008

只读取 screening，沿用当时的候选动作和统计口径；不重跑仿真，不插值。
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, patheffects
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, PercentFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnostics.plot_density_heatmaps import METRICS, NOTES, color_limits, prepare_metrics


PANEL_NAMES = {
    "baseline_request_num": ("Bubble requests", "=", r"Requests ($10^3$)"),
    "request_rate": ("Order request rate", r"$\uparrow$", "%"),
    "driver_accept_rate": ("Driver acceptance", r"$\uparrow$", "%"),
    "order_completion_rate": ("Order completion", r"$\uparrow$", "%"),
    "mean_wait_to_match_seconds": ("Mean waiting time", r"$\downarrow$", "Time (s)"),
    "platform_revenue": ("Platform revenue", r"$\uparrow$", r"Revenue ($10^3$ USD)"),
}
SCENARIOS = {"low": "Small-scale", "medium": "Medium-scale", "high": "Large-scale"}
REFERENCE_PRICE, REFERENCE_RADIUS = 1.5, 3.0
MISSING_COLOR = "#D9DDE1"


def _font_family():
    for family in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family),
                                  fallback_to_default=False)
            return family
        except ValueError:
            continue
    raise ValueError


def draw_paper_heatmap(fig, ax, data, metric, prices, radii, limits, index):
    """候选动作等宽排列，色标三种规模共享；冒泡数保留逐格标注。"""
    values = data.loc[data.metric == metric].pivot(
        index="price_multiplier", columns="matching_radius_km", values="value").reindex(
            index=prices, columns=radii).to_numpy()
    name, direction, unit = PANEL_NAMES[metric]
    cmap_name = "viridis_r" if metric == "mean_wait_to_match_seconds" else "viridis"
    cmap = plt.get_cmap(cmap_name).with_extremes(bad=MISSING_COLOR)
    # 每个动作格导出为矢量色块，PDF/SVG 放大不会把热力图变成位图。
    im = ax.pcolormesh(np.arange(len(radii) + 1) - 0.5, np.arange(len(prices) + 1) - 0.5,
                       np.ma.masked_invalid(values), shading="flat", cmap=cmap,
                       vmin=limits[0], vmax=limits[1], edgecolors="none",
                       antialiased=False, rasterized=False)
    ax.set_xlim(-0.5, len(radii) - 0.5)
    ax.set_ylim(-0.5, len(prices) - 0.5)
    ax.set_title(f"({chr(97 + index)}) {name} {direction}", loc="left", fontsize=8.5, pad=7)
    ax.set_xticks(range(len(radii)), [f"{value:g}" for value in radii])
    ax.set_yticks(range(len(prices)), [f"{value:g}" for value in prices])
    ax.tick_params(length=2, width=0.45, pad=2, labelsize=7)
    if index % 3 == 0:
        ax.set_ylabel("Price multiplier", fontsize=8.5, labelpad=3)
    else:
        ax.tick_params(labelleft=False)
    if index >= 3:
        ax.set_xlabel("Matching radius (km)", fontsize=8.5, labelpad=4)
    else:
        ax.tick_params(labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#444444")
    if REFERENCE_PRICE in prices and REFERENCE_RADIUS in radii:
        # 冒泡数格内有文字，参考框用细线，避免白色描边压住数字。
        annotated = metric == "baseline_request_num"
        reference = Rectangle((radii.index(REFERENCE_RADIUS) - 0.5,
                               prices.index(REFERENCE_PRICE) - 0.5), 1, 1,
                              fill=False, edgecolor="#222222", linewidth=0.6 if annotated else 1.0,
                              linestyle=(0, (3, 2)), zorder=5)
        reference.set_path_effects([patheffects.Stroke(linewidth=1.1 if annotated else 2.1,
                                                      foreground="white"),
                                    patheffects.Normal()])
        ax.add_patch(reference)
    # 冒泡数沿用原图的逐格显示，避免固定需求被一块纯色和中央白框替代。
    # 不加千位分隔符，给五位订单数留出空间；色条仍以千单为单位。
    if metric == "baseline_request_num":
        ax.set_xticks(np.arange(len(radii) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(prices) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=0.2, alpha=0.25)
        ax.tick_params(which="minor", length=0)
        for (row, column), value in np.ndenumerate(values):
            if np.isfinite(value):
                label = f"{value:.0f}" if value == int(value) else f"{value:g}"
                red, green, blue, _ = im.cmap(im.norm(value))
                color = ("#222222" if 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.55
                         else "white")
            else:
                label, color = "—", "#444444"
            ax.text(column, row, label, ha="center", va="center", fontsize=4.2,
                    color=color, clip_on=True)
    cbar = fig.colorbar(im, ax=ax, fraction=0.043, pad=0.035, aspect=28)
    cbar.set_label(unit, fontsize=7, labelpad=3)
    cbar.ax.tick_params(labelsize=7, length=2, width=0.45, pad=2)
    cbar.outline.set_linewidth(0.45)
    cbar.set_ticks(np.linspace(limits[0], limits[1], 5))
    if metric.endswith("rate"):
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    elif metric in ("baseline_request_num", "platform_revenue"):
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.1f}"))
    else:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    # Matplotlib 默认会把连续色条栅格化，这里也保留为矢量。
    cbar.solids.set_rasterized(False)
    return im


def plot_paper_heatmaps(input_dir="experiments/density_action_sweep_20261008", output_dir=None):
    source = Path(input_dir)
    report = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    config = report["config"]
    if report["status"] != "complete" or not set(config["scenarios"]).issubset(SCENARIOS):
        raise ValueError
    data = prepare_metrics(pd.read_csv(source / "screening_runs.csv"), config)
    prices, radii = sorted(config["prices"]), sorted(config["radii"])
    limits = {metric: color_limits(data.loc[data.metric == metric, "value"], metric) for metric in METRICS}
    output_dir = (Path(output_dir) if output_dir is not None
                  else source / "six_metric_heatmaps" / "paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "heatmap_values.csv", index=False)
    family = _font_family()
    seed_count = len(config["screening_seeds"])
    reference_present = REFERENCE_PRICE in prices and REFERENCE_RADIUS in radii
    files, captions = [], []
    style = {
        "font.family": family, "font.size": 8.5, "mathtext.fontset": "stix",
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "image.composite_image": False,
    }
    with plt.rc_context(style), PdfPages(output_dir / "all_scenarios_six_metrics.pdf") as combined:
        for scenario in config["scenarios"]:
            settings = report["scenario_settings"][scenario]
            title = SCENARIOS[scenario]
            fig, axes = plt.subplots(2, 3, figsize=(7.16, 6.1))
            fig.subplots_adjust(left=0.072, right=0.92, bottom=0.145, top=0.865,
                                hspace=0.26, wspace=0.50)
            for index, (ax, metric) in enumerate(zip(axes.flat, METRICS)):
                draw_paper_heatmap(fig, ax, data.loc[data.scenario == scenario], metric,
                                   prices, radii, limits[metric], index)
            fig.suptitle(f"{title} scenario", fontsize=11, y=0.987)
            fig.text(0.5, 0.947,
                     f"{settings['actual_sampled_orders']:,} sampled requests  |  "
                     f"{settings['driver_num']:,} drivers  |  6 grids", ha="center", fontsize=8.5)
            handles = [Patch(facecolor=MISSING_COLOR, edgecolor="#AAAAAA", label="Undefined")]
            if reference_present:
                handles.insert(0, Patch(facecolor="white", edgecolor="#222222", linestyle="--",
                                        label="Reference: radius = 3 km, price multiplier = 1.5"))
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.047),
                       ncol=2, frameon=False, fontsize=7.5, handlelength=1.6, columnspacing=2)
            fig.text(0.5, 0.022,
                     r"$\uparrow$ Higher is better     $\downarrow$ Lower is better     = Exogenous demand",
                     ha="center", fontsize=8)
            for extension in ("pdf", "svg", "png"):
                path = output_dir / f"{scenario}_six_metrics.{extension}"
                fig.savefig(path, dpi=600, facecolor="white")
                files.append(path.name)
            combined.savefig(fig, facecolor="white")
            plt.close(fig)
            captions.append(
                f"{title} scenario: six-metric fixed-action screening over {settings['actual_sampled_orders']:,} "
                f"sampled requests and {settings['driver_num']:,} drivers. Panels show (a) recorded bubble "
                "requests, (b) order request rate, (c) driver acceptance rate, (d) order completion rate, "
                "(e) mean waiting time to matching, and (f) platform revenue. "
                f"Each cell represents the mean of {seed_count} screening run(s) "
                "for the indicated price multiplier and matching radius.\n")
    files.append("all_scenarios_six_metrics.pdf")
    hour, minute = map(int, config["start_time"].split(":"))
    end = hour * 60 + minute + config["simulation_steps"]
    notes = (
        f"Common setup: {config['date']}, {config['start_time']}–{end // 60:02d}:{end % 60:02d}, "
        "six grids, one fixed action applied across all grids and decision steps. These figures use "
        "only the complete screening grid; independent-seed confirmation of revenue-shortlisted "
        "actions is not mixed into the heatmaps. Candidates are those actually tested in this sweep, "
        "not any later revision of the training action set.\n\n"
        "All six panels share the action axes. Unevenly spaced numerical action candidates are displayed "
        "as equally spaced categorical cells, without interpolation or smoothing. The color scale for "
        "each metric is shared across the three scenarios. Percentages are displayed without clipping "
        "at 100%; count and revenue colorbars use units of 1,000. Gray cells are undefined, not zero. "
        "Dashed boxes identify the reference action when present. Arrows describe individual service "
        "preferences, not simultaneous or overall optimality. Bubble requests are exogenous and have "
        "no preference direction. Their panel retains the original viridis palette and annotates "
        "each tested action with the recorded request count (unscaled, without thousands separators); "
        "identical colors and counts within a scenario reflect fixed exogenous demand, not missing data.\n\n"
        "Recorded bubble requests exclude the final generated batch not yet entered into minute-level "
        "statistics, so counts can be smaller than the sampled input shown in each heading. "
        "Order request rate is submitted requests divided by recorded bubble requests; demand resampling "
        "can make it exceed 100%, so it is not a probability. Driver acceptance rate is accepts divided "
        "by offers. Order completion rate is retained matches divided by candidate attempts plus "
        "unmatched cancellations: retries repeat in the denominator, and this is not unique-order "
        "trip completion. Mean waiting time is weighted by retained matches and excludes unserved "
        "passengers and pickup time. Platform revenue is commission on matched fares, not net profit.\n\n"
        "Limitations: these are descriptive screening results, not a multi-seed stability or significance "
        "analysis. Drivers and requests change together across scenarios; the figures do not isolate "
        "a causal effect of demand density. Exact values, seed counts and valid-seed counts are provided "
        "in heatmap_values.csv.\n"
    )
    (output_dir / "captions.txt").write_text("\n".join(captions) + "\n" + notes, encoding="utf-8")
    manifest = {
        "phase": "screening", "scenarios": config["scenarios"], "seeds": config["screening_seeds"],
        "prices": prices, "radii_km": radii, "color_limits_shared_across_scenarios": limits,
        "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in ("screening_runs.csv", "manifest.json")},
        "plot_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metric_preparation_code_sha256": hashlib.sha256(
            Path(__file__).with_name("plot_density_heatmaps.py").read_bytes()).hexdigest(),
        "font_family": family, "figure_size_inches": [7.16, 6.1], "png_dpi": 600,
        "interpolation": "none", "heatmap_rendering": "vector cells", "smoothing": False,
        "bubble_request_display": {"colormap": "viridis", "annotations": "each action cell",
                                   "annotation_units": "requests", "colorbar_units": "1000 requests"},
        "reference_action": {"price_multiplier": REFERENCE_PRICE, "matching_radius_km": REFERENCE_RADIUS,
                             "present": reference_present},
        "notes": NOTES, "files": files + ["heatmap_values.csv", "captions.txt"],
    }
    (output_dir / "heatmap_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"论文版六指标热力图已保存至：{output_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="导出三种规模的论文版六指标热力图")
    parser.add_argument("--input", dest="input_dir", default="experiments/density_action_sweep_20261008")
    parser.add_argument("--output", dest="output_dir")
    plot_paper_heatmaps(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
